"""Metrics for the field-map surrogate.

Everything here is computed on the 4-channel physical representation, so image
quality and physical accuracy are measured on the same object:

  * SSIM, overall and PER CHANNEL (von Mises / PEEQ / deflection / mask)
  * impact-position error, in pixels AND in millimetres via the EXACT 3.000 mm
    cell size -- no camera model, no fitView zoom, no isometric projection
  * per-channel physical error in native units (MPa, strain, mm)
  * image-vs-scalar consistency: the peak value decoded from the generated
    field compared against the physics head's scalar prediction. Because the
    channel mapping is invertible this is a real cross-check between the two
    branches, which are otherwise independent.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .data import (CELL_MM, CHANNEL_NAMES, MASK_CHANNEL, PLATE_MM,
                  U3_CHANNEL, hard_intact_peak_mm, norm_to_fields)

_WINDOW_CACHE: dict = {}


def _gauss_window(channels: int, size: int = 11, sigma: float = 1.5, device=None, dtype=None):
    key = (channels, size, sigma, str(device), str(dtype))
    if key not in _WINDOW_CACHE:
        ax = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(ax ** 2) / (2 * sigma ** 2))
        k = torch.outer(g, g)
        k = (k / k.sum()).to(device=device, dtype=dtype)
        _WINDOW_CACHE[key] = k.expand(channels, 1, size, size).contiguous()
    return _WINDOW_CACHE[key]


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0,
         per_channel: bool = False):
    """SSIM for tensors in [-1, 1]. Returns (B,) or (B, C) if per_channel."""
    b, c, _, _ = pred.shape
    w = _gauss_window(c, device=pred.device, dtype=pred.dtype)
    pad = w.shape[-1] // 2
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    mu_x = F.conv2d(pred, w, padding=pad, groups=c)
    mu_y = F.conv2d(target, w, padding=pad, groups=c)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sx = F.conv2d(pred * pred, w, padding=pad, groups=c) - mu_xx
    sy = F.conv2d(target * target, w, padding=pad, groups=c) - mu_yy
    sxy = F.conv2d(pred * target, w, padding=pad, groups=c) - mu_xy
    m = ((2 * mu_xy + C1) * (2 * sxy + C2)) / ((mu_xx + mu_yy + C1) * (sx + sy + C2))
    return m.mean(dim=(2, 3)) if per_channel else m.mean(dim=(1, 2, 3))


# MS-SSIM weights: the standard five-scale set, truncated to four and
# renormalised. At 200x200 a fifth scale would leave a 12x12 image under an
# 11x11 window, which measures almost nothing.
MSSSIM_WEIGHTS = (0.0517, 0.3295, 0.3462, 0.2726)


def ms_ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 2.0,
            weights=MSSSIM_WEIGHTS, per_channel: bool = False):
    """Multi-scale SSIM for tensors in [-1, 1]. Differentiable.

    Single-scale SSIM sees one window size, so it is blind to agreement that
    only exists at coarser scales. The measured spectra say that is exactly
    where the reproducible content of this field lives: 83.5% of the von Mises
    signal power sits above 16 cells while only 11.9% of the run-to-run
    disagreement does. A term evaluated at one scale spends most of its
    gradient on the band that does not repeat between simulations.

    Contrast and structure are taken at every scale, luminance only at the
    coarsest, which is the usual construction; the alternative double-counts a
    quantity the L1 term already supervises.
    """
    x = pred.float()
    y = target.float()
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    c = x.shape[1]
    cs_all = []
    lum = None
    for i in range(len(weights)):
        w = _gauss_window(c, device=x.device, dtype=x.dtype)
        pad = w.shape[-1] // 2
        mu_x = F.conv2d(x, w, padding=pad, groups=c)
        mu_y = F.conv2d(y, w, padding=pad, groups=c)
        mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
        sx = F.conv2d(x * x, w, padding=pad, groups=c) - mu_xx
        sy = F.conv2d(y * y, w, padding=pad, groups=c) - mu_yy
        sxy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy
        cs = (2 * sxy + C2) / (sx + sy + C2)
        cs_all.append(cs.clamp(min=1e-6).mean(dim=(2, 3)))
        if i == len(weights) - 1:
            lum = ((2 * mu_xy + C1) / (mu_xx + mu_yy + C1))
            lum = lum.clamp(min=1e-6).mean(dim=(2, 3))
        else:
            x = F.avg_pool2d(x, 2)
            y = F.avg_pool2d(y, 2)
    out = lum ** weights[-1]
    for i, wt in enumerate(weights):
        out = out * (cs_all[i] ** wt)
    return out if per_channel else out.mean(dim=1)


# ---------------------------------------------------------------- position
# The impact locator, and why the obvious one does not work.
#
# `soft_centroid` on the deflection channel is a natural first choice, and it is
# an earlier training target. Measured against the impact
# coordinates the simulation was actually given, it is wrong by 40.6 mm on
# GROUND-TRUTH field maps. The reason is physical: when the plate perforates,
# the material at the strike point is deleted, so the surviving deflection peak
# sits on the hole rim rather than the impact. Even a hard argmax of u3 is
# 32.6 mm off. The hole is the signal, not the deflection peak.
#
# Adding the missing-material map to a sharpened deflection map fixes both
# regimes at once: the hole term dominates where a hole exists, and vanishes
# where it does not. Ground-truth floor 1.11 mm (0.09 mm non-perforating,
# 1.32 mm perforated), and it is differentiable, so it can also be trained on.
IMPACT_POW = 8.0


def impact_saliency(fields: torch.Tensor, power: float = IMPACT_POW):
    """(B,4,H,W) normalised -> (B,H,W) non-negative map peaked at the impact."""
    u3 = norm_to_fields(fields)[:, U3_CHANNEL]
    u3n = u3 / u3.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    hole = (1.0 - (fields[:, MASK_CHANNEL] + 1.0) / 2.0).clamp(0.0, 1.0)
    return hole + u3n ** power


def _centroid(w: torch.Tensor):
    """(B,H,W) weights -> (B,2) centroid as (index along axis0, along axis1)."""
    b, n0, n1 = w.shape
    s = w.sum(dim=(1, 2)).clamp_min(1e-9)
    a0 = torch.arange(n0, device=w.device, dtype=w.dtype).view(1, n0)
    a1 = torch.arange(n1, device=w.device, dtype=w.dtype).view(1, n1)
    return torch.stack([(w.sum(2) * a0).sum(1) / s, (w.sum(1) * a1).sum(1) / s], dim=1)


def locate_impact_mm(fields: torch.Tensor, power: float = IMPACT_POW):
    """(B,4,H,W) -> (B,2) impact position in mm from the plate centre.

    Axis 0 of the field array indexes x and axis 1 indexes y; both map as
    idx * 3.000 - 298.5 mm. That was established by regressing the deflection
    argmax against the recorded impact coordinates, not assumed.
    """
    c = _centroid(impact_saliency(fields, power))
    return (c + 0.5) * CELL_MM - PLATE_MM / 2.0


def impact_position_loss(fake: torch.Tensor, true_xy_mm: torch.Tensor,
                         power: float = IMPACT_POW):
    """Distance from the GENERATED impact location to the true coordinates.

    The impact coordinates are an INPUT to the generator and are recorded per
    run, so the exact answer is available; matching a soft centroid of the real
    image would inherit that locator's 40 mm error. Scaled to decimetres to keep the term O(0.1).
    """
    return F.smooth_l1_loss(locate_impact_mm(fake, power) / 100.0,
                            true_xy_mm / 100.0)


@torch.no_grad()
def impact_position_error_mm(fields: torch.Tensor, true_xy_mm: torch.Tensor,
                             power: float = IMPACT_POW):
    d = locate_impact_mm(fields, power) - true_xy_mm
    return torch.linalg.vector_norm(d, dim=1)


def soft_centroid(fields: torch.Tensor, channel: int = 2, temperature: float = 12.0):
    """Soft-argmax of one channel -> (B,2) centroid in pixel coordinates.

    Channel 2 is |U3|: the plate's deflection peaks at the strike point, which
    makes it the physically correct locator (and it is present whether or not
    the plate perforates).
    """
    x = fields[:, channel]
    b, h, w = x.shape
    flat = x.reshape(b, -1)
    flat = flat - flat.max(dim=1, keepdim=True).values
    p = torch.softmax(flat * temperature, dim=1).view(b, h, w)
    ys = torch.arange(h, device=x.device, dtype=p.dtype)
    xs = torch.arange(w, device=x.device, dtype=p.dtype)
    cy = (p.sum(dim=2) * ys).sum(dim=1)
    cx = (p.sum(dim=1) * xs).sum(dim=1)
    return torch.stack([cx, cy], dim=1)


def position_loss(pred, target, temperature: float = 12.0):
    a = soft_centroid(pred, temperature=temperature)
    b = soft_centroid(target, temperature=temperature).detach()
    return F.smooth_l1_loss(a / 100.0, b / 100.0)


@torch.no_grad()
def position_error_px(pred, target, temperature: float = 12.0):
    a = soft_centroid(pred, temperature=temperature)
    b = soft_centroid(target, temperature=temperature)
    return torch.linalg.vector_norm(a - b, dim=1)


def px_to_mm(px):
    """Exact: the grid is the mesh, one cell = 3.000 mm."""
    return px * CELL_MM


# ------------------------------------------------------- physical accuracy
@torch.no_grad()
def channel_physical_error(pred, target):
    """Per-channel MAE in native units, over the union of non-trivial cells."""
    p = norm_to_fields(pred)
    t = norm_to_fields(target)
    out = {}
    for i, name in enumerate(CHANNEL_NAMES):
        out[f"mae_{name}"] = (p[:, i] - t[:, i]).abs().mean().item()
    return out


@torch.no_grad()
def image_peaks(fields):
    """Peak of each physical channel, decoded from a normalised (B,4,H,W) batch.

    Each readout must match the definition of the scalar it is compared with.
    `max_mises_mpa` and `max_peeq` are plain field maxima, so those two are
    plain maxima here. `max_deflection_mm` is NOT: it is the peak over material
    at least two cells from any damage front, and taking a raw maximum instead
    compares two different quantities. Measured on ground truth, the raw
    maximum disagrees with its own stored scalar by 197 %, while this readout
    agrees to 0.06 %.
    """
    phys = norm_to_fields(fields)
    return torch.stack([phys[:, 0].amax(dim=(1, 2)),
                        phys[:, 1].amax(dim=(1, 2)),
                        hard_intact_peak_mm(fields)], dim=1)


def image_scalar_consistency(pred_fields, scalar_pred, scaler):
    """Peak decoded from the generated image vs the physics head's scalar.

    The two branches share no weights, so agreement here is evidence the image
    is physically consistent with the regressed response rather than merely
    plausible-looking. Returns relative disagreement per quantity.

    scalar_pred is in normalised space; channel order (mises, peeq, u3) matches
    TARGET_COLS = (max_mises_mpa, max_peeq, max_deflection_mm).
    """
    img_peak = image_peaks(pred_fields)
    head = scaler.inv_targets(scalar_pred)
    if not isinstance(head, torch.Tensor):
        head = torch.as_tensor(head, device=img_peak.device, dtype=img_peak.dtype)
    rel = (img_peak - head).abs() / torch.clamp(head.abs(), min=1e-6)
    ok = torch.isfinite(rel)
    out = {}
    for i, nm in enumerate(["mises", "peeq", "defl"]):
        m = ok[:, i]
        out["consistency_mape_" + nm] = (100 * rel[m, i].mean().item()
                                         if m.any() else float("nan"))
    return out
