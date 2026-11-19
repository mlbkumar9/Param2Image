"""Dataset and normalisation for the Param2Image v3 field-map surrogate.

REPRESENTATION
--------------
Each sample is a 4 x 200 x 200 tensor of *physical* fields on the plate's own
finite-element grid, one pixel per element at exactly 3.000 mm:

    ch 0  mises  von Mises stress          MPa
    ch 1  peeq   equivalent plastic strain  -
    ch 2  u3     |out-of-plane deflection|  mm
    ch 3  mask   surviving material         0/1

Channels are mapped linearly to [-1, 1] using FIXED ranges stored in the
checkpoint, so the mapping is exactly invertible: a predicted pixel converts
back to MPa / strain / mm without reference to the training set. This is what
makes per-channel physical error reporting and the image-vs-scalar consistency
check possible, and it answers the "units, normalisation and inverse mapping"
question directly.

CHANNEL CAPS
------------
`u3` is capped at 60 mm. Restricting displacement to surviving elements removes
ejected debris, but a one-cell rim at a hole boundary is damage-front material
and reaches ~456 mm. Measured over 1200 training runs the cap touches 21.5 % of
runs but only 0.0064 % of all cells (~2.6 cells per affected run), i.e. exactly
that rim, while the largest physical intact deflection in the dataset is
45.7 mm. `mises` (1200 MPa) and `peeq` (2.0) clip nothing at all.

PER-CHANNEL TRANSFORM
---------------------
A purely linear map to [-1, 1] is a poor use of the range for the two sparse
channels. Measured over 400 runs, `u3` occupies 28 % of its cap at p99.9 and
`peeq` only 15 %, with a median of exactly zero in both: the plate is
undeformed almost everywhere. In normalised space that left `u3` contributing
0.27 % of the total L1 signal and `peeq` 0.14 %, against 93 % for the binary
mask, so the generator received almost no gradient on either. The consequence
was measurable: at 300 epochs the generated deflection field ran ~45 % low at
every percentile up to p99.9 while its maximum overshot by 199 %, and the peak
correlated only 0.10 with truth, all while scoring SSIM 0.979, because a
near-empty prediction matches a near-empty field.

Those two channels are therefore mapped through a square root, which is
monotone and has a closed-form inverse, so a predicted pixel still converts
back to strain and millimetres exactly. It moves p99.9 of `u3` from -0.54 to
-0.05 in normalised space and lifts the populated low range clear of the
float16 quantisation floor of the packed array. `mises` already used 67 % of
its cap and `mask` is binary, so both stay linear.

Weights for the reconstruction loss are not fixed here. They are measured from
the training split at run time (inverse mean absolute deviation) and stored in
the checkpoint, so no channel can silently dominate again.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

# name, lo, hi, unit, transform
CHANNEL_SPEC = [
    ("mises", 0.0, 1200.0, "MPa", "linear"),
    ("peeq", 0.0, 2.0, "-", "sqrt"),
    ("u3", 0.0, 60.0, "mm", "sqrt"),
    ("mask", 0.0, 1.0, "-", "linear"),
]
CHANNEL_NAMES = [c[0] for c in CHANNEL_SPEC]
CHANNEL_LO = np.array([c[1] for c in CHANNEL_SPEC], dtype=np.float32)
CHANNEL_HI = np.array([c[2] for c in CHANNEL_SPEC], dtype=np.float32)
CHANNEL_UNITS = [c[3] for c in CHANNEL_SPEC]
CHANNEL_TRANSFORM = [c[4] for c in CHANNEL_SPEC]
# kept as an array so the forward and inverse maps stay vectorised
CHANNEL_SQRT = np.array([t == "sqrt" for t in CHANNEL_TRANSFORM], dtype=bool)

MASK_CHANNEL = CHANNEL_NAMES.index("mask")
U3_CHANNEL = CHANNEL_NAMES.index("u3")
EROSION_CELLS = 2        # matches the defl_intact_mm target definition

GRID = 200
CELL_MM = 3.0            # exact: 600 mm plate / 200 cells
PLATE_MM = GRID * CELL_MM

PARAM_COLS = ["velocity_kmh", "proj_radius_mm", "impact_x_mm", "impact_y_mm"]
TARGET_COLS = ["max_mises_mpa", "max_peeq", "max_deflection_mm"]
TARGET_UNITS = ["MPa", "-", "mm"]


def fields_to_norm(arr: np.ndarray) -> np.ndarray:
    """(4,H,W) physical -> [-1,1]. Invertible up to the documented caps.

    Linear on `mises` and `mask`, square root on `peeq` and `u3`; see the
    module docstring. Both branches are monotone with a closed-form inverse.
    """
    lo = CHANNEL_LO[:, None, None]
    hi = CHANNEL_HI[:, None, None]
    u = (np.clip(arr, lo, hi) - lo) / (hi - lo)                 # [0,1]
    u = np.where(CHANNEL_SQRT[:, None, None], np.sqrt(np.clip(u, 0.0, None)), u)
    return (2.0 * u - 1.0).astype(np.float32)


def norm_to_fields(t):
    """[-1,1] -> physical units. Works on numpy or torch, batched or not."""
    if isinstance(t, torch.Tensor):
        lo = torch.as_tensor(CHANNEL_LO, device=t.device, dtype=t.dtype).view(-1, 1, 1)
        hi = torch.as_tensor(CHANNEL_HI, device=t.device, dtype=t.dtype).view(-1, 1, 1)
        sq = torch.as_tensor(CHANNEL_SQRT, device=t.device).view(-1, 1, 1)
        if t.dim() == 4:
            lo, hi, sq = lo.unsqueeze(0), hi.unsqueeze(0), sq.unsqueeze(0)
        u = (t.clamp(-1, 1) + 1.0) / 2.0
        return torch.where(sq, u * u, u) * (hi - lo) + lo
    lo = CHANNEL_LO[:, None, None]
    hi = CHANNEL_HI[:, None, None]
    sq = CHANNEL_SQRT[:, None, None]
    if np.ndim(t) == 4:
        lo, hi, sq = lo[None], hi[None], sq[None]
    u = (np.clip(t, -1, 1) + 1.0) / 2.0
    return np.where(sq, u * u, u) * (hi - lo) + lo


def load_field_npz(path: str) -> np.ndarray:
    z = np.load(path)
    return np.stack([z[n] for n in CHANNEL_NAMES], axis=0).astype(np.float32)


# --------------------------------------------------------------------------- #
@dataclass
class TabularScaler:
    """Standardise inputs; log1p-standardise the positive scalar responses.

    Fitted on the TRAINING split only -- no statistic ever crosses from val or
    test into training.
    """
    param_mean: list
    param_std: list
    target_mean: list
    target_std: list
    log1p_targets: bool = True

    @staticmethod
    def fit(df: pd.DataFrame, log1p_targets: bool = True) -> "TabularScaler":
        p = df[PARAM_COLS].to_numpy(np.float64)
        y = df[TARGET_COLS].to_numpy(np.float64)
        if log1p_targets:
            y = np.log1p(np.clip(y, 0, None))
        ps = p.std(0); ps[ps < 1e-8] = 1.0
        ys = y.std(0); ys[ys < 1e-8] = 1.0
        return TabularScaler(p.mean(0).tolist(), ps.tolist(),
                             y.mean(0).tolist(), ys.tolist(), log1p_targets)

    def tx_params(self, p):
        return ((np.asarray(p, np.float64) - self.param_mean) / self.param_std).astype(np.float32)

    def tx_targets(self, y):
        y = np.asarray(y, np.float64)
        if self.log1p_targets:
            y = np.log1p(np.clip(y, 0, None))
        return ((y - self.target_mean) / self.target_std).astype(np.float32)

    def inv_targets(self, y):
        """Back to physical units; accepts numpy or torch."""
        if isinstance(y, torch.Tensor):
            m = torch.as_tensor(self.target_mean, dtype=y.dtype, device=y.device)
            s = torch.as_tensor(self.target_std, dtype=y.dtype, device=y.device)
            out = y * s + m
            return torch.expm1(out.clamp(max=60.0)).clamp_min(0) if self.log1p_targets else out
        out = np.asarray(y, np.float64) * self.target_std + self.target_mean
        return np.maximum(np.expm1(np.clip(out, None, 60.0)), 0) if self.log1p_targets else out

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return TabularScaler(**d)


# --------------------------------------------------------------------------- #
class FieldMapDataset(Dataset):
    """Field maps + impact parameters + scalar responses for one split.

    Reads from a packed float16 memmap when available (fast, one file, ideal for
    remote runtimes), otherwise falls back to the individual .npz files.
    """

    def __init__(self, csv_path, split, scaler=None, pack_path=None, root=None, smooth=0, smooth_mask=False,
                 cache_ram=False):
        df = pd.read_csv(csv_path)
        if split not in (None, "all"):
            df = df[df.split == split]
        self.df = df.reset_index(drop=True)
        self.root = root or os.path.dirname(os.path.abspath(csv_path))
        self.scaler = scaler

        # The memmap is opened LAZILY, once per worker process. Opening it in
        # __init__ and letting DataLoader pickle it breaks under the 'spawn'
        # start method (Windows, and macOS since 3.8) with
        # "UnpicklingError: pickle data was truncated". Storing only the path
        # keeps the dataset trivially picklable on every platform.
        self.smooth = int(smooth or 0)
        self.smooth_mask = bool(smooth_mask)
        self.pack_path = pack_path if (pack_path and os.path.exists(pack_path)) else None
        self.pack = None
        self.pack_rows = None
        if self.pack_path:
            idx_path = os.path.splitext(self.pack_path)[0] + "_index.csv"
            order = pd.read_csv(idx_path).run_id.tolist()
            pos = {r: i for i, r in enumerate(order)}
            missing = [r for r in self.df.run_id if r not in pos]
            if missing:
                raise KeyError("%d run_id not in pack index (e.g. %s)"
                               % (len(missing), missing[:3]))
            self.pack_rows = np.array([pos[r] for r in self.df.run_id], dtype=np.int64)

        # Optional: hold this split's slice of the pack in RAM. The maps are
        # float16, so the training split costs ~1.6 GB and the whole dataset
        # ~1.9 GB, comfortably inside a high-RAM runtime. Reading from a
        # memmap was the throughput limit on an A100 (0.20 s/step at batch 32
        # while using 6.6 GB of 40 GB), and this removes the disk entirely. It
        # changes no arithmetic whatsoever, only where the bytes come from.
        self.cache = None
        if cache_ram and self.pack_path is not None:
            src = np.load(self.pack_path, mmap_mode="r")
            self.cache = torch.from_numpy(
                np.ascontiguousarray(src[self.pack_rows], dtype=np.float16))

        self.params = self.df[PARAM_COLS].to_numpy(np.float64)
        self.targets = self.df[TARGET_COLS].to_numpy(np.float64)

    def __getstate__(self):
        st = self.__dict__.copy()
        st["pack"] = None          # never pickle an open memmap
        return st

    def __len__(self):
        return len(self.df)

    def _load_fields(self, i):
        if self.cache is not None:
            return self.cache[i]
        if self.pack_path is not None:
            if self.pack is None:                     # first touch in this worker
                self.pack = np.load(self.pack_path, mmap_mode="r")
            return np.asarray(self.pack[self.pack_rows[i]], dtype=np.float32)
        path = self.df.field_npz.iloc[i]
        if not os.path.isabs(path):
            path = os.path.join(self.root, path)
        return fields_to_norm(load_field_npz(path))

    def __getitem__(self, i):
        f = self._load_fields(i)
        img = f.float() if torch.is_tensor(f) else torch.from_numpy(f)
        if self.smooth > 1:
            img = smooth_fields(img, self.smooth, self.smooth_mask)
        p = self.params[i]
        y = self.targets[i]
        if self.scaler is not None:
            p = self.scaler.tx_params(p)
            y = self.scaler.tx_targets(y)
        return (torch.as_tensor(np.asarray(p, np.float32)),
                img,
                torch.as_tensor(np.asarray(y, np.float32)))


# ------------------------------------------------------- target smoothing
SMOOTH_CELLS = 0          # 0 or 1 disables; 3 and 5 are the measured options
SMOOTH_MASK = False       # keep the mask binary so hole area stays integral


def smooth_fields(x, k: int = SMOOTH_CELLS, smooth_mask: bool = SMOOTH_MASK):
    """Box-blur the normalised field maps by k cells, without crossing the hole.

    Applied to (C,H,W) or (B,C,H,W) in the NORMALISED representation, which is
    what SSIM is computed on and what the ceiling measurement used.

    The average is normalised by the intact-cell count in each window. Deleted
    cells hold exactly zero, which denotes absent material and not low stress,
    so letting them into the average biases every cell near the boundary
    downward: measured at k=3 on a perforating case, von Mises on the rim falls
    by 83 MPa, and the rim carries the plastic annulus and 7.3% of the model's
    error. Far from a hole the masked and unmasked results agree exactly.

    Why smoothing at all: two ABAQUS runs of the same impact agree to only 0.897
    on the von Mises channel, so a network conditioned on the four impact
    parameters cannot do better on the raw field. The residual is element-level
    speckle that a re-run does not reproduce. Smoothing raises the
    reproducibility of the target, and therefore the reachable score.
    """
    if k is None or k <= 1:
        return x
    single = (x.dim() == 3)
    if single:
        x = x[None]
    b, c, h, w_ = x.shape
    pad = k // 2
    ker = torch.ones(1, 1, k, k, dtype=x.dtype, device=x.device) / float(k * k)

    keep = (x[:, MASK_CHANNEL:MASK_CHANNEL + 1] > 0.0).to(x.dtype)
    phys = [i for i in range(c) if i != MASK_CHANNEL]
    v = x[:, phys]
    n = len(phys)

    def blur(t):
        t = F.pad(t, (pad, pad, pad, pad), mode="replicate")
        return F.conv2d(t.reshape(-1, 1, t.shape[-2], t.shape[-1]), ker) \
                .reshape(t.shape[0], -1, h, w_)

    num = blur(v * keep)
    den = blur(keep).clamp(min=1e-6)
    out = x.clone()
    smoothed = num / den
    # Only intact cells are smoothed. A deleted cell denotes absent material and
    # must keep its zero: letting it take the average of its intact neighbours
    # would paint stress into the hole and corrupt the perforation readout.
    out[:, phys] = torch.where(keep.expand(-1, n, -1, -1).bool(), smoothed, v)
    if smooth_mask:
        out[:, MASK_CHANNEL] = blur(x[:, MASK_CHANNEL:MASK_CHANNEL + 1])[:, 0]
    return out[0] if single else out


def channel_spec_dict(smooth_cells: int = SMOOTH_CELLS,
                      smooth_mask: bool = SMOOTH_MASK):
    """Channel spec for the checkpoint, including the smoothing actually used.

    The smoothing arguments must be passed by the caller. Reading the module
    defaults here recorded smooth_cells 0 for a run trained with --smooth 3,
    and evaluate.py trusts this field to prepare its references: the model was
    scored against raw targets while reporting nothing unusual.
    """
    return {"names": CHANNEL_NAMES, "lo": CHANNEL_LO.tolist(),
            "hi": CHANNEL_HI.tolist(), "units": CHANNEL_UNITS,
            "transform": CHANNEL_TRANSFORM, "erosion_cells": EROSION_CELLS,
            "grid": GRID, "cell_mm": CELL_MM, "plate_mm": PLATE_MM,
            "smooth_cells": int(smooth_cells or 0),
            "smooth_mask": bool(smooth_mask)}


# --------------------------------------------------------------- intact peak
def erode_mask(mask: torch.Tensor, k: int = EROSION_CELLS) -> torch.Tensor:
    """Shrink a (B,H,W) boolean intact-material mask by k cells, edges excluded.

    Identical rule to scripts/compute_deflection_metrics.py, which defines the
    `defl_intact_mm` target: 4-neighbourhood erosion with the plate boundary
    dropped. One implementation means the image readout and the regression
    target cannot drift apart again.
    """
    m = mask
    for _ in range(k):
        e = m.clone()
        e[:, 1:, :] &= m[:, :-1, :]
        e[:, :-1, :] &= m[:, 1:, :]
        e[:, :, 1:] &= m[:, :, :-1]
        e[:, :, :-1] &= m[:, :, 1:]
        e[:, 0, :] = False
        e[:, -1, :] = False
        e[:, :, 0] = False
        e[:, :, -1] = False
        m = e
    return m


def intact_mask_from_norm(fields: torch.Tensor, k: int = EROSION_CELLS) -> torch.Tensor:
    """Eroded intact-material mask from a normalised (B,4,H,W) batch.

    The mask channel is linear on [0,1], so its normalised midpoint is 0.0.
    """
    return erode_mask(fields[:, MASK_CHANNEL] > 0.0, k)


def soft_peak(x: torch.Tensor, mask: torch.Tensor, temperature: float = 20.0):
    """Differentiable maximum of (B,H,W) `x` restricted to `mask`.

    A hard amax passes gradient to exactly one pixel, far too sparse to train
    on, so this is a temperature-weighted soft maximum. Weights are formed in
    float32: a bf16 softmax over 40,000 logits loses the peak entirely.
    """
    fx = x.flatten(1)
    fm = mask.flatten(1)
    logits = torch.where(fm, fx.float() * temperature,
                         torch.full(fx.shape, -1e4, dtype=torch.float32, device=fx.device))
    w = torch.softmax(logits, dim=1).to(fx.dtype)
    return (w * fx).sum(1)


@torch.no_grad()
def hard_intact_peak_mm(fields: torch.Tensor, k: int = EROSION_CELLS):
    """Peak |U3| in mm over eroded intact material; NaN where nothing survives.

    This is the readout that matches the `max_deflection_mm` target exactly, and
    it is what the image-vs-scalar consistency check must use.
    """
    u3 = norm_to_fields(fields)[:, U3_CHANNEL]
    m = erode_mask(fields[:, MASK_CHANNEL] > 0.0, k)
    v = torch.where(m, u3, torch.full_like(u3, -1.0)).amax(dim=(1, 2))
    return torch.where(m.any(dim=(1, 2)), v, torch.full_like(v, float("nan")))


N_PHYS_FEATURES = 9


def physics_features(p):
    """(B,4) raw [velocity_kmh, radius_mm, x_mm, y_mm] -> (B,9) derived features.

    Every added column is a deterministic function of the four inputs, so this
    adds no information. What it adds is the ability to express the response in
    the variables the physics is actually written in: an impact is governed by
    kinetic energy and by where the strike sits relative to the clamped
    boundary, not by velocity and a Cartesian coordinate pair.

    Measured on the validation split, this lifts deflection R2 from 0.935 to
    0.949 and its MAPE from 9.50 % to 8.83 % once the Fourier features are gone.
    """
    is_t = isinstance(p, torch.Tensor)
    v, r = p[..., 0], p[..., 1]
    x, y = p[..., 2], p[..., 3]
    ms = v / 3.6                                  # velocity in m/s
    # Radius and velocity are positive by construction, but clamp anyway: a
    # negative kinetic energy would make log1p return NaN and poison the batch
    # silently rather than raise.
    if is_t:
        ke = ((r.clamp_min(0.0) ** 3) * ms ** 2).clamp_min(0.0)
        rad = torch.sqrt(x * x + y * y)
        edge = PLATE_MM / 2 - torch.maximum(x.abs(), y.abs())
        cols = [v, r, x, y, torch.log1p(ke), ke ** (1.0 / 3.0),
                rad, edge, ms * r ** 2]
        return torch.stack(cols, dim=-1)
    ke = np.maximum((np.maximum(r, 0.0) ** 3) * ms ** 2, 0.0)
    rad = np.sqrt(x * x + y * y)
    edge = PLATE_MM / 2 - np.maximum(np.abs(x), np.abs(y))
    cols = [v, r, x, y, np.log1p(ke), np.cbrt(ke), rad, edge, ms * r ** 2]
    return np.stack(cols, axis=-1)


def measure_channel_weights(loader, max_batches: int = 40):
    """Inverse mean-absolute-deviation weights, measured on the training split.

    Returns weights normalised to mean 1.0, so `--lambda-l1` keeps its meaning
    while every channel contributes comparably to the reconstruction term.
    """
    tot, n = None, 0
    for bi, (_, real, _) in enumerate(loader):
        d = (real - real.mean(dim=(2, 3), keepdim=True)).abs().mean(dim=(2, 3))
        tot = d.sum(0) if tot is None else tot + d.sum(0)
        n += real.shape[0]
        if bi + 1 >= max_batches:
            break
    w = 1.0 / (tot / max(n, 1)).clamp_min(1e-6)
    return (w / w.mean()).tolist()
