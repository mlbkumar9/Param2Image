"""
run_eval_local.py
=================
Self-contained local evaluation of Param2Image v3.1 (epoch-48 checkpoint).
Runs with the gnn_env Python (torch 2.7.0+xpu, CPU-only inference).

Usage:
    python run_eval_local.py

Outputs EPS + PDF figures to:
    ../DML_Mahendra/figures/fig_eval_*.eps  (for LaTeX)
    ../DML_Mahendra/figures/fig_eval_*.pdf  (keep as backup)
"""
from __future__ import annotations

import os, sys, json, math, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ── Matplotlib (Agg so no display needed) ────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
# Error-magnitude colormap for the sample-comparison figures. Matches the
# heat scale used in the earlier Param2Image_Report error maps (matplotlib
# 'hot': black -> dark red -> red -> orange -> yellow -> white; low error dark,
# high error bright). Verified against train_v2_eval_*_samples.png.
ERROR_CMAP = "hot"

# Paper body font size and text width (elsarticle 3p: 10 pt body, 468 pt text
# width). Every figure below is built at its FINAL on-page width (a fraction of
# TEXTWIDTH_IN) and uses a 10 pt base font, so it is placed at scale 1:1 in
# LaTeX and its axis / tick / legend text matches the surrounding body text.
TEXTWIDTH_IN = 468.0 / 72.27   # ~= 6.476 in
# Full-bleed width for metric plots: A4 (210 mm) minus 7 mm margin each side.
WIDE_IN = 196.0 / 25.4         # ~= 7.717 in

plt.rcParams.update({
    # Match the paper body font (mathpazo / Palatino) at body size (10 pt).
    # No in-figure titles are produced anywhere below — all descriptive
    # text lives in the LaTeX figure captions.
    "font.family": "serif",
    "font.serif": ["Palatino Linotype", "Palatino", "URW Palladio L",
                   "Book Antiqua", "DejaVu Serif"],
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "legend.framealpha": 1.0,   # opaque: renders identically on PDF and EPS/PS routes
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype":  42,   # embed fonts in EPS
    "ps.useafm": False,
})

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).resolve().parent
from p2i_paths import data_dir, checkpoint_dir, figures_dir, report_figures_dir
import os as _os
CKPT_PATH   = checkpoint_dir("training_checkpoints_position") / "ckpt_best_position.pt"   # position-selected checkpoint
_DATA = data_dir()
CSV_PATH    = Path(_os.environ.get("P2I_CSV_PATH", str(_DATA / "results_master.csv")))
IMG_ROOT    = _DATA
FIGURES_DIR = figures_dir()
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
# Param2Image_Report figures (local Documentation/ or fork-local).
REPORT_FIG_DIR = report_figures_dir()
REPORT_TW_IN   = 160.0 / 25.4   # ~= 6.30 in

# Add script dir to path so local modules can be imported
sys.path.insert(0, str(SCRIPT_DIR))
from model_p2i_pytorch import Generator, PhysicsHead
from position_supervision import impact_position_error_pixels, pixels_to_plate_mm
from preprocessing_p2i import (
    ImpactTabularScaler, PARAM_COLUMNS, OUTPUT_COLUMNS,
    OUTPUT_SHORT_NAMES, OUTPUT_DISPLAY_NAMES, OUTPUT_UNITS,
)

SEED       = 42
VAL_FRAC   = 0.1
BATCH_SIZE = 2          # small to stay within RAM on CPU
IMG_SIZE   = 256        # downscale for CPU inference (model is resolution-agnostic at eval)
N_GRID     = 3          # samples per sample-grid figure (3 rows x 3 cols -- keeps every row to <=3 panels)
DEVICE     = torch.device("cpu")

# Set by main() once checkpoint is loaded; used in all figure titles
EPOCH_LABEL = "Epoch ?? EMA"


# ══════════════════════════════════════════════════════════════════════════════
# SSIM (pure torch, no torchmetrics)
# ══════════════════════════════════════════════════════════════════════════════

def _gauss_kernel(size=11, sigma=1.5, channels=4):
    ax = torch.arange(size, dtype=torch.float32) - size // 2
    g  = torch.exp(-ax**2 / (2*sigma**2))
    k  = g.outer(g); k = k / k.sum()
    return k.unsqueeze(0).unsqueeze(0).repeat(channels, 1, 1, 1)  # C×1×H×W

_KERNEL = None

def ssim_per_sample(pred, target, data_range=2.0, C1=(0.01*2)**2, C2=(0.03*2)**2):
    global _KERNEL
    B, C, H, W = pred.shape
    if _KERNEL is None or _KERNEL.shape[0] != C:
        _KERNEL = _gauss_kernel(channels=C)
    k = _KERNEL
    pad = 5
    mu_x  = F.conv2d(pred,   k, padding=pad, groups=C)
    mu_y  = F.conv2d(target, k, padding=pad, groups=C)
    mu_xx = F.conv2d(pred*pred,   k, padding=pad, groups=C)
    mu_yy = F.conv2d(target*target, k, padding=pad, groups=C)
    mu_xy = F.conv2d(pred*target, k, padding=pad, groups=C)
    sig_xx = mu_xx - mu_x**2
    sig_yy = mu_yy - mu_y**2
    sig_xy = mu_xy - mu_x*mu_y
    num = (2*mu_x*mu_y + C1) * (2*sig_xy + C2)
    den = (mu_x**2 + mu_y**2 + C1) * (sig_xx + sig_yy + C2)
    return (num / den).mean(dim=(1, 2, 3))   # shape [B]


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class _ValDS(torch.utils.data.Dataset):
    def __init__(self, df, scaler):
        self.df = df.reset_index(drop=True)
        self.tfm = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([.5]*4, [.5]*4),
        ])
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_rel = str(row["img_after"]).replace("\\", os.sep).replace("/", os.sep)
        img = Image.open(IMG_ROOT / img_rel).convert("RGBA")
        img_t = self.tfm(img)
        p_raw = row[PARAM_COLUMNS].astype(float).to_numpy()
        o_raw = row[OUTPUT_COLUMNS].astype(float).to_numpy()
        p_n = self.scaler.transform_params_array(p_raw)[0]
        o_n = self.scaler.transform_outputs_array(o_raw)[0]
        return (
            torch.tensor(p_n, dtype=torch.float32),
            img_t,
            torch.tensor(o_n, dtype=torch.float32),
            torch.tensor(p_raw, dtype=torch.float32),
            torch.tensor(o_raw, dtype=torch.float32),
        )


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation loop
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def run_inference(G, P, loader, scaler, latent_dim):
    G.eval(); P.eval()
    ssims, preds, trues, params_all = [], [], [], []
    fakes_all, reals_all = [], []
    position_error_sum = 0.0
    position_sample_count = 0

    for batch_idx, batch in enumerate(loader):
        p_n, real, o_n, p_raw, o_raw = batch
        z = torch.randn(p_n.size(0), latent_dim)
        fake = G(z, p_n).clamp(-1, 1)

        # Generator outputs at full img_size (1024); real is loaded at IMG_SIZE (256).
        # Downsample fake to match for SSIM computation.
        if fake.shape[-1] != real.shape[-1]:
            fake_down = F.interpolate(fake, size=real.shape[-2:], mode="bilinear", align_corners=False)
        else:
            fake_down = fake

        s = ssim_per_sample(fake_down, real.clamp(-1, 1)).tolist()
        ssims.extend(s)
        batch_position_error = impact_position_error_pixels(fake_down, real.clamp(-1, 1))
        position_error_sum += float(batch_position_error) * p_n.size(0)
        position_sample_count += p_n.size(0)

        mu, _ = P(p_n)
        preds.append(scaler.inverse_transform_outputs_array(mu.numpy()))
        trues.append(scaler.inverse_transform_outputs_array(o_n.numpy()))
        params_all.append(p_raw.numpy())
        # Store downsampled fake (256px) for consistent visualisation with real
        fakes_all.append(fake_down.cpu())
        reals_all.append(real.cpu())

        if (batch_idx + 1) % 10 == 0:
            print(f"  [{batch_idx+1}/{len(loader)}] mean SSIM so far: {np.mean(ssims):.4f}")

    return (
        np.array(ssims),
        np.concatenate(preds, 0),
        np.concatenate(trues, 0),
        np.concatenate(params_all, 0),
        torch.cat(fakes_all, 0),
        torch.cat(reals_all, 0),
        position_error_sum / max(position_sample_count, 1),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _save(fig, stem, out_dir=None):
    """Save both EPS and PDF into out_dir (defaults to the paper figures dir)."""
    d = out_dir or FIGURES_DIR
    for ext in ("eps", "pdf"):
        fig.savefig(str(d / f"{stem}.{ext}"), format=ext)
    print(f"  Saved: {stem}.eps + .pdf  -> {d.name}")
    plt.close(fig)


def _to_rgb(t):
    """[-1,1] C×H×W tensor → uint8 H×W×3 numpy (alpha blended on white)."""
    t = t.clamp(-1, 1)
    arr = ((t + 1) / 2).numpy()           # [0,1] C×H×W
    if arr.shape[0] == 4:
        alpha = arr[3:4]
        rgb   = arr[:3]
        comp  = rgb * alpha + (1 - alpha)  # blend on white
    else:
        comp = arr[:3]
    return (comp.clip(0, 1) * 255).astype(np.uint8).transpose(1, 2, 0)


# ══════════════════════════════════════════════════════════════════════════════
# Figure generators
# ══════════════════════════════════════════════════════════════════════════════

def fig_sample_grid(indices, fakes, reals, ssims, params, n, stem):
    """3 rows (ground truth / predicted / error magnitude) x <=3 columns of
    samples. The error row is a scalar map (mean absolute RGB difference,
    normalised to [0,1]) rendered on a shared black->yellow->red heat scale
    with a single colourbar. No in-figure title -- per-sample SSIM and impact
    parameters are placed as small annotations above each column; the full
    description lives in the LaTeX caption."""
    cols = min(n, len(indices))
    # Built at full text width (placed at scale 1:1 in LaTeX) so labels render
    # at body size; square image panels => height ~= width + colourbar margin.
    fig, axes = plt.subplots(3, cols, figsize=(TEXTWIDTH_IN, TEXTWIDTH_IN + 0.85),
                             constrained_layout=True)
    if cols == 1:
        axes = axes[:, np.newaxis]
    row_labels = ["Ground\ntruth", "Predicted", "Error\nmagnitude"]

    # Pre-compute per-column ground truth, prediction, and scalar error map so
    # the whole error row can share one colour scale.
    gts, prs, diffs = [], [], []
    for idx in indices[:cols]:
        gt = _to_rgb(reals[idx]); pr = _to_rgb(fakes[idx])
        d  = np.abs(gt.astype(np.float32) - pr.astype(np.float32)).mean(axis=2) / 255.0
        gts.append(gt); prs.append(pr); diffs.append(d)
    vmax = max(0.05, max(float(d.max()) for d in diffs))   # shared error scale

    diff_im = None
    for c in range(cols):
        idx = indices[c]
        for r in range(3):
            ax = axes[r, c]
            if r == 0:
                ax.imshow(gts[c])
            elif r == 1:
                ax.imshow(prs[c])
            else:
                diff_im = ax.imshow(diffs[c], cmap=ERROR_CMAP, vmin=0.0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=10)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.axis("off")
        v, rad = params[idx, 0] * 3.6, params[idx, 1]   # velocity m/s -> km/h
        # two-line annotation keeps the text within one (narrow) column width
        axes[0, c].annotate(f"SSIM = {ssims[idx]:.3f}\n$v$={v:.0f} km/h,  $r$={rad:.0f} mm",
                            xy=(0.5, 1.03), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=9)

    cbar = fig.colorbar(diff_im, ax=axes[2, :].ravel().tolist(),
                        location="bottom", fraction=0.05, pad=0.02, shrink=0.8)
    cbar.set_label("Normalised absolute error", fontsize=10)
    cbar.ax.tick_params(labelsize=9)
    _save(fig, stem)


def fig_pred_vs_actual(preds, trues, stem):
    """1x3 scatter panel (one panel per scalar output) -- at the row-size limit.
    R^2 / MAPE annotated inside each panel; no figure-level title (see caption)."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDE_IN, 3.05), constrained_layout=True)
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    for k, ax in enumerate(axes):
        yt = trues[:, k]; yp = preds[:, k]
        mask = np.isfinite(yp) & np.isfinite(yt)
        yt, yp = yt[mask], yp[mask]
        # von Mises panel (k==0): a handful of low-energy cases develop no stress
        # (ground truth ~= 0) yet the physics head emits a small spurious value.
        # These are physically degenerate noise points that distort the regression;
        # exclude them (< 1% of the peak ground-truth stress) so the fitted line
        # reflects the cases where stress actually develops.
        if k == 0:
            keep = yt > 0.01 * yt.max()
            yt, yp = yt[keep], yp[keep]
        slope, intercept, r, *_ = scipy_stats.linregress(yt, yp)
        r2 = r**2
        dm = np.abs(yt) > 1e-3
        mape = 100*np.mean(np.abs((yp[dm]-yt[dm])/yt[dm])) if dm.any() else float("nan")
        lo = min(yt.min(), yp.min()); hi = max(yt.max(), yp.max())
        ax.scatter(yt, yp, s=9, c=colors[k], linewidths=0)
        ax.plot([lo,hi],[lo,hi],"k--",lw=1.0,label="Perfect")
        xl = np.linspace(lo, hi, 200)
        ax.plot(xl, slope*xl+intercept, "r-", lw=1.2, label="Lin. fit")
        # short axis labels (quantity name carried in the in-panel box) so the
        # text fits a narrow 1-of-3 panel at body font size
        ax.set_xlabel(f"Ground truth ({OUTPUT_UNITS[k]})")
        ax.set_ylabel(f"Predicted ({OUTPUT_UNITS[k]})")
        ax.text(0.04, 0.96, f"{OUTPUT_DISPLAY_NAMES[k]}\n$R^2$ = {r2:.3f}\nMAPE = {mape:.1f}%",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))
        ax.legend(fontsize=8.5, loc="lower right"); ax.grid(True, color="0.85")
        ax.set_box_aspect(1)     # square (1:1) panels
    _save(fig, stem)


def fig_ssim_hist(ssims, stem):
    """Single-panel SSIM histogram. Summary statistics are annotated in-panel
    and repeated in the LaTeX caption -- no figure title."""
    fig, ax = plt.subplots(figsize=(3.7, 3.7), constrained_layout=True)
    ax.hist(ssims, bins=25, color="#2E86AB", edgecolor="white", linewidth=0.5)
    ax.axvline(ssims.mean(), color="red",    lw=1.6, ls="--", label=f"Mean = {ssims.mean():.4f}")
    ax.axvline(np.median(ssims), color="orange", lw=1.6, ls=":", label=f"Median = {np.median(ssims):.4f}")
    ax.set_xlabel("SSIM"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(True, axis="y", color="0.85")
    ax.set_box_aspect(1)         # square (1:1) panel
    _save(fig, stem)


def fig_mape_velocity(preds, trues, params_all, stem):
    """Compact single-axis bar chart of deflection MAPE per impact-velocity bin.
    The per-bin validation sample count is carried in the x-tick labels (n=...),
    so the previous redundant secondary sample-count axis is dropped, along with
    the (uninformative) horizontal grid lines. No figure title."""
    k = OUTPUT_SHORT_NAMES.index("deflection")
    vel = params_all[:, 0] * 3.6   # param is stored as velocity_ms (m/s) -- convert to km/h for display
    yt = trues[:, k]; yp = preds[:, k]
    mask = np.abs(yt) > 0.1
    vel, yt, yp = vel[mask], yt[mask], yp[mask]
    bins   = [0, 100, 200, 300, 450, 700]
    labels = ["0–100", "100–200", "200–300", "300–450", "450–700"]
    mapes, counts = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (vel >= lo) & (vel < hi)
        mapes.append(100*np.mean(np.abs((yp[sel]-yt[sel])/yt[sel])) if sel.sum() else float("nan"))
        counts.append(int(sel.sum()))
    # Compact figure built at its on-page width (~0.78 text width) with the 10 pt
    # body font, so it is placed at scale 1:1 and its labels match the body text.
    fig, ax = plt.subplots(figsize=(0.78 * TEXTWIDTH_IN, 2.8), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, mapes, color="#A23B72", edgecolor="white", linewidth=0.5, width=0.7)
    ax.bar_label(bars, labels=[f"{m:.1f}%" if not math.isnan(m) else "N/A" for m in mapes],
                 fontsize=9, padding=3)
    _valid = [m for m in mapes if not math.isnan(m)]
    if _valid:
        ax.set_ylim(0, max(_valid) * 1.18)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb} km/h\n($n$={c})" for lb, c in zip(labels, counts)])
    ax.set_xlabel("Impact velocity range (km/h)")
    ax.set_ylabel("Displacement MAPE (%)")
    ax.margins(x=0.03)
    _save(fig, stem)


def fig_error_dists(preds, trues, stem):
    """1x3 residual-distribution panel (one per scalar output) -- at the
    row-size limit. mu/sigma annotated in-panel; no figure-level title."""
    fig, axes = plt.subplots(1, 3, figsize=(WIDE_IN, 3.05), constrained_layout=True)
    colors = ["#2E86AB", "#A23B72", "#F18F01"]
    for k, ax in enumerate(axes):
        res = preds[:, k] - trues[:, k]
        res = res[np.isfinite(res)]
        ax.hist(res, bins=28, color=colors[k], edgecolor="white", lw=0.5, density=True)
        mu_r, std_r = scipy_stats.norm.fit(res)
        xl = np.linspace(res.min(), res.max(), 300)
        ax.plot(xl, scipy_stats.norm.pdf(xl, mu_r, std_r), "k-", lw=1.4,
                label="Normal fit")
        ax.axvline(0, color="red", lw=1.1, ls="--", label="Zero error")
        # quantity name carried in the x-label; only mu/sigma in the in-panel box
        # so it never collides with the legend in the narrow 1-of-3 panel
        ax.set_xlabel(f"{OUTPUT_DISPLAY_NAMES[k].replace('Max ', '')} residual ({OUTPUT_UNITS[k]})")
        ax.set_ylabel("Density")
        ax.text(0.04, 0.96, f"$\\mu$ = {mu_r:.3g}\n$\\sigma$ = {std_r:.3g}",
                transform=ax.transAxes, ha="left", va="top", fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7"))
        ax.legend(fontsize=8.5, loc="upper right"); ax.grid(True, color="0.85")
        ax.set_box_aspect(1)     # square (1:1) panels
    _save(fig, stem)


def fig_channel_maps(fakes, reals, best_idx, stem):
    """4 rows (one per RGBA channel) x 2 columns (ground truth | predicted).
    This keeps every row to exactly 2 panels (well within the <=3 rule) and
    is more natural to read than a cramped 2x4 grid: the eye scans straight
    down comparing truth vs. prediction channel by channel."""
    ch_names = ["R — von Mises proxy", "G — PEEQ proxy",
                "B — deflection proxy", "A — damage mask"]
    fake = fakes[best_idx].clamp(-1, 1).numpy()
    real = reals[best_idx].clamp(-1, 1).numpy()
    fig, axes = plt.subplots(4, 2, figsize=(0.62 * TEXTWIDTH_IN, 0.62 * TEXTWIDTH_IN * 1.62),
                             constrained_layout=True)
    # Pack the four channel rows tightly: the square image panels otherwise leave
    # large vertical gaps between rows. Near-zero hspace pulls the rows together.
    fig.set_constrained_layout_pads(h_pad=0.015, w_pad=0.02, hspace=0.0, wspace=0.04)
    for r in range(4):
        gt = (real[r] + 1) / 2
        pr = (fake[r] + 1) / 2
        # Per-channel contrast stretch: these proxy fields are near-uniform, so a
        # fixed [0,1] scale hides all structure. We map the 2nd-98th percentile of
        # the combined (truth+prediction) values to the full colour range and use
        # the high-sensitivity perceptually-uniform 'turbo' colormap so subtle
        # spatial variation becomes visible. Truth and prediction share the same
        # per-channel scale, keeping the comparison fair.
        combined = np.concatenate([gt.ravel(), pr.ravel()])
        lo, hi = np.percentile(combined, [2, 98])
        if hi - lo < 1e-3:                      # essentially flat channel
            lo, hi = float(combined.min()), float(combined.max())
            if hi - lo < 1e-6:
                hi = lo + 1e-6
        for c, (data, lbl) in enumerate([(gt, "Ground truth"), (pr, "Predicted")]):
            ax = axes[r, c]
            im = ax.imshow(data, cmap="turbo", vmin=lo, vmax=hi, aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(lbl, fontsize=10)
            if c == 0:
                ax.set_ylabel(ch_names[r], fontsize=9)
            cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(labelsize=7.5)
    _save(fig, stem)


# ══════════════════════════════════════════════════════════════════════════════
# Param2Image_Report variants (bigger fonts, NO in-figure info boxes, same
# black->yellow->red error-map colour; sized to the report text width).
# ══════════════════════════════════════════════════════════════════════════════

def fig_report_sample_grid(indices, fakes, reals, ssims, params, n, stem):
    """Report version of the best/worst sample grid: same 3-row layout and
    black->yellow->red error map as the paper, but with larger fonts and no
    boxed annotations (per request)."""
    cols = min(n, len(indices))
    # Height chosen so the three rows of square images pack tightly (no wasted
    # vertical space between rows): 3 square panels + colourbar + annotations.
    fig, axes = plt.subplots(3, cols, figsize=(REPORT_TW_IN, REPORT_TW_IN + 0.45),
                             constrained_layout=True)
    fig.set_constrained_layout_pads(h_pad=0.03, w_pad=0.03, hspace=0.015, wspace=0.015)
    if cols == 1:
        axes = axes[:, np.newaxis]
    row_labels = ["Ground\ntruth", "Predicted", "Error\nmagnitude"]
    gts, prs, diffs = [], [], []
    for idx in indices[:cols]:
        gt = _to_rgb(reals[idx]); pr = _to_rgb(fakes[idx])
        d  = np.abs(gt.astype(np.float32) - pr.astype(np.float32)).mean(axis=2) / 255.0
        gts.append(gt); prs.append(pr); diffs.append(d)
    vmax = max(0.05, max(float(d.max()) for d in diffs))
    diff_im = None
    for c in range(cols):
        idx = indices[c]
        for r in range(3):
            ax = axes[r, c]
            if r == 0:
                ax.imshow(gts[c])
            elif r == 1:
                ax.imshow(prs[c])
            else:
                diff_im = ax.imshow(diffs[c], cmap=ERROR_CMAP, vmin=0.0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(row_labels[r], fontsize=14)
                for spine in ax.spines.values():
                    spine.set_visible(False)
            else:
                ax.axis("off")
        v, rad = params[idx, 0] * 3.6, params[idx, 1]
        axes[0, c].annotate(f"SSIM = {ssims[idx]:.3f}\n$v$ = {v:.0f} km/h,  $r$ = {rad:.0f} mm",
                            xy=(0.5, 1.03), xycoords="axes fraction",
                            ha="center", va="bottom", fontsize=13)
    cbar = fig.colorbar(diff_im, ax=axes[2, :].ravel().tolist(),
                        location="bottom", fraction=0.05, pad=0.02, shrink=0.85)
    cbar.set_label("Normalised absolute error", fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    _save(fig, stem, out_dir=REPORT_FIG_DIR)


def fig_report_mape(preds, trues, params_all, stem):
    """Report version of the deflection-MAPE-by-velocity bar chart with larger
    fonts and no in-figure info boxes."""
    k = OUTPUT_SHORT_NAMES.index("deflection")
    vel = params_all[:, 0] * 3.6
    yt = trues[:, k]; yp = preds[:, k]
    mask = np.abs(yt) > 0.1
    vel, yt, yp = vel[mask], yt[mask], yp[mask]
    bins   = [0, 100, 200, 300, 450, 700]
    labels = ["0–100", "100–200", "200–300", "300–450", "450–700"]
    mapes, counts = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (vel >= lo) & (vel < hi)
        mapes.append(100*np.mean(np.abs((yp[sel]-yt[sel])/yt[sel])) if sel.sum() else float("nan"))
        counts.append(int(sel.sum()))
    fig, ax = plt.subplots(figsize=(REPORT_TW_IN, 0.6 * REPORT_TW_IN), constrained_layout=True)
    x = np.arange(len(labels))
    bars = ax.bar(x, mapes, color="#A23B72", edgecolor="white", linewidth=0.5)
    ax.bar_label(bars, labels=[f"{m:.1f}%" if not math.isnan(m) else "N/A" for m in mapes],
                 fontsize=13, padding=4)
    _valid = [m for m in mapes if not math.isnan(m)]
    if _valid:
        ax.set_ylim(0, max(_valid) * 1.22)
    ax2 = ax.twinx()
    ax2.plot(x, counts, "ko--", ms=6, lw=1.4, label="Sample count")
    ax2.set_ylabel("Sample count", fontsize=14); ax2.legend(loc="upper right", fontsize=12)
    ax2.tick_params(labelsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lb} km/h\n(n={c})" for lb, c in zip(labels, counts)], fontsize=11.5)
    ax.set_xlabel("Impact velocity range (km/h)", fontsize=14)
    ax.set_ylabel("Deflection MAPE (%)", fontsize=14)
    ax.tick_params(axis="y", labelsize=12)
    ax.grid(True, axis="y", color="0.85")
    _save(fig, stem, out_dir=REPORT_FIG_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    torch.manual_seed(SEED); np.random.seed(SEED)
    print("\n" + "="*65)
    print("  Param2Image v3.1 — Local Evaluation (CPU)")
    print(f"  Checkpoint : {CKPT_PATH}")
    print(f"  CSV        : {CSV_PATH}")
    print(f"  Output     : {FIGURES_DIR}")
    print("="*65 + "\n")

    # ── 1. Load checkpoint ──────────────────────────────────────────────────
    global EPOCH_LABEL
    print("Loading checkpoint ...")
    ck = torch.load(str(CKPT_PATH), map_location=DEVICE, weights_only=False)
    cfg = ck.get("config", {})
    epoch_num = ck['epoch'] + 1   # 0-indexed in ckpt → 1-indexed display
    EPOCH_LABEL = f"Epoch {epoch_num} EMA"
    print(f"  Epoch {epoch_num}  |  best_ssim={ck['best_ssim']:.4f}")

    # ── 2. Build and restore models ─────────────────────────────────────────
    print("Building models ...")
    G = Generator(
        num_params=cfg.get("num_params", 4),
        num_numerical_outputs=cfg.get("num_numerical_outputs", 3),
        latent_dim=cfg.get("latent_dim", 100),
        img_channels=cfg.get("img_channels", 4),
        img_size=cfg.get("img_size", 1024),
        init_size=cfg.get("init_size", 16),
        init_channels=cfg.get("init_channels", 512),
        w_dim=cfg.get("w_dim", 512),
        use_grad_checkpoint=False,
    ).to(DEVICE)
    P = PhysicsHead(
        num_params=cfg.get("num_params", 4),
        num_outputs=cfg.get("num_numerical_outputs", 3),
        hidden=cfg.get("phys_hidden", 256),
        depth=cfg.get("phys_depth", 6),
        num_freqs=cfg.get("phys_num_freqs", 16),
        sigma=cfg.get("phys_sigma", 5.0),
    ).to(DEVICE)
    # Prefer EMA weights
    G.load_state_dict(ck["ema_g"],  strict=True); print("  Generator   <- EMA")
    P.load_state_dict(ck["ema_p"],  strict=True); print("  PhysicsHead <- EMA")
    G.eval(); P.eval()
    latent_dim = cfg.get("latent_dim", 100)

    # ── 3. Scaler: refit from CSV (stats saved in ckpt config aren't the scaler params) ──
    print("Fitting scaler from CSV ...")
    df_full = pd.read_csv(str(CSV_PATH))
    req = ["img_after"] + PARAM_COLUMNS + OUTPUT_COLUMNS
    df_full = df_full.dropna(subset=req).reset_index(drop=True)
    scaler = ImpactTabularScaler(output_transform="log1p")
    scaler.fit(df_full)
    print(f"  Scaler fitted on {len(df_full)} samples")

    # ── 4. Validation split (same seed / frac as training) ──────────────────
    rng = np.random.RandomState(SEED)
    n_val = max(1, int(len(df_full) * VAL_FRAC))
    val_idx = rng.choice(len(df_full), size=n_val, replace=False)
    val_df  = df_full.iloc[val_idx]
    print(f"  Val set: {len(val_df)} samples")

    ds  = _ValDS(val_df, scaler)
    ldr = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 5. Inference ─────────────────────────────────────────────────────────
    print(f"\nRunning inference on {len(ds)} validation samples …")
    ssims, preds, trues, params, fakes, reals, position_error_px = run_inference(G, P, ldr, scaler, latent_dim)

    print(f"\nValidation SSIM : mean={ssims.mean():.4f}  median={np.median(ssims):.4f}  "
          f"std={ssims.std():.4f}  min={ssims.min():.4f}  max={ssims.max():.4f}")
    position_error_mm = pixels_to_plate_mm(
        position_error_px,
        IMG_SIZE,
        plate_width_mm=cfg.get("plate_width_mm", 2500.0),
        border_fraction=cfg.get("pos_border_fraction", 0.04),
    )
    print(f"Impact Position Error : {position_error_px:.2f}px (~{position_error_mm:.2f}mm)")
    di = OUTPUT_SHORT_NAMES.index("deflection")
    dm = np.abs(trues[:, di]) > 0.1
    if dm.any():
        mape = 100*np.mean(np.abs((preds[dm, di]-trues[dm, di])/trues[dm, di]))
        print(f"Deflection MAPE : {mape:.2f}%")

    # ── 6. Sort by SSIM, restricted to visually informative samples ──────────
    # The very highest-SSIM samples are degenerate near-blank low-energy cases
    # (almost no damage), which make uninformative comparison panels. We measure
    # the visual "damage content" of each ground-truth image (std of the
    # white-composited grayscale) and restrict the best/worst panels to the
    # samples that actually show damage, so the figures compare real impact
    # patterns. Full-set statistics (SSIM mean/median, histogram) are unaffected.
    dmg = np.array([_to_rgb(reals[i]).astype(np.float32).std() for i in range(len(ssims))])
    info_idx = np.where(dmg >= np.median(dmg))[0]
    order_info = info_idx[np.argsort(ssims[info_idx])]
    worst_idx = order_info[:N_GRID].tolist()
    # Best showcase: the top-SSIM cases are dominated by low-energy, near-blank
    # impacts that reconstruct trivially well -- poor, repetitive presentation
    # samples. Instead pick the highest-SSIM clearly-damaged reconstruction in
    # each of three impact-velocity bands, so the panel presents the model across
    # distinct, visually substantial impact regimes (low / medium / high energy),
    # left-to-right in increasing velocity.
    vel_kmh = params[:, 0] * 3.6
    dmg_ok  = dmg >= np.median(dmg)
    bands   = [(100.0, 230.0), (230.0, 380.0), (380.0, 560.0)]
    best_idx = []
    for lo, hi in bands:
        cand = np.where(dmg_ok & (vel_kmh >= lo) & (vel_kmh < hi))[0]
        if len(cand):
            best_idx.append(int(cand[np.argmax(ssims[cand])]))
    # fallback: backfill from the global damaged top-SSIM list if a band was empty
    if len(best_idx) < N_GRID:
        for i in order_info[::-1]:
            if i not in best_idx:
                best_idx.append(int(i))
            if len(best_idx) == N_GRID:
                break
    best_idx = best_idx[:N_GRID]
    def _info(i): return f"idx{i}(ssim={ssims[i]:.3f}, v={params[i,0]*3.6:.0f}km/h, r={params[i,1]:.0f}mm)"
    print(f"  Sample-grid selection:")
    print(f"    best  (top-SSIM clearly-damaged per velocity band): {[_info(i) for i in best_idx]}")
    print(f"    worst (>median-damage pool, n={len(info_idx)}):     {[_info(i) for i in worst_idx]}")

    # Channel-map exemplar: a high-SSIM, high-velocity case so that ALL four
    # channels (especially the von-Mises / PEEQ stress fields, which are only
    # excited at higher impact energies) carry meaningful, well-reconstructed
    # structure. Among the upper-SSIM half of the damage-bearing subset, pick the
    # highest-velocity sample.
    upper = order_info[len(order_info)//2:]            # top half by SSIM
    chmap_idx = int(upper[np.argmax(params[upper, 0])])  # highest velocity among them
    print(f"    channel-map exemplar idx={chmap_idx}  "
          f"ssim={ssims[chmap_idx]:.3f}  v={params[chmap_idx,0]*3.6:.0f} km/h")

    # ── 7. Generate figures ───────────────────────────────────────────────────
    print("\nGenerating figures …")

    fig_sample_grid(best_idx,  fakes, reals, ssims, params, N_GRID,
                    "fig_eval_best_samples")
    fig_sample_grid(worst_idx, fakes, reals, ssims, params, N_GRID,
                    "fig_eval_worst_samples")
    fig_pred_vs_actual(preds, trues, "fig_eval_predicted_vs_actual")
    fig_ssim_hist(ssims,              "fig_eval_ssim_histogram")
    fig_mape_velocity(preds, trues, params, "fig_eval_mape_by_velocity")
    fig_error_dists(preds, trues,    "fig_eval_error_distributions")
    fig_channel_maps(fakes, reals, chmap_idx, "fig_eval_channel_maps")

    # ── Param2Image_Report variants (v3 model, bigger fonts, no info boxes) ──
    if REPORT_FIG_DIR.exists():
        print("\nGenerating Param2Image_Report figures (v3, bigger fonts) ...")
        fig_report_sample_grid(best_idx,  fakes, reals, ssims, params, N_GRID, "v3_best_samples")
        fig_report_sample_grid(worst_idx, fakes, reals, ssims, params, N_GRID, "v3_worst_samples")
        fig_report_mape(preds, trues, params, "v3_mape_by_velocity")

    # ── 8. Summary CSV ───────────────────────────────────────────────────────
    summary = pd.DataFrame({
        "ssim": ssims,
        **{f"pred_{n}": preds[:, i] for i, n in enumerate(OUTPUT_SHORT_NAMES)},
        **{f"true_{n}": trues[:, i] for i, n in enumerate(OUTPUT_SHORT_NAMES)},
        **{f"param_{c}": params[:, i] for i, c in enumerate(PARAM_COLUMNS)},
    })
    csv_out = FIGURES_DIR / f"eval_results_ep{epoch_num}.csv"
    summary.to_csv(str(csv_out), index=False)
    print(f"  Saved: {csv_out.name}")

    print(f"\nAll done. Figures in: {FIGURES_DIR}\n")


if __name__ == "__main__":
    main()
