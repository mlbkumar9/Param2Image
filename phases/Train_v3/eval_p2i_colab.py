"""
eval_p2i_colab.py  —  Param2Image v3.1 Evaluation Script
=========================================================
Run this in Google Colab after mounting Google Drive.  It loads the epoch-48
EMA checkpoint, evaluates on the full validation split, and saves publication-
quality figures to  /content/drive/MyDrive/p2i_v3/eval_figures/.

Usage (Colab cell):
    !python eval_p2i_colab.py \
        --checkpoint /content/drive/MyDrive/p2i_v3/training_checkpoints_position/ckpt_best_position.pt \
        --csv        /content/drive/MyDrive/p2i_v3/data/impact_dataset.csv \
        --data_dir   /content/drive/MyDrive/p2i_v3/data/images \
        --stats      /content/drive/MyDrive/p2i_v3/training_checkpoints/normalization_stats.json \
        --out_dir    /content/drive/MyDrive/p2i_v3/eval_figures \
        --n_samples  16

Figures generated
-----------------
  fig_eval_best_samples.pdf      — 4×4 grid: params / GT / pred / diff
  fig_eval_worst_samples.pdf     — 4×4 grid for worst-SSIM samples
  fig_eval_predicted_vs_actual.pdf — scatter plots (3 outputs, R²)
  fig_eval_ssim_histogram.pdf    — SSIM distribution over val set
  fig_eval_mape_by_velocity.pdf  — deflection MAPE binned by velocity
  fig_eval_error_distributions.pdf — residual histograms for 3 outputs
  fig_eval_channel_maps.pdf      — per-channel RGBA breakdown (best sample)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid

# ---------------------------------------------------------------------------
# Matplotlib settings (publication style)
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats as scipy_stats

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,   # embed fonts in PDF
})

# ---------------------------------------------------------------------------
# Add script directory to path so we can import local modules
# ---------------------------------------------------------------------------
script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir))

from model_p2i_pytorch import Generator, PhysicsHead
from position_supervision import impact_position_error_pixels, pixels_to_plate_mm
from preprocessing_p2i import (
    ImpactTabularScaler,
    PARAM_COLUMNS,
    OUTPUT_COLUMNS,
    OUTPUT_SHORT_NAMES,
    OUTPUT_DISPLAY_NAMES,
    OUTPUT_UNITS,
    SCALER_FILENAME,
)

# Default architecture hyper-parameters (match training defaults in model file)
NUM_PARAMS  = 4
LATENT_DIM  = 100
W_DIM       = 512
IMG_CHANNELS = 4
INIT_CHANNELS = 512   # base_channels / init_channels in Generator


# ---------------------------------------------------------------------------
# SSIM helper (pure PyTorch, no torchmetrics dependency)
# ---------------------------------------------------------------------------

def _gaussian_kernel(size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    k = g.outer(g)
    return (k / k.sum()).unsqueeze(0).unsqueeze(0)


def ssim_torch(pred: torch.Tensor, target: torch.Tensor,
               data_range: float = 2.0, K1: float = 0.01, K2: float = 0.03,
               window_size: int = 11) -> torch.Tensor:
    """SSIM averaged over channels and spatial dimensions. Input in [-1,1]."""
    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    B, C, H, W = pred.shape
    kernel = _gaussian_kernel(window_size).to(pred.device)

    ssim_vals = []
    for c in range(C):
        x = pred[:, c:c+1]
        y = target[:, c:c+1]
        mu_x  = F.conv2d(x, kernel, padding=window_size // 2)
        mu_y  = F.conv2d(y, kernel, padding=window_size // 2)
        mu_xx = F.conv2d(x * x, kernel, padding=window_size // 2)
        mu_yy = F.conv2d(y * y, kernel, padding=window_size // 2)
        mu_xy = F.conv2d(x * y, kernel, padding=window_size // 2)
        sig_xx = mu_xx - mu_x ** 2
        sig_yy = mu_yy - mu_y ** 2
        sig_xy = mu_xy - mu_x * mu_y
        num = (2 * mu_x * mu_y + C1) * (2 * sig_xy + C2)
        den = (mu_x ** 2 + mu_y ** 2 + C1) * (sig_xx + sig_yy + C2)
        ssim_vals.append((num / den).mean())
    return torch.stack(ssim_vals).mean()


# ---------------------------------------------------------------------------
# Dataset (minimal, no training augmentation)
# ---------------------------------------------------------------------------

class ValDataset(torch.utils.data.Dataset):
    def __init__(self, df, data_dir, scaler: ImpactTabularScaler, img_size: int = 1024):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.scaler = scaler
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*4, [0.5]*4),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, str(row["img_after"]).replace("\\", "/"))
        img = Image.open(img_path).convert("RGBA")
        img_t = self.transform(img)

        params_raw = row[PARAM_COLUMNS].astype(float).to_numpy()
        outputs_raw = row[OUTPUT_COLUMNS].astype(float).to_numpy()
        params_n = self.scaler.transform_params_array(params_raw)[0]
        outputs_n = self.scaler.transform_outputs_array(outputs_raw)[0]
        return (
            torch.tensor(params_n, dtype=torch.float32),
            img_t,
            torch.tensor(outputs_n, dtype=torch.float32),
            torch.tensor(params_raw, dtype=torch.float32),
            torch.tensor(outputs_raw, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    return ckpt


def build_models(device: torch.device, cfg: dict | None = None):
    """Build Generator and PhysicsHead from config dict (or fall back to defaults)."""
    c = cfg or {}
    G = Generator(
        num_params          = c.get("num_params",          NUM_PARAMS),
        num_numerical_outputs = c.get("num_numerical_outputs", 3),
        latent_dim          = c.get("latent_dim",          LATENT_DIM),
        img_channels        = c.get("img_channels",        IMG_CHANNELS),
        img_size            = c.get("img_size",            1024),
        init_size           = c.get("init_size",           16),
        init_channels       = c.get("init_channels",       INIT_CHANNELS),
        w_dim               = c.get("w_dim",               W_DIM),
        use_grad_checkpoint = False,   # not needed for inference
    ).to(device)
    P = PhysicsHead(
        num_params  = c.get("num_params",  NUM_PARAMS),
        num_outputs = c.get("num_numerical_outputs", 3),
        hidden      = c.get("phys_hidden", 256),
        depth       = c.get("phys_depth",  6),
        num_freqs   = c.get("phys_freqs",  16),
        sigma       = c.get("phys_sigma",  5.0),
    ).to(device)
    return G, P


def restore_ema_weights(G, P, ckpt: dict):
    """Load EMA shadow weights into G and P (fall back to plain weights)."""
    # Training script saves:
    #   ckpt["ema_g"]  -> EMA.state_dict() == shadow {name: tensor}
    #   ckpt["ema_p"]  -> same for physics head
    #   ckpt["generator"]   -> plain generator weights
    #   ckpt["physics_head"] -> plain physics weights
    def _load(model, key_ema, key_plain, label):
        if key_ema in ckpt:
            model.load_state_dict(ckpt[key_ema], strict=True)
            print(f"  {label} loaded from EMA ({key_ema})")
        elif key_plain in ckpt:
            model.load_state_dict(ckpt[key_plain], strict=True)
            print(f"  {label} loaded from plain weights ({key_plain})")
        else:
            raise KeyError(f"Checkpoint missing both '{key_ema}' and '{key_plain}'")

    _load(G, "ema_g", "generator",    "Generator")
    _load(P, "ema_p", "physics_head", "PhysicsHead")


# ---------------------------------------------------------------------------
# Core evaluation loop
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(G, P, loader, device, scaler: ImpactTabularScaler, latent_dim: int):
    G.eval(); P.eval()
    all_ssim, all_pred, all_true, all_raw_params = [], [], [], []
    all_fake_imgs, all_real_imgs = [], []
    position_error_sum = 0.0
    position_sample_count = 0

    for batch in loader:
        params_n, real_img, outputs_n, params_raw, outputs_raw = batch
        params_n  = params_n.to(device)
        real_img  = real_img.to(device)
        outputs_n = outputs_n.to(device)

        z = torch.randn(params_n.size(0), latent_dim, device=device)
        fake_img = G(z, params_n)
        if fake_img.shape[-2:] != real_img.shape[-2:]:
            fake_img = F.interpolate(fake_img, size=real_img.shape[-2:], mode="bilinear", align_corners=False)
        batch_position_error = impact_position_error_pixels(fake_img, real_img)
        position_error_sum += float(batch_position_error) * params_n.size(0)
        position_sample_count += params_n.size(0)

        # SSIM per sample
        for i in range(params_n.size(0)):
            s = ssim_torch(
                fake_img[i:i+1].clamp(-1, 1),
                real_img[i:i+1].clamp(-1, 1),
            ).item()
            all_ssim.append(s)

        # Physics head predictions (normalised space)
        mu, log_var = P(params_n)
        # Inverse-transform to physical units
        mu_np = mu.cpu().numpy()
        outputs_n_np = outputs_n.cpu().numpy()
        # inverse transform
        mu_phys   = scaler.inverse_transform_outputs_array(mu_np)
        true_phys = scaler.inverse_transform_outputs_array(outputs_n_np)
        all_pred.append(mu_phys)
        all_true.append(true_phys)
        all_raw_params.append(params_raw.numpy())

        # Keep images for visualisation (on CPU)
        all_fake_imgs.append(fake_img.cpu())
        all_real_imgs.append(real_img.cpu())

    all_ssim   = np.array(all_ssim)
    all_pred   = np.concatenate(all_pred, axis=0)
    all_true   = np.concatenate(all_true, axis=0)
    all_params = np.concatenate(all_raw_params, axis=0)
    all_fake   = torch.cat(all_fake_imgs, dim=0)
    all_real   = torch.cat(all_real_imgs, dim=0)

    position_error_px = position_error_sum / max(position_sample_count, 1)
    return all_ssim, all_pred, all_true, all_params, all_fake, all_real, position_error_px


# ---------------------------------------------------------------------------
# Helper: tensor -> PIL (for grid)
# ---------------------------------------------------------------------------

def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert [-1,1] C×H×W tensor to uint8 PIL (RGB, dropping alpha)."""
    t = t.clamp(-1, 1).cpu()
    arr = ((t + 1) / 2 * 255).byte().numpy()
    if arr.shape[0] == 4:
        # blend alpha onto white background
        alpha = arr[3:4].astype(np.float32) / 255.0
        rgb   = arr[:3].astype(np.float32) / 255.0
        comp  = rgb * alpha + np.ones_like(rgb) * (1 - alpha)
        arr   = (comp * 255).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr.transpose(1, 2, 0))  # H×W×C


# ---------------------------------------------------------------------------
# Figure 1 — Best / Worst sample grids
# ---------------------------------------------------------------------------

def fig_sample_grid(indices, all_fake, all_real, all_ssim, all_params,
                    n: int, title: str, out_path: str):
    """n samples from `indices` shown as GT | Pred | |Diff|."""
    cols = min(n, len(indices))
    fig, axes = plt.subplots(3, cols, figsize=(cols * 2.2, 3 * 2.2 + 0.5))
    if cols == 1:
        axes = axes[:, np.newaxis]

    row_labels = ["Ground\ntruth", "Predicted", "|Difference|"]
    for col, idx in enumerate(indices[:cols]):
        real_pil = tensor_to_pil(all_real[idx])
        fake_pil = tensor_to_pil(all_fake[idx])
        diff_arr = np.abs(
            np.array(real_pil, dtype=np.float32) - np.array(fake_pil, dtype=np.float32)
        ).astype(np.uint8)
        diff_pil = Image.fromarray(diff_arr)

        for row, im in enumerate([real_pil, fake_pil, diff_pil]):
            ax = axes[row, col]
            ax.imshow(im)
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=8, rotation=90, va="center")
            v, r, x, y = all_params[idx]
            ax.set_title(
                f"SSIM={all_ssim[idx]:.3f}\n"
                f"v={v:.0f}km/h r={r:.0f}mm",
                fontsize=7, pad=2,
            )

    fig.suptitle(title, fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 2 — Predicted vs Actual scatter (3 outputs)
# ---------------------------------------------------------------------------

def fig_pred_vs_actual(all_pred, all_true, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for k, ax in enumerate(axes):
        y_true = all_true[:, k]
        y_pred = all_pred[:, k]
        # Filter out extreme outlier predictions for plotting
        mask = np.isfinite(y_pred) & np.isfinite(y_true)
        yt, yp = y_true[mask], y_pred[mask]

        # Stats
        slope, intercept, r, p_val, _ = scipy_stats.linregress(yt, yp)
        r2 = r ** 2
        mape_mask = np.abs(yt) > 1e-3
        mape = 100 * np.mean(np.abs((yp[mape_mask] - yt[mape_mask]) / yt[mape_mask])) if mape_mask.any() else float("nan")

        ax.scatter(yt, yp, s=10, alpha=0.5, c=colors[k], linewidths=0, label=None)
        # Perfect line
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, label="Perfect")
        # Fit line
        x_line = np.linspace(lo, hi, 200)
        ax.plot(x_line, slope * x_line + intercept, "r-", lw=1.0, label=f"OLS fit")

        ax.set_xlabel(f"Ground truth ({OUTPUT_UNITS[k]})", fontsize=8)
        ax.set_ylabel(f"Predicted ({OUTPUT_UNITS[k]})", fontsize=8)
        ax.set_title(
            f"{OUTPUT_DISPLAY_NAMES[k]}\n"
            f"$R^2$={r2:.4f}  MAPE={mape:.1f}%",
            fontsize=8,
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted vs Actual — Validation Set (Epoch 48 EMA)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 3 — SSIM histogram
# ---------------------------------------------------------------------------

def fig_ssim_histogram(all_ssim: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 4))
    n, bins, _ = ax.hist(all_ssim, bins=20, color="#2E86AB", edgecolor="white",
                         linewidth=0.5, alpha=0.85)
    ax.axvline(all_ssim.mean(), color="red", lw=1.5, linestyle="--",
               label=f"Mean = {all_ssim.mean():.4f}")
    ax.axvline(np.median(all_ssim), color="orange", lw=1.5, linestyle=":",
               label=f"Median = {np.median(all_ssim):.4f}")
    ax.set_xlabel("SSIM", fontsize=9)
    ax.set_ylabel("Count", fontsize=9)
    ax.set_title(
        f"SSIM Distribution — Validation Set\n"
        f"N={len(all_ssim)}  mean={all_ssim.mean():.4f}  "
        f"std={all_ssim.std():.4f}  min={all_ssim.min():.4f}  max={all_ssim.max():.4f}",
        fontsize=9,
    )
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 4 — Deflection MAPE binned by velocity
# ---------------------------------------------------------------------------

def fig_mape_by_velocity(all_pred, all_true, all_params, out_path: str):
    defl_idx = OUTPUT_SHORT_NAMES.index("deflection")
    vel = all_params[:, 0]   # km/h
    y_true = all_true[:, defl_idx]
    y_pred = all_pred[:, defl_idx]

    # Only include samples where ground-truth deflection > 0.1 mm
    mask = np.abs(y_true) > 0.1
    vel, y_true, y_pred = vel[mask], y_true[mask], y_pred[mask]

    bins = [0, 100, 200, 300, 450, 700]
    bin_labels = ["0–100", "100–200", "200–300", "300–450", "450–700"]
    mapes, counts = [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        sel = (vel >= lo) & (vel < hi)
        if sel.sum() > 0:
            m = 100 * np.mean(np.abs((y_pred[sel] - y_true[sel]) / y_true[sel]))
            mapes.append(m)
        else:
            mapes.append(float("nan"))
        counts.append(int(sel.sum()))

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(bin_labels))
    bars = ax.bar(x, mapes, color="#A23B72", edgecolor="white", linewidth=0.5, alpha=0.85)
    ax.bar_label(bars, labels=[f"{m:.1f}%" for m in mapes], fontsize=8, padding=2)
    ax2 = ax.twinx()
    ax2.plot(x, counts, "ko--", markersize=5, lw=1.2, label="Sample count")
    ax2.set_ylabel("Sample count", fontsize=9)
    ax2.legend(loc="upper right", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lbl} km/h\n(n={c})" for lbl, c in zip(bin_labels, counts)], fontsize=8)
    ax.set_xlabel("Impact velocity bin", fontsize=9)
    ax.set_ylabel("Deflection MAPE (%)", fontsize=9)
    ax.set_title("Max-Deflection MAPE by Velocity Regime — Epoch 48 EMA", fontsize=9, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 5 — Residual error distributions (3 outputs)
# ---------------------------------------------------------------------------

def fig_error_distributions(all_pred, all_true, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    colors = ["#2E86AB", "#A23B72", "#F18F01"]

    for k, ax in enumerate(axes):
        y_true = all_true[:, k]
        y_pred = all_pred[:, k]
        residual = y_pred - y_true
        mask = np.isfinite(residual)
        res = residual[mask]

        ax.hist(res, bins=25, color=colors[k], edgecolor="white", linewidth=0.5, alpha=0.85, density=True)
        # Fit normal
        mu_r, std_r = scipy_stats.norm.fit(res)
        x_fit = np.linspace(res.min(), res.max(), 300)
        ax.plot(x_fit, scipy_stats.norm.pdf(x_fit, mu_r, std_r), "k-", lw=1.5, label=f"N({mu_r:.2f}, {std_r:.2f})")
        ax.axvline(0, color="red", lw=1, linestyle="--", label="Zero error")
        ax.set_xlabel(f"Residual ({OUTPUT_UNITS[k]})", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.set_title(
            f"{OUTPUT_DISPLAY_NAMES[k]}\n"
            f"μ={mu_r:.3f}  σ={std_r:.3f}",
            fontsize=8,
        )
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Prediction Residual Distributions — Validation Set (Epoch 48 EMA)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Figure 6 — Per-channel RGBA breakdown for best sample
# ---------------------------------------------------------------------------

def fig_channel_maps(all_fake, all_real, best_idx: int, out_path: str):
    channels = ["R (Mises proxy)", "G (PEEQ proxy)", "B (Deflection proxy)", "A (Damage mask)"]
    fake = all_fake[best_idx].clamp(-1, 1).numpy()  # 4×H×W in [-1,1]
    real = all_real[best_idx].clamp(-1, 1).numpy()

    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    cmaps = ["Reds", "Greens", "Blues", "Greys"]
    for c in range(4):
        for row, (data, lbl) in enumerate([(real, "Ground truth"), (fake, "Predicted")]):
            ax = axes[row, c]
            ch = (data[c] + 1) / 2  # [0,1]
            im = ax.imshow(ch, cmap=cmaps[c], vmin=0, vmax=1, aspect="equal")
            if row == 0:
                ax.set_title(channels[c], fontsize=8)
            if c == 0:
                ax.set_ylabel(lbl, fontsize=8)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle("Per-Channel Breakdown — Best SSIM Sample (Epoch 48 EMA)", fontsize=10, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Param2Image v3.1 evaluation")
    p.add_argument("--checkpoint", required=True, help="Path to ckpt_best.pt")
    p.add_argument("--csv",        required=True, help="Path to impact_dataset.csv")
    p.add_argument("--data_dir",   required=True, help="Directory containing impact images")
    p.add_argument("--stats",      required=True, help="Path to normalization_stats.json")
    p.add_argument("--out_dir",    default="./eval_figures", help="Output directory for figures")
    p.add_argument("--n_samples",  type=int, default=16, help="Number of samples to show in grids")
    p.add_argument("--batch_size", type=int, default=4,  help="Eval batch size")
    p.add_argument("--img_size",   type=int, default=256, help="Image size for eval (256 saves VRAM)")
    p.add_argument("--val_frac",   type=float, default=0.1, help="Fraction of data for validation")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nParam2Image v3.1 Evaluation  —  device: {device}")
    print(f"Checkpoint: {args.checkpoint}")

    # --- Output dir ---
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Scaler ---
    scaler = ImpactTabularScaler.load(args.stats)
    print(f"Scaler loaded from: {args.stats}")

    # --- Dataset ---
    import pandas as pd
    df = pd.read_csv(args.csv)
    required = ["img_after"] + PARAM_COLUMNS + OUTPUT_COLUMNS
    df = df.dropna(subset=required).reset_index(drop=True)
    n_val = max(1, int(len(df) * args.val_frac))
    rng = np.random.RandomState(args.seed)
    val_idx = rng.choice(len(df), size=n_val, replace=False)
    val_df  = df.iloc[val_idx]
    print(f"Validation set: {len(val_df)} samples (val_frac={args.val_frac:.0%})")

    val_ds = ValDataset(val_df, args.data_dir, scaler, img_size=args.img_size)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True,
    )

    # --- Models ---
    ckpt  = load_checkpoint(args.checkpoint, device)
    print(f"Checkpoint epoch: {ckpt.get('epoch', 'unknown')}")
    cfg = ckpt.get("config", {})
    G, P = build_models(device, cfg)
    restore_ema_weights(G, P, ckpt)
    G.eval(); P.eval()
    print(f"Models loaded. Generator params: {sum(p.numel() for p in G.parameters()):,}")
    print(f"PhysicsHead params: {sum(p.numel() for p in P.parameters()):,}")

    # --- Evaluation loop ---
    print("\nRunning inference on validation set...")
    latent_dim = cfg.get("latent_dim", LATENT_DIM)
    all_ssim, all_pred, all_true, all_params, all_fake, all_real, position_error_px = evaluate(
        G, P, val_loader, device, scaler, latent_dim,
    )
    print(f"Val SSIM: mean={all_ssim.mean():.4f}  median={np.median(all_ssim):.4f}  "
          f"std={all_ssim.std():.4f}")
    position_error_mm = pixels_to_plate_mm(
        position_error_px,
        args.img_size,
        plate_width_mm=cfg.get("plate_width_mm", 2500.0),
        border_fraction=cfg.get("pos_border_fraction", 0.04),
    )
    print(f"Val Impact Position Error: {position_error_px:.2f}px (~{position_error_mm:.2f}mm)")

    # MAPE on deflection (denom > 0.1 mm)
    defl_idx = OUTPUT_SHORT_NAMES.index("deflection")
    d_true = all_true[:, defl_idx]
    d_pred = all_pred[:, defl_idx]
    mask_d = np.abs(d_true) > 0.1
    if mask_d.any():
        defl_mape = 100 * np.mean(np.abs((d_pred[mask_d] - d_true[mask_d]) / d_true[mask_d]))
        print(f"Val Deflection MAPE: {defl_mape:.1f}%")

    # Sort by SSIM
    ssim_order = np.argsort(all_ssim)
    worst_idx  = ssim_order[:args.n_samples].tolist()
    best_idx   = ssim_order[-args.n_samples:][::-1].tolist()

    # --- Figures ---
    print("\nGenerating figures...")

    fig_sample_grid(
        best_idx, all_fake, all_real, all_ssim, all_params,
        n=args.n_samples,
        title="Best SSIM Samples — Epoch 48 EMA",
        out_path=str(out_dir / "fig_eval_best_samples.pdf"),
    )
    fig_sample_grid(
        worst_idx, all_fake, all_real, all_ssim, all_params,
        n=args.n_samples,
        title="Worst SSIM Samples — Epoch 48 EMA",
        out_path=str(out_dir / "fig_eval_worst_samples.pdf"),
    )
    fig_pred_vs_actual(
        all_pred, all_true,
        out_path=str(out_dir / "fig_eval_predicted_vs_actual.pdf"),
    )
    fig_ssim_histogram(
        all_ssim,
        out_path=str(out_dir / "fig_eval_ssim_histogram.pdf"),
    )
    fig_mape_by_velocity(
        all_pred, all_true, all_params,
        out_path=str(out_dir / "fig_eval_mape_by_velocity.pdf"),
    )
    fig_error_distributions(
        all_pred, all_true,
        out_path=str(out_dir / "fig_eval_error_distributions.pdf"),
    )
    fig_channel_maps(
        all_fake, all_real, best_idx[0],
        out_path=str(out_dir / "fig_eval_channel_maps.pdf"),
    )

    # --- Summary CSV ---
    import pandas as pd
    summary = pd.DataFrame({
        "val_ssim": all_ssim,
        **{f"pred_{n}": all_pred[:, i] for i, n in enumerate(OUTPUT_SHORT_NAMES)},
        **{f"true_{n}": all_true[:, i] for i, n in enumerate(OUTPUT_SHORT_NAMES)},
        **{f"param_{c}": all_params[:, i] for i, c in enumerate(PARAM_COLUMNS)},
    })
    csv_path = str(out_dir / "eval_results.csv")
    summary.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    print(f"\nAll figures saved to: {out_dir}/")
    print("Done.")


if __name__ == "__main__":
    main()
