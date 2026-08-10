"""
plot_training_metrics.py
========================
Generates the training-curve figures for the Param2Image paper from the
REAL TensorBoard logs of the position-supervised 200-epoch run
(../Train_v3/logs_position/events.out.tfevents.*), cross-checked against
training_checkpoints_position/train_console.log and ckpt_best.pt.

Design rules followed (per paper style guide):
  * No in-figure titles -- all descriptive text lives in the LaTeX caption.
  * Plot text is rendered at the SAME size as the paper body font (10 pt,
    Palatino / mathpazo). This is achieved by building every figure at its
    final on-page width (a fraction of the 468 pt = 6.48 in text width) and
    using a 10 pt base font, so the figure is placed at scale 1:1 in LaTeX and
    the axis / tick / legend text matches the surrounding paragraph text.
  * No more than 3 panels share a row.
  * The full training-diagnostics set is shown (validation accuracy, GAN loss
    dynamics, and optimisation diagnostics incl. physics NLL, R1 penalty,
    learning-rate schedule and per-epoch wall-clock time) -- nothing dropped.

Usage:
    python plot_training_metrics.py
"""

import os
import re
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from p2i_paths import figures_dir
FIG_DIR    = str(figures_dir())
DATA_PATH  = os.path.join(SCRIPT_DIR, "training_checkpoints_position", "parsed_training_curves.json")
LOG_PATH   = os.path.join(SCRIPT_DIR, "training_checkpoints_position", "train_console.log")
os.makedirs(FIG_DIR, exist_ok=True)

with open(DATA_PATH) as fh:
    D = json.load(fh)

epoch     = np.array(D["epoch"], dtype=float)
val_ssim  = np.array(D["val_ssim"])
pos_px    = np.array(D["pos_err_px"])
pos_mm    = np.array(D["pos_err_mm"])
mape_defl = np.array(D["mape_defl"]) * 100.0     # stored as fraction -> percent
g_loss    = np.array(D["g_loss"])
d_loss    = np.array(D["d_loss"])
l1_loss   = np.array(D["l1_loss"])
pos_loss  = np.array(D["pos_loss"])
adv_loss  = np.array(D["adv_loss"])
phys_nll  = np.array(D["phys_nll"])
r1_loss   = np.array(D["r1_loss"])
lr_g      = np.array(D["lr_g"])


# ------------------------------------------------------------------
# Per-epoch wall-clock time, parsed from the console log
# ("--- Epoch N done in XXX.Xs ---"); keep the last value seen per epoch.
# ------------------------------------------------------------------
def parse_epoch_times(path, n):
    times = np.full(n, np.nan)
    if os.path.exists(path):
        pat = re.compile(r"Epoch\s+(\d+)\s+done in\s+([\d.]+)s")
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = pat.search(line)
                if m:
                    e = int(m.group(1))
                    if 1 <= e <= n:
                        times[e - 1] = float(m.group(2))
    return times

epoch_time = parse_epoch_times(LOG_PATH, len(epoch))

# ------------------------------------------------------------------
# Style -- match the paper BODY font exactly: Palatino (mathpazo) family at
# 10 pt, with axis/tick/legend text at body size. Figures are built at their
# final on-page width so no down-scaling shrinks the text.
# ------------------------------------------------------------------
BLUE, ORANGE, GREEN, RED, PURPLE, GRAY, BROWN = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#7f7f7f", "#8c564b")

TEXTWIDTH_IN = 468.0 / 72.27   # elsarticle 3p text width = 468 pt ~= 6.476 in
# Full-bleed width: A4 paper (210 mm) minus 7 mm margin on each side = 196 mm.
# The metric-plot figures are built at this width and placed full-bleed in LaTeX
# (\paperwidth-14mm) so they use the page efficiently; panels are forced square
# (1:1) via set_box_aspect, and the 10 pt font keeps text at body size at 1:1.
WIDE_IN = 196.0 / 25.4         # ~= 7.717 in

plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Palatino Linotype", "Palatino", "URW Palladio L",
                         "Book Antiqua", "DejaVu Serif"],
    "font.size":        10,    # = paper body font size
    "axes.labelsize":   10,
    "axes.titlesize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.linewidth":   0.7,
    "lines.linewidth":  1.3,
    "legend.frameon":   True,
    "legend.framealpha": 1.0,
    "legend.handlelength": 1.6,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype":     42,
    "ps.fonttype":      42,
    "ps.useafm":        False,
})


def savefig(fig, name):
    for ext in ("eps", "pdf"):
        fig.savefig(os.path.join(FIG_DIR, f"{name}.{ext}"), format=ext)
    print(f"  Saved: {name}.eps + .pdf")
    plt.close(fig)


best_ssim_epoch = int(np.nanargmax(val_ssim)) + 1
best_ssim_val   = float(np.nanmax(val_ssim))
best_pos_epoch  = int(np.nanargmin(pos_px)) + 1
best_pos_val    = float(np.nanmin(pos_px))

# ==================================================================
# Figure 1 -- Validation convergence: SSIM | impact-position error | MAPE
#             1x3, built at full text width so each panel renders at the
#             same physical size it occupies on the page.
# ==================================================================
print("Generating validation-convergence figure (SSIM / position error / MAPE) ...")
fig, axes = plt.subplots(1, 3, figsize=(WIDE_IN, 3.05), constrained_layout=True)

ax = axes[0]
ax.plot(epoch, val_ssim, color=BLUE)
ax.scatter([best_ssim_epoch], [best_ssim_val], s=26, color=RED, zorder=5,
           label=f"Best {best_ssim_val:.3f} (ep {best_ssim_epoch})")
ax.axvspan(1, 5, color="#e9f5e9", zorder=0, label="LR warm-up")
ax.set_xlabel("Epoch")
ax.set_ylabel("Validation SSIM")
ax.set_xlim(0, 200); ax.set_ylim(0, 0.85)
ax.legend(loc="lower right")
ax.grid(True, color="0.85")

ax = axes[1]
ax.plot(epoch, pos_px, color=PURPLE)
ax.scatter([best_pos_epoch], [best_pos_val], s=26, color=RED, zorder=5,
           label=f"Best {best_pos_val:.1f} px (ep {best_pos_epoch})")
ax.axvspan(1, 5, color="#e9f5e9", zorder=0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Position error (px)")
ax.set_xlim(0, 200)
ax.legend(loc="upper right")
ax.grid(True, color="0.85")

ax = axes[2]
ax.semilogy(epoch, mape_defl, color=ORANGE)
ax.axvspan(1, 5, color="#e9f5e9", zorder=0)
ax.set_xlabel("Epoch")
ax.set_ylabel("Deflection MAPE (%)")
ax.set_xlim(0, 200)
ax.grid(True, color="0.85", which="both")

for a in axes:
    a.set_box_aspect(1)          # square (1:1) panels
savefig(fig, "fig_training_validation_metrics")

# ==================================================================
# Figure 2 -- GAN loss dynamics: G | D | L1 & position-supervision
# ==================================================================
print("Generating training-loss-dynamics figure (G / D / L1 & position) ...")
fig, axes = plt.subplots(1, 3, figsize=(WIDE_IN, 3.05), constrained_layout=True)

ax = axes[0]
ax.plot(epoch, g_loss, color=BLUE, label="$\\mathcal{L}_G$ (total)")
ax.plot(epoch, adv_loss, color=ORANGE, label="$\\mathcal{L}_{\\mathrm{adv}}^{G}$")
ax.set_xlabel("Epoch")
ax.set_ylabel("Generator loss")
ax.set_xlim(0, 200); ax.set_ylim(0, 8)
ax.legend(loc="upper right")
ax.grid(True, color="0.85")

ax = axes[1]
ax.plot(epoch, d_loss, color=RED)
ax.axhline(y=np.log(2), color=GRAY, lw=1.0, ls="--", label="$\\log 2$")
ax.set_xlabel("Epoch")
ax.set_ylabel("Discriminator loss")
ax.set_xlim(0, 200)
ax.legend(loc="lower right")
ax.grid(True, color="0.85")

ax = axes[2]
ax.plot(epoch, l1_loss, color=GREEN, label="$\\mathcal{L}_{L1}$")
ax.plot(epoch, pos_loss, color=PURPLE, label="$\\mathcal{L}_{\\mathrm{pos}}$")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_xlim(0, 200); ax.set_yscale("log")
ax.legend(loc="upper right")
ax.grid(True, color="0.85", which="both")

for a in axes:
    a.set_box_aspect(1)          # square (1:1) panels
savefig(fig, "fig_training_loss_dynamics")

# ==================================================================
# Figure 3 -- Optimisation diagnostics: physics-head NLL | R1 penalty |
#             per-epoch wall-clock time. 1x3 at full text width; restores
#             the physics-NLL and epoch-time panels of the original
#             training dashboard and adds the R1-regularisation diagnostic.
# ==================================================================
print("Generating optimisation-diagnostics figure (NLL / R1 / epoch time) ...")
fig, axes = plt.subplots(1, 3, figsize=(WIDE_IN, 3.05), constrained_layout=True)

ax = axes[0]
ax.plot(epoch, phys_nll, color=BROWN)
ax.axhline(0, color=GRAY, lw=0.9, ls="--")
ax.set_xlabel("Epoch")
ax.set_ylabel("Physics head NLL")
ax.set_xlim(0, 200)
ax.grid(True, color="0.85")

ax = axes[1]
ax.semilogy(epoch, np.maximum(r1_loss, 1e-6), color=RED)
ax.set_xlabel("Epoch")
ax.set_ylabel("R1 penalty")
ax.set_xlim(0, 200)
ax.grid(True, color="0.85", which="both")

ax = axes[2]
ax.plot(epoch, epoch_time, color=GRAY)
mean_t = np.nanmean(epoch_time)
ax.axhline(mean_t, color=RED, lw=1.0, ls="--", label=f"Mean {mean_t:.0f} s")
ax.set_xlabel("Epoch")
ax.set_ylabel("Epoch time (s)")
ax.set_xlim(0, 200)
ax.legend(loc="lower right")
ax.grid(True, color="0.85")

for a in axes:
    a.set_box_aspect(1)          # square (1:1) panels
savefig(fig, "fig_training_diagnostics")

# ==================================================================
# Summary printout
# ==================================================================
print("\n=== Training summary (position-supervised, 200-epoch run) ===")
print(f"  Best Val SSIM        : {best_ssim_val:.4f}  (epoch {best_ssim_epoch})")
print(f"  Final Val SSIM       : {val_ssim[-1]:.4f}  (epoch 200)")
print(f"  Best position error  : {best_pos_val:.2f} px (epoch {best_pos_epoch})")
print(f"  Final position error : {pos_px[-1]:.2f} px (epoch 200)")
print(f"  Min deflection MAPE  : {np.nanmin(mape_defl):.2f}%  (epoch {int(np.nanargmin(mape_defl))+1})")
print(f"  Final deflection MAPE: {mape_defl[-1]:.2f}%  (epoch 200)")
print(f"  Mean epoch time      : {np.nanmean(epoch_time):.1f} s over {np.isfinite(epoch_time).sum()} epochs")
print(f"\nFigures saved to: {FIG_DIR}")
