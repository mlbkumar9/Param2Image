"""
make_placeholders.py — create placeholder PDFs for eval figures.
Run this once so main.tex compiles while the real figures are being
generated in Colab.  Replace each PDF with the real output from
eval_p2i_colab.py after running it in Colab.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from p2i_paths import figures_dir
FIGS_DIR = figures_dir()
FIGS_DIR.mkdir(parents=True, exist_ok=True)

PLACEHOLDERS = {
    "fig_eval_best_samples.pdf":       (14, 9,  "Best SSIM Samples\n(Ground truth | Predicted | |Diff|)\nGenerate with: eval_p2i_colab.py"),
    "fig_eval_worst_samples.pdf":      (14, 9,  "Worst SSIM Samples\n(Ground truth | Predicted | |Diff|)\nGenerate with: eval_p2i_colab.py"),
    "fig_eval_channel_maps.pdf":       (14, 7,  "Per-Channel RGBA Breakdown\n(R=Mises | G=PEEQ | B=Deflection | A=Mask)\nGenerate with: eval_p2i_colab.py"),
    "fig_eval_predicted_vs_actual.pdf":(12, 4,  "Predicted vs Actual — 3 Outputs\n(Scatter + R² + MAPE)\nGenerate with: eval_p2i_colab.py"),
    "fig_eval_mape_by_velocity.pdf":   (7,  4,  "Deflection MAPE by Velocity Bin\nGenerate with: eval_p2i_colab.py"),
    "fig_eval_error_distributions.pdf":(13, 4,  "Residual Distributions — 3 Outputs\nGenerate with: eval_p2i_colab.py"),
    "fig_eval_ssim_histogram.pdf":     (6,  4,  "SSIM Histogram — Validation Set\nGenerate with: eval_p2i_colab.py"),
}

for fname, (w, h, msg) in PLACEHOLDERS.items():
    out = FIGS_DIR / fname
    if out.exists():
        print(f"  Skip (exists): {fname}")
        continue
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_facecolor("#f5f5f5")
    ax.text(0.5, 0.5, msg, ha="center", va="center", fontsize=12,
            color="#555555", family="monospace", transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#aaaaaa"))
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_color("#cccccc")
    ax.set_title(f"[PLACEHOLDER — replace with Colab output]", fontsize=9, color="#999999")
    fig.tight_layout()
    fig.savefig(str(out), format="pdf")
    plt.close(fig)
    print(f"  Created: {fname}")

print("Done.")
