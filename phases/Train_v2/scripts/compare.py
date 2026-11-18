"""
Publication-oriented comparison of Param2Image predictions against Abaqus data.

Outputs:
  - comparison_panel.{png,pdf,svg,eps,fig}
  - numerical_parity.{png,pdf,svg,eps,fig}
  - comparison_results.csv
"""
from __future__ import annotations

import argparse
import os
import pickle
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision import transforms
from torchvision.utils import save_image

from p2i.model import load_generator_checkpoint
from p2i.paths import default_data_dir, default_generator_path, repo_root, results_dir
from p2i.preprocessing import (
    OUTPUT_DISPLAY_NAMES,
    OUTPUT_SHORT_NAMES,
    OUTPUT_UNITS,
    ImpactTabularScaler,
    default_scaler_path_for_model,
    safe_absolute_percentage_error,
)

REPO_ROOT = repo_root(Path(__file__))

NUM_PARAMS = 4
NUM_NUMERICAL_OUTPUTS = 3
LATENT_DIM = 100
IMG_CHANNELS = 4
IMG_SIZE = 512
REQUIRED_COLS = [
    "img_after",
    "velocity_ms",
    "proj_radius_mm",
    "impact_x_mm",
    "impact_y_mm",
    "max_mises_mpa",
    "max_peeq",
    "max_deflection_mm",
]


def parse_args():
    default_data = default_data_dir(Path(__file__))
    parser = argparse.ArgumentParser(
        description="Create research-quality comparison plots for Param2Image cGAN predictions."
    )
    parser.add_argument("--data_dir", type=Path, default=default_data)
    parser.add_argument(
        "--model_path",
        type=Path,
        default=default_generator_path(Path(__file__)),
    )
    parser.add_argument(
        "--scaler_path",
        type=Path,
        default=None,
        help="Path to normalization_stats.json. Defaults to the model checkpoint folder.",
    )
    parser.add_argument("--num_samples", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--selection",
        choices=["stratified", "random", "tail"],
        default="stratified",
        help="stratified samples cover the nonlinear deflection response range.",
    )
    parser.add_argument(
        "--num_latent_samples",
        type=int,
        default=8,
        help="Average numerical predictions across latent draws for legacy stochastic heads.",
    )
    parser.add_argument("--dpi", type=int, default=450)
    return parser.parse_args()


def configure_publication_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "axes.titleweight": "bold",
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "figure.titlesize": 12,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 1.0,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_ground_truth_image(img_path, img_size=512):
    img = Image.open(img_path).convert("RGBA")
    transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ]
    )
    return transform(img)


def tensor_to_numpy(tensor):
    img = tensor.detach().cpu().numpy()
    if img.min() < 0:
        img = (img + 1.0) / 2.0
    img = np.clip(img, 0, 1)
    return np.transpose(img, (1, 2, 0))


def compute_image_metrics(generated, ground_truth):
    gen_rgb = generated[..., :3].astype(np.float32)
    gt_rgb = ground_truth[..., :3].astype(np.float32)
    l1 = float(np.mean(np.abs(gen_rgb - gt_rgb)))
    mse = float(np.mean((gen_rgb - gt_rgb) ** 2))
    metrics = {"image_l1_rgb": l1, "image_mse_rgb": mse}
    try:
        metrics["image_psnr_rgb"] = float(psnr(gt_rgb, gen_rgb, data_range=1.0))
    except ValueError:
        metrics["image_psnr_rgb"] = np.nan
    try:
        metrics["image_ssim_rgb"] = float(
            ssim(gt_rgb, gen_rgb, channel_axis=-1, data_range=1.0)
        )
    except TypeError:
        metrics["image_ssim_rgb"] = float(
            ssim(gt_rgb, gen_rgb, multichannel=True, data_range=1.0)
        )
    except ValueError:
        metrics["image_ssim_rgb"] = np.nan
    return metrics


def select_samples(df, num_samples, seed, strategy):
    n = min(num_samples, len(df))
    if strategy == "random":
        return df.sample(n=n, random_state=seed).reset_index(drop=True)
    if strategy == "tail":
        return df.tail(n).reset_index(drop=True)

    ranked = df.sort_values("max_deflection_mm").reset_index(drop=True)
    if n == 1:
        return ranked.iloc[[len(ranked) // 2]].reset_index(drop=True)
    positions = np.linspace(0, len(ranked) - 1, n).round().astype(int)
    return ranked.iloc[np.unique(positions)].reset_index(drop=True)


def prepare_model_params(row, scaler):
    raw = [[row["velocity_ms"], row["proj_radius_mm"], row["impact_x_mm"], row["impact_y_mm"]]]
    return scaler.transform_params_array(raw) if scaler is not None else raw


def predict_sample(generator, row, device, scaler, num_latent_samples):
    model_params = torch.tensor(
        prepare_model_params(row, scaler), dtype=torch.float32, device=device
    )
    model_params = model_params.repeat(num_latent_samples, 1)
    z = torch.randn(num_latent_samples, LATENT_DIM, device=device)

    with torch.no_grad():
        pred_images, pred_numerical = generator(z, model_params)

    numerical = pred_numerical.detach().cpu().numpy()
    if scaler is not None:
        numerical = scaler.inverse_transform_outputs_array(numerical)

    return pred_images[0].detach().cpu(), numerical.mean(axis=0), numerical.std(axis=0)


def save_figure_bundle(fig, base_path, dpi):
    base_path = Path(base_path)
    saved = []
    for suffix in [".png", ".pdf", ".svg", ".eps"]:
        out_path = base_path.with_suffix(suffix)
        fig.savefig(out_path, dpi=dpi, facecolor="white")
        saved.append(out_path)

    fig_path = base_path.with_suffix(".fig")
    with open(fig_path, "wb") as handle:
        pickle.dump(fig, handle, protocol=pickle.HIGHEST_PROTOCOL)
    saved.append(fig_path)
    return saved


def add_image_axis(ax, image, title):
    ax.imshow(image, interpolation="lanczos")
    ax.set_title(title, pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)


def add_numeric_table(ax, sample):
    ax.axis("off")
    rows = []
    compact_labels = ["Mises (MPa)", "PEEQ", "Defl. (mm)"]
    for label, short in zip(compact_labels, OUTPUT_SHORT_NAMES):
        gt = sample[f"gt_{short}"]
        pred = sample[f"pred_{short}"]
        err = sample[f"{short}_ape_pct"]
        fmt = "{:.4f}" if short == "peeq" else "{:.1f}"
        rows.append([label, fmt.format(gt), fmt.format(pred), f"{err:.1f}"])

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Abaqus", "Model", "APE (%)"],
        loc="center",
        cellLoc="center",
        colLoc="center",
        colWidths=[0.34, 0.22, 0.22, 0.22],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.2)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        cell.set_edgecolor("#B8B8B8")
        cell.set_linewidth(0.4)
        if row == 0:
            cell.set_facecolor("#EFEFEF")
            cell.set_text_props(weight="bold")

    input_text = (
        f"{sample['run_id']} | v={sample['velocity_ms']:.1f} m/s, "
        f"r={sample['proj_radius_mm']:.1f} mm, "
        f"x={sample['impact_x_mm']:.0f} mm, y={sample['impact_y_mm']:.0f} mm"
    )
    ax.text(0.5, 0.08, input_text, ha="center", va="center", fontsize=7.2, transform=ax.transAxes)


def create_comparison_panel(samples, run_dir, dpi):
    n = len(samples)
    fig = plt.figure(figsize=(9.2, max(2.15 * n, 2.5)), constrained_layout=True)
    gs = gridspec.GridSpec(n, 4, figure=fig, width_ratios=[1, 1, 1, 1.65])

    for i, sample in enumerate(samples):
        ax_gt = fig.add_subplot(gs[i, 0])
        ax_pred = fig.add_subplot(gs[i, 1])
        ax_err = fig.add_subplot(gs[i, 2])
        ax_num = fig.add_subplot(gs[i, 3])

        add_image_axis(ax_gt, sample["ground_truth"], "Abaqus" if i == 0 else "")
        add_image_axis(ax_pred, sample["prediction"], "Prediction" if i == 0 else "")

        err_map = np.mean(
            np.abs(sample["prediction"][..., :3] - sample["ground_truth"][..., :3]),
            axis=-1,
        )
        im = ax_err.imshow(
            err_map, cmap="magma", vmin=0, vmax=max(0.25, float(np.nanpercentile(err_map, 99)))
        )
        ax_err.set_title("RGB absolute error" if i == 0 else "", pad=5)
        ax_err.set_xticks([])
        ax_err.set_yticks([])
        ax_err.grid(False)
        for spine in ax_err.spines.values():
            spine.set_visible(False)
        cbar = fig.colorbar(im, ax=ax_err, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6)
        add_numeric_table(ax_num, sample)

    fig.suptitle("Param2Image Train_v2: Abaqus-to-model comparison", y=1.01)
    return save_figure_bundle(fig, run_dir / "comparison_panel", dpi)


def _axis_limits(gt, pred):
    values = np.concatenate([gt, pred])
    min_val = float(np.nanmin(values))
    max_val = float(np.nanmax(values))
    span = max(max_val - min_val, 1e-6)
    return min_val - 0.05 * span, max_val + 0.05 * span


def _regression_summary(gt, pred):
    err = pred - gt
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    ss_res = float(np.sum(err**2))
    ss_tot = float(np.sum((gt - np.mean(gt)) ** 2))
    r2 = np.nan if len(gt) < 2 or ss_tot < 1e-12 else 1.0 - (ss_res / ss_tot)
    mape = float(np.mean([safe_absolute_percentage_error(p, a) for p, a in zip(pred, gt)]))
    return mae, rmse, r2, mape


def create_numerical_parity_plot(results_df, run_dir, dpi):
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    color = "#2F5597"

    for ax, label, unit, short in zip(axes, OUTPUT_DISPLAY_NAMES, OUTPUT_UNITS, OUTPUT_SHORT_NAMES):
        gt = results_df[f"gt_{short}"].to_numpy(dtype=float)
        pred = results_df[f"pred_{short}"].to_numpy(dtype=float)
        mae, rmse, r2, mape = _regression_summary(gt, pred)
        lo, hi = _axis_limits(gt, pred)

        ax.scatter(gt, pred, s=26, color=color, edgecolor="white", linewidth=0.5)
        ax.plot([lo, hi], [lo, hi], color="#B00020", linewidth=1.2, linestyle="--", label="1:1")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        unit_text = f" ({unit})" if unit != "-" else ""
        ax.set_xlabel(f"Abaqus{unit_text}")
        ax.set_ylabel(f"Model{unit_text}")
        ax.set_title(label)
        r2_text = "N/A" if np.isnan(r2) else f"{r2:.3f}"
        ax.text(
            0.04,
            0.96,
            f"$R^2$={r2_text}\nMAE={mae:.3g}\nMAPE={mape:.1f}%",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="#B8B8B8", linewidth=0.4),
        )
        ax.legend(loc="lower right", frameon=False)

    fig.suptitle("Numerical response parity", y=1.04)
    return save_figure_bundle(fig, run_dir / "numerical_parity", dpi)


def main():
    args = parse_args()
    configure_publication_style()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    csv_path = args.data_dir / "results_master.csv"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = results_dir(Path(__file__)) / "comparison" / f"comparison_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {args.data_dir}")
    print(f"Model path: {args.model_path}")
    print(f"Output directory: {run_dir}")

    device = get_device()
    print(f"Device: {device}")

    df = pd.read_csv(csv_path).dropna(subset=REQUIRED_COLS)
    selected = select_samples(df, args.num_samples, args.seed, args.selection)
    print(f"Selected {len(selected)} samples using '{args.selection}' sampling")

    generator, head_mode = load_generator_checkpoint(
        args.model_path,
        device,
        num_params=NUM_PARAMS,
        num_numerical_outputs=NUM_NUMERICAL_OUTPUTS,
        latent_dim=LATENT_DIM,
        img_channels=IMG_CHANNELS,
        numerical_head_mode="auto",
    )
    print(f"Numerical head mode: {head_mode}")

    scaler_path = args.scaler_path or default_scaler_path_for_model(args.model_path)
    scaler = ImpactTabularScaler.load(scaler_path) if Path(scaler_path).exists() else None
    if scaler is not None:
        print(f"Loaded tabular scaler: {scaler_path}")
    else:
        print("No tabular scaler found; using raw input/output scale")

    samples = []
    results = []
    for i, (_, row) in enumerate(selected.iterrows(), start=1):
        gt_img_path = args.data_dir / row["img_after"].replace("\\", "/")
        if not gt_img_path.exists():
            print(f"Skipping missing image: {gt_img_path}")
            continue

        gt_tensor = load_ground_truth_image(gt_img_path, IMG_SIZE)
        pred_tensor, pred_values, pred_std = predict_sample(
            generator, row, device, scaler, max(args.num_latent_samples, 1)
        )
        gt_img = tensor_to_numpy(gt_tensor)
        pred_img = tensor_to_numpy(pred_tensor)
        img_metrics = compute_image_metrics(pred_img, gt_img)

        gt_values = np.array(
            [row["max_mises_mpa"], row["max_peeq"], row["max_deflection_mm"]], dtype=float
        )
        sample = {
            "run_id": row["run_id"],
            "ground_truth": gt_img,
            "prediction": pred_img,
            "velocity_ms": row["velocity_ms"],
            "proj_radius_mm": row["proj_radius_mm"],
            "impact_x_mm": row["impact_x_mm"],
            "impact_y_mm": row["impact_y_mm"],
            "gt_mises": gt_values[0],
            "gt_peeq": gt_values[1],
            "gt_deflection": gt_values[2],
            "pred_mises": pred_values[0],
            "pred_peeq": pred_values[1],
            "pred_deflection": pred_values[2],
            "pred_mises_std": pred_std[0],
            "pred_peeq_std": pred_std[1],
            "pred_deflection_std": pred_std[2],
            **img_metrics,
        }
        for short in OUTPUT_SHORT_NAMES:
            sample[f"{short}_ape_pct"] = safe_absolute_percentage_error(
                sample[f"pred_{short}"], sample[f"gt_{short}"]
            )

        samples.append(sample)
        results.append({k: v for k, v in sample.items() if k not in {"ground_truth", "prediction"}})

        save_image(
            torch.stack([gt_tensor, pred_tensor]),
            run_dir / f"sample_{i:02d}_{row['run_id']}.png",
            nrow=2,
            normalize=True,
        )
        print(
            f"{i:02d}. {row['run_id']}: "
            f"Mises APE={sample['mises_ape_pct']:.1f}%, "
            f"PEEQ APE={sample['peeq_ape_pct']:.1f}%, "
            f"Deflection APE={sample['deflection_ape_pct']:.1f}%"
        )

    if not samples:
        raise RuntimeError("No valid samples were available for comparison")

    results_df = pd.DataFrame(results)
    results_df["numerical_head_mode"] = head_mode
    results_df["scaler_path"] = str(scaler_path) if scaler is not None else ""
    results_df.to_csv(run_dir / "comparison_results.csv", index=False)

    panel_files = create_comparison_panel(samples, run_dir, args.dpi)
    parity_files = create_numerical_parity_plot(results_df, run_dir, args.dpi)

    print("\nSaved comparison artifacts:")
    for path in panel_files + parity_files:
        print(f"  {path}")
    print(f"  {run_dir / 'comparison_results.csv'}")

    print("\nSummary:")
    for short, label in zip(OUTPUT_SHORT_NAMES, OUTPUT_DISPLAY_NAMES):
        print(f"  {label} mean APE: {results_df[f'{short}_ape_pct'].mean():.2f}%")


if __name__ == "__main__":
    main()
