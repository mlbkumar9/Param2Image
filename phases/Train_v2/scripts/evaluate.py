"""
Comprehensive Model Evaluation Script for Param2Image cGAN (Train_v2)

This script:
1. Generates predictions on validation/test samples
2. Computes image quality metrics (SSIM, PSNR, L1, L2)
3. Computes numerical prediction metrics (MAE, RMSE, R²)
4. Creates comparison visualizations
5. Saves all results to a report

Adapted for Train_v2's 4-parameter model (velocity, radius, impact_x, impact_y)
with 512x512 RGBA images.
"""

import os
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from scipy.stats import pearsonr

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from p2i.model import load_generator_checkpoint
from p2i.paths import default_data_dir, default_generator_path, repo_root, results_dir
from p2i.preprocessing import (
    ImpactTabularScaler,
    default_scaler_path_for_model,
    safe_absolute_percentage_error,
)

REPO_ROOT = repo_root(Path(__file__))

NUM_PARAMS = 4
NUM_NUMERICAL_OUTPUTS = 3
LATENT_DIM = 100
IMG_CHANNELS = 4  # RGBA
IMG_SIZE = 512


def get_device():
    """Detects and returns the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(model_path, device):
    """Load the trained generator model."""
    generator, head_mode = load_generator_checkpoint(
        model_path,
        device,
        num_params=NUM_PARAMS,
        num_numerical_outputs=NUM_NUMERICAL_OUTPUTS,
        latent_dim=LATENT_DIM,
        img_channels=IMG_CHANNELS,
        numerical_head_mode="auto",
    )
    return generator, head_mode


def tensor_to_numpy(tensor, channels=4):
    """Convert tensor to numpy image array."""
    img = tensor.cpu().numpy()
    img = (img + 1) / 2.0
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    return img


def load_and_preprocess_image(img_path, img_size=512, channels=4):
    """Load and preprocess a ground truth image."""
    img = Image.open(img_path)
    if channels == 4:
        img = img.convert('RGBA')
    else:
        img = img.convert('RGB')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return img_array


def compute_image_metrics(generated, ground_truth):
    """Compute various image quality metrics."""
    gen = generated.astype(np.float32)
    gt = ground_truth.astype(np.float32)

    gen_rgb = gen[..., :3]
    gt_rgb = gt[..., :3]

    metrics = {}
    metrics['l1'] = np.mean(np.abs(gen - gt))
    metrics['l1_rgb'] = np.mean(np.abs(gen_rgb - gt_rgb))
    metrics['l2'] = np.mean((gen - gt) ** 2)
    metrics['mse'] = metrics['l2']

    try:
        metrics['psnr'] = psnr(gt_rgb, gen_rgb, data_range=1.0)
    except Exception:
        metrics['psnr'] = 0.0

    try:
        ssim_vals = []
        for c in range(min(gen.shape[-1], gt.shape[-1])):
            s = ssim(gt[..., c], gen[..., c], data_range=1.0)
            ssim_vals.append(s)
        metrics['ssim'] = np.mean(ssim_vals)
    except Exception:
        metrics['ssim'] = 0.0

    return metrics


def compute_numerical_metrics(predictions, ground_truth, output_names):
    """Compute regression metrics for numerical outputs."""
    metrics = {}

    for i, name in enumerate(output_names):
        pred = predictions[:, i]
        gt = ground_truth[:, i]

        mae = np.mean(np.abs(pred - gt))
        metrics[f'{name}_mae'] = mae

        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        metrics[f'{name}_rmse'] = rmse

        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = np.nan if len(gt) < 2 or ss_tot < 1e-8 else 1 - (ss_res / ss_tot)
        metrics[f'{name}_r2'] = r2

        if len(pred) > 2:
            try:
                corr, _ = pearsonr(pred, gt)
                metrics[f'{name}_corr'] = corr
            except Exception:
                metrics[f'{name}_corr'] = 0.0
        else:
            metrics[f'{name}_corr'] = 0.0

        valid_mask = np.abs(gt) > 1e-6
        if valid_mask.sum() > 0:
            mape = np.mean(np.abs((gt[valid_mask] - pred[valid_mask]) / gt[valid_mask])) * 100
        else:
            mape = 0.0
        metrics[f'{name}_mape'] = mape

    return metrics


def create_comparison_figure(samples, save_path, num_show=6, dpi=300):
    """Create a visual comparison of generated vs ground truth images (high quality)."""
    n = min(len(samples), num_show)

    fig = plt.figure(figsize=(14, max(3.2 * n, 3.4)), dpi=dpi)
    gs = gridspec.GridSpec(n, 4, width_ratios=[1, 1, 1, 0.8])

    for i in range(n):
        sample = samples[i]

        ax1 = fig.add_subplot(gs[i, 0])
        ax1.imshow(sample['ground_truth'], interpolation='lanczos')
        ax1.set_title(f"Ground Truth\n{sample['run_id']}", fontsize=10, fontweight='bold')
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[i, 1])
        ax2.imshow(sample['generated'], interpolation='lanczos')
        ax2.set_title(
            f"Generated\nSSIM: {sample['ssim']:.3f} | PSNR: {sample['psnr']:.1f}dB",
            fontsize=10,
            fontweight='bold',
        )
        ax2.axis('off')

        ax3 = fig.add_subplot(gs[i, 2])
        diff = np.abs(sample['generated'][..., :3] - sample['ground_truth'][..., :3])
        diff_gray = np.mean(diff, axis=-1)
        im = ax3.imshow(diff_gray, cmap='hot', vmin=0, vmax=0.3, interpolation='lanczos')
        ax3.set_title(f"Error Map\nL1: {sample['l1']:.4f}", fontsize=10, fontweight='bold')
        ax3.axis('off')
        cbar = plt.colorbar(im, ax=ax3, fraction=0.046)
        cbar.ax.tick_params(labelsize=8)

        ax4 = fig.add_subplot(gs[i, 3])
        ax4.axis('off')
        info_text = (
            f"Input Parameters:\n"
            f"  Velocity: {sample['velocity']:.1f} m/s\n"
            f"  Radius: {sample['radius']:.1f} mm\n"
            f"  Impact: ({sample['x']:.0f}, {sample['y']:.0f})\n\n"
            f"Predicted Outputs:\n"
            f"  Mises: {sample['pred_mises']:.1f} MPa\n"
            f"  PEEQ: {sample['pred_peeq']:.4f}\n"
            f"  Deflection: {sample['pred_defl']:.1f} mm\n\n"
            f"Ground Truth:\n"
            f"  Mises: {sample['gt_mises']:.1f} MPa\n"
            f"  PEEQ: {sample['gt_peeq']:.4f}\n"
            f"  Deflection: {sample['gt_defl']:.1f} mm\n\n"
            f"Errors:\n"
            f"  Mises: {safe_absolute_percentage_error(sample['pred_mises'], sample['gt_mises']):.1f}%\n"
            f"  PEEQ: {safe_absolute_percentage_error(sample['pred_peeq'], sample['gt_peeq']):.1f}%\n"
            f"  Defl: {safe_absolute_percentage_error(sample['pred_defl'], sample['gt_defl']):.1f}%"
        )
        ax4.text(
            0.05,
            0.5,
            info_text,
            fontsize=7.5,
            verticalalignment='center',
            family='monospace',
            transform=ax4.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    svg_path = str(save_path).replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    pdf_path = str(save_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved comparison figure: {save_path} (+ .svg, .pdf)")


def create_metrics_plots(num_metrics_dict, num_plot_data, output_path, dpi=300):
    """Create plots for numerical prediction accuracy (high quality)."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), dpi=dpi)

    output_names = ['Max Mises (MPa)', 'Max PEEQ', 'Max Deflection (mm)']
    short_names = ['mises', 'peeq', 'deflection']

    for i, (name, short) in enumerate(zip(output_names, short_names)):
        ax = axes[0, i]
        pred = num_plot_data[f'pred_{i}']
        gt = num_plot_data[f'gt_{i}']

        ax.scatter(gt, pred, alpha=0.65, s=28, c='blue', edgecolors='white', linewidth=0.4)
        min_val = min(gt.min(), pred.min())
        max_val = max(gt.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, label='Perfect')
        ax.set_xlabel(f'Ground Truth {name}', fontsize=9)
        ax.set_ylabel(f'Predicted {name}', fontsize=9)
        r2 = num_metrics_dict.get(f'{short}_r2', 0)
        r2_label = 'N/A' if np.isnan(r2) else f'{r2:.4f}'
        ax.set_title(f'{name}\nR² = {r2_label}', fontsize=10, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

    for i, (name, short) in enumerate(zip(output_names, short_names)):
        ax = axes[1, i]
        pred = num_plot_data[f'pred_{i}']
        gt = num_plot_data[f'gt_{i}']
        errors = pred - gt
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(x=0, color='r', linestyle='--', linewidth=1.2)
        ax.set_xlabel('Prediction Error', fontsize=9)
        ax.set_ylabel('Frequency', fontsize=9)
        mae = np.mean(np.abs(errors))
        ax.set_title(f'{name} Error Distribution\nMAE = {mae:.2f}', fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    svg_path = str(output_path).replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved metrics plots: {output_path} (+ .svg)")


def create_summary_plot(avg_img_metrics, num_metrics_dict, output_path, epoch_info="", dpi=300):
    """Create a summary dashboard plot."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=dpi)

    ax1 = axes[0]
    img_metric_names = ['SSIM', 'PSNR (dB/50)', 'L1 Error', 'L2 Error']
    img_metric_values = [
        avg_img_metrics.get('ssim', 0),
        avg_img_metrics.get('psnr', 0) / 50,
        avg_img_metrics.get('l1', 0),
        avg_img_metrics.get('l2', 0),
    ]
    colors = ['green' if i < 2 else 'red' for i in range(4)]
    bars = ax1.bar(img_metric_names, img_metric_values, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Metric Value', fontsize=9)
    ax1.set_title(f'Image Quality Metrics{epoch_info}', fontsize=10, fontweight='bold')
    ax1.tick_params(axis='both', labelsize=8)
    ax1.set_ylim(0, 1.2)

    for i, (bar, val) in enumerate(
        zip(
            bars,
            [
                avg_img_metrics.get('ssim', 0),
                avg_img_metrics.get('psnr', 0),
                avg_img_metrics.get('l1', 0),
                avg_img_metrics.get('l2', 0),
            ],
        )
    ):
        label = f'{val:.1f} dB' if 'PSNR' in img_metric_names[i] else f'{val:.4f}'
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            label,
            ha='center',
            fontsize=8,
            fontweight='bold',
        )

    ax2 = axes[1]
    num_metric_names = ['Mises R²', 'PEEQ R²', 'Deflection R²']
    raw_num_metric_values = [
        num_metrics_dict.get('mises_r2', 0),
        num_metrics_dict.get('peeq_r2', 0),
        num_metrics_dict.get('deflection_r2', 0),
    ]
    num_metric_values = [0.0 if np.isnan(v) else v for v in raw_num_metric_values]
    colors = [
        'gray' if np.isnan(v) else 'green' if v > 0.5 else 'orange' if v > 0 else 'red'
        for v in raw_num_metric_values
    ]
    bars = ax2.bar(num_metric_names, num_metric_values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_ylabel('R² Score', fontsize=9)
    ax2.set_title(f'Numerical Prediction Quality{epoch_info}', fontsize=10, fontweight='bold')
    ax2.tick_params(axis='both', labelsize=8)
    ax2.set_ylim(-0.5, 1.1)
    ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax2.axhline(y=0.5, color='green', linestyle='--', linewidth=1, alpha=0.5)

    for bar, raw_val in zip(bars, raw_num_metric_values):
        label = 'N/A' if np.isnan(raw_val) else f'{raw_val:.4f}'
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            max(bar.get_height(), 0) + 0.03,
            label,
            ha='center',
            fontsize=8,
            fontweight='bold',
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    svg_path = str(output_path).replace('.png', '.svg')
    plt.savefig(svg_path, format='svg', bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved summary plot: {output_path} (+ .svg)")


def evaluate_model(args):
    """Main evaluation function."""
    print("=" * 70)
    print("    Param2Image cGAN Model Evaluation (Train_v2)")
    print("=" * 70)

    device = get_device()
    print(f"Device: {device}")

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = default_data_dir(Path(__file__))

    if args.model_path:
        model_path = Path(args.model_path)
    else:
        model_path = default_generator_path(Path(__file__))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = results_dir(Path(__file__)) / 'evaluation' / f'eval_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nData directory: {data_dir}")
    print(f"Model path: {model_path}")
    print(f"Output directory: {output_dir}")

    print("\nLoading model...")
    generator, head_mode = load_model(model_path, device)
    print("  Model loaded successfully")
    print(f"  Numerical head mode: {head_mode}")
    print(f"  Image size: {IMG_SIZE}x{IMG_SIZE}")
    print(f"  Channels: {IMG_CHANNELS} (RGBA)")
    print(f"  Parameters: {NUM_PARAMS}")

    scaler_path = Path(args.scaler_path) if args.scaler_path else default_scaler_path_for_model(model_path)
    scaler = ImpactTabularScaler.load(scaler_path) if scaler_path.exists() else None
    if scaler is not None:
        print(f"  Tabular scaler: {scaler_path}")
    else:
        print("  Tabular scaler: not found; using raw input/output scale")

    csv_path = data_dir / 'results_master.csv'
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    required_cols = [
        'img_after',
        'velocity_ms',
        'proj_radius_mm',
        'impact_x_mm',
        'impact_y_mm',
        'max_mises_mpa',
        'max_peeq',
        'max_deflection_mm',
    ]
    df = df.dropna(subset=required_cols)
    print(f"Total valid simulations: {len(df)}")

    if args.num_samples:
        np.random.seed(42)
        eval_indices = np.random.choice(len(df), min(args.num_samples, len(df)), replace=False)
        eval_df = df.iloc[eval_indices]
    else:
        eval_df = df.tail(50)

    print(f"Evaluating on {len(eval_df)} samples")

    all_samples = []
    image_metrics_list = []
    all_pred_outputs = []
    all_gt_outputs = []

    with torch.no_grad():
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc="Evaluating"):
            img_after_path = data_dir / row['img_after'].replace('\\', '/')
            if not img_after_path.exists():
                print(f"  Warning: Image not found: {img_after_path}")
                continue

            gt_img = load_and_preprocess_image(img_after_path, IMG_SIZE, IMG_CHANNELS)

            raw_params = [[
                row['velocity_ms'],
                row['proj_radius_mm'],
                row['impact_x_mm'],
                row['impact_y_mm'],
            ]]
            model_params = scaler.transform_params_array(raw_params) if scaler is not None else raw_params
            input_params = torch.tensor(model_params, dtype=torch.float32).to(device)

            z = torch.randn(1, LATENT_DIM, device=device)
            gen_img_tensor, pred_numerical = generator(z, input_params)
            gen_img = tensor_to_numpy(gen_img_tensor[0], IMG_CHANNELS)

            pred_values = pred_numerical.cpu().numpy()
            if scaler is not None:
                pred_values = scaler.inverse_transform_outputs_array(pred_values)
            pred_values = pred_values.flatten()

            gt_values = np.array([
                row['max_mises_mpa'],
                row['max_peeq'],
                row['max_deflection_mm'],
            ])

            img_metrics = compute_image_metrics(gen_img, gt_img)
            image_metrics_list.append(img_metrics)
            all_pred_outputs.append(pred_values)
            all_gt_outputs.append(gt_values)

            sample_info = {
                'run_id': row['run_id'],
                'generated': gen_img,
                'ground_truth': gt_img,
                'velocity': row['velocity_ms'],
                'radius': row['proj_radius_mm'],
                'x': row['impact_x_mm'],
                'y': row['impact_y_mm'],
                'pred_mises': pred_values[0],
                'pred_peeq': pred_values[1],
                'pred_defl': pred_values[2],
                'gt_mises': gt_values[0],
                'gt_peeq': gt_values[1],
                'gt_defl': gt_values[2],
                **img_metrics,
            }
            all_samples.append(sample_info)

    if len(all_samples) == 0:
        print("ERROR: No samples were evaluated!")
        return None

    print("\n" + "=" * 70)
    print("    Image Quality Metrics (averaged)")
    print("=" * 70)

    avg_img_metrics = {}
    for key in image_metrics_list[0].keys():
        values = [m[key] for m in image_metrics_list]
        avg_img_metrics[key] = np.mean(values)
        avg_img_metrics[f'{key}_std'] = np.std(values)
        print(f"  {key:12s}: {avg_img_metrics[key]:.4f} ± {avg_img_metrics[f'{key}_std']:.4f}")

    print("\n" + "=" * 70)
    print("    Numerical Prediction Metrics")
    print("=" * 70)

    all_pred_outputs = np.array(all_pred_outputs)
    all_gt_outputs = np.array(all_gt_outputs)

    output_names = ['mises', 'peeq', 'deflection']
    num_metrics_dict = compute_numerical_metrics(all_pred_outputs, all_gt_outputs, output_names)

    for name in output_names:
        r2_value = num_metrics_dict[f'{name}_r2']
        r2_label = 'N/A' if np.isnan(r2_value) else f'{r2_value:.4f}'
        print(f"\n  {name.upper()}:")
        print(f"    MAE:  {num_metrics_dict[f'{name}_mae']:.4f}")
        print(f"    RMSE: {num_metrics_dict[f'{name}_rmse']:.4f}")
        print(f"    R²:   {r2_label}")
        print(f"    Corr: {num_metrics_dict[f'{name}_corr']:.4f}")
        print(f"    MAPE: {num_metrics_dict[f'{name}_mape']:.1f}%")

    metrics_df = pd.DataFrame([
        {k: v for k, v in s.items() if k not in ['generated', 'ground_truth']}
        for s in all_samples
    ])
    metrics_df.to_csv(output_dir / 'detailed_metrics.csv', index=False)
    print(f"\nSaved detailed metrics: {output_dir / 'detailed_metrics.csv'}")

    summary = {
        **avg_img_metrics,
        **num_metrics_dict,
        'num_samples': len(all_samples),
        'model_path': str(model_path),
        'numerical_head_mode': head_mode,
        'scaler_path': str(scaler_path) if scaler is not None else '',
        'image_size': IMG_SIZE,
        'channels': IMG_CHANNELS,
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(output_dir / 'summary_metrics.csv', index=False)
    print(f"Saved summary metrics: {output_dir / 'summary_metrics.csv'}")

    print("\nGenerating high-quality visualizations...")
    epoch_info = "\n(Model after training)" if not args.epoch else f"\n(Epoch {args.epoch})"
    all_samples.sort(key=lambda x: x['ssim'], reverse=True)

    create_comparison_figure(all_samples[:6], output_dir / 'best_samples.png', num_show=6, dpi=args.dpi)
    create_comparison_figure(all_samples[-6:], output_dir / 'worst_samples.png', num_show=6, dpi=args.dpi)

    import random
    random.seed(42)
    random_samples = random.sample(all_samples, min(6, len(all_samples)))
    create_comparison_figure(random_samples, output_dir / 'random_samples.png', num_show=6, dpi=args.dpi)

    num_plot_data = {
        'pred_0': all_pred_outputs[:, 0],
        'pred_1': all_pred_outputs[:, 1],
        'pred_2': all_pred_outputs[:, 2],
        'gt_0': all_gt_outputs[:, 0],
        'gt_1': all_gt_outputs[:, 1],
        'gt_2': all_gt_outputs[:, 2],
    }
    create_metrics_plots(num_metrics_dict, num_plot_data, output_dir / 'numerical_predictions.png', dpi=args.dpi)
    create_summary_plot(
        avg_img_metrics,
        num_metrics_dict,
        output_dir / 'summary_dashboard.png',
        epoch_info=epoch_info,
        dpi=args.dpi,
    )

    pred_img_dir = output_dir / 'predictions'
    pred_img_dir.mkdir(exist_ok=True)
    print(f"\nSaving prediction images to: {pred_img_dir}")
    for sample in tqdm(all_samples[:20], desc="Saving images"):
        gen_img_uint8 = (sample['generated'] * 255).astype(np.uint8)
        img = Image.fromarray(gen_img_uint8, mode='RGBA')
        img.save(pred_img_dir / f"pred_{sample['run_id']}.png")

        gt_img_uint8 = (sample['ground_truth'] * 255).astype(np.uint8)
        gt_pil = Image.fromarray(gt_img_uint8, mode='RGBA')
        combined = Image.new('RGBA', (IMG_SIZE * 2 + 20, IMG_SIZE), (255, 255, 255, 255))
        combined.paste(gt_pil, (0, 0))
        combined.paste(img, (IMG_SIZE + 20, 0))
        combined.save(pred_img_dir / f"compare_{sample['run_id']}.png")

    print("\n" + "=" * 70)
    print("    EVALUATION COMPLETE")
    print("=" * 70)
    print("\nKey Results:")
    print(f"  Average SSIM:     {avg_img_metrics['ssim']:.4f}")
    print(f"  Average PSNR:     {avg_img_metrics['psnr']:.2f} dB")
    print(f"  Average L1:       {avg_img_metrics['l1']:.4f}")
    mises_r2 = num_metrics_dict['mises_r2']
    peeq_r2 = num_metrics_dict['peeq_r2']
    deflection_r2 = num_metrics_dict['deflection_r2']
    mises_r2_label = 'N/A' if np.isnan(mises_r2) else f'{mises_r2:.4f}'
    peeq_r2_label = 'N/A' if np.isnan(peeq_r2) else f'{peeq_r2:.4f}'
    deflection_r2_label = 'N/A' if np.isnan(deflection_r2) else f'{deflection_r2:.4f}'
    print(f"\n  Mises R²:         {mises_r2_label}")
    print(f"  PEEQ R²:          {peeq_r2_label}")
    print(f"  Deflection R²:    {deflection_r2_label}")
    print(f"\nResults saved to: {output_dir}")

    print("\n" + "-" * 70)
    print("Interpretation Guide:")
    print("-" * 70)
    print("  SSIM: 0.0-0.3 (poor), 0.3-0.6 (moderate), 0.6-0.8 (good), 0.8-1.0 (excellent)")
    print("  PSNR: <20dB (poor), 20-30dB (moderate), 30-40dB (good), >40dB (excellent)")
    print("  R²:   <0 (worse than mean), 0-0.5 (weak), 0.5-0.8 (moderate), 0.8-1.0 (strong)")
    print("-" * 70)

    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate Param2Image cGAN model (Train_v2)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to impact_dataset directory (default: auto-detect)')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to generator model (default: training_checkpoints/cGAN_generator.pth)')
    parser.add_argument('--scaler_path', type=str, default=None,
                        help='Path to normalization_stats.json (default: model checkpoint folder)')
    parser.add_argument('--num_samples', type=int, default=50,
                        help='Number of samples to evaluate (default: 50)')
    parser.add_argument('--epoch', type=int, default=None,
                        help='Epoch number for labeling (optional)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Figure export DPI for raster outputs (default: 300)')

    args = parser.parse_args()
    evaluate_model(args)
