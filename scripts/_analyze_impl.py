"""
Comprehensive Analysis Script for Param2Image cGAN Report

Generates detailed research-quality visualizations and metrics for the LaTeX report:
1. Per-parameter error analysis
2. Correlation matrices
3. Error distribution plots
4. Sample-by-sample comparison grids
5. Metric progression visualization
6. Statistical summaries
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from PIL import Image
import torch
from tqdm import tqdm
from datetime import datetime
from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

# Setup paths
from p2i.model import load_generator_checkpoint
from p2i.paths import (
    default_data_dir,
    default_generator_path,
    repo_root,
    results_dir,
)
from p2i.preprocessing import ImpactTabularScaler, default_scaler_path_for_model

REPO_ROOT = repo_root(Path(__file__))
DATA_DIR = default_data_dir(Path(__file__))
REPORT_FIGURES_DIR = results_dir(Path(__file__)) / 'analysis' / 'report_figures'
OUTPUT_DIR = results_dir(Path(__file__)) / 'analysis' / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

NUM_PARAMS = 4
NUM_NUMERICAL_OUTPUTS = 3
LATENT_DIM = 100
IMG_CHANNELS = 4
IMG_SIZE = 512

# Publication-quality settings
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'font.family': 'serif',
    'text.usetex': False,
    'axes.grid': True,
    'grid.alpha': 0.3
})


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_model(model_path, device):
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


def tensor_to_numpy(tensor):
    img = tensor.cpu().numpy()
    img = (img + 1) / 2.0
    img = np.clip(img, 0, 1)
    img = np.transpose(img, (1, 2, 0))
    return img


def load_and_preprocess_image(img_path, img_size=512):
    img = Image.open(img_path).convert('RGBA')
    img = img.resize((img_size, img_size), Image.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def compute_all_metrics(gen, gt):
    """Compute comprehensive image metrics."""
    metrics = {}
    
    # L1 and L2
    metrics['l1'] = np.mean(np.abs(gen - gt))
    metrics['l2'] = np.mean((gen - gt) ** 2)
    
    # PSNR
    try:
        metrics['psnr'] = psnr(gt[..., :3], gen[..., :3], data_range=1.0)
    except:
        metrics['psnr'] = 0.0
    
    # SSIM per channel
    ssim_vals = []
    for c in range(min(gen.shape[-1], gt.shape[-1])):
        try:
            s = ssim(gt[..., c], gen[..., c], data_range=1.0)
            ssim_vals.append(s)
        except:
            ssim_vals.append(0.0)
    metrics['ssim'] = np.mean(ssim_vals)
    metrics['ssim_rgb'] = np.mean(ssim_vals[:3])
    
    return metrics


def generate_all_samples(generator, df, device, num_samples=None, scaler=None):
    """Generate predictions for all samples and collect metrics."""
    results = []
    
    if num_samples:
        np.random.seed(42)
        indices = np.random.choice(len(df), min(num_samples, len(df)), replace=False)
        df = df.iloc[indices]
    
    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating predictions"):
            img_path = DATA_DIR / row['img_after'].replace('\\', '/')
            if not img_path.exists():
                continue
            
            gt_img = load_and_preprocess_image(img_path)
            
            raw_params = [[
                row['velocity_ms'],
                row['proj_radius_mm'],
                row['impact_x_mm'],
                row['impact_y_mm']
            ]]
            model_params = scaler.transform_params_array(raw_params) if scaler is not None else raw_params
            input_params = torch.tensor(model_params, dtype=torch.float32).to(device)
            
            z = torch.randn(1, LATENT_DIM, device=device)
            gen_img_tensor, pred_numerical = generator(z, input_params)
            gen_img = tensor_to_numpy(gen_img_tensor[0])
            pred_values = pred_numerical.cpu().numpy()
            if scaler is not None:
                pred_values = scaler.inverse_transform_outputs_array(pred_values)
            pred_values = pred_values.flatten()
            
            img_metrics = compute_all_metrics(gen_img, gt_img)
            
            result = {
                'run_id': row['run_id'],
                'velocity_ms': row['velocity_ms'],
                'proj_radius_mm': row['proj_radius_mm'],
                'impact_x_mm': row['impact_x_mm'],
                'impact_y_mm': row['impact_y_mm'],
                'gt_mises': row['max_mises_mpa'],
                'gt_peeq': row['max_peeq'],
                'gt_deflection': row['max_deflection_mm'],
                'pred_mises': pred_values[0],
                'pred_peeq': pred_values[1],
                'pred_deflection': pred_values[2],
                'generated': gen_img,
                'ground_truth': gt_img,
                **img_metrics
            }
            
            # Compute percentage errors
            for output in ['mises', 'peeq', 'deflection']:
                gt_val = result[f'gt_{output}']
                pred_val = result[f'pred_{output}']
                if abs(gt_val) > 1e-6:
                    result[f'{output}_error_pct'] = abs(pred_val - gt_val) / abs(gt_val) * 100
                else:
                    result[f'{output}_error_pct'] = 0.0
            
            results.append(result)
    
    return results


def plot_1_parameter_sensitivity(results, output_dir):
    """Plot how errors vary with each input parameter."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['generated', 'ground_truth']} 
                       for r in results])
    
    params = ['velocity_ms', 'proj_radius_mm', 'impact_x_mm', 'impact_y_mm']
    param_names = ['Velocity (m/s)', 'Projectile Radius (mm)', 'Impact X (mm)', 'Impact Y (mm)']
    outputs = ['mises', 'peeq', 'deflection']
    output_names = ['Max Mises Stress', 'Max PEEQ', 'Max Deflection']
    
    fig, axes = plt.subplots(4, 3, figsize=(18, 20))
    
    for i, (param, pname) in enumerate(zip(params, param_names)):
        for j, (output, oname) in enumerate(zip(outputs, output_names)):
            ax = axes[i, j]
            
            x = df[param]
            y = df[f'{output}_error_pct']
            
            ax.scatter(x, y, alpha=0.6, s=40, c='steelblue', edgecolors='white', linewidth=0.5)
            
            # Trend line
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), 'r--', linewidth=2, label='Trend')
            
            ax.set_xlabel(pname)
            ax.set_ylabel(f'{oname} Error (%)')
            ax.set_title(f'{oname} vs {pname}')
            
            # Add correlation coefficient
            corr, _ = pearsonr(x, y)
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'parameter_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: parameter_sensitivity.png/.pdf")


def plot_2_correlation_matrix(results, output_dir):
    """Plot correlation matrix between all variables."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['generated', 'ground_truth']} 
                       for r in results])
    
    # Select relevant columns
    cols = ['velocity_ms', 'proj_radius_mm', 'impact_x_mm', 'impact_y_mm',
            'gt_mises', 'gt_peeq', 'gt_deflection',
            'pred_mises', 'pred_peeq', 'pred_deflection',
            'ssim', 'psnr', 'l1']
    
    labels = ['Velocity', 'Radius', 'X', 'Y',
              'GT Mises', 'GT PEEQ', 'GT Defl',
              'Pred Mises', 'Pred PEEQ', 'Pred Defl',
              'SSIM', 'PSNR', 'L1']
    
    corr_matrix = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    
    # Use seaborn if available, otherwise use imshow
    if HAS_SEABORN:
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, square=True,
                    xticklabels=labels, yticklabels=labels, ax=ax,
                    annot_kws={'size': 9})
    else:
        # Fallback without seaborn
        corr_masked = np.ma.array(corr_matrix.values, mask=mask)
        im = ax.imshow(corr_masked, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax)
        # Add text annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                if not mask[i, j]:
                    ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                           ha='center', va='center', fontsize=8)
    
    plt.title('Correlation Matrix: Parameters, Outputs, and Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'correlation_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: correlation_matrix.png/.pdf")


def plot_3_error_distributions(results, output_dir):
    """Plot detailed error distributions for each output."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['generated', 'ground_truth']} 
                       for r in results])
    
    outputs = ['mises', 'peeq', 'deflection']
    output_names = ['Max Mises Stress (MPa)', 'Max PEEQ', 'Max Deflection (mm)']
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    
    for i, (output, oname, color) in enumerate(zip(outputs, output_names, colors)):
        gt = df[f'gt_{output}']
        pred = df[f'pred_{output}']
        error = pred - gt
        error_pct = df[f'{output}_error_pct']
        
        # Absolute error histogram
        ax1 = axes[i, 0]
        ax1.hist(error, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='black', linestyle='--', linewidth=2)
        ax1.axvline(x=error.mean(), color='red', linestyle='-', linewidth=2, label=f'Mean: {error.mean():.2f}')
        ax1.set_xlabel('Absolute Error')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{oname}\nAbsolute Error Distribution')
        ax1.legend()
        
        # Percentage error histogram
        ax2 = axes[i, 1]
        ax2.hist(error_pct, bins=30, color=color, alpha=0.7, edgecolor='black')
        ax2.axvline(x=error_pct.mean(), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {error_pct.mean():.1f}%')
        ax2.axvline(x=error_pct.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {error_pct.median():.1f}%')
        ax2.set_xlabel('Percentage Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{oname}\nPercentage Error Distribution')
        ax2.legend()
        
        # Q-Q plot
        ax3 = axes[i, 2]
        sorted_error = np.sort(error)
        theoretical_quantiles = np.linspace(0.01, 0.99, len(sorted_error))
        theoretical_values = np.quantile(np.random.normal(error.mean(), error.std(), 10000), 
                                         theoretical_quantiles)
        ax3.scatter(theoretical_values, sorted_error, alpha=0.5, s=30, c=color)
        
        # Reference line
        min_val = min(theoretical_values.min(), sorted_error.min())
        max_val = max(theoretical_values.max(), sorted_error.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
        ax3.set_xlabel('Theoretical Quantiles')
        ax3.set_ylabel('Sample Quantiles')
        ax3.set_title(f'{oname}\nQ-Q Plot')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'error_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: error_distributions.png/.pdf")


def plot_4_predicted_vs_actual(results, output_dir):
    """Create scatter plots of predicted vs actual for each output."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['generated', 'ground_truth']} 
                       for r in results])
    
    outputs = ['mises', 'peeq', 'deflection']
    output_names = ['Max Mises Stress (MPa)', 'Max PEEQ', 'Max Deflection (mm)']
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (output, oname, color) in enumerate(zip(outputs, output_names, colors)):
        ax = axes[i]
        
        gt = df[f'gt_{output}']
        pred = df[f'pred_{output}']
        
        ax.scatter(gt, pred, alpha=0.6, s=50, c=color, edgecolors='white', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(gt.min(), pred.min())
        max_val = max(gt.max(), pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2, label='Perfect')
        
        # Regression line
        z = np.polyfit(gt, pred, 1)
        p = np.poly1d(z)
        ax.plot([min_val, max_val], [p(min_val), p(max_val)], 'r-', linewidth=2, label='Fit')
        
        # Compute R²
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        
        # Compute Pearson correlation
        corr, _ = pearsonr(gt, pred)
        
        ax.set_xlabel(f'Ground Truth {oname}')
        ax.set_ylabel(f'Predicted {oname}')
        ax.set_title(f'{oname}\nR² = {r2:.4f}, Pearson r = {corr:.4f}')
        ax.legend(loc='upper left')
        
        # Add statistics box
        mae = np.mean(np.abs(pred - gt))
        rmse = np.sqrt(np.mean((pred - gt) ** 2))
        stats_text = f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}'
        ax.annotate(stats_text, xy=(0.95, 0.05), xycoords='axes fraction',
                   fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'predicted_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'predicted_vs_actual.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: predicted_vs_actual.png/.pdf")


def plot_5_image_quality_analysis(results, output_dir):
    """Analyze image quality metrics distribution."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['generated', 'ground_truth']} 
                       for r in results])
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # SSIM distribution
    ax1 = axes[0, 0]
    ax1.hist(df['ssim'], bins=25, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(x=df['ssim'].mean(), color='red', linewidth=2, label=f'Mean: {df["ssim"].mean():.3f}')
    ax1.axvline(x=0.85, color='green', linestyle='--', linewidth=2, label='Target: 0.85')
    ax1.set_xlabel('SSIM')
    ax1.set_ylabel('Frequency')
    ax1.set_title('SSIM Distribution')
    ax1.legend()
    
    # PSNR distribution
    ax2 = axes[0, 1]
    ax2.hist(df['psnr'], bins=25, color='coral', alpha=0.7, edgecolor='black')
    ax2.axvline(x=df['psnr'].mean(), color='red', linewidth=2, label=f'Mean: {df["psnr"].mean():.1f} dB')
    ax2.axvline(x=28, color='green', linestyle='--', linewidth=2, label='Target: 28 dB')
    ax2.set_xlabel('PSNR (dB)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('PSNR Distribution')
    ax2.legend()
    
    # L1 Error distribution
    ax3 = axes[1, 0]
    ax3.hist(df['l1'], bins=25, color='mediumseagreen', alpha=0.7, edgecolor='black')
    ax3.axvline(x=df['l1'].mean(), color='red', linewidth=2, label=f'Mean: {df["l1"].mean():.4f}')
    ax3.axvline(x=0.03, color='green', linestyle='--', linewidth=2, label='Target: 0.03')
    ax3.set_xlabel('L1 Error')
    ax3.set_ylabel('Frequency')
    ax3.set_title('L1 Error Distribution')
    ax3.legend()
    
    # SSIM vs PSNR scatter
    ax4 = axes[1, 1]
    scatter = ax4.scatter(df['psnr'], df['ssim'], c=df['l1'], cmap='RdYlGn_r', 
                         s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('L1 Error')
    ax4.set_xlabel('PSNR (dB)')
    ax4.set_ylabel('SSIM')
    ax4.set_title('SSIM vs PSNR (colored by L1 Error)')
    
    # Add quadrant lines
    ax4.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'image_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'image_quality_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: image_quality_analysis.png/.pdf")


def plot_6_best_worst_cases(results, output_dir, n=4):
    """Show detailed best and worst case samples."""
    # Sort by SSIM
    sorted_results = sorted(results, key=lambda x: x['ssim'], reverse=True)
    
    best = sorted_results[:n]
    worst = sorted_results[-n:]
    
    fig = plt.figure(figsize=(24, 16))
    
    for i, sample in enumerate(best):
        # Ground truth
        ax1 = fig.add_subplot(4, 8, i*2 + 1)
        ax1.imshow(sample['ground_truth'])
        ax1.set_title(f"GT: {sample['run_id']}", fontsize=10)
        ax1.axis('off')
        
        # Generated
        ax2 = fig.add_subplot(4, 8, i*2 + 2)
        ax2.imshow(sample['generated'])
        ax2.set_title(f"SSIM: {sample['ssim']:.3f}", fontsize=10)
        ax2.axis('off')
    
    for i, sample in enumerate(worst):
        # Ground truth
        ax1 = fig.add_subplot(4, 8, 16 + i*2 + 1)
        ax1.imshow(sample['ground_truth'])
        ax1.set_title(f"GT: {sample['run_id']}", fontsize=10)
        ax1.axis('off')
        
        # Generated
        ax2 = fig.add_subplot(4, 8, 16 + i*2 + 2)
        ax2.imshow(sample['generated'])
        ax2.set_title(f"SSIM: {sample['ssim']:.3f}", fontsize=10)
        ax2.axis('off')
    
    fig.text(0.5, 0.52, 'Best Samples (Highest SSIM)', ha='center', fontsize=16, fontweight='bold')
    fig.text(0.5, 0.02, 'Worst Samples (Lowest SSIM)', ha='center', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_dir / 'best_worst_cases.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: best_worst_cases.png")


def plot_7_sample_comparison_grid(results, output_dir, n=6):
    """Create a publication-ready comparison grid."""
    np.random.seed(42)
    samples = np.random.choice(results, min(n, len(results)), replace=False)
    
    fig = plt.figure(figsize=(20, 4*n))
    
    for i, sample in enumerate(samples):
        # Ground Truth
        ax1 = fig.add_subplot(n, 4, i*4 + 1)
        ax1.imshow(sample['ground_truth'])
        ax1.set_title('Ground Truth', fontsize=12, fontweight='bold')
        ax1.set_ylabel(f"{sample['run_id']}", fontsize=11, rotation=0, labelpad=50, va='center')
        ax1.set_xticks([])
        ax1.set_yticks([])
        
        # Generated
        ax2 = fig.add_subplot(n, 4, i*4 + 2)
        ax2.imshow(sample['generated'])
        ax2.set_title('Generated', fontsize=12, fontweight='bold')
        ax2.axis('off')
        
        # Error map
        ax3 = fig.add_subplot(n, 4, i*4 + 3)
        diff = np.mean(np.abs(sample['generated'][..., :3] - sample['ground_truth'][..., :3]), axis=-1)
        im = ax3.imshow(diff, cmap='hot', vmin=0, vmax=0.3)
        ax3.set_title('Error Map', fontsize=12, fontweight='bold')
        ax3.axis('off')
        cbar = plt.colorbar(im, ax=ax3, fraction=0.046)
        cbar.ax.tick_params(labelsize=8)
        
        # Metrics text
        ax4 = fig.add_subplot(n, 4, i*4 + 4)
        ax4.axis('off')
        
        mises_err = sample['mises_error_pct']
        defl_err = sample['deflection_error_pct']
        
        info = (
            f"Image Metrics:\n"
            f"  SSIM: {sample['ssim']:.4f}\n"
            f"  PSNR: {sample['psnr']:.2f} dB\n"
            f"  L1: {sample['l1']:.4f}\n\n"
            f"Numerical Errors:\n"
            f"  Mises: {mises_err:.1f}%\n"
            f"  Deflection: {defl_err:.1f}%\n\n"
            f"Input Parameters:\n"
            f"  V={sample['velocity_ms']:.0f} m/s\n"
            f"  R={sample['proj_radius_mm']:.1f} mm"
        )
        ax4.text(0.1, 0.5, info, fontsize=10, verticalalignment='center',
                family='monospace', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_comparison_grid.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'sample_comparison_grid.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: sample_comparison_grid.png/.pdf")


def plot_8_summary_dashboard(results, output_dir):
    """Create a comprehensive summary dashboard."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['generated', 'ground_truth']} 
                       for r in results])
    
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.3)
    
    # 1. Image quality metrics bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['SSIM', 'PSNR/50', 'L1×10']
    values = [df['ssim'].mean(), df['psnr'].mean()/50, df['l1'].mean()*10]
    targets = [0.85, 28/50, 0.03*10]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, values, width, label='Achieved', color='steelblue', alpha=0.8)
    bars2 = ax1.bar(x + width/2, targets, width, label='Target', color='coral', alpha=0.8)
    
    ax1.set_ylabel('Normalized Value')
    ax1.set_title('Image Quality Metrics')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.set_ylim(0, 1.2)
    
    # 2. Numerical R² scores
    ax2 = fig.add_subplot(gs[0, 1])
    outputs = ['Mises', 'PEEQ', 'Deflection']
    r2_scores = []
    for out in ['mises', 'peeq', 'deflection']:
        gt = df[f'gt_{out}']
        pred = df[f'pred_{out}']
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2 = 1 - (ss_res / (ss_tot + 1e-8))
        r2_scores.append(r2)
    
    colors = ['green' if r > 0.5 else 'orange' if r > 0 else 'red' for r in r2_scores]
    bars = ax2.bar(outputs, r2_scores, color=colors, alpha=0.8, edgecolor='black')
    ax2.axhline(y=0, color='gray', linestyle='--')
    ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.5)
    ax2.set_ylabel('R² Score')
    ax2.set_title('Numerical Prediction R²')
    ax2.set_ylim(-1, 1.1)
    
    for bar, score in zip(bars, r2_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, max(score, 0) + 0.05, 
                f'{score:.3f}', ha='center', fontsize=10)
    
    # 3. MAPE scores
    ax3 = fig.add_subplot(gs[0, 2])
    mapes = [df['mises_error_pct'].mean(), df['peeq_error_pct'].mean(), df['deflection_error_pct'].mean()]
    colors = ['green' if m < 20 else 'orange' if m < 50 else 'red' for m in mapes]
    bars = ax3.bar(outputs, mapes, color=colors, alpha=0.8, edgecolor='black')
    ax3.axhline(y=20, color='green', linestyle='--', alpha=0.5)
    ax3.set_ylabel('MAPE (%)')
    ax3.set_title('Mean Absolute Percentage Error')
    
    for bar, mape in zip(bars, mapes):
        ax3.text(bar.get_x() + bar.get_width()/2, mape + 2, 
                f'{mape:.1f}%', ha='center', fontsize=10)
    
    # 4. Sample counts
    ax4 = fig.add_subplot(gs[0, 3])
    good_ssim = (df['ssim'] > 0.85).sum()
    good_psnr = (df['psnr'] > 25).sum()
    good_mises = (df['mises_error_pct'] < 20).sum()
    good_defl = (df['deflection_error_pct'] < 20).sum()
    
    categories = ['SSIM>0.85', 'PSNR>25dB', 'Mises<20%', 'Defl<20%']
    counts = [good_ssim, good_psnr, good_mises, good_defl]
    total = len(df)
    
    bars = ax4.barh(categories, counts, color='steelblue', alpha=0.8)
    ax4.axvline(x=total*0.5, color='orange', linestyle='--', label='50%')
    ax4.axvline(x=total*0.8, color='green', linestyle='--', label='80%')
    ax4.set_xlabel('Number of Samples')
    ax4.set_title(f'Quality Thresholds (n={total})')
    ax4.legend()
    
    for bar, count in zip(bars, counts):
        ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                f'{count}/{total} ({count/total*100:.1f}%)', va='center', fontsize=10)
    
    # 5. Deflection scatter (best performing)
    ax5 = fig.add_subplot(gs[1, 0:2])
    gt = df['gt_deflection']
    pred = df['pred_deflection']
    ax5.scatter(gt, pred, alpha=0.6, s=50, c='steelblue', edgecolors='white')
    
    min_val, max_val = min(gt.min(), pred.min()), max(gt.max(), pred.max())
    ax5.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
    
    ss_res = np.sum((gt - pred) ** 2)
    ss_tot = np.sum((gt - np.mean(gt)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    ax5.set_xlabel('Ground Truth Deflection (mm)')
    ax5.set_ylabel('Predicted Deflection (mm)')
    ax5.set_title(f'Deflection: Predicted vs Ground Truth (R² = {r2:.4f})')
    ax5.legend()
    
    # 6. Statistics table
    ax6 = fig.add_subplot(gs[1, 2:4])
    ax6.axis('off')
    
    stats_data = [
        ['Metric', 'Mean', 'Std', 'Min', 'Max'],
        ['SSIM', f"{df['ssim'].mean():.4f}", f"{df['ssim'].std():.4f}", 
         f"{df['ssim'].min():.4f}", f"{df['ssim'].max():.4f}"],
        ['PSNR (dB)', f"{df['psnr'].mean():.2f}", f"{df['psnr'].std():.2f}",
         f"{df['psnr'].min():.2f}", f"{df['psnr'].max():.2f}"],
        ['L1 Error', f"{df['l1'].mean():.4f}", f"{df['l1'].std():.4f}",
         f"{df['l1'].min():.4f}", f"{df['l1'].max():.4f}"],
        ['Mises Err %', f"{df['mises_error_pct'].mean():.1f}", f"{df['mises_error_pct'].std():.1f}",
         f"{df['mises_error_pct'].min():.1f}", f"{df['mises_error_pct'].max():.1f}"],
        ['Defl Err %', f"{df['deflection_error_pct'].mean():.1f}", f"{df['deflection_error_pct'].std():.1f}",
         f"{df['deflection_error_pct'].min():.1f}", f"{df['deflection_error_pct'].max():.1f}"],
    ]
    
    table = ax6.table(cellText=stats_data, loc='center', cellLoc='center',
                     colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header row
    for i in range(5):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Param2Image cGAN Model Evaluation Summary', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_dashboard.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: summary_dashboard.png/.pdf")


def generate_latex_tables(results, output_dir):
    """Generate LaTeX-formatted tables for the report."""
    df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['generated', 'ground_truth']} 
                       for r in results])
    
    # Compute aggregate metrics
    ssim_mean = df['ssim'].mean()
    ssim_std = df['ssim'].std()
    psnr_mean = df['psnr'].mean()
    psnr_std = df['psnr'].std()
    l1_mean = df['l1'].mean()
    l1_std = df['l1'].std()
    
    # R² scores
    r2_scores = {}
    for out in ['mises', 'peeq', 'deflection']:
        gt = df[f'gt_{out}']
        pred = df[f'pred_{out}']
        ss_res = np.sum((gt - pred) ** 2)
        ss_tot = np.sum((gt - np.mean(gt)) ** 2)
        r2_scores[out] = 1 - (ss_res / (ss_tot + 1e-8))
    
    # MAPE scores
    mapes = {
        'mises': df['mises_error_pct'].mean(),
        'peeq': df['peeq_error_pct'].mean(),
        'deflection': df['deflection_error_pct'].mean()
    }
    
    latex_content = f"""% Auto-generated LaTeX tables from scripts/_analyze_impl.py
% Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
% Number of samples evaluated: {len(df)}

% Table 1: Image Quality Metrics
\\begin{{table}}[h!]
\\centering
\\renewcommand{{\\arraystretch}}{{1.3}}
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\rowcolor{{orange!70}}
\\textbf{{Metric}} & \\textbf{{Mean}} & \\textbf{{Std Dev}} & \\textbf{{Target}} \\\\
\\hline
SSIM & {ssim_mean:.4f} & $\\pm${ssim_std:.4f} & $>$0.85 \\\\
\\hline
PSNR (dB) & {psnr_mean:.2f} & $\\pm${psnr_std:.2f} & $>$28 \\\\
\\hline
L1 Error & {l1_mean:.4f} & $\\pm${l1_std:.4f} & $<$0.03 \\\\
\\hline
\\end{{tabular}}
\\caption{{Image quality metrics on {len(df)} validation samples.}}
\\label{{tab:image_quality_metrics}}
\\end{{table}}

% Table 2: Numerical Prediction Accuracy
\\begin{{table}}[h!]
\\centering
\\renewcommand{{\\arraystretch}}{{1.3}}
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\rowcolor{{orange!70}}
\\textbf{{Output}} & \\textbf{{R²}} & \\textbf{{MAPE (\\%)}} & \\textbf{{MAE}} & \\textbf{{RMSE}} \\\\
\\hline
Max Mises (MPa) & {r2_scores['mises']:.4f} & {mapes['mises']:.1f} & {df['gt_mises'].sub(df['pred_mises']).abs().mean():.2f} & {np.sqrt(((df['gt_mises'] - df['pred_mises'])**2).mean()):.2f} \\\\
\\hline
Max PEEQ & {r2_scores['peeq']:.4f} & {mapes['peeq']:.1f} & {df['gt_peeq'].sub(df['pred_peeq']).abs().mean():.4f} & {np.sqrt(((df['gt_peeq'] - df['pred_peeq'])**2).mean()):.4f} \\\\
\\hline
Max Deflection (mm) & {r2_scores['deflection']:.4f} & {mapes['deflection']:.1f} & {df['gt_deflection'].sub(df['pred_deflection']).abs().mean():.2f} & {np.sqrt(((df['gt_deflection'] - df['pred_deflection'])**2).mean()):.2f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Numerical output prediction accuracy metrics.}}
\\label{{tab:numerical_metrics}}
\\end{{table}}

% Table 3: Sample Performance Distribution
\\begin{{table}}[h!]
\\centering
\\renewcommand{{\\arraystretch}}{{1.3}}
\\begin{{tabular}}{{|l|c|c|}}
\\hline
\\rowcolor{{orange!70}}
\\textbf{{Threshold}} & \\textbf{{Count}} & \\textbf{{Percentage}} \\\\
\\hline
SSIM $>$ 0.85 & {(df['ssim'] > 0.85).sum()} & {(df['ssim'] > 0.85).mean()*100:.1f}\\% \\\\
\\hline
SSIM $>$ 0.90 & {(df['ssim'] > 0.90).sum()} & {(df['ssim'] > 0.90).mean()*100:.1f}\\% \\\\
\\hline
PSNR $>$ 25 dB & {(df['psnr'] > 25).sum()} & {(df['psnr'] > 25).mean()*100:.1f}\\% \\\\
\\hline
Mises Error $<$ 20\\% & {(df['mises_error_pct'] < 20).sum()} & {(df['mises_error_pct'] < 20).mean()*100:.1f}\\% \\\\
\\hline
Deflection Error $<$ 20\\% & {(df['deflection_error_pct'] < 20).sum()} & {(df['deflection_error_pct'] < 20).mean()*100:.1f}\\% \\\\
\\hline
\\end{{tabular}}
\\caption{{Distribution of samples meeting quality thresholds (n={len(df)}).}}
\\label{{tab:threshold_distribution}}
\\end{{table}}
"""
    
    with open(output_dir / 'latex_tables.tex', 'w') as f:
        f.write(latex_content)
    
    print(f"Saved: latex_tables.tex")
    return latex_content


def main():
    print("=" * 70)
    print("    Comprehensive Analysis for Param2Image Report")
    print("=" * 70)
    
    device = get_device()
    print(f"Device: {device}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report figures: {REPORT_FIGURES_DIR}")
    
    # Load model
    model_path = default_generator_path(Path(__file__))
    print(f"\nLoading model from: {model_path}")
    generator, head_mode = load_model(model_path, device)
    print(f"Numerical head mode: {head_mode}")
    scaler_path = default_scaler_path_for_model(model_path)
    scaler = ImpactTabularScaler.load(scaler_path) if scaler_path.exists() else None
    if scaler is not None:
        print(f"Loaded tabular scaler: {scaler_path}")
    else:
        print("No tabular scaler found; using raw input/output scale")
    
    # Load data
    csv_path = DATA_DIR / 'results_master.csv'
    print(f"Loading data from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    required_cols = ['img_after', 'velocity_ms', 'proj_radius_mm', 'impact_x_mm', 'impact_y_mm',
                     'max_mises_mpa', 'max_peeq', 'max_deflection_mm']
    df = df.dropna(subset=required_cols)
    print(f"Total valid samples: {len(df)}")
    
    # Generate predictions for all samples
    print("\nGenerating predictions for analysis...")
    results = generate_all_samples(generator, df, device, num_samples=200, scaler=scaler)
    print(f"Analyzed {len(results)} samples")
    
    # Generate all plots
    print("\nGenerating analysis plots...")
    
    plot_1_parameter_sensitivity(results, OUTPUT_DIR)
    plot_2_correlation_matrix(results, OUTPUT_DIR)
    plot_3_error_distributions(results, OUTPUT_DIR)
    plot_4_predicted_vs_actual(results, OUTPUT_DIR)
    plot_5_image_quality_analysis(results, OUTPUT_DIR)
    plot_6_best_worst_cases(results, OUTPUT_DIR)
    plot_7_sample_comparison_grid(results, OUTPUT_DIR)
    plot_8_summary_dashboard(results, OUTPUT_DIR)
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    latex_tables = generate_latex_tables(results, OUTPUT_DIR)
    
    # Save raw results
    results_df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['generated', 'ground_truth']} 
                               for r in results])
    results_df.to_csv(OUTPUT_DIR / 'all_results.csv', index=False)
    print(f"Saved: all_results.csv")
    
    # Copy key figures to report directory
    print(f"\nCopying figures to report directory: {REPORT_FIGURES_DIR}")
    
    import shutil
    figures_to_copy = [
        'summary_dashboard.png',
        'predicted_vs_actual.png', 
        'sample_comparison_grid.png',
        'error_distributions.png',
        'correlation_matrix.png',
        'parameter_sensitivity.png',
        'image_quality_analysis.png'
    ]
    
    for fig in figures_to_copy:
        src = OUTPUT_DIR / fig
        if src.exists():
            # Copy with train_v2 prefix
            dst = REPORT_FIGURES_DIR / f'train_v2_{fig}'
            shutil.copy(src, dst)
            print(f"  Copied: {fig} -> train_v2_{fig}")
    
    print("\n" + "=" * 70)
    print("    ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Report figures copied to: {REPORT_FIGURES_DIR}")


if __name__ == '__main__':
    main()
