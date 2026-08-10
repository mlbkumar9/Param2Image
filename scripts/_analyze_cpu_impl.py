"""
Comprehensive Analysis Script - CPU Analysis Only (No PyTorch)

This script analyzes the pre-computed evaluation results from the CSV files
without requiring PyTorch to generate new predictions.
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime
from scipy.stats import pearsonr
from PIL import Image
import shutil

# Setup paths via portable helpers (works from scripts/ or any checkout).
from p2i.paths import default_data_dir, repo_root, results_dir

REPO_ROOT = repo_root(Path(__file__))
DATA_DIR = default_data_dir(Path(__file__))
REPORT_FIGURES_DIR = results_dir(Path(__file__)) / 'analysis' / 'report_figures'

# Find latest evaluation results
EVAL_DIR = results_dir(Path(__file__)) / 'evaluation'
eval_folders = sorted(EVAL_DIR.glob('eval_*'))
if eval_folders:
    LATEST_EVAL = eval_folders[-1]
else:
    print("No evaluation results found!")
    sys.exit(1)

OUTPUT_DIR = results_dir(Path(__file__)) / 'analysis' / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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


def load_evaluation_data():
    """Load pre-computed evaluation metrics from CSV."""
    detailed_csv = LATEST_EVAL / 'detailed_metrics.csv'
    summary_csv = LATEST_EVAL / 'summary_metrics.csv'
    
    if not detailed_csv.exists():
        print(f"Error: {detailed_csv} not found")
        sys.exit(1)
    
    df = pd.read_csv(detailed_csv)
    
    # Rename columns to standard format
    column_mapping = {
        'velocity': 'velocity_ms',
        'radius': 'proj_radius_mm',
        'x': 'impact_x_mm',
        'y': 'impact_y_mm',
        'pred_defl': 'pred_deflection',
        'gt_defl': 'gt_deflection'
    }
    
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Add percentage errors if not present
    for output in ['mises', 'peeq', 'deflection']:
        gt_col = f'gt_{output}'
        pred_col = f'pred_{output}'
        err_col = f'{output}_error_pct'
        
        if gt_col in df.columns and pred_col in df.columns:
            if err_col not in df.columns:
                mask = df[gt_col].abs() > 1e-6
                df.loc[mask, err_col] = ((df.loc[mask, pred_col] - df.loc[mask, gt_col]).abs() / 
                                         df.loc[mask, gt_col].abs() * 100)
                df.loc[~mask, err_col] = 0
    
    print(f"Loaded {len(df)} samples from {detailed_csv}")
    print(f"Columns: {list(df.columns)}")
    return df


def plot_1_parameter_sensitivity(df, output_dir):
    """Plot how errors vary with each input parameter."""
    params = ['velocity_ms', 'proj_radius_mm', 'impact_x_mm', 'impact_y_mm']
    param_names = ['Velocity (m/s)', 'Projectile Radius (mm)', 'Impact X (mm)', 'Impact Y (mm)']
    outputs = ['mises', 'peeq', 'deflection']
    output_names = ['Max Mises Stress', 'Max PEEQ', 'Max Deflection']
    
    # Check which columns exist
    valid_params = [p for p in params if p in df.columns]
    valid_outputs = [o for o in outputs if f'{o}_error_pct' in df.columns]
    
    if not valid_params or not valid_outputs:
        print("Skipping parameter sensitivity plot - missing columns")
        return
    
    fig, axes = plt.subplots(len(valid_params), len(valid_outputs), 
                             figsize=(6*len(valid_outputs), 5*len(valid_params)))
    
    if len(valid_params) == 1:
        axes = axes.reshape(1, -1)
    if len(valid_outputs) == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (param, pname) in enumerate(zip(valid_params, 
                                            [param_names[params.index(p)] for p in valid_params])):
        for j, (output, oname) in enumerate(zip(valid_outputs,
                                                 [output_names[outputs.index(o)] for o in valid_outputs])):
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
            
            corr, _ = pearsonr(x, y)
            ax.annotate(f'r = {corr:.3f}', xy=(0.05, 0.95), xycoords='axes fraction',
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'parameter_sensitivity.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: parameter_sensitivity.png/.pdf")


def plot_2_correlation_matrix(df, output_dir):
    """Plot correlation matrix between all variables."""
    cols_to_check = ['velocity_ms', 'proj_radius_mm', 'impact_x_mm', 'impact_y_mm',
                     'gt_mises', 'gt_peeq', 'gt_deflection',
                     'pred_mises', 'pred_peeq', 'pred_deflection',
                     'ssim', 'psnr', 'l1']
    
    labels_map = {
        'velocity_ms': 'Velocity', 'proj_radius_mm': 'Radius', 
        'impact_x_mm': 'X', 'impact_y_mm': 'Y',
        'gt_mises': 'GT Mises', 'gt_peeq': 'GT PEEQ', 'gt_deflection': 'GT Defl',
        'pred_mises': 'Pred Mises', 'pred_peeq': 'Pred PEEQ', 'pred_deflection': 'Pred Defl',
        'ssim': 'SSIM', 'psnr': 'PSNR', 'l1': 'L1'
    }
    
    cols = [c for c in cols_to_check if c in df.columns]
    labels = [labels_map[c] for c in cols]
    
    if len(cols) < 3:
        print("Skipping correlation matrix - not enough columns")
        return
    
    corr_matrix = df[cols].corr()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    corr_masked = np.ma.array(corr_matrix.values, mask=mask)
    
    im = ax.imshow(corr_masked, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticklabels(labels)
    plt.colorbar(im, ax=ax, label='Correlation')
    
    # Add text annotations
    for i in range(len(labels)):
        for j in range(len(labels)):
            if not mask[i, j]:
                val = corr_matrix.iloc[i, j]
                color = 'white' if abs(val) > 0.5 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    
    plt.title('Correlation Matrix: Parameters, Outputs, and Metrics', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'correlation_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: correlation_matrix.png/.pdf")


def plot_3_error_distributions(df, output_dir):
    """Plot detailed error distributions for each output."""
    outputs = ['mises', 'peeq', 'deflection']
    output_names = ['Max Mises Stress (MPa)', 'Max PEEQ', 'Max Deflection (mm)']
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    
    valid_outputs = [(o, n, c) for o, n, c in zip(outputs, output_names, colors) 
                     if f'gt_{o}' in df.columns and f'pred_{o}' in df.columns]
    
    if not valid_outputs:
        print("Skipping error distributions - missing columns")
        return
    
    fig, axes = plt.subplots(len(valid_outputs), 3, figsize=(18, 5*len(valid_outputs)))
    
    if len(valid_outputs) == 1:
        axes = axes.reshape(1, -1)
    
    for i, (output, oname, color) in enumerate(valid_outputs):
        gt = df[f'gt_{output}']
        pred = df[f'pred_{output}']
        error = pred - gt
        
        if f'{output}_error_pct' in df.columns:
            error_pct = df[f'{output}_error_pct']
        else:
            error_pct = (error.abs() / gt.abs().clip(lower=1e-6)) * 100
        
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
        ax2.hist(error_pct.clip(upper=200), bins=30, color=color, alpha=0.7, edgecolor='black')
        ax2.axvline(x=error_pct.mean(), color='red', linestyle='-', linewidth=2, 
                   label=f'Mean: {error_pct.mean():.1f}%')
        ax2.axvline(x=error_pct.median(), color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {error_pct.median():.1f}%')
        ax2.set_xlabel('Percentage Error (%)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{oname}\nPercentage Error Distribution')
        ax2.legend()
        
        # Predicted vs Ground Truth scatter
        ax3 = axes[i, 2]
        ax3.scatter(gt, pred, alpha=0.5, s=30, c=color, edgecolors='white')
        min_val, max_val = min(gt.min(), pred.min()), max(gt.max(), pred.max())
        ax3.plot([min_val, max_val], [min_val, max_val], 'k--', linewidth=2)
        ax3.set_xlabel('Ground Truth')
        ax3.set_ylabel('Predicted')
        ax3.set_title(f'{oname}\nPredicted vs Actual')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_distributions.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'error_distributions.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: error_distributions.png/.pdf")


def plot_4_predicted_vs_actual(df, output_dir):
    """Create scatter plots of predicted vs actual for each output."""
    outputs = ['mises', 'peeq', 'deflection']
    output_names = ['Max Mises Stress (MPa)', 'Max PEEQ', 'Max Deflection (mm)']
    colors = ['#E74C3C', '#27AE60', '#3498DB']
    
    valid_outputs = [(o, n, c) for o, n, c in zip(outputs, output_names, colors) 
                     if f'gt_{o}' in df.columns and f'pred_{o}' in df.columns]
    
    if not valid_outputs:
        print("Skipping predicted vs actual - missing columns")
        return
    
    fig, axes = plt.subplots(1, len(valid_outputs), figsize=(6*len(valid_outputs), 6))
    
    if len(valid_outputs) == 1:
        axes = [axes]
    
    for i, (output, oname, color) in enumerate(valid_outputs):
        ax = axes[i]
        
        gt = df[f'gt_{output}']
        pred = df[f'pred_{output}']
        
        ax.scatter(gt, pred, alpha=0.6, s=50, c=color, edgecolors='white', linewidth=0.5)
        
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
        
        corr, _ = pearsonr(gt, pred)
        
        ax.set_xlabel(f'Ground Truth {oname}')
        ax.set_ylabel(f'Predicted {oname}')
        ax.set_title(f'{oname}\nR² = {r2:.4f}, Pearson r = {corr:.4f}')
        ax.legend(loc='upper left')
        
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


def plot_5_image_quality_analysis(df, output_dir):
    """Analyze image quality metrics distribution."""
    if 'ssim' not in df.columns or 'psnr' not in df.columns:
        print("Skipping image quality analysis - missing columns")
        return
    
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
    if 'l1' in df.columns:
        ax3.hist(df['l1'], bins=25, color='mediumseagreen', alpha=0.7, edgecolor='black')
        ax3.axvline(x=df['l1'].mean(), color='red', linewidth=2, label=f'Mean: {df["l1"].mean():.4f}')
        ax3.axvline(x=0.03, color='green', linestyle='--', linewidth=2, label='Target: 0.03')
        ax3.set_xlabel('L1 Error')
        ax3.set_ylabel('Frequency')
        ax3.set_title('L1 Error Distribution')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, 'L1 data not available', ha='center', va='center', transform=ax3.transAxes)
        ax3.axis('off')
    
    # SSIM vs PSNR scatter
    ax4 = axes[1, 1]
    if 'l1' in df.columns:
        scatter = ax4.scatter(df['psnr'], df['ssim'], c=df['l1'], cmap='RdYlGn_r', 
                             s=50, alpha=0.7, edgecolors='white', linewidth=0.5)
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('L1 Error')
    else:
        ax4.scatter(df['psnr'], df['ssim'], s=50, alpha=0.7, c='steelblue')
    
    ax4.set_xlabel('PSNR (dB)')
    ax4.set_ylabel('SSIM')
    ax4.set_title('SSIM vs PSNR')
    ax4.axhline(y=0.85, color='gray', linestyle='--', alpha=0.5)
    ax4.axvline(x=25, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'image_quality_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'image_quality_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: image_quality_analysis.png/.pdf")


def plot_6_summary_dashboard(df, output_dir):
    """Create a comprehensive summary dashboard."""
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 4, figure=fig, wspace=0.3, hspace=0.3)
    
    # 1. Image quality metrics bar chart
    ax1 = fig.add_subplot(gs[0, 0])
    if 'ssim' in df.columns and 'psnr' in df.columns and 'l1' in df.columns:
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
    else:
        ax1.text(0.5, 0.5, 'Metrics not available', ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Numerical R² scores
    ax2 = fig.add_subplot(gs[0, 1])
    outputs = ['Mises', 'PEEQ', 'Deflection']
    r2_scores = []
    for out in ['mises', 'peeq', 'deflection']:
        if f'gt_{out}' in df.columns and f'pred_{out}' in df.columns:
            gt = df[f'gt_{out}']
            pred = df[f'pred_{out}']
            ss_res = np.sum((gt - pred) ** 2)
            ss_tot = np.sum((gt - np.mean(gt)) ** 2)
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            r2_scores.append(r2)
        else:
            r2_scores.append(0)
    
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
    mapes = []
    for out in ['mises', 'peeq', 'deflection']:
        if f'{out}_error_pct' in df.columns:
            mapes.append(df[f'{out}_error_pct'].mean())
        else:
            mapes.append(0)
    
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
    total = len(df)
    
    categories = []
    counts = []
    
    if 'ssim' in df.columns:
        categories.append('SSIM>0.85')
        counts.append((df['ssim'] > 0.85).sum())
    
    if 'psnr' in df.columns:
        categories.append('PSNR>25dB')
        counts.append((df['psnr'] > 25).sum())
    
    if 'mises_error_pct' in df.columns:
        categories.append('Mises<20%')
        counts.append((df['mises_error_pct'] < 20).sum())
    
    if 'deflection_error_pct' in df.columns:
        categories.append('Defl<20%')
        counts.append((df['deflection_error_pct'] < 20).sum())
    
    if categories:
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
    if 'gt_deflection' in df.columns and 'pred_deflection' in df.columns:
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
    
    stats_data = [['Metric', 'Mean', 'Std', 'Min', 'Max']]
    
    if 'ssim' in df.columns:
        stats_data.append(['SSIM', f"{df['ssim'].mean():.4f}", f"{df['ssim'].std():.4f}", 
                          f"{df['ssim'].min():.4f}", f"{df['ssim'].max():.4f}"])
    
    if 'psnr' in df.columns:
        stats_data.append(['PSNR (dB)', f"{df['psnr'].mean():.2f}", f"{df['psnr'].std():.2f}",
                          f"{df['psnr'].min():.2f}", f"{df['psnr'].max():.2f}"])
    
    if 'l1' in df.columns:
        stats_data.append(['L1 Error', f"{df['l1'].mean():.4f}", f"{df['l1'].std():.4f}",
                          f"{df['l1'].min():.4f}", f"{df['l1'].max():.4f}"])
    
    if 'mises_error_pct' in df.columns:
        stats_data.append(['Mises Err %', f"{df['mises_error_pct'].mean():.1f}", 
                          f"{df['mises_error_pct'].std():.1f}",
                          f"{df['mises_error_pct'].min():.1f}", 
                          f"{df['mises_error_pct'].max():.1f}"])
    
    if 'deflection_error_pct' in df.columns:
        stats_data.append(['Defl Err %', f"{df['deflection_error_pct'].mean():.1f}", 
                          f"{df['deflection_error_pct'].std():.1f}",
                          f"{df['deflection_error_pct'].min():.1f}", 
                          f"{df['deflection_error_pct'].max():.1f}"])
    
    if len(stats_data) > 1:
        table = ax6.table(cellText=stats_data, loc='center', cellLoc='center',
                         colWidths=[0.25, 0.18, 0.18, 0.18, 0.18])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)
        
        for i in range(5):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax6.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.suptitle('Param2Image cGAN Model Evaluation Summary', fontsize=18, fontweight='bold', y=0.98)
    plt.savefig(output_dir / 'summary_dashboard.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'summary_dashboard.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: summary_dashboard.png/.pdf")


def generate_latex_tables(df, output_dir):
    """Generate LaTeX-formatted tables for the report."""
    # Compute aggregate metrics
    ssim_mean = df['ssim'].mean() if 'ssim' in df.columns else 0
    ssim_std = df['ssim'].std() if 'ssim' in df.columns else 0
    psnr_mean = df['psnr'].mean() if 'psnr' in df.columns else 0
    psnr_std = df['psnr'].std() if 'psnr' in df.columns else 0
    l1_mean = df['l1'].mean() if 'l1' in df.columns else 0
    l1_std = df['l1'].std() if 'l1' in df.columns else 0
    
    # R² scores
    r2_scores = {}
    mapes = {}
    maes = {}
    rmses = {}
    
    for out in ['mises', 'peeq', 'deflection']:
        if f'gt_{out}' in df.columns and f'pred_{out}' in df.columns:
            gt = df[f'gt_{out}']
            pred = df[f'pred_{out}']
            ss_res = np.sum((gt - pred) ** 2)
            ss_tot = np.sum((gt - np.mean(gt)) ** 2)
            r2_scores[out] = 1 - (ss_res / (ss_tot + 1e-8))
            maes[out] = np.mean(np.abs(pred - gt))
            rmses[out] = np.sqrt(np.mean((pred - gt) ** 2))
        else:
            r2_scores[out] = 0
            maes[out] = 0
            rmses[out] = 0
        
        if f'{out}_error_pct' in df.columns:
            mapes[out] = df[f'{out}_error_pct'].mean()
        else:
            mapes[out] = 0
    
    latex_content = f"""% Auto-generated LaTeX tables from scripts/_analyze_cpu_impl.py
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
\\caption{{Image quality metrics on {len(df)} validation samples (Train\\_v2 model at 512$\\times$512).}}
\\label{{tab:train_v2_image_quality_metrics}}
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
Max Mises (MPa) & {r2_scores['mises']:.4f} & {mapes['mises']:.1f} & {maes['mises']:.2f} & {rmses['mises']:.2f} \\\\
\\hline
Max PEEQ & {r2_scores['peeq']:.4f} & {mapes['peeq']:.1f} & {maes['peeq']:.4f} & {rmses['peeq']:.4f} \\\\
\\hline
Max Deflection (mm) & {r2_scores['deflection']:.4f} & {mapes['deflection']:.1f} & {maes['deflection']:.2f} & {rmses['deflection']:.2f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Numerical output prediction accuracy metrics (Train\\_v2 model).}}
\\label{{tab:train_v2_numerical_metrics}}
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
SSIM $>$ 0.85 & {(df['ssim'] > 0.85).sum() if 'ssim' in df.columns else 0} & {((df['ssim'] > 0.85).mean()*100) if 'ssim' in df.columns else 0:.1f}\\% \\\\
\\hline
SSIM $>$ 0.90 & {(df['ssim'] > 0.90).sum() if 'ssim' in df.columns else 0} & {((df['ssim'] > 0.90).mean()*100) if 'ssim' in df.columns else 0:.1f}\\% \\\\
\\hline
PSNR $>$ 25 dB & {(df['psnr'] > 25).sum() if 'psnr' in df.columns else 0} & {((df['psnr'] > 25).mean()*100) if 'psnr' in df.columns else 0:.1f}\\% \\\\
\\hline
Mises Error $<$ 20\\% & {(df['mises_error_pct'] < 20).sum() if 'mises_error_pct' in df.columns else 0} & {((df['mises_error_pct'] < 20).mean()*100) if 'mises_error_pct' in df.columns else 0:.1f}\\% \\\\
\\hline
Deflection Error $<$ 20\\% & {(df['deflection_error_pct'] < 20).sum() if 'deflection_error_pct' in df.columns else 0} & {((df['deflection_error_pct'] < 20).mean()*100) if 'deflection_error_pct' in df.columns else 0:.1f}\\% \\\\
\\hline
\\end{{tabular}}
\\caption{{Distribution of samples meeting quality thresholds (n={len(df)}).}}
\\label{{tab:train_v2_threshold_distribution}}
\\end{{table}}
"""
    
    with open(output_dir / 'latex_tables.tex', 'w') as f:
        f.write(latex_content)
    
    print(f"Saved: latex_tables.tex")
    return latex_content


def main():
    print("=" * 70)
    print("    CPU-Only Analysis for Param2Image Report")
    print("=" * 70)
    
    print(f"Using evaluation results from: {LATEST_EVAL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Report figures: {REPORT_FIGURES_DIR}")
    
    # Load evaluation data
    df = load_evaluation_data()
    
    # Generate all plots
    print("\nGenerating analysis plots...")
    
    plot_1_parameter_sensitivity(df, OUTPUT_DIR)
    plot_2_correlation_matrix(df, OUTPUT_DIR)
    plot_3_error_distributions(df, OUTPUT_DIR)
    plot_4_predicted_vs_actual(df, OUTPUT_DIR)
    plot_5_image_quality_analysis(df, OUTPUT_DIR)
    plot_6_summary_dashboard(df, OUTPUT_DIR)
    
    # Generate LaTeX tables
    print("\nGenerating LaTeX tables...")
    latex_tables = generate_latex_tables(df, OUTPUT_DIR)
    
    # Copy key figures to report directory
    print(f"\nCopying figures to report directory: {REPORT_FIGURES_DIR}")
    
    figures_to_copy = [
        'summary_dashboard.png',
        'predicted_vs_actual.png', 
        'error_distributions.png',
        'correlation_matrix.png',
        'parameter_sensitivity.png',
        'image_quality_analysis.png'
    ]
    
    for fig in figures_to_copy:
        src = OUTPUT_DIR / fig
        if src.exists():
            dst = REPORT_FIGURES_DIR / f'train_v2_{fig}'
            shutil.copy(src, dst)
            print(f"  Copied: {fig} -> train_v2_{fig}")
    
    # Also copy existing evaluation figures
    eval_figures = [
        'best_samples.png',
        'worst_samples.png',
        'random_samples.png',
        'summary_dashboard.png'
    ]
    
    for fig in eval_figures:
        src = LATEST_EVAL / fig
        if src.exists():
            dst = REPORT_FIGURES_DIR / f'train_v2_eval_{fig}'
            shutil.copy(src, dst)
            print(f"  Copied from eval: {fig} -> train_v2_eval_{fig}")
    
    print("\n" + "=" * 70)
    print("    ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print(f"Report figures copied to: {REPORT_FIGURES_DIR}")


if __name__ == '__main__':
    main()
