#!/usr/bin/env python3
"""
Analyze Multi-Client Benchmark Results
=======================================
Computes mean ± std for PSNR, SSIM, LPIPS across multiple client attacks.
Generates summary statistics and optionally a grid of best/median/worst results.
"""

import argparse
import os
import glob
import json
import numpy as np
from pathlib import Path

try:
    from PIL import Image
    import matplotlib.pyplot as plt
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def parse_metrics_file(metrics_path: str) -> dict:
    """Parse a metrics.txt file and return metrics as a dict."""
    metrics = {}
    with open(metrics_path, 'r') as f:
        for line in f:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    metrics[key] = value
    return metrics


def collect_all_metrics(results_dir: str) -> list:
    """Collect metrics from all client subdirectories."""
    all_metrics = []
    
    # Find all bmk_c* directories
    pattern = os.path.join(results_dir, 'bmk_c*')
    client_dirs = sorted(glob.glob(pattern))
    
    if not client_dirs:
        # Try without bmk_ prefix
        pattern = os.path.join(results_dir, 'c*')
        client_dirs = sorted(glob.glob(pattern))
    
    if not client_dirs:
        print(f"No client directories found in {results_dir}")
        return []
    
    for client_dir in client_dirs:
        metrics_path = os.path.join(client_dir, 'metrics.txt')
        if os.path.exists(metrics_path):
            metrics = parse_metrics_file(metrics_path)
            metrics['_dir'] = client_dir
            metrics['_client'] = os.path.basename(client_dir)
            all_metrics.append(metrics)
        else:
            print(f"Warning: No metrics.txt in {client_dir}")
    
    return all_metrics


def compute_statistics(all_metrics: list) -> dict:
    """Compute mean ± std for key metrics."""
    stats = {}
    
    key_metrics = ['PSNR', 'SSIM', 'LPIPS', 'MSE', 'LabelMatch']
    
    for metric_name in key_metrics:
        values = [m.get(metric_name) for m in all_metrics if m.get(metric_name) is not None]
        # Filter out nan values
        values = [v for v in values if isinstance(v, (int, float)) and not np.isnan(v)]
        
        if values:
            stats[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'n': len(values),
                'values': values
            }
    
    return stats


def find_best_median_worst(all_metrics: list, by_metric: str = 'PSNR') -> dict:
    """Find best, median, and worst reconstructions by a given metric."""
    # Filter metrics that have the target metric
    valid = [(i, m) for i, m in enumerate(all_metrics) 
             if m.get(by_metric) is not None and not np.isnan(m.get(by_metric, float('nan')))]
    
    if not valid:
        return {}
    
    # Sort by metric (higher PSNR/SSIM is better, lower LPIPS is better)
    reverse = by_metric in ['PSNR', 'SSIM', 'LabelMatch']
    sorted_metrics = sorted(valid, key=lambda x: x[1][by_metric], reverse=reverse)
    
    result = {
        'best': sorted_metrics[0][1],
        'worst': sorted_metrics[-1][1],
        'median': sorted_metrics[len(sorted_metrics) // 2][1],
    }
    
    return result


def create_summary_grid(results_dir: str, all_metrics: list, output_path: str):
    """Create a visual grid showing best/median/worst reconstructions."""
    if not HAS_PIL:
        print("PIL/matplotlib not available, skipping visual grid generation")
        return
    
    rankings = find_best_median_worst(all_metrics, by_metric='PSNR')
    if not rankings:
        print("Could not determine rankings for grid")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    categories = ['best', 'median', 'worst']
    
    for col, cat in enumerate(categories):
        client_dir = rankings[cat]['_dir']
        
        # Find the attack result image
        img_path = os.path.join(client_dir, 'baseline_attack_result.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            # The image has original on top, reconstruction on bottom
            # We'll show the full image
            axes[0, col].imshow(img)
            axes[0, col].axis('off')
            
            psnr = rankings[cat].get('PSNR', 'N/A')
            ssim = rankings[cat].get('SSIM', 'N/A')
            lpips = rankings[cat].get('LPIPS', 'N/A')
            
            if isinstance(psnr, float):
                psnr = f"{psnr:.2f}"
            if isinstance(ssim, float):
                ssim = f"{ssim:.4f}"
            if isinstance(lpips, float):
                lpips = f"{lpips:.4f}"
            
            title = f"{cat.upper()}\n{rankings[cat]['_client']}"
            axes[0, col].set_title(title, fontsize=12, fontweight='bold')
            
            # Metrics in second row
            axes[1, col].text(0.5, 0.6, f"PSNR: {psnr} dB", ha='center', va='center', fontsize=14)
            axes[1, col].text(0.5, 0.4, f"SSIM: {ssim}", ha='center', va='center', fontsize=14)
            axes[1, col].text(0.5, 0.2, f"LPIPS: {lpips}", ha='center', va='center', fontsize=14)
            axes[1, col].axis('off')
        else:
            axes[0, col].text(0.5, 0.5, f"Image not found\n{cat}", ha='center', va='center')
            axes[0, col].axis('off')
            axes[1, col].axis('off')
    
    plt.suptitle('Multi-Client Attack Results: Best / Median / Worst', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Summary grid saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze multi-client benchmark results')
    parser.add_argument('--results-dir', type=str, default='results/multi_client',
                        help='Directory containing bmk_c* subdirectories')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for summary (default: <results-dir>/summary.txt)')
    parser.add_argument('--grid', action='store_true',
                        help='Generate visual grid of best/median/worst')
    parser.add_argument('--json', action='store_true',
                        help='Also output statistics as JSON')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    if not os.path.isdir(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    print(f"Analyzing results in: {results_dir}")
    print("-" * 60)
    
    # Collect all metrics
    all_metrics = collect_all_metrics(results_dir)
    
    if not all_metrics:
        print("No metrics found!")
        return
    
    print(f"Found {len(all_metrics)} client results")
    print()
    
    # Compute statistics
    stats = compute_statistics(all_metrics)
    
    # Print summary
    print("=" * 60)
    print("  AGGREGATE STATISTICS (mean ± std)")
    print("=" * 60)
    
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("  MULTI-CLIENT BENCHMARK RESULTS")
    summary_lines.append(f"  N = {len(all_metrics)} clients")
    summary_lines.append("=" * 60)
    summary_lines.append("")
    
    for metric_name in ['PSNR', 'SSIM', 'LPIPS', 'MSE', 'LabelMatch']:
        if metric_name in stats:
            s = stats[metric_name]
            line = f"  {metric_name:12s}: {s['mean']:8.4f} ± {s['std']:.4f}  (min={s['min']:.4f}, max={s['max']:.4f})"
            print(line)
            summary_lines.append(line)
    
    print()
    summary_lines.append("")
    
    # Find best/median/worst
    rankings = find_best_median_worst(all_metrics, by_metric='PSNR')
    if rankings:
        print("-" * 60)
        print("  RANKINGS (by PSNR)")
        print("-" * 60)
        summary_lines.append("-" * 60)
        summary_lines.append("  RANKINGS (by PSNR)")
        summary_lines.append("-" * 60)
        
        for cat in ['best', 'median', 'worst']:
            m = rankings[cat]
            psnr = m.get('PSNR', 'N/A')
            ssim = m.get('SSIM', 'N/A')
            lpips = m.get('LPIPS', 'N/A')
            line = f"  {cat.upper():8s}: {m['_client']:12s}  PSNR={psnr:.2f}  SSIM={ssim:.4f}  LPIPS={lpips:.4f}"
            print(line)
            summary_lines.append(line)
    
    print()
    summary_lines.append("")
    
    # Save summary
    output_path = args.output or os.path.join(results_dir, 'summary.txt')
    with open(output_path, 'w') as f:
        f.write('\n'.join(summary_lines))
    print(f"Summary saved to: {output_path}")
    
    # Optionally save JSON
    if args.json:
        json_path = output_path.replace('.txt', '.json')
        json_data = {
            'n_clients': len(all_metrics),
            'statistics': {k: {kk: vv for kk, vv in v.items() if kk != 'values'} 
                          for k, v in stats.items()},
            'per_client': [{k: v for k, v in m.items() if not k.startswith('_')} 
                          for m in all_metrics]
        }
        with open(json_path, 'w') as f:
            json.dump(json_data, f, indent=2)
        print(f"JSON saved to: {json_path}")
    
    # Optionally create visual grid
    if args.grid:
        grid_path = os.path.join(results_dir, 'summary_grid.png')
        create_summary_grid(results_dir, all_metrics, grid_path)


if __name__ == '__main__':
    main()
