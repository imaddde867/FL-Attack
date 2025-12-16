#!/usr/bin/env python3
"""
Analyze Ablation Study Results
==============================
Computes mean ± std for PSNR, SSIM, LPIPS across ablation conditions.
Produces a clean comparison table suitable for poster/paper.
"""

import argparse
import os
import glob
import json
import numpy as np
from pathlib import Path


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


def collect_ablation_metrics(results_dir: str) -> dict:
    """Collect metrics from all ablation conditions."""
    ablation_results = {}
    
    # Find all ablation directories
    for ablation_dir in sorted(glob.glob(os.path.join(results_dir, '*'))):
        if not os.path.isdir(ablation_dir):
            continue
            
        ablation_name = os.path.basename(ablation_dir)
        client_metrics = []
        
        # Find all client subdirectories
        for client_dir in sorted(glob.glob(os.path.join(ablation_dir, 'c*'))):
            metrics_path = os.path.join(client_dir, 'metrics.txt')
            if os.path.exists(metrics_path):
                metrics = parse_metrics_file(metrics_path)
                metrics['_client'] = os.path.basename(client_dir)
                client_metrics.append(metrics)
        
        if client_metrics:
            ablation_results[ablation_name] = client_metrics
    
    return ablation_results


def compute_stats(values: list) -> tuple:
    """Compute mean and std for a list of values."""
    valid = [v for v in values if v is not None and not np.isnan(v)]
    if not valid:
        return float('nan'), float('nan')
    return np.mean(valid), np.std(valid)


def summarize_ablation(ablation_results: dict) -> dict:
    """Compute summary statistics for each ablation condition."""
    summary = {}
    
    for ablation_name, client_metrics in ablation_results.items():
        psnr_vals = [m.get('PSNR') for m in client_metrics]
        ssim_vals = [m.get('SSIM') for m in client_metrics]
        lpips_vals = [m.get('LPIPS') for m in client_metrics]
        
        psnr_mean, psnr_std = compute_stats(psnr_vals)
        ssim_mean, ssim_std = compute_stats(ssim_vals)
        lpips_mean, lpips_std = compute_stats(lpips_vals)
        
        summary[ablation_name] = {
            'n_clients': len(client_metrics),
            'PSNR': (psnr_mean, psnr_std),
            'SSIM': (ssim_mean, ssim_std),
            'LPIPS': (lpips_mean, lpips_std),
        }
    
    return summary


def print_comparison_table(summary: dict, sort_by: str = 'PSNR'):
    """Print a formatted comparison table."""
    if not summary:
        print("No results to display.")
        return
    
    # Sort by metric (higher PSNR is better)
    reverse = sort_by in ['PSNR', 'SSIM']
    sorted_ablations = sorted(
        summary.items(),
        key=lambda x: x[1][sort_by][0] if not np.isnan(x[1][sort_by][0]) else -999,
        reverse=reverse
    )
    
    # Print header
    print("\n" + "=" * 80)
    print("ABLATION COMPARISON (sorted by {})".format(sort_by))
    print("=" * 80)
    print(f"{'Ablation':<20} {'N':>3} {'PSNR':>14} {'SSIM':>14} {'LPIPS':>14}")
    print("-" * 80)
    
    for ablation_name, stats in sorted_ablations:
        n = stats['n_clients']
        psnr_m, psnr_s = stats['PSNR']
        ssim_m, ssim_s = stats['SSIM']
        lpips_m, lpips_s = stats['LPIPS']
        
        psnr_str = f"{psnr_m:>6.2f} ± {psnr_s:<5.2f}" if not np.isnan(psnr_m) else "N/A"
        ssim_str = f"{ssim_m:>5.3f} ± {ssim_s:<5.3f}" if not np.isnan(ssim_m) else "N/A"
        lpips_str = f"{lpips_m:>5.3f} ± {lpips_s:<5.3f}" if not np.isnan(lpips_m) else "N/A"
        
        print(f"{ablation_name:<20} {n:>3} {psnr_str:>14} {ssim_str:>14} {lpips_str:>14}")
    
    print("=" * 80)
    print()


def export_latex_table(summary: dict, output_path: str):
    """Export results as a LaTeX table."""
    # Sort by PSNR descending
    sorted_ablations = sorted(
        summary.items(),
        key=lambda x: x[1]['PSNR'][0] if not np.isnan(x[1]['PSNR'][0]) else -999,
        reverse=True
    )
    
    with open(output_path, 'w') as f:
        f.write("% Ablation Study Results\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("\\toprule\n")
        f.write("Configuration & PSNR (dB) $\\uparrow$ & SSIM $\\uparrow$ & LPIPS $\\downarrow$ \\\\\n")
        f.write("\\midrule\n")
        
        for ablation_name, stats in sorted_ablations:
            psnr_m, psnr_s = stats['PSNR']
            ssim_m, ssim_s = stats['SSIM']
            lpips_m, lpips_s = stats['LPIPS']
            
            # Format name for LaTeX
            name = ablation_name.replace('_', '\\_')
            
            psnr = f"${psnr_m:.2f} \\pm {psnr_s:.2f}$" if not np.isnan(psnr_m) else "N/A"
            ssim = f"${ssim_m:.3f} \\pm {ssim_s:.3f}$" if not np.isnan(ssim_m) else "N/A"
            lpips = f"${lpips_m:.3f} \\pm {lpips_s:.3f}$" if not np.isnan(lpips_m) else "N/A"
            
            f.write(f"{name} & {psnr} & {ssim} & {lpips} \\\\\n")
        
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
    
    print(f"LaTeX table saved to: {output_path}")


def export_csv(summary: dict, output_path: str):
    """Export results as CSV for easy import."""
    sorted_ablations = sorted(
        summary.items(),
        key=lambda x: x[1]['PSNR'][0] if not np.isnan(x[1]['PSNR'][0]) else -999,
        reverse=True
    )
    
    with open(output_path, 'w') as f:
        f.write("ablation,n,psnr_mean,psnr_std,ssim_mean,ssim_std,lpips_mean,lpips_std\n")
        for ablation_name, stats in sorted_ablations:
            psnr_m, psnr_s = stats['PSNR']
            ssim_m, ssim_s = stats['SSIM']
            lpips_m, lpips_s = stats['LPIPS']
            f.write(f"{ablation_name},{stats['n_clients']},{psnr_m:.4f},{psnr_s:.4f},{ssim_m:.4f},{ssim_s:.4f},{lpips_m:.4f},{lpips_s:.4f}\n")
    
    print(f"CSV saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze ablation study results")
    parser.add_argument('--results-dir', type=str, default='results/ablation',
                        help='Directory containing ablation results')
    parser.add_argument('--export-latex', type=str, default=None,
                        help='Export LaTeX table to this path')
    parser.add_argument('--export-csv', type=str, default=None,
                        help='Export CSV to this path')
    parser.add_argument('--sort-by', type=str, default='PSNR',
                        choices=['PSNR', 'SSIM', 'LPIPS'])
    args = parser.parse_args()
    
    print(f"Analyzing ablation results from: {args.results_dir}")
    
    # Collect results
    ablation_results = collect_ablation_metrics(args.results_dir)
    
    if not ablation_results:
        print("No ablation results found!")
        return
    
    print(f"Found {len(ablation_results)} ablation conditions")
    
    # Compute summary
    summary = summarize_ablation(ablation_results)
    
    # Print comparison table
    print_comparison_table(summary, sort_by=args.sort_by)
    
    # Export if requested
    if args.export_latex:
        export_latex_table(summary, args.export_latex)
    
    if args.export_csv:
        export_csv(summary, args.export_csv)
    
    # Auto-export to results dir
    csv_path = os.path.join(args.results_dir, 'ablation_summary.csv')
    export_csv(summary, csv_path)


if __name__ == '__main__':
    main()
