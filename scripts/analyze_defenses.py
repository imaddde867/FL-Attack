#!/usr/bin/env python3
"""
Analyze defense experiment results and generate comparison table.
"""
import argparse
import json
import os
from pathlib import Path


def parse_metrics(metrics_path: str) -> dict:
    """Parse metrics.txt file."""
    metrics = {}
    if not os.path.exists(metrics_path):
        return metrics
    with open(metrics_path, 'r') as f:
        for line in f:
            if ':' in line:
                key, val = line.strip().split(':', 1)
                try:
                    metrics[key.strip()] = float(val.strip())
                except ValueError:
                    metrics[key.strip()] = val.strip()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Analyze defense experiments")
    parser.add_argument('--results-dir', type=str, default='results/defenses',
                        help='Directory containing defense results')
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Expected conditions
    conditions = [
        ('baseline', 'No Protection'),
        ('dp_eps8', 'DP ε=8.0'),
        ('dp_eps1', 'DP ε=1.0'),
        ('dp_eps01', 'DP ε=0.1'),
        ('he', 'HE Only'),
        ('dp_he', 'DP ε=1.0 + HE'),
    ]
    
    print("\n" + "="*80)
    print("  DEFENSE EXPERIMENT RESULTS")
    print("="*80)
    print("")
    
    # Collect results
    all_results = []
    for cond_dir, cond_name in conditions:
        cond_path = results_dir / cond_dir
        metrics_path = cond_path / 'metrics.txt'
        
        if metrics_path.exists():
            metrics = parse_metrics(str(metrics_path))
            all_results.append((cond_name, metrics))
        else:
            all_results.append((cond_name, None))
    
    # Print table header
    print(f"{'Defense':<20} {'PSNR↓':>10} {'SSIM↓':>10} {'LPIPS↑':>10} {'MSE↑':>12} {'Label':>8}")
    print("-"*80)
    
    for cond_name, metrics in all_results:
        if metrics is None:
            print(f"{cond_name:<20} {'N/A':>10} {'N/A':>10} {'N/A':>10} {'N/A':>12} {'N/A':>8}")
        else:
            psnr = metrics.get('PSNR', float('nan'))
            ssim = metrics.get('SSIM', float('nan'))
            lpips = metrics.get('LPIPS', float('nan'))
            mse = metrics.get('MSE', float('nan'))
            label = metrics.get('LabelMatch', float('nan'))
            
            print(f"{cond_name:<20} {psnr:>10.2f} {ssim:>10.4f} {lpips:>10.4f} {mse:>12.6f} {label:>8.2f}")
    
    print("-"*80)
    print("")
    
    # Analysis summary
    baseline_metrics = None
    for cond_name, metrics in all_results:
        if cond_name == 'No Protection' and metrics:
            baseline_metrics = metrics
            break
    
    if baseline_metrics:
        baseline_psnr = baseline_metrics.get('PSNR', 0)
        print("Analysis:")
        print(f"  • Baseline attack achieves PSNR={baseline_psnr:.2f} dB")
        print("")
        print("  Defense Effectiveness (lower PSNR = better protection):")
        
        for cond_name, metrics in all_results:
            if metrics and cond_name != 'No Protection':
                psnr = metrics.get('PSNR', 0)
                psnr_drop = baseline_psnr - psnr
                effectiveness = "✓✓✓ STRONG" if psnr < 15 else "✓✓ GOOD" if psnr < 20 else "✓ MODERATE" if psnr < 25 else "⚠ WEAK"
                print(f"    - {cond_name}: PSNR dropped {psnr_drop:+.2f} dB → {effectiveness}")
    
    print("")
    print("="*80)
    print("")
    
    # Generate LaTeX table for paper
    print("LaTeX Table (for paper):")
    print("-"*40)
    print(r"\begin{table}[h]")
    print(r"\centering")
    print(r"\caption{Attack success under various privacy defenses}")
    print(r"\begin{tabular}{lcccc}")
    print(r"\toprule")
    print(r"Defense & PSNR$\downarrow$ & SSIM$\downarrow$ & LPIPS$\uparrow$ & MSE$\uparrow$ \\")
    print(r"\midrule")
    
    for cond_name, metrics in all_results:
        if metrics:
            psnr = metrics.get('PSNR', float('nan'))
            ssim = metrics.get('SSIM', float('nan'))
            lpips = metrics.get('LPIPS', float('nan'))
            mse = metrics.get('MSE', float('nan'))
            print(f"{cond_name} & {psnr:.2f} & {ssim:.3f} & {lpips:.3f} & {mse:.4f} \\\\")
    
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")
    print("")


if __name__ == '__main__':
    main()
