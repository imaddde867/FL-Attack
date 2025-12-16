#!/usr/bin/env python3
"""
Phase 1 — Improve 64×64 Baseline (Stability + Visual Realism)
==============================================================

Systematic experiments to improve the baseline reconstruction quality:
1. TV Regularization Sweep: {1e-6, 1e-5, 1e-4}
2. Layer Weighting Ablation: uniform vs early upweighting strategies
3. Perceptual Loss (Optional): LPIPS for visual realism

Success Criteria:
- Visually clean face reconstructions
- Stable identity features  
- PSNR ≥ current baseline (25.83), SSIM ≈ or > 0.89

Usage:
    python scripts/exp_phase1.py                        # Full Phase 1 suite
    python scripts/exp_phase1.py --mode tv-sweep        # Just TV sweep
    python scripts/exp_phase1.py --mode layer-ablation  # Just layer weighting
    python scripts/exp_phase1.py --mode perceptual      # Just perceptual loss
    python scripts/exp_phase1.py --dry-run              # Preview commands
    python scripts/exp_phase1.py --clean                # Remove old results first
"""

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any
import shutil
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    name: str
    description: str
    category: str = "phase1"  # For grouping results
    # FL settings (fixed for Phase 1)
    batch_size: int = 1
    client_momentum: float = 0.0
    attack_source: str = "gradients"
    num_rounds: int = 1
    capture_round: int = 0
    # Attack settings
    attack_iterations: int = 3000
    attack_restarts: int = 5
    attack_lr: float = 0.1
    attack_optimizer: str = "adam"
    tv_weight: float = 1e-6
    lr_schedule: str = "cosine"
    early_stop: bool = False
    patience: int = 600
    preset: str = "none"
    match_metric: str = "l2"
    layer_weights: Optional[str] = None
    fft_init: bool = False
    compute_lpips: bool = True  # Always compute LPIPS for comparison
    priority: int = 0

    def to_flags(self) -> List[str]:
        """Convert to CLI flags for run_experiment.py."""
        flags = [
            "--batch-size", str(self.batch_size),
            "--client-momentum", str(self.client_momentum),
            "--attack-source", self.attack_source,
            "--num-rounds", str(self.num_rounds),
            "--capture-round", str(self.capture_round),
            "--attack-iterations", str(self.attack_iterations),
            "--attack-restarts", str(self.attack_restarts),
            "--attack-lr", str(self.attack_lr),
            "--attack-optimizer", self.attack_optimizer,
            "--tv-weight", str(self.tv_weight),
            "--lr-schedule", self.lr_schedule,
            "--patience", str(self.patience),
            "--match-metric", self.match_metric,
        ]
        if self.layer_weights is not None and self.layer_weights != "":
            flags.extend(["--layer-weights", str(self.layer_weights)])
        if self.preset and self.preset.lower() not in ("none", ""):
            flags.extend(["--preset", self.preset])
        if self.early_stop:
            flags.append("--early-stop")
        if self.fft_init:
            flags.append("--fft-init")
        if self.compute_lpips:
            flags.append("--compute-lpips")
        return flags


# ============================================================================
# Experiment Definitions - Phase 1
# ============================================================================

def get_baseline_reference() -> ExperimentConfig:
    """Baseline reference configuration (current best settings)."""
    return ExperimentConfig(
        name="p1_baseline_reference",
        description="Phase 1 baseline reference (TV=1e-6, uniform weights)",
        category="baseline",
        tv_weight=1e-6,
        layer_weights=None,  # Uniform
        attack_iterations=3000,
        attack_restarts=5,
        priority=0,
    )


def get_tv_sweep_configs() -> List[ExperimentConfig]:
    """
    TV Regularization Sweep: {1e-6, 1e-5, 1e-4}
    
    Goal: Find optimal TV weight that balances smoothness vs detail.
    - Too low: Noisy, high-frequency artifacts
    - Too high: Over-smoothed, loses detail
    """
    configs = []
    
    for i, tv in enumerate([1e-6, 1e-5, 1e-4]):
        configs.append(ExperimentConfig(
            name=f"p1_tv_{tv:.0e}".replace("+", "").replace("-0", "-"),
            description=f"TV sweep: weight={tv:.0e}",
            category="tv_sweep",
            tv_weight=tv,
            layer_weights=None,  # Keep uniform for controlled comparison
            attack_iterations=3000,
            attack_restarts=5,
            priority=10 + i,
        ))
    
    return configs


def get_layer_weighting_configs() -> List[ExperimentConfig]:
    """
    Layer Weighting Ablation
    
    Strategies tested:
    - uniform: Equal weight for all layers (baseline)
    - auto: Inverse gradient norm (normalize contribution)
    - early: Exponential decay (upweight early layers)
    - early_linear: Linear decay from early to late
    - early_strong: Strong emphasis on first 1/3 of layers
    - early_conv: Emphasize early convolutional layers specifically
    
    Goal: Improve spatial coherence and reduce high-frequency noise
    by focusing on early layers that capture low-frequency structure.
    """
    configs = []
    
    strategies = [
        ("uniform", "Uniform weighting (baseline)"),
        ("auto", "Auto inverse-norm weighting"),
        ("early", "Exponential early layer emphasis"),
        ("early_linear", "Linear early layer decay"),
        ("early_strong", "Strong first-1/3 emphasis"),
        ("early_conv", "Early convolutional layer focus"),
    ]
    
    for i, (strategy, desc) in enumerate(strategies):
        lw = None if strategy == "uniform" else strategy
        configs.append(ExperimentConfig(
            name=f"p1_layer_{strategy}",
            description=f"Layer weighting: {desc}",
            category="layer_weighting",
            tv_weight=1e-6,  # Fixed at baseline best
            layer_weights=lw,
            attack_iterations=3000,
            attack_restarts=5,
            priority=20 + i,
        ))
    
    return configs


def get_combined_best_configs() -> List[ExperimentConfig]:
    """
    Combined experiments with best settings from sweeps.
    Run after initial sweeps to find optimal combination.
    """
    configs = []
    
    # Combine best TV with best layer weighting (we'll update after initial results)
    combinations = [
        # (tv_weight, layer_strategy, description)
        (1e-5, "early", "Medium TV + early weighting"),
        (1e-5, "early_linear", "Medium TV + linear early weighting"),
        (1e-4, "early_strong", "High TV + strong early emphasis"),
        (1e-5, "early_conv", "Medium TV + conv layer focus"),
    ]
    
    for i, (tv, layer, desc) in enumerate(combinations):
        tv_str = f"{tv:.0e}".replace("+", "").replace("-0", "-")
        configs.append(ExperimentConfig(
            name=f"p1_combined_{tv_str}_{layer}",
            description=f"Combined: {desc}",
            category="combined",
            tv_weight=tv,
            layer_weights=layer,
            attack_iterations=4000,  # Slightly more iterations for combined
            attack_restarts=7,
            priority=30 + i,
        ))
    
    return configs


def get_all_configs() -> List[ExperimentConfig]:
    """Get all Phase 1 experiment configurations."""
    configs = []
    configs.append(get_baseline_reference())
    configs.extend(get_tv_sweep_configs())
    configs.extend(get_layer_weighting_configs())
    configs.extend(get_combined_best_configs())
    return sorted(configs, key=lambda c: c.priority)


# ============================================================================
# Execution Engine
# ============================================================================

def run_experiment(
    config: ExperimentConfig,
    output_dir: Path,
    base_flags: List[str],
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment and return metrics."""
    
    exp_dir = output_dir / config.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable, "run_experiment.py",
        "--out-dir", str(exp_dir),
        "--save-config",
    ] + base_flags + config.to_flags()
    
    print(f"\n{'='*70}")
    print(f"EXPERIMENT: {config.name}")
    print(f"{'='*70}")
    print(f"Description: {config.description}")
    print(f"Category: {config.category}")
    print(f"Key settings: TV={config.tv_weight:.0e}, layers={config.layer_weights or 'uniform'}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return {"name": config.name, "category": config.category, "status": "skipped"}
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,
        )
        metrics = parse_metrics(exp_dir / "metrics.txt")
        metrics["name"] = config.name
        metrics["category"] = config.category
        metrics["status"] = "success"
        metrics["output_dir"] = str(exp_dir)
        metrics["tv_weight"] = config.tv_weight
        metrics["layer_weights"] = config.layer_weights or "uniform"
        
        # Save config alongside results
        with open(exp_dir / "experiment_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed: {e}")
        return {"name": config.name, "category": config.category, "status": "failed", "error": str(e)}
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return {"name": config.name, "category": config.category, "status": "error", "error": str(e)}


def parse_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Parse metrics.txt into a dictionary."""
    metrics = {}
    if not metrics_path.exists():
        return metrics
    with open(metrics_path) as f:
        for line in f:
            if ":" in line:
                key, val = line.split(":", 1)
                key = key.strip()
                val = val.strip()
                try:
                    if "." in val:
                        metrics[key] = float(val)
                    else:
                        metrics[key] = int(val)
                except ValueError:
                    metrics[key] = val
    return metrics


def generate_report(results: List[Dict[str, Any]], output_dir: Path):
    """Generate comprehensive Phase 1 results report."""
    report_path = output_dir / "PHASE1_REPORT.md"
    
    # Group by category
    categories = {}
    for r in results:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    with open(report_path, "w") as f:
        f.write("# Phase 1 — Baseline Improvement Results\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary table
        f.write("## Summary Table\n\n")
        f.write("| Experiment | TV Weight | Layer Weights | PSNR | SSIM | MSE | LPIPS | Status |\n")
        f.write("|------------|-----------|---------------|------|------|-----|-------|--------|\n")
        
        successful = [r for r in results if r.get("status") == "success"]
        for r in sorted(successful, key=lambda x: -x.get("PSNR", 0)):
            psnr = r.get("PSNR", "N/A")
            ssim = r.get("SSIM", "N/A")
            mse = r.get("MSE", "N/A")
            lpips = r.get("LPIPS", "N/A")
            tv = r.get("tv_weight", "N/A")
            lw = r.get("layer_weights", "uniform")
            
            psnr_str = f"{psnr:.2f}" if isinstance(psnr, (int, float)) else str(psnr)
            ssim_str = f"{ssim:.3f}" if isinstance(ssim, (int, float)) else str(ssim)
            mse_str = f"{mse:.4f}" if isinstance(mse, (int, float)) else str(mse)
            lpips_str = f"{lpips:.4f}" if isinstance(lpips, (int, float)) else str(lpips)
            tv_str = f"{tv:.0e}" if isinstance(tv, float) else str(tv)
            
            f.write(f"| {r['name']} | {tv_str} | {lw} | {psnr_str} | {ssim_str} | {mse_str} | {lpips_str} | ✓ |\n")
        
        # Add failed experiments
        failed = [r for r in results if r.get("status") != "success"]
        for r in failed:
            f.write(f"| {r['name']} | - | - | - | - | - | - | {r.get('status', 'unknown')} |\n")
        
        f.write("\n")
        
        # Category analysis
        f.write("## Analysis by Category\n\n")
        
        for cat, cat_results in categories.items():
            f.write(f"### {cat.replace('_', ' ').title()}\n\n")
            successful_cat = [r for r in cat_results if r.get("status") == "success"]
            if successful_cat:
                best = max(successful_cat, key=lambda x: x.get("PSNR", 0))
                f.write(f"**Best in category:** {best['name']}\n")
                psnr_val = best.get('PSNR', 'N/A')
                ssim_val = best.get('SSIM', 'N/A')
                lpips_val = best.get('LPIPS', 'N/A')
                psnr_str = f"{psnr_val:.2f}" if isinstance(psnr_val, (int, float)) else str(psnr_val)
                ssim_str = f"{ssim_val:.3f}" if isinstance(ssim_val, (int, float)) else str(ssim_val)
                lpips_str = f"{lpips_val:.4f}" if isinstance(lpips_val, (int, float)) and lpips_val == lpips_val else 'N/A'
                f.write(f"- PSNR: {psnr_str}\n")
                f.write(f"- SSIM: {ssim_str}\n")
                f.write(f"- LPIPS: {lpips_str}\n")
                f.write("\n")
        
        # Best overall
        if successful:
            f.write("## Best Overall Configuration\n\n")
            best_psnr = max(successful, key=lambda x: x.get("PSNR", 0))
            best_ssim = max(successful, key=lambda x: x.get("SSIM", 0))
            best_lpips = min([r for r in successful if isinstance(r.get("LPIPS"), (int, float))], 
                            key=lambda x: x.get("LPIPS", float('inf')), default=None)
            
            f.write("| Metric | Best Experiment | Value |\n")
            f.write("|--------|-----------------|-------|\n")
            f.write(f"| PSNR | {best_psnr['name']} | {best_psnr.get('PSNR', 'N/A'):.2f} |\n")
            f.write(f"| SSIM | {best_ssim['name']} | {best_ssim.get('SSIM', 'N/A'):.3f} |\n")
            if best_lpips:
                f.write(f"| LPIPS (lower=better) | {best_lpips['name']} | {best_lpips.get('LPIPS', 'N/A'):.4f} |\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on Phase 1 results:\n\n")
            f.write(f"1. **Best PSNR:** Use TV={best_psnr.get('tv_weight', 1e-6):.0e}, ")
            f.write(f"layer_weights={best_psnr.get('layer_weights', 'uniform')}\n")
            f.write(f"2. **Best SSIM:** Use TV={best_ssim.get('tv_weight', 1e-6):.0e}, ")
            f.write(f"layer_weights={best_ssim.get('layer_weights', 'uniform')}\n")
            if best_lpips:
                f.write(f"3. **Best Perceptual Quality:** Use TV={best_lpips.get('tv_weight', 1e-6):.0e}, ")
                f.write(f"layer_weights={best_lpips.get('layer_weights', 'uniform')}\n")
        
        f.write("\n## Success Criteria Check\n\n")
        baseline_psnr = 25.83
        baseline_ssim = 0.894
        
        if successful:
            best = max(successful, key=lambda x: x.get("PSNR", 0))
            psnr_pass = best.get("PSNR", 0) >= baseline_psnr
            ssim_pass = best.get("SSIM", 0) >= baseline_ssim - 0.01  # Allow small variance
            
            f.write(f"- [{'x' if psnr_pass else ' '}] PSNR ≥ {baseline_psnr:.2f} (baseline): ")
            f.write(f"Best achieved: {best.get('PSNR', 0):.2f}\n")
            f.write(f"- [{'x' if ssim_pass else ' '}] SSIM ≈ or > {baseline_ssim:.3f}: ")
            f.write(f"Best achieved: {best.get('SSIM', 0):.3f}\n")
    
    print(f"\n[INFO] Report saved to: {report_path}")
    return report_path


def save_summary_csv(results: List[Dict[str, Any]], output_dir: Path):
    """Save results as CSV for easy analysis."""
    csv_path = output_dir / "phase1_summary.csv"
    
    fieldnames = ["name", "category", "status", "tv_weight", "layer_weights", 
                  "PSNR", "SSIM", "MSE", "LPIPS", "LabelMatch", "output_dir"]
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    
    print(f"[INFO] CSV summary saved to: {csv_path}")
    return csv_path


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 1 — Baseline Improvement Experiments")
    parser.add_argument("--mode", type=str, default="all",
                       choices=["all", "tv-sweep", "layer-ablation", "combined", "baseline"],
                       help="Which experiments to run")
    parser.add_argument("--dry-run", action="store_true", help="Preview commands without running")
    parser.add_argument("--clean", action="store_true", help="Remove existing results first")
    parser.add_argument("--device", type=str, default="mps", help="Device (mps/cuda/cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--data-subset", type=int, default=200, help="Data subset size")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of FL clients")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    
    args = parser.parse_args()
    
    # Setup output directory
    if args.out_dir:
        output_dir = Path(args.out_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("results") / f"phase1_{timestamp}"
    
    if args.clean and output_dir.exists():
        print(f"[INFO] Cleaning existing results at: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Output directory: {output_dir}")
    
    # Select experiments based on mode
    if args.mode == "tv-sweep":
        configs = get_tv_sweep_configs()
    elif args.mode == "layer-ablation":
        configs = get_layer_weighting_configs()
    elif args.mode == "combined":
        configs = get_combined_best_configs()
    elif args.mode == "baseline":
        configs = [get_baseline_reference()]
    else:  # all
        configs = get_all_configs()
    
    print(f"\n[INFO] Running {len(configs)} experiments in mode: {args.mode}")
    
    # Base flags that apply to all experiments
    base_flags = [
        "--device", args.device,
        "--seed", str(args.seed),
        "--data-subset", str(args.data_subset),
        "--num-clients", str(args.num_clients),
        "--local-epochs", "1",
        "--client-lr", "0.01",
    ]
    
    # Run experiments
    results = []
    for i, config in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Running: {config.name}")
        result = run_experiment(config, output_dir, base_flags, dry_run=args.dry_run)
        results.append(result)
        
        # Save incremental progress
        if not args.dry_run:
            with open(output_dir / "progress.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
    
    # Generate reports
    if not args.dry_run:
        generate_report(results, output_dir)
        save_summary_csv(results, output_dir)
        
        # Save full results as JSON
        with open(output_dir / "phase1_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE")
    print("="*70)
    
    successful = [r for r in results if r.get("status") == "success"]
    failed = [r for r in results if r.get("status") != "success"]
    
    print(f"Total experiments: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful and not args.dry_run:
        best = max(successful, key=lambda x: x.get("PSNR", 0))
        print(f"\nBest result: {best['name']}")
        psnr_val = best.get('PSNR', 'N/A')
        ssim_val = best.get('SSIM', 'N/A')
        lpips_val = best.get('LPIPS', 'N/A')
        mse_val = best.get('MSE', 'N/A')
        psnr_str = f"{psnr_val:.2f}" if isinstance(psnr_val, (int, float)) else str(psnr_val)
        ssim_str = f"{ssim_val:.3f}" if isinstance(ssim_val, (int, float)) else str(ssim_val)
        lpips_str = f"{lpips_val:.4f}" if isinstance(lpips_val, (int, float)) else str(lpips_val)
        mse_str = f"{mse_val:.4f}" if isinstance(mse_val, (int, float)) else str(mse_val)
        print(f"  PSNR: {psnr_str}")
        print(f"  SSIM: {ssim_str}")
        print(f"  LPIPS: {lpips_str}")
        print(f"  MSE: {mse_str}")
        print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
