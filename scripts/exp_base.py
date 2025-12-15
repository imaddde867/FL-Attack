#!/usr/bin/env python3
"""
Baseline Experimentation Suite for Federated Learning Gradient Inversion Attack
================================================================================

A clean, concise, and efficient script for producing:
1. High-quality baseline reconstruction (showcase result)
2. Strategic ablations for diagrams and comparison studies

Optimized for 16GB M4 Mac with MPS acceleration.

KEY INSIGHTS FOR HIGH-QUALITY RECONSTRUCTION:
- Use UNTRAINED model (num_rounds=1, capture_round=0) - gradients leak most from fresh models
- Use LBFGS optimizer - much better convergence than Adam for this problem
- Low TV weight (1e-4 to 1e-5) - avoid over-smoothing
- Many iterations (8000+) without early stopping for best quality
- batch_size=1 and momentum=0 are critical

Run Order (efficient memory/time):
1. Quick validation run (sanity check)
2. High-quality baseline (final showcase image)
3. Defense-relevant ablations (batch size, momentum effects)

Usage:
    python scripts/exp_base.py                    # Full suite
    python scripts/exp_base.py --mode showcase    # Just the best result
    python scripts/exp_base.py --mode ablation    # Just ablations
    python scripts/exp_base.py --dry-run          # Preview commands
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

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    name: str
    description: str
    # FL settings
    batch_size: int = 1
    client_momentum: float = 0.0
    attack_source: str = "gradients"
    num_rounds: int = 1          # KEY: fewer rounds = untrained model = better attack
    capture_round: int = 0       # Capture from first round (untrained model)
    # Attack settings (higher = better quality but slower)
    attack_iterations: int = 2000
    attack_restarts: int = 3
    attack_lr: float = 1.0       # Higher LR for LBFGS
    attack_optimizer: str = "lbfgs"  # LBFGS is much better for gradient inversion
    tv_weight: float = 1e-4      # Low TV to avoid over-smoothing
    # Enhanced settings for quality
    lr_schedule: str = "none"    # No schedule for LBFGS
    early_stop: bool = False     # Let it converge fully for best quality
    patience: int = 600
    preset: str = "none"         # No preset - use raw optimization
    match_metric: str = "l2"     # Pure L2 matching works well with LBFGS
    layer_weights: str = "auto"
    fft_init: bool = False       # Random init works better with LBFGS
    # Flags
    is_showcase: bool = False
    priority: int = 0  # Lower = runs first

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
            "--layer-weights", self.layer_weights,
        ]
        # Only pass preset if it's not "none" - otherwise it overrides tv_weight
        if self.preset and self.preset.lower() not in ("none", ""):
            flags.extend(["--preset", self.preset])
        if self.early_stop:
            flags.append("--early-stop")
        if self.fft_init:
            flags.append("--fft-init")
        return flags


# ============================================================================
# Experiment Definitions
# ============================================================================

def get_showcase_config() -> ExperimentConfig:
    """
    High-quality showcase configuration.
    
    KEY SETTINGS FOR BEST VISUAL QUALITY:
    - num_rounds=1, capture_round=0: Attack untrained model (gradients leak most)
    - LBFGS optimizer: Superior convergence for this optimization problem
    - Low TV weight (1e-5): Avoid over-smoothing that causes blur
    - Many iterations (8000): Allow full convergence
    - No early stopping: Let it run to completion
    - Multiple restarts: Best of 5 random initializations
    """
    return ExperimentConfig(
        name="showcase_best",
        description="High-quality baseline attack - optimal settings for visual results",
        # CRITICAL: Attack untrained model for maximum gradient leakage
        num_rounds=1,
        capture_round=0,
        # Optimal FL settings for successful gradient inversion
        batch_size=1,           # Critical: single sample for iDLG
        client_momentum=0.0,    # No momentum = clean gradients
        attack_source="gradients",  # Direct gradients (most information)
        # Aggressive attack settings for best quality
        attack_iterations=8000,     # Many iterations for full convergence
        attack_restarts=5,          # Multiple restarts for robustness
        attack_optimizer="lbfgs",   # LBFGS is superior for gradient inversion
        attack_lr=1.0,              # Higher LR works with LBFGS
        tv_weight=1e-5,             # Very low TV to preserve details
        lr_schedule="none",         # No LR decay with LBFGS
        early_stop=False,           # Run to completion for best quality
        preset="none",              # No clamping preset
        match_metric="l2",          # Pure L2 works well with LBFGS
        layer_weights="auto",       # Inverse-norm weighting
        fft_init=False,             # Random init works better with LBFGS
        is_showcase=True,
        priority=0,
    )


def get_ablation_configs() -> List[ExperimentConfig]:
    """
    Strategic ablation configurations.
    
    Selected to demonstrate:
    1. Why batch_size=1 is necessary (compare bs=1 vs bs=2)
    2. Impact of momentum on gradient quality
    3. Effect of training rounds (untrained vs trained model)
    
    All use optimal attack settings (LBFGS, low TV) to isolate FL parameter effects.
    """
    configs = []
    
    # Ablation 1: Baseline reference with optimal settings
    configs.append(ExperimentConfig(
        name="abl_baseline_optimal",
        description="Baseline: optimal attack on untrained model (bs=1, mom=0, round=0)",
        num_rounds=1,
        capture_round=0,
        batch_size=1,
        client_momentum=0.0,
        attack_source="gradients",
        attack_optimizer="lbfgs",
        attack_iterations=5000,
        attack_restarts=3,
        attack_lr=1.0,
        tv_weight=1e-5,
        early_stop=False,
        priority=1,
    ))
    
    # Ablation 2: Effect of model training (attack after 5 rounds)
    configs.append(ExperimentConfig(
        name="abl_trained_model",
        description="Ablation: attack after 5 training rounds - shows defense from training",
        num_rounds=5,
        capture_round=4,  # Last round
        batch_size=1,
        client_momentum=0.0,
        attack_source="gradients",
        attack_optimizer="lbfgs",
        attack_iterations=5000,
        attack_restarts=3,
        attack_lr=1.0,
        tv_weight=1e-5,
        early_stop=False,
        priority=2,
    ))
    
    # Ablation 3: Batch size impact (bs=2)
    configs.append(ExperimentConfig(
        name="abl_batchsize_2",
        description="Ablation: bs=2 - demonstrate batch size defense",
        num_rounds=1,
        capture_round=0,
        batch_size=2,
        client_momentum=0.0,
        attack_source="gradients",
        attack_optimizer="adam",  # Adam for batch>1
        attack_iterations=3000,
        attack_restarts=2,
        attack_lr=0.1,
        tv_weight=1e-4,
        match_metric="both",
        early_stop=True,
        patience=800,
        priority=3,
    ))
    
    # Ablation 4: Momentum impact (mom=0.9)
    configs.append(ExperimentConfig(
        name="abl_momentum_high",
        description="Ablation: momentum=0.9 - typical FL setting degrades attack",
        num_rounds=1,
        capture_round=0,
        batch_size=1,
        client_momentum=0.9,
        attack_source="gradients",
        attack_optimizer="lbfgs",
        attack_iterations=5000,
        attack_restarts=3,
        attack_lr=1.0,
        tv_weight=1e-5,
        early_stop=False,
        priority=4,
    ))
    
    # Ablation 5: Combined realistic FL scenario
    configs.append(ExperimentConfig(
        name="abl_realistic_fl",
        description="Ablation: realistic FL (bs=4, mom=0.9, trained) - typical protection",
        num_rounds=5,
        capture_round=4,
        batch_size=4,
        client_momentum=0.9,
        attack_source="gradients",
        attack_optimizer="adam",
        attack_iterations=2000,
        attack_restarts=2,
        attack_lr=0.1,
        tv_weight=1e-3,
        match_metric="both",
        early_stop=True,
        patience=600,
        priority=5,
    ))
    
    return configs


def get_quick_validation_config() -> ExperimentConfig:
    """Quick validation to ensure everything works before long runs.
    
    Uses Adam (faster than LBFGS) with fewer iterations for quick feedback.
    """
    return ExperimentConfig(
        name="validation_quick",
        description="Quick validation run - sanity check (should get PSNR > 20)",
        num_rounds=1,
        capture_round=0,
        batch_size=1,
        client_momentum=0.0,
        attack_source="gradients",
        attack_optimizer="adam",   # Adam is much faster for validation
        attack_iterations=500,      # Fewer iterations for quick check
        attack_restarts=1,
        attack_lr=0.1,              # Lower LR for Adam
        tv_weight=1e-4,
        lr_schedule="cosine",
        early_stop=True,
        patience=200,
        match_metric="both",
        priority=-1,  # First
    )


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
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}")
    
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return {"name": config.name, "status": "skipped"}
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=False,  # Let output stream to terminal
        )
        metrics = parse_metrics(exp_dir / "metrics.txt")
        metrics["name"] = config.name
        metrics["status"] = "success"
        metrics["output_dir"] = str(exp_dir)
        
        # Save config alongside results
        with open(exp_dir / "experiment_config.json", "w") as f:
            json.dump(asdict(config), f, indent=2)
        
        return metrics
        
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed: {e}")
        return {"name": config.name, "status": "failed", "error": str(e)}
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return {"name": config.name, "status": "error", "error": str(e)}


def parse_metrics(metrics_path: Path) -> Dict[str, Any]:
    """Parse metrics.txt into a dictionary."""
    metrics = {}
    if not metrics_path.exists():
        return metrics
    
    for line in metrics_path.read_text().strip().splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value
    
    return metrics


def save_summary(results: List[Dict], output_dir: Path):
    """Save experiment summary as CSV and JSON."""
    
    # CSV summary
    csv_path = output_dir / "experiment_summary.csv"
    if results:
        fieldnames = sorted(set(k for r in results for k in r.keys()))
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved CSV summary: {csv_path}")
    
    # JSON summary (for programmatic access)
    json_path = output_dir / "experiment_summary.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved JSON summary: {json_path}")


def generate_comparison_report(results: List[Dict], output_dir: Path):
    """Generate a markdown report for quick comparison."""
    
    report_path = output_dir / "RESULTS_REPORT.md"
    
    successful = [r for r in results if r.get("status") == "success"]
    
    lines = [
        "# Baseline Experimentation Results",
        "",
        f"Generated from `exp_base.py`",
        "",
        "## Summary",
        "",
        f"- Total experiments: {len(results)}",
        f"- Successful: {len(successful)}",
        f"- Failed: {len(results) - len(successful)}",
        "",
        "## Results Table",
        "",
        "| Experiment | PSNR | SSIM | MSE | Label Match |",
        "|------------|------|------|-----|-------------|",
    ]
    
    for r in successful:
        psnr = r.get("PSNR", "N/A")
        ssim = r.get("SSIM", "N/A")
        mse = r.get("MSE", "N/A")
        label = r.get("LabelMatch", "N/A")
        
        if isinstance(psnr, float):
            psnr = f"{psnr:.2f}"
        if isinstance(ssim, float):
            ssim = f"{ssim:.3f}"
        if isinstance(mse, float):
            mse = f"{mse:.4f}"
        if isinstance(label, float):
            label = f"{label:.0%}"
        
        lines.append(f"| {r['name']} | {psnr} | {ssim} | {mse} | {label} |")
    
    lines.extend([
        "",
        "## Key Findings",
        "",
        "- **Best reconstruction**: Look for highest PSNR and SSIM",
        "- **PSNR > 20 dB**: Generally recognizable reconstruction",
        "- **SSIM > 0.5**: Good structural similarity",
        "- **Batch size**: Attack quality degrades rapidly with bs > 1",
        "- **Momentum**: SGD momentum disrupts gradient information",
        "",
        "## Visualizations",
        "",
        "Check the `showcase_best/` folder for the highest quality reconstruction.",
        "",
    ])
    
    report_path.write_text("\n".join(lines))
    print(f"Generated report: {report_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Baseline FL Gradient Attack Experimentation Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["full", "showcase", "ablation", "validate"],
        default="full",
        help="Experiment mode: full (all), showcase (best only), ablation (comparisons), validate (quick test)",
    )
    
    # Output
    parser.add_argument(
        "--output-root",
        type=str,
        default="results/baseline_final",
        help="Root directory for experiment outputs",
    )
    
    # Execution
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--clean", action="store_true", help="Remove existing output directory first")
    
    # Hardware
    parser.add_argument("--device", type=str, default="mps", help="Device: mps, cuda, cpu")
    
    # FL settings (shared across experiments)
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num-clients", type=int, default=10, help="Number of FL clients")
    parser.add_argument("--data-subset", type=int, default=200, help="Data subset size")
    parser.add_argument("--client-lr", type=float, default=0.01, help="Client learning rate")
    parser.add_argument("--local-epochs", type=int, default=1, help="Local training epochs")
    
    # Quality options
    parser.add_argument("--compute-lpips", action="store_true", help="Compute LPIPS metric (requires lpips package)")
    parser.add_argument("--no-augment", action="store_true", help="Disable data augmentation")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    output_dir = Path(args.output_root)
    
    # Clean if requested
    if args.clean and output_dir.exists():
        print(f"Cleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Base flags shared by all experiments (excluding num-rounds which is per-experiment)
    base_flags = [
        "--device", args.device,
        "--seed", str(args.seed),
        "--num-clients", str(args.num_clients),
        "--data-subset", str(args.data_subset),
        "--client-lr", str(args.client_lr),
        "--local-epochs", str(args.local_epochs),
    ]
    if args.compute_lpips:
        base_flags.append("--compute-lpips")
    if args.no_augment:
        base_flags.append("--no-augment")
    
    # Collect experiments based on mode
    configs = []
    
    if args.mode == "validate":
        configs = [get_quick_validation_config()]
    elif args.mode == "showcase":
        configs = [get_showcase_config()]
    elif args.mode == "ablation":
        configs = get_ablation_configs()
    else:  # full
        configs = [
            get_quick_validation_config(),
            get_showcase_config(),
        ] + get_ablation_configs()
    
    # Sort by priority
    configs.sort(key=lambda c: c.priority)
    
    print("\n" + "="*70)
    print("BASELINE EXPERIMENTATION SUITE")
    print("="*70)
    print(f"Mode: {args.mode}")
    print(f"Output: {output_dir}")
    print(f"Device: {args.device}")
    print(f"Experiments: {len(configs)}")
    print("="*70)
    
    # Run experiments
    results = []
    for i, config in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running: {config.name}")
        result = run_experiment(config, output_dir, base_flags, args.dry_run)
        results.append(result)
        
        # Early exit on validation failure
        if config.name == "validation_quick" and result.get("status") != "success":
            print("\n[ERROR] Validation failed! Fix issues before running full suite.")
            break
    
    # Save summary
    if not args.dry_run:
        save_summary(results, output_dir)
        generate_comparison_report(results, output_dir)
    
    # Print final summary
    print("\n" + "="*70)
    print("EXPERIMENT SUITE COMPLETE")
    print("="*70)
    
    successful = [r for r in results if r.get("status") == "success"]
    print(f"Successful: {len(successful)}/{len(results)}")
    
    if successful:
        # Find best result by PSNR
        best = max(successful, key=lambda r: r.get("PSNR", 0))
        print(f"\nBest reconstruction: {best['name']}")
        print(f"  PSNR: {best.get('PSNR', 'N/A'):.2f} dB")
        print(f"  SSIM: {best.get('SSIM', 'N/A'):.3f}")
        print(f"  Output: {best.get('output_dir', 'N/A')}")
    
    print(f"\nResults saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()
