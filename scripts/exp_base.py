#!/usr/bin/env python3
"""
Baseline Experimentation Suite for Federated Learning Gradient Inversion Attack
================================================================================

A clean, concise, and efficient script for producing:
1. High-quality baseline reconstruction (showcase result)
2. Strategic ablations for diagrams and comparison studies

Optimized for 16GB M4 Mac with MPS acceleration.

Run Order (efficient memory/time):
1. Quick validation run (sanity check)
2. High-quality baseline (final showcase image)
3. Defense-relevant ablations (batch size, momentum effects)

Usage:
    python scripts/run_baseline_final.py                    # Full suite
    python scripts/run_baseline_final.py --mode showcase    # Just the best result
    python scripts/run_baseline_final.py --mode ablation    # Just ablations
    python scripts/run_baseline_final.py --dry-run          # Preview commands
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
    # Attack settings (higher = better quality but slower)
    attack_iterations: int = 2000
    attack_restarts: int = 3
    attack_lr: float = 0.1
    tv_weight: float = 1e-3
    # Enhanced settings for quality
    lr_schedule: str = "cosine"
    early_stop: bool = True
    patience: int = 600
    preset: str = "soft"
    match_metric: str = "both"
    layer_weights: str = "auto"
    fft_init: bool = False
    # Flags
    is_showcase: bool = False
    priority: int = 0  # Lower = runs first

    def to_flags(self) -> List[str]:
        """Convert to CLI flags for run_experiment.py."""
        flags = [
            "--batch-size", str(self.batch_size),
            "--client-momentum", str(self.client_momentum),
            "--attack-source", self.attack_source,
            "--attack-iterations", str(self.attack_iterations),
            "--attack-restarts", str(self.attack_restarts),
            "--attack-lr", str(self.attack_lr),
            "--tv-weight", str(self.tv_weight),
            "--lr-schedule", self.lr_schedule,
            "--patience", str(self.patience),
            "--preset", self.preset,
            "--match-metric", self.match_metric,
            "--layer-weights", self.layer_weights,
        ]
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
    
    Settings optimized for producing the best possible reconstruction
    that can be used as the "successful attack" demonstration.
    """
    return ExperimentConfig(
        name="showcase_best",
        description="High-quality baseline attack - optimal settings for visual results",
        # Optimal FL settings for successful gradient inversion
        batch_size=1,           # Critical: single sample for iDLG
        client_momentum=0.0,    # No momentum = clean gradients
        attack_source="gradients",  # Direct gradients (most information)
        # Aggressive attack settings for best quality
        attack_iterations=4000,     # Higher iterations for convergence
        attack_restarts=5,          # Multiple restarts for robustness
        attack_lr=0.1,
        tv_weight=5e-4,             # Moderate smoothing
        lr_schedule="cosine",       # Better convergence
        early_stop=True,
        patience=800,               # More patience for fine details
        preset="soft",              # Relaxed clamping
        match_metric="both",        # L2 + cosine loss
        layer_weights="auto",       # Inverse-norm weighting
        fft_init=True,              # Better initialization
        is_showcase=True,
        priority=0,
    )


def get_ablation_configs() -> List[ExperimentConfig]:
    """
    Strategic ablation configurations.
    
    Selected to demonstrate:
    1. Why batch_size=1 is necessary (compare bs=1 vs bs=2)
    2. Impact of momentum on gradient quality
    3. Attack source comparison (gradients vs updates)
    
    NOT included (wasteful):
    - bs=4+ (known to fail completely)
    - Multiple momentum values (binary comparison sufficient)
    """
    configs = []
    
    # Ablation 1: Baseline reference (same as showcase but fewer iterations)
    configs.append(ExperimentConfig(
        name="abl_baseline_reference",
        description="Baseline: bs=1, mom=0 - reference for comparisons",
        batch_size=1,
        client_momentum=0.0,
        attack_source="gradients",
        attack_iterations=2500,
        attack_restarts=3,
        priority=1,
    ))
    
    # Ablation 2: Batch size impact (bs=2 - showing degradation starts here)
    configs.append(ExperimentConfig(
        name="abl_batchsize_2",
        description="Ablation: bs=2 - demonstrate batch size sensitivity",
        batch_size=2,
        client_momentum=0.0,
        attack_source="gradients",
        attack_iterations=2000,
        attack_restarts=2,
        priority=2,
    ))
    
    # Ablation 3: Momentum impact (mom=0.9 - common FL setting)
    configs.append(ExperimentConfig(
        name="abl_momentum_0.9",
        description="Ablation: momentum=0.9 - typical FL setting degrades attack",
        batch_size=1,
        client_momentum=0.9,
        attack_source="gradients",
        attack_iterations=2000,
        attack_restarts=2,
        priority=3,
    ))
    
    # Ablation 4: One-step update (common in FedAvg)
    configs.append(ExperimentConfig(
        name="abl_source_onestep",
        description="Ablation: one_step_update source - FedAvg scenario",
        batch_size=1,
        client_momentum=0.0,
        attack_source="one_step_update",
        attack_iterations=2000,
        attack_restarts=2,
        priority=4,
    ))
    
    # Ablation 5: Combined defense factors (bs=2 + momentum)
    configs.append(ExperimentConfig(
        name="abl_combined_defense",
        description="Ablation: bs=2 + momentum=0.5 - realistic FL protection",
        batch_size=2,
        client_momentum=0.5,
        attack_source="gradients",
        attack_iterations=1500,
        attack_restarts=2,
        priority=5,
    ))
    
    return configs


def get_quick_validation_config() -> ExperimentConfig:
    """Quick validation to ensure everything works before long runs."""
    return ExperimentConfig(
        name="validation_quick",
        description="Quick validation run - sanity check",
        batch_size=1,
        client_momentum=0.0,
        attack_source="gradients",
        attack_iterations=500,
        attack_restarts=1,
        tv_weight=1e-3,
        early_stop=True,
        patience=200,
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
        f"Generated from `run_baseline_final.py`",
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
    parser.add_argument("--num-rounds", type=int, default=5, help="FL training rounds")
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
    
    # Base flags shared by all experiments
    base_flags = [
        "--device", args.device,
        "--seed", str(args.seed),
        "--num-clients", str(args.num_clients),
        "--num-rounds", str(args.num_rounds),
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
