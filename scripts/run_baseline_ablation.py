#!/usr/bin/env python3
"""
Automates the baseline + ablation sweep described in the research plan.

It escalates through:
    1. Warm-up sanity run (verifies pipeline end-to-end).
    2. Client hyperparameter sweep (momentum × local epochs).
    3. Attacker-depth sweep on the top client configs.
    4. Batch-size degradation study.
    5. Signal-type comparison (gradients vs one-step update).
    6. Extended-round leakage check.

Each run stores outputs in a structured directory tree and appends a CSV summary
for downstream plotting/analysis.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import itertools
import json
import math
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the baseline ablation suite for FL + gradient attacks."
    )
    parser.add_argument("--python-bin", default=sys.executable, help="Python binary to invoke.")
    parser.add_argument("--device", default=None, help="Forwarded to run_experiment (cuda|mps|cpu).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-clients", type=int, default=6)
    parser.add_argument("--small-subset", type=int, default=200, help="Quick sanity subset size.")
    parser.add_argument("--large-subset", type=int, default=1000, help="Richer sweep subset size.")
    parser.add_argument("--base-out-dir", default=None, help="Root directory for all runs.")
    parser.add_argument("--compute-lpips", action="store_true", help="Enable LPIPS on final configs.")
    return parser.parse_args()


def timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def sanitize_label(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", text)


def build_cli_args(cfg: Dict[str, object]) -> List[str]:
    args: List[str] = []
    for key, value in cfg.items():
        flag = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                args.append(flag)
            continue
        if value is None:
            continue
        args.extend([flag, str(value)])
    return args


def parse_metrics_file(path: Path) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    if not path.exists():
        return metrics
    with path.open() as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.strip().split(":", 1)
            key = key.strip()
            val = val.strip()
            try:
                metrics[key] = float(val)
            except ValueError:
                metrics[key] = math.nan if val.lower() == "nan" else val
    return metrics


ACCURACY_RE = re.compile(r"Accuracy\s*=\s*([0-9.]+)%")


def extract_accuracy(log: str) -> Optional[float]:
    matches = ACCURACY_RE.findall(log)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def run_once(
    python_bin: str,
    cfg: Dict[str, object],
    out_dir: Path,
) -> Tuple[subprocess.CompletedProcess[str], float]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cli_args = build_cli_args({**cfg, "out_dir": str(out_dir)})
    cmd = [python_bin, "run_experiment.py"] + cli_args
    print(f"\n[RUN] {' '.join(cmd)}")
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    elapsed = time.perf_counter() - start
    if proc.stdout:
        print(proc.stdout)
    if proc.stderr:
        print(proc.stderr, file=sys.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"run_experiment failed (exit={proc.returncode})")
    return proc, elapsed


def result_score(metrics: Dict[str, float]) -> Tuple[float, float, float, float]:
    def safe_get(name: str, default: float) -> float:
        value = metrics.get(name, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    label_match = safe_get("LabelMatch", -1.0)
    ssim = safe_get("SSIM", float("-inf"))
    psnr = safe_get("PSNR", float("-inf"))
    mse = safe_get("MSE", float("inf"))
    return (label_match, ssim, psnr, -mse)


def select_top(results: Sequence[Dict], top_k: int = 1) -> List[Dict]:
    ordered = sorted(results, key=lambda r: result_score(r["metrics"]), reverse=True)
    return list(ordered[:top_k])


def strip_out_dir(cfg: Dict[str, object]) -> Dict[str, object]:
    clean = dict(cfg)
    clean.pop("out_dir", None)
    return clean


def summarize(
    stage: str,
    label: str,
    cfg: Dict[str, object],
    proc: subprocess.CompletedProcess[str],
    runtime_sec: float,
    out_dir: Path,
) -> Dict:
    metrics = parse_metrics_file(out_dir / "metrics.txt")
    accuracy = extract_accuracy(proc.stdout)
    record = {
        "stage": stage,
        "label": label,
        "out_dir": str(out_dir),
        "runtime_sec": runtime_sec,
        "accuracy": accuracy,
        "metrics": metrics,
        "config": dict(cfg),
        "command": proc.args,
    }
    return record


def stage_runner(
    python_bin: str,
    stage: str,
    base_out: Path,
    configs: Sequence[Tuple[str, Dict[str, object]]],
) -> List[Dict]:
    stage_results: List[Dict] = []
    for label, cfg in configs:
        sub_dir = base_out / stage / sanitize_label(label)
        proc, elapsed = run_once(python_bin, cfg, sub_dir)
        record = summarize(stage, label, cfg, proc, elapsed, sub_dir)
        stage_results.append(record)
    return stage_results


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_out_dir or f"results/baseline_ablation_{timestamp()}")
    base_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing results under {base_dir}")

    base_cfg: Dict[str, object] = {
        "seed": args.seed,
        "num_clients": args.num_clients,
        "batch_size": 1,
        "data_subset": args.small_subset,
        "num_rounds": 1,
        "local_epochs": 1,
        "client_lr": 0.05,
        "client_momentum": 0.0,
        "capture_client": 0,
        "capture_round": 0,
        "attack_source": "gradients",
        "attack_iterations": 2000,
        "attack_lr": 0.1,
        "attack_optimizer": "adam",
        "attack_restarts": 1,
        "attack_batch": 1,
        "label_strategy": "auto",
        "tv_weight": 5e-4,
        "no_augment": True,
    }
    if args.device:
        base_cfg["device"] = args.device

    all_results: List[Dict] = []

    # Stage 1: Warm-up sanity.
    warmup_cfgs = [
        (
            "warmup_m0.0_ep1_lr0.05",
            deepcopy(base_cfg),
        )
    ]
    all_results += stage_runner(args.python_bin, "01_warmup", base_dir, warmup_cfgs)

    # Stage 2: Client hyper sweep (momentum × epochs).
    stage2_cfgs = []
    for mom, ep in itertools.product([0.0, 0.5, 0.9], [1, 2]):
        cfg = deepcopy(base_cfg)
        cfg.update({"client_momentum": mom, "local_epochs": ep})
        label = f"mom{mom}_ep{ep}"
        stage2_cfgs.append((label, cfg))
    stage2_results = stage_runner(args.python_bin, "02_client_sweep", base_dir, stage2_cfgs)
    all_results += stage2_results
    top_clients = select_top(stage2_results, top_k=2)
    print(
        "[INFO] Top client configs:",
        json.dumps(
            [
                {
                    "label": r["label"],
                    "metrics": r["metrics"],
                    "accuracy": r["accuracy"],
                }
                for r in top_clients
            ],
            indent=2,
        ),
    )

    # Stage 3: Attacker-depth sweep on top client configs with larger subset.
    attack_grid = [
        {"attack_iterations": 2000, "attack_restarts": 1, "tv_weight": 5e-4, "attack_optimizer": "adam"},
        {"attack_iterations": 2000, "attack_restarts": 3, "tv_weight": 5e-4, "attack_optimizer": "adam"},
        {"attack_iterations": 4000, "attack_restarts": 1, "tv_weight": 1e-3, "attack_optimizer": "adam"},
        {"attack_iterations": 4000, "attack_restarts": 3, "tv_weight": 1e-3, "attack_optimizer": "lbfgs"},
    ]
    stage3_cfgs = []
    for client_res in top_clients:
        client_cfg = strip_out_dir(client_res["config"])
        client_cfg.update({"data_subset": args.large_subset})
        for idx, atk in enumerate(attack_grid):
            cfg = deepcopy(client_cfg)
            cfg.update(atk)
            label = f"{client_res['label']}_atk{idx}"
            stage3_cfgs.append((label, cfg))
    stage3_results = stage_runner(args.python_bin, "03_attack_depth", base_dir, stage3_cfgs)
    all_results += stage3_results
    best_config = select_top(stage3_results, top_k=1)[0]
    print(
        "[INFO] Best baseline candidate:",
        json.dumps(
            {
                "label": best_config["label"],
                "stage": best_config["stage"],
                "metrics": best_config["metrics"],
                "accuracy": best_config["accuracy"],
            },
            indent=2,
        ),
    )

    best_base_cfg = strip_out_dir(best_config["config"])

    # Stage 4: Batch-size degradation study.
    batch_cfg = deepcopy(best_base_cfg)
    batch_cfg.update(
        {
            "batch_size": 2,
            "attack_batch": 2,
            "label_strategy": "optimize",
            "attack_iterations": 4000,
            "attack_restarts": 3,
        }
    )
    stage4_results = stage_runner(
        args.python_bin,
        "04_batch_effect",
        base_dir,
        [("batch2_attack2", batch_cfg)],
    )
    all_results += stage4_results

    # Stage 5: Signal-type comparison (gradients vs one_step_update).
    signal_cfgs = []
    grad_cfg = deepcopy(best_base_cfg)
    if args.compute_lpips:
        grad_cfg["compute_lpips"] = True
    signal_cfgs.append(("gradients_ref", grad_cfg))
    one_step_cfg = deepcopy(grad_cfg if args.compute_lpips else best_base_cfg)
    one_step_cfg.update(
        {
            "attack_source": "one_step_update",
            "client_momentum": 0.0,
            "local_epochs": 1,
        }
    )
    if args.compute_lpips and "compute_lpips" not in one_step_cfg:
        one_step_cfg["compute_lpips"] = True
    signal_cfgs.append(("one_step_update", one_step_cfg))
    stage5_results = stage_runner(args.python_bin, "05_signal_type", base_dir, signal_cfgs)
    all_results += stage5_results

    # Stage 6: Extended rounds leakage check.
    extended_cfg = deepcopy(best_base_cfg)
    extended_cfg.update(
        {
            "num_rounds": 5,
            "capture_round": 4,
        }
    )
    if args.compute_lpips:
        extended_cfg["compute_lpips"] = True
    stage6_results = stage_runner(
        args.python_bin,
        "06_steady_state",
        base_dir,
        [("rounds5_capture4", extended_cfg)],
    )
    all_results += stage6_results

    # Persist summary CSV.
    summary_path = base_dir / "ablation_summary.csv"
    summary_fields = [
        "stage",
        "label",
        "out_dir",
        "runtime_sec",
        "accuracy",
        "MSE",
        "PSNR",
        "SSIM",
        "LabelMatch",
        "command",
    ]
    with summary_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=summary_fields)
        writer.writeheader()
        for record in all_results:
            metrics = record.get("metrics", {})
            cmd = record.get("command")
            cmd_str = " ".join(cmd) if isinstance(cmd, list) else str(cmd)
            writer.writerow(
                {
                    "stage": record.get("stage"),
                    "label": record.get("label"),
                    "out_dir": record.get("out_dir"),
                    "runtime_sec": f"{record.get('runtime_sec', 0.0):.2f}",
                    "accuracy": record.get("accuracy"),
                    "MSE": metrics.get("MSE"),
                    "PSNR": metrics.get("PSNR"),
                    "SSIM": metrics.get("SSIM"),
                    "LabelMatch": metrics.get("LabelMatch"),
                    "command": cmd_str,
                }
            )

    print(f"[DONE] Summary written to {summary_path}")


if __name__ == "__main__":
    main()
