"""Shared subprocess-execution helpers for the exp_*.py experiment scripts.

Kept deliberately small: only the parts of run_experiment()/parse_metrics()
that were byte-for-byte identical (or safely unifiable) between exp_base.py
and exp_phase1.py live here. Each script's own config dataclass, flag
building, and report generation stay separate — they differ enough that
forcing a shared abstraction isn't worth it, especially since neither
script can be run end-to-end in an environment without GPU/CelebA to
re-verify the merge didn't change behavior.
"""
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def run_experiment_subprocess(cmd: List[str], dry_run: bool) -> Tuple[str, Optional[str]]:
    """Run one experiment's run_experiment.py invocation.

    Returns (status, error) where status is one of
    "skipped" / "success" / "failed" / "error", and error is None unless
    status is "failed" or "error".
    """
    if dry_run:
        print("[DRY RUN] Skipping execution")
        return "skipped", None

    try:
        subprocess.run(cmd, check=True, capture_output=False)
        return "success", None
    except subprocess.CalledProcessError as e:
        print(f"[ERROR] Experiment failed: {e}")
        return "failed", str(e)
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        return "error", str(e)


def parse_metrics_file(path: Path) -> Dict[str, Any]:
    """Parse a metrics.txt file ('key: value' per line) into a dict,
    converting each value to int (no '.') or float where possible."""
    metrics: Dict[str, Any] = {}
    if not path.exists():
        return metrics
    with open(path) as f:
        for line in f:
            if ":" not in line:
                continue
            key, val = line.split(":", 1)
            key, val = key.strip(), val.strip()
            try:
                metrics[key] = int(val) if "." not in val else float(val)
            except ValueError:
                metrics[key] = val
    return metrics


def build_experiment_command(config_name: str, exp_dir: Path,
                              base_flags: List[str], config_flags: List[str]) -> List[str]:
    """Build the run_experiment.py subprocess command shared by both scripts."""
    return (
        [sys.executable, "run_experiment.py", "--out-dir", str(exp_dir), "--save-config"]
        + base_flags
        + config_flags
    )
