# Federated Learning Baseline + Gradient Inversion (R&D)

This project demonstrates how a standard federated learning baseline (no differential privacy or homomorphic encryption) leaks training examples via gradient‑based image reconstruction (DLG/iDLG) and a one‑step update variant. The project now uses CelebA (attributes classification; binary by default, e.g., `Male`) exclusively.

Dataset setup (CelebA):
- Place the CelebA images under `./data/img_align_celeba/` (the code also accepts the nested form `./data/img_align_celeba/img_align_celeba/`).
- Ensure the following CSV files exist in `./data/`:
  - `list_attr_celeba.csv`
  - `list_eval_partition.csv`
The loader expects these files and will raise an error if they are missing.

## Attack/Signal Variants

- One-step update attack: reconstruct from a single local SGD step by approximating gradients as −Δ/η (server sees model deltas). `--attack-source one_step_update`.
- Aggregated-update attack: capture FedAvg delta and approximate gradients via −Δavg/η to mimic secure-aggregation vantage (`--attack-source agg_update`).
- Multi-restarts + optimizer variants: randomized restarts, Adam or LBFGS, tunable TV weight/clamping.
- Batch-size sensitivity: sweep `--batch-size` and `--attack-batch` (optimizing k dummy samples) to show degradation beyond batch size 1.
- Label leakage toggles: iDLG (single-sample) vs soft-label optimization (`--label-strategy optimize`).

## FL/Training Controls

- Local optimizer knobs: vary client LR, momentum, and local epochs; momentum/more epochs obfuscate single‑sample signals.
- Data augmentation toggle: `--no-augment` to disable augmentation; augmentation typically degrades reconstruction.
- IID vs non‑IID (future): add non‑IID partitioning to study heterogeneity effects.

## Measurement & Visualization

- Metrics: MSE, PSNR, SSIM, optional LPIPS (`--compute-lpips` adds it to metrics file).
- Visuals: tiled grid (up to 4 samples) showing original vs reconstruction and predicted labels.
- Failure modes: explore degradation vs batch size, momentum, epochs, augmentation.

## Reproducibility & UX

- CLI-driven runs: device, seeds, attack iterations/LR, TV weight, restarts, capture mode, attack batch, label strategy.
- Artifact logging: timestamped results folder with images and metrics (CSV-friendly).
- Sweep runner: `python scripts/sweep_baseline.py` sweeps batch sizes × momenta × attack sources, writes `sweep_summary.csv`, and auto-plots the requested metric per momentum setting.
- Config snapshots (future): dump full args/env for exact repro.

## Model/Architecture Levers

- Pooling/BN toggles: AvgPool vs MaxPool, with/without BatchNorm/Dropout to test sensitivity.
- Datasets: CelebA (current). Future ideas: MNIST (easier) and Tiny‑ImageNet (scale).

- CLI for experiments, output dirs, seeds, attack batch, label strategy (`run_experiment.py`).
- Capture modes: raw `gradients`, `one_step_update`, and aggregated update (FedAvg delta) with metadata capture for comparisons (`fl_system.py`).
- Gradient attacker: multi-restarts, Adam/LBFGS, tunable TV/clamp, batch>1 reconstruction with soft-label optimization (`gradient_attack.py`).
- Metrics: MSE/PSNR/SSIM (+ optional LPIPS), label-match rate, tiled visualizations, timestamped artifacts (`run_experiment.py`).
- Sweep script for batch size × momentum × attack-source matrices (`scripts/sweep_baseline.py`).

## Quick Usage

- Baseline (batch size 1, gradients, Adam, TV):
  - `python run_experiment.py --batch-size 1 --attack-iterations 2000 --attack-restarts 3 --tv-weight 0.001 --attack-optimizer adam`
- One-step update attack (no momentum, single epoch):
  - `python run_experiment.py --attack-source one_step_update --client-momentum 0.0 --local-epochs 1`
- Aggregated-update attack (FedAvg delta, metadata from client 0):
  - `python run_experiment.py --attack-source agg_update --capture-client 0 --client-momentum 0.0 --local-epochs 1`
- Batch>1 reconstruction (optimize labels):
  - `python run_experiment.py --attack-batch 2 --label-strategy optimize --attack-iterations 4000`
- Show augmentation impact:
  - `python run_experiment.py --no-augment`
- Stress with larger batch sizes:
  - `python run_experiment.py --batch-size 4 --attack-iterations 4000 --attack-restarts 5`
- Sweep + plots (MSE vs batch size for each momentum):
  - `python scripts/sweep_baseline.py --batch-sizes 1 2 4 --momenta 0.0 0.9 --plot-metric MSE`

## Outputs

- Saved under `results/<timestamp>/` (or your `--out-dir`):
  - `baseline_attack_result.png` — tiled original vs reconstructed samples
  - `metrics.txt` — MSE, PSNR, SSIM, optional LPIPS, label-match rate (if labels known)
  - `sweep_summary.csv` (from sweeper) — aggregated metrics per configuration

## Requirements

- Python 3.9+
- PyTorch + torchvision, matplotlib
- Install: `pip install torch torchvision matplotlib`

Notes:
- Images are normalized with mean/std (0.5, 0.5, 0.5) for CelebA 64×64 crops.
- Default task is binary attribute classification (e.g., `Male`), adjustable in code.
