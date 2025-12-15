# Federated Learning Baseline + Gradient Inversion (R&D)

This project demonstrates how a standard federated learning baseline (no differential privacy or homomorphic encryption) leaks training examples via gradient‑based image reconstruction (DLG/iDLG) and a one‑step update variant. CIFAR‑10 is used and downloaded automatically to `./data`.

## Attack/Signal Variants

- One‑step update attack: reconstruct from a single local SGD step by approximating gradients as −Δ/η (server sees model deltas). Activate with `--attack-source one_step_update`.
- Multi‑restarts + optimizer variants: randomized restarts, Adam or LBFGS, tunable TV weight and clamping to improve recon quality.
- Batch‑size sensitivity: sweep `--batch-size` in {1, 2, 4, 8} to show sharp quality drop beyond 1.
- Label leakage toggles: compare with/without iDLG label inference to isolate label leakage contribution (baseline uses iDLG‑style inference).
- Aggregated signal (future): attempt from FedAvg‑aggregated updates (start with one local step, no momentum; approximate average grad via −Δavg/η).

## FL/Training Controls

- Local optimizer knobs: vary client LR, momentum, and local epochs; momentum/more epochs obfuscate single‑sample signals.
- Data augmentation toggle: `--no-augment` to disable augmentation; augmentation typically degrades reconstruction.
- IID vs non‑IID (future): add non‑IID partitioning to study heterogeneity effects.

## Measurement & Visualization

- Metrics: MSE and PSNR are saved; SSIM/LPIPS can be added if needed.
- Visuals: side‑by‑side original vs reconstruction saved per run.
- Failure modes: explore degradation vs batch size, momentum, epochs, augmentation.

## Reproducibility & UX

- CLI‑driven runs: device, seeds, attack iterations/LR, TV weight, restarts, capture mode.
- Artifacts: timestamped results folder with images and metrics.
- Config snapshots (future): dump full args/env for exact repro.
- Experiment sweeps (future): small matrix runner to produce summary plots.

## Model/Architecture Levers

- Pooling/BN toggles: AvgPool vs MaxPool, with/without BatchNorm/Dropout to test sensitivity.
- Datasets: CIFAR‑10 (current), consider MNIST (easier) and Tiny‑ImageNet (scale).

## What’s Implemented Now

- CLI for experiments, output dirs, and seeds (`run_experiment.py`).
- Capture modes: raw `gradients` and `one_step_update` (grads ≈ −Δ/η) (`fl_system.py`, `gradient_attack.py`).
- Multi‑restarts, Adam/LBFGS, tunable TV/clamp (`gradient_attack.py`).
- Client optimizer/epochs and augmentation toggles (`fl_system.py`, `run_experiment.py`).
- Results and metrics saved under `results/<timestamp>/` (`run_experiment.py`).

## Quick Usage

- Baseline (batch size 1, gradients, Adam, TV):
  - `python run_experiment.py --batch-size 1 --attack-iterations 2000 --attack-restarts 3 --tv-weight 0.001 --attack-optimizer adam`
- One‑step update attack (no momentum, single epoch):
  - `python run_experiment.py --attack-source one_step_update --client-momentum 0.0 --local-epochs 1`
- Show augmentation impact:
  - `python run_experiment.py --no-augment --batch-size 1`
- Stress with larger batch sizes:
  - `python run_experiment.py --batch-size 4 --attack-iterations 4000 --attack-restarts 5`

## Outputs

- Saved under `results/<timestamp>/`:
  - `baseline_attack_result.png` — original vs reconstructed image
  - `metrics.txt` — MSE, PSNR, label‑match flag

## Requirements

- Python 3.9+
- PyTorch + torchvision, matplotlib
- Install: `pip install torch torchvision matplotlib`
