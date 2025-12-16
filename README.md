# Federated Learning Gradient Inversion + Defenses

Compact R&D sandbox showing how a simple FL setup on CelebA leaks training examples via gradient-based reconstruction (DLG/iDLG, one-step, aggregated updates), and how basic defenses (DP and HE) change the outcome. Apple M‑series/MPS is supported.

## Data
- Place images in `data/img_align_celeba/` (nested `data/img_align_celeba/img_align_celeba/` also works).
- Put `data/list_attr_celeba.csv` and `data/list_eval_partition.csv` alongside images.

## What’s Included
- Attack signals: `gradients`, `one_step_update`, `agg_update` (FedAvg delta approximation).
- Attacker: multi-restarts, Adam/LBFGS/SGD/AdamW, TV/clamp, cosine LR, early stopping, FFT init, layer selection/weights, batch>1 with soft labels.
- Defenses: Differential Privacy (clip + Gaussian noise) and Homomorphic Encryption (Paillier‑like, fixed‑point, optional encrypted noise) applied to the captured signal before reconstruction.
- Automation: curated experiment suites and analysis scripts for showcase, ablations, multi‑client, and defenses.

## Install
- Python 3.9+
- PyTorch, torchvision, matplotlib (optional: `lpips` for perceptual metric)
- Example: `pip install torch torchvision matplotlib lpips`

## Quick Start
- Best single‑face “showcase” (recommended): `bash scripts/run_showcase.sh`
- Fast benchmark config: `bash scripts/run_benchmark.sh`
- Attack many clients + summarize: `bash scripts/run_multi_client.sh` then `python scripts/analyze_multi_client.py --results-dir results/multi_client`

Or drive directly:
- Baseline (batch=1, gradients): `python run_experiment.py --num-rounds 1 --capture-round 0 --batch-size 1 --client-momentum 0.0 --attack-iterations 3000 --attack-restarts 3 --tv-weight 1e-5 --lr-schedule cosine --compute-lpips`
- One‑step update: `python run_experiment.py --attack-source one_step_update --client-momentum 0.0 --local-epochs 1`
- Aggregated update: `python run_experiment.py --attack-source agg_update --capture-client 0 --client-momentum 0.0 --local-epochs 1`

## Privacy Defenses (flags)
- Differential Privacy: `--dp-epsilon <ε> --dp-delta <δ> --dp-max-norm <L2>`
- Homomorphic Encryption: `--use-he [--he-bits 512 --he-precision 1000000]`
- End‑to‑end defense sweeps: `bash scripts/run_defenses.sh` and analyze with `python scripts/analyze_defenses.py --results-dir results/defenses`

## Results & Outputs
- Saved under `results/<name>/` or your `--out-dir`:
  - `baseline_attack_result.png` – original vs reconstruction (plus heatmap row if ground truth available)
  - `metrics.txt` – MSE, PSNR, SSIM, optional LPIPS, LabelMatch
  - `config.json` – exact run arguments

## Repo Map
- `fl_system.py` – minimal CelebA FL loop + capture modes
- `gradient_attack.py` – gradient inversion core and options
- `Differential_privacy.py` – clip, noise calibration, DP aggregation helpers
- `homomorphic_encryptor.py` – lightweight additive HE for vectors
- `run_experiment.py` – CLI for end‑to‑end runs (attacks + defenses)
- `scripts/exp_base.py` – curated baseline suite (showcase + ablations + report)
- `scripts/exp_phase1.py` – refinement sweeps (TV, layer weighting, perceptual)
- `scripts/run_*.sh` – convenience wrappers for common experiment sets

Notes
- CelebA is loaded as 64×64 crops normalized with mean/std (0.5, 0.5, 0.5).
- Default task is a binary attribute (e.g., `Male`).
