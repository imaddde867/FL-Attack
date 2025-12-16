# Security in Federated Learning

[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue?logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **[🔗 Live Dashboard](https://imaddde867.github.io/FL-Attack/)**

A research framework for studying **gradient inversion attacks** and **privacy defenses** in federated learning systems.

![Research Poster](results/report/poster_4k.png)

## Key Findings

| Configuration | PSNR (dB) | LPIPS ↓ |
|---------------|-----------|---------|
| Baseline (no defense) | 29.38 | 0.117 |
| Differential Privacy (ε=1) | 8.12 | 0.714 |
| Homomorphic Encryption | 12.45 | 0.623 |
| **DP + HE (combined)** | **6.37** | **0.824** |

- Baseline attacks successfully reconstruct recognizable faces
- Privacy defenses significantly degrade reconstruction quality
- Combined DP+HE provides strongest protection

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run showcase experiment
bash scripts/run_showcase.sh

# Generate poster visualization
python scripts/make_poster.py

# Launch local dashboard
python -m http.server --directory docs 8000
```

## Project Structure

```text
├── run_experiment.py          # Main experiment runner
├── fl_system.py               # Federated learning simulation
├── gradient_attack.py         # DLG/iDLG attack implementation
├── differential_privacy.py    # Gaussian mechanism for DP
├── homomorphic_encryptor.py   # Paillier-like HE
├── device_utils.py            # Auto device detection
├── scripts/                   # Experiment & analysis scripts
├── results/                   # Experiment outputs
│   └── report/                # Generated reports & dashboard
├── docs/                      # GitHub Pages dashboard
└── data/                      # CelebA dataset (not included)
```

## Experiments

| Script | Description |
|--------|-------------|
| `run_showcase.sh` | Single high-quality attack demo |
| `run_multi_client.sh` | Benchmark across 10 clients |
| `run_defenses.sh` | DP/HE defense evaluation |
| `run_ablation.sh` | Attack hyperparameter study |

## Usage

```bash
# Basic attack (no defense)
python run_experiment.py --attack-iterations 3000

# With differential privacy
python run_experiment.py --dp-epsilon 1.0

# With homomorphic encryption
python run_experiment.py --use-he

# Combined defenses
python run_experiment.py --dp-epsilon 1.0 --use-he
```

See `python run_experiment.py --help` for all options.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- CelebA dataset in `data/`

## Notes

- Results are specific to this experimental setup
- DP/HE implementations are research-grade, not production-ready
- See the interactive dashboard for detailed visualizations
