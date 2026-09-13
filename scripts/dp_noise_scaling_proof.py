#!/usr/bin/env python3
"""
Synthetic proof: why the DP epsilon sweep in results/report/summary.csv is flat.

No CelebA, no GPU, no torch. Reproduces the analytic Gaussian mechanism
formula from differential_privacy.gaussian_sigma_for_dp (duplicated here,
not imported, so this script has zero dependency on torch being installed)
and shows that at this project's actual model dimensionality, injected
noise dwarfs the clipped signal at every epsilon that was tested — this is
an experimental-design finding, not a bug in the DP mechanism (see
differential_privacy.aggregate_clipped_noisy's docstring and
run_experiment.py's DP block comment for the full threat-model discussion).
"""
import math

import matplotlib.pyplot as plt
import numpy as np

# SimpleCNN's actual parameter count (fl_system.py: 3 conv blocks + 2 linear layers).
MODEL_PARAM_COUNT = 8_760_962
DELTA = 1e-5
MAX_NORM = 1.0
TESTED_EPSILONS = (8.0, 1.0, 0.1)


def gaussian_sigma(epsilon: float, delta: float, max_norm: float) -> float:
    """Analytic Gaussian mechanism noise std, same formula as
    differential_privacy.gaussian_sigma_for_dp."""
    return max_norm * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def expected_noise_l2_norm(sigma: float, d: int) -> float:
    """Expected L2 norm of a d-dimensional i.i.d. N(0, sigma^2) noise vector."""
    return sigma * math.sqrt(d)


def main() -> None:
    d_values = np.logspace(1, 8, 200)

    fig, ax = plt.subplots(figsize=(9, 6))

    for epsilon in TESTED_EPSILONS:
        sigma = gaussian_sigma(epsilon, DELTA, MAX_NORM)
        noise_norms = [expected_noise_l2_norm(sigma, d) for d in d_values]
        ax.plot(d_values, noise_norms, label=f"ε={epsilon} (σ={sigma:.3f})", linewidth=2)

    ax.axhline(MAX_NORM, color="black", linestyle="--", linewidth=1.5,
               label=f"clipped signal norm (max_norm={MAX_NORM})")
    ax.axvline(MODEL_PARAM_COUNT, color="gray", linestyle=":", linewidth=1.5,
               label=f"this model's d = {MODEL_PARAM_COUNT:,}")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("parameter count (d)")
    ax.set_ylabel("expected noise L2 norm")
    ax.set_title("Per-coordinate DP noise vs. clipped signal, by model size")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    out_path = "results/report/figures/dp_noise_scaling.png"
    fig.savefig(out_path, dpi=150)
    print(f"[INFO] Wrote {out_path}")

    print("\nAt this model's dimensionality (d={:,}):".format(MODEL_PARAM_COUNT))
    for epsilon in TESTED_EPSILONS:
        sigma = gaussian_sigma(epsilon, DELTA, MAX_NORM)
        noise_norm = expected_noise_l2_norm(sigma, MODEL_PARAM_COUNT)
        print(f"  epsilon={epsilon:<5} sigma={sigma:>10.4f}  "
              f"noise_norm={noise_norm:>12.1f}  (signal capped at {MAX_NORM})")


if __name__ == "__main__":
    main()
