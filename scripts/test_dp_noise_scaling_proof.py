"""Tests for the DP noise-vs-dimensionality synthetic proof (no torch needed)."""
import math

from dp_noise_scaling_proof import gaussian_sigma, expected_noise_l2_norm


def test_gaussian_sigma_matches_analytic_formula():
    # sigma = max_norm * sqrt(2 ln(1.25/delta)) / epsilon
    sigma = gaussian_sigma(epsilon=8.0, delta=1e-5, max_norm=1.0)
    expected = math.sqrt(2 * math.log(1.25 / 1e-5)) / 8.0
    assert abs(sigma - expected) < 1e-9


def test_gaussian_sigma_at_tested_epsilons():
    # The three epsilons actually used in results/report/summary.csv.
    sigma_8 = gaussian_sigma(epsilon=8.0, delta=1e-5, max_norm=1.0)
    sigma_1 = gaussian_sigma(epsilon=1.0, delta=1e-5, max_norm=1.0)
    sigma_01 = gaussian_sigma(epsilon=0.1, delta=1e-5, max_norm=1.0)
    assert 0.60 < sigma_8 < 0.61
    assert 4.8 < sigma_1 < 4.9
    assert 48.0 < sigma_01 < 49.0


def test_noise_norm_scales_with_sqrt_d():
    sigma = 1.0
    norm_at_1 = expected_noise_l2_norm(sigma, d=1)
    norm_at_100 = expected_noise_l2_norm(sigma, d=100)
    assert abs(norm_at_1 - 1.0) < 1e-9
    assert abs(norm_at_100 - 10.0) < 1e-9


def test_model_dimensionality_dominates_signal_at_every_tested_epsilon():
    # d = 8,760,962 is SimpleCNN's actual parameter count (fl_system.py).
    d = 8_760_962
    max_norm = 1.0
    for epsilon in (8.0, 1.0, 0.1):
        sigma = gaussian_sigma(epsilon=epsilon, delta=1e-5, max_norm=max_norm)
        noise_norm = expected_noise_l2_norm(sigma, d)
        # Even the weakest tested epsilon (8.0) buries the clipped signal.
        assert noise_norm > 1000 * max_norm
