"""
Differential Privacy utilities for Federated Learning.

Implements Gaussian mechanism for DP-SGD / DP-FedAvg style gradient perturbation.
"""

import math
from typing import List
import torch


def gaussian_sigma_for_dp(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """
    Compute Gaussian noise σ for (ε, δ)-differential privacy.
    
    Uses the analytic Gaussian mechanism:
        σ = sensitivity × √(2 ln(1.25/δ)) / ε
    
    Args:
        epsilon: Privacy budget (ε > 0)
        delta: Failure probability (0 < δ < 1)
        sensitivity: L2 sensitivity of the query
    
    Returns:
        Standard deviation for Gaussian noise
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon


def clip_gradients(gradients: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    """
    Clip gradient tensors to have total L2 norm ≤ max_norm.
    
    Args:
        gradients: List of gradient tensors
        max_norm: Maximum allowed L2 norm
    
    Returns:
        Clipped gradients with same structure
    """
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in gradients))
    clip_factor = min(max_norm / (total_norm + 1e-8), 1.0)
    return [g * clip_factor for g in gradients]


def add_gaussian_noise(gradients: List[torch.Tensor], sigma: float) -> List[torch.Tensor]:
    """
    Add i.i.d. Gaussian noise N(0, σ²) to each gradient tensor.
    
    Args:
        gradients: List of gradient tensors
        sigma: Standard deviation of noise
    
    Returns:
        Noisy gradients
    """
    return [g + torch.randn_like(g) * sigma for g in gradients]


def aggregate_clipped_noisy(
    client_gradients: List[List[torch.Tensor]],
    max_norm: float,
    sigma: float,
    device: torch.device = None
) -> List[torch.Tensor]:
    """
    Central DP-FedAvg aggregation with per-client clipping and noise addition.

    This is NOT the mechanism used by run_experiment.py's --dp-epsilon flag.
    That flag implements *local* DP: it clips and noises one client's raw
    gradient before any adversary (including this project's gradient-inversion
    attacker) can read it — the correct mechanism when the adversary's read
    point is a single client's pre-aggregation update, which is what every
    published result in this project tests (attack_source="gradients").

    This function implements *central* DP-FedAvg instead: it clips each
    client's gradient, averages the clipped values, and adds noise scaled
    down by num_clients — a guarantee that covers only the released
    aggregate. It provides no protection to an adversary who reads any
    individual client's update before aggregation, so it must not be used to
    "fix" the local-DP path above; doing so would understate the noise
    actually needed and produce a broken privacy guarantee for that release
    point.

    This function is implemented but not evaluated anywhere in this project.
    It's the mechanism a secure-aggregation / agg_update-style experiment
    would need (see fl_system.py's capture_mode='agg_update'), which this
    project has never run — no GPU/CelebA compute was available to do so.

    Steps:
        1. Clip each client's gradients to max_norm (L2)
        2. Average clipped gradients
        3. Add calibrated Gaussian noise

    Args:
        client_gradients: List of gradient lists, shape [num_clients × num_params]
        max_norm: Per-client gradient clipping threshold
        sigma: Base noise scale (adjusted by num_clients)
        device: Target device for computation

    Returns:
        Aggregated noisy gradients
    """
    if not client_gradients:
        raise ValueError("No client gradients provided")
    
    device = device or torch.device("cpu")
    num_clients = len(client_gradients)
    num_params = len(client_gradients[0])

    # Clip per client
    clipped = [
        clip_gradients([g.to(device) for g in grads], max_norm) 
        for grads in client_gradients
    ]

    # Average per-parameter
    aggregated = [
        torch.stack([grads[i] for grads in clipped]).mean(dim=0)
        for i in range(num_params)
    ]

    # Scale noise by sensitivity (max_norm / num_clients)
    noise_scale = sigma * max_norm / num_clients
    return add_gaussian_noise(aggregated, noise_scale)


def compute_dp_budget(sigma: float, delta: float, sensitivity: float = 1.0) -> float:
    """
    Compute ε given σ and δ (inverse of gaussian_sigma_for_dp).
    
    Args:
        sigma: Gaussian noise standard deviation
        delta: Failure probability
        sensitivity: L2 sensitivity
    
    Returns:
        Privacy budget ε (inf if σ ≤ 0)
    """
    if sigma <= 0:
        return float("inf")
    return sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / sigma

