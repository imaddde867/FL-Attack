"""
Differential Privacy utilities for Federated Learning.

Implements Gaussian mechanism for DP-SGD style gradient perturbation.
"""

import math
import torch
from typing import List, OrderedDict


def gaussian_sigma_for_dp(epsilon: float, delta: float, sensitivity: float = 1.0) -> float:
    """
    Compute the Gaussian noise standard deviation for (epsilon, delta)-DP.
    
    Uses the analytic Gaussian mechanism formula:
    sigma >= sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    
    Args:
        epsilon: Privacy budget (smaller = more private)
        delta: Failure probability (typically 1e-5 or smaller)
        sensitivity: L2 sensitivity of the query (default 1.0 for clipped gradients)
    
    Returns:
        sigma: Standard deviation for Gaussian noise
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if delta <= 0 or delta >= 1:
        raise ValueError("delta must be in (0, 1)")
    
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    return sigma


def clip_gradients(gradients: List[torch.Tensor], max_norm: float) -> List[torch.Tensor]:
    """
    Clip gradients to have maximum L2 norm.
    
    Args:
        gradients: List of gradient tensors
        max_norm: Maximum L2 norm for clipping
    
    Returns:
        clipped_gradients: List of clipped gradient tensors
    """
    # Compute total L2 norm
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in gradients))
    
    # Clip factor
    clip_factor = max_norm / (total_norm + 1e-8)
    clip_factor = min(clip_factor, 1.0)
    
    # Apply clipping
    clipped = [g * clip_factor for g in gradients]
    return clipped


def add_gaussian_noise(gradients: List[torch.Tensor], sigma: float) -> List[torch.Tensor]:
    """
    Add Gaussian noise to gradients for differential privacy.
    
    Args:
        gradients: List of gradient tensors
        sigma: Standard deviation of Gaussian noise
    
    Returns:
        noisy_gradients: List of gradient tensors with added noise
    """
    noisy = []
    for g in gradients:
        noise = torch.randn_like(g) * sigma
        noisy.append(g + noise)
    return noisy


def aggregate_clipped_noisy(
    client_gradients: List[List[torch.Tensor]],
    max_norm: float,
    sigma: float,
    device: torch.device = None,
) -> List[torch.Tensor]:
    """
    Aggregate gradients from multiple clients with clipping and noise for DP.
    
    Implements DP-FedAvg style aggregation:
    1. Clip each client's gradients to max_norm
    2. Average clipped gradients
    3. Add Gaussian noise calibrated for DP
    
    Args:
        client_gradients: List of gradient lists from each client
        max_norm: Maximum L2 norm for per-client gradient clipping
        sigma: Standard deviation for Gaussian noise
        device: Device to place tensors on
    
    Returns:
        aggregated: List of aggregated gradient tensors with DP noise
    """
    if not client_gradients:
        raise ValueError("No client gradients provided")
    
    num_clients = len(client_gradients)
    num_params = len(client_gradients[0])
    
    # Clip each client's gradients
    clipped_gradients = [clip_gradients(grads, max_norm) for grads in client_gradients]
    
    # Average clipped gradients
    aggregated = []
    for param_idx in range(num_params):
        stacked = torch.stack([grads[param_idx] for grads in clipped_gradients])
        avg = stacked.mean(dim=0)
        aggregated.append(avg)
    
    # Add noise (scaled for the aggregation)
    # For averaging, the sensitivity is max_norm / num_clients
    noise_scale = sigma * max_norm / num_clients
    noisy_aggregated = add_gaussian_noise(aggregated, noise_scale)
    
    return noisy_aggregated


def compute_dp_budget(
    sigma: float,
    delta: float,
    sensitivity: float = 1.0,
) -> float:
    """
    Compute the epsilon (privacy budget) given sigma and delta.
    
    Inverse of gaussian_sigma_for_dp.
    
    Args:
        sigma: Standard deviation of Gaussian noise
        delta: Failure probability
        sensitivity: L2 sensitivity
    
    Returns:
        epsilon: Privacy budget
    """
    if sigma <= 0:
        return float('inf')  # No privacy
    
    epsilon = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / sigma
    return epsilon
