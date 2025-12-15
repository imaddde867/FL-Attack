"""
Differential Privacy utilities for Federated Learning.

Gaussian mechanism for DP-SGD / DP-FedAvg style gradient perturbation.
"""

import math
from typing import List

import torch


def gaussian_sigma_for_dp(epsilon, delta, sensitivity=1.0):
    """
    Compute Gaussian noise std for (epsilon, delta)-DP.

    sigma = sensitivity * sqrt(2 * ln(1.25 / delta)) / epsilon
    """
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if not (0 < delta < 1):
        raise ValueError("delta must be in (0, 1)")
    sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
    return sigma


def clip_gradients(gradients, max_norm):
    """
    Clip a list of gradient tensors to have total L2 norm <= max_norm.
    gradients: list[torch.Tensor]
    """
    total_norm = torch.sqrt(sum(g.pow(2).sum() for g in gradients))
    clip_factor = max_norm / (total_norm + 1e-8)
    clip_factor = min(clip_factor, 1.0)
    return [g * clip_factor for g in gradients]


def add_gaussian_noise(gradients, sigma):
    """
    Add i.i.d. N(0, sigma^2) noise to each gradient tensor.
    gradients: list[torch.Tensor]
    """
    noisy = []
    for g in gradients:
        noise = torch.randn_like(g) * sigma
        noisy.append(g + noise)
    return noisy


def aggregate_clipped_noisy(client_gradients, max_norm, sigma, device=None):
    """
    DP-FedAvg style aggregation:

    client_gradients: list of list of tensors, shape:
        num_clients x num_params

    Steps:
      1. Clip each client's gradients to max_norm (L2).
      2. Average clipped gradients.
      3. Add Gaussian noise calibrated for DP.
    """
    if not client_gradients:
        raise ValueError("No client gradients provided")

    if device is None:
        device = torch.device("cpu")

    num_clients = len(client_gradients)
    num_params = len(client_gradients[0])

    # clip per client
    clipped_gradients = []
    for grads in client_gradients:
        grads_dev = [g.to(device) for g in grads]
        clipped_gradients.append(clip_gradients(grads_dev, max_norm))

    # average per-parameter
    aggregated = []
    for p_idx in range(num_params):
        stacked = torch.stack([grads[p_idx] for grads in clipped_gradients])
        avg = stacked.mean(dim=0)
        aggregated.append(avg)

    # for average, sensitivity ~ max_norm / num_clients
    noise_scale = sigma * max_norm / num_clients
    noisy_aggregated = add_gaussian_noise(aggregated, noise_scale)

    return noisy_aggregated


def compute_dp_budget(sigma, delta, sensitivity=1.0):
    """
    Invert gaussian_sigma_for_dp: given sigma, delta, sensitivity -> epsilon.
    """
    if sigma <= 0:
        return float("inf")
    epsilon = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / sigma
    return epsilon

"""Usage example (uncomment to run)"""
# if __name__ == "__main__":
    # Tiny DP sanity check (does NOT affect how FL uses this module)

    # eps, delta = 1.0, 1e-5
    # max_norm = 1.0
    # sensitivity = max_norm

    # sigma = gaussian_sigma_for_dp(eps, delta, sensitivity=sensitivity)
    # print("sigma:", sigma)

    # fake client gradients: 3 clients, 2 "parameters" each
   #  g1 = [torch.tensor([0.5, -0.3]), torch.tensor([0.1])]
   #  g2 = [torch.tensor([1.2, 0.1]), torch.tensor([-0.2])]
   #  g3 = [torch.tensor([-0.6, 0.4]), torch.tensor([0.0])]
    # client_grads = [g1, g2, g3]

    # noisy_agg = aggregate_clipped_noisy(client_grads, max_norm=max_norm, sigma=sigma)

    # print("noisy aggregated param 0:", noisy_agg[0])
    # print("noisy aggregated param 1:", noisy_agg[1])
