"""
Federated Learning Gradient Attack Experiment Runner.

Command-line interface for running gradient inversion attacks on federated
learning systems with optional privacy defenses (DP and HE).
"""

import argparse
import datetime as _dt
import json
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from device_utils import resolve_device
from differential_privacy import add_gaussian_noise, clip_gradients, gaussian_sigma_for_dp
from fl_system import FederatedLearningSystem
from gradient_attack import GradientInversionAttack
from homomorphic_encryptor import HomomorphicEncryptor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HE_SAMPLE_LIMIT = 10_000  # Max gradient elements for real HE (else use simulation)
DEFAULT_DENORM = (0.5, 0.5, 0.5)  # Default CelebA normalization

# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def denormalize_tensor(
    tensor: torch.Tensor,
    mean: Tuple[float, ...],
    std: Tuple[float, ...]
) -> torch.Tensor:
    """Denormalize a tensor from normalized space to [0, 1] pixel range."""
    mean_t = tensor.new_tensor(mean).view(-1, 1, 1)
    std_t = tensor.new_tensor(std).view(-1, 1, 1)
    return tensor * std_t + mean_t


def compute_simple_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 4.0
) -> float:
    """Compute simplified SSIM between prediction and target tensors."""
    dims = list(range(1, pred.ndim))
    mu_x = pred.mean(dim=dims)
    mu_y = target.mean(dim=dims)
    sigma_x = pred.var(dim=dims, unbiased=False)
    sigma_y = target.var(dim=dims, unbiased=False)
    sigma_xy = (
        (pred - mu_x.view(-1, *([1] * (pred.ndim - 1))))
        * (target - mu_y.view(-1, *([1] * (target.ndim - 1))))
    ).mean(dim=dims)
    
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim = (
        (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    ) / (
        (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)
    )
    return ssim.mean().item()


def maybe_init_lpips(device: torch.device, enabled: bool) -> Optional[Any]:
    """Initialize LPIPS model if enabled and available."""
    if not enabled:
        return None
    try:
        import lpips
    except ImportError:
        print("[WARN] LPIPS requested but package is not installed.")
        return None
    model = lpips.LPIPS(net='alex').to(device).eval()
    return model


def compute_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    compute_lpips: bool = False,
    lpips_model: Optional[Any] = None,
    denorm_mean: Optional[Tuple[float, ...]] = None,
    denorm_std: Optional[Tuple[float, ...]] = None
) -> Dict[str, float]:
    """Compute reconstruction quality metrics (MSE, PSNR, SSIM, optionally LPIPS)."""
    metrics: Dict[str, float] = {}
    mse = F.mse_loss(pred, target).item()
    metrics['MSE'] = mse
    
    max_range = 4.0  # Approximate data range for normalized tensors
    metrics['PSNR'] = 10 * np.log10((max_range ** 2) / max(mse, 1e-12))
    metrics['SSIM'] = compute_simple_ssim(pred, target, data_range=max_range)
    
    if compute_lpips:
        if lpips_model is None:
            metrics['LPIPS'] = float('nan')
        else:
            with torch.no_grad():
                if denorm_mean is not None and denorm_std is not None:
                    pred_pix = torch.clamp(denormalize_tensor(pred, denorm_mean, denorm_std), 0, 1)
                    target_pix = torch.clamp(denormalize_tensor(target, denorm_mean, denorm_std), 0, 1)
                    pred_lp = pred_pix * 2.0 - 1.0
                    target_lp = target_pix * 2.0 - 1.0
                else:
                    pred_lp = torch.clamp(pred, -1, 1)
                    target_lp = torch.clamp(target, -1, 1)
                metrics['LPIPS'] = lpips_model(pred_lp, target_lp).mean().item()
    return metrics


def _parse_layer_weights(arg_val: Optional[str]) -> Optional[Union[str, List[float]]]:
    """Parse layer weights argument (string mode or list of floats)."""
    if arg_val is None:
        return None
    
    val = str(arg_val).strip().lower()
    valid_modes = {
        'auto', 'auto_norm', 'inv_norm', 'early', 'early_linear',
        'early_strong', 'early_conv', 'spatial', 'uniform', 'none'
    }
    
    if val in valid_modes:
        return None if val in ('uniform', 'none') else val
    
    try:
        parts = [p.strip() for p in val.replace(';', ',').split(',') if p.strip()]
        return [float(p) for p in parts]
    except ValueError:
        print(f"[WARN] Could not parse --layer-weights='{arg_val}', using uniform.")
        return None


# ---------------------------------------------------------------------------
# Main Experiment Runner
# ---------------------------------------------------------------------------


def run_baseline_experiment(args: argparse.Namespace) -> None:
    """Run baseline FL training with gradient inversion attack."""
    print("-" * 60)
    print("EXPERIMENT: BASELINE (Gradient Inversion Attack)")
    print("-" * 60)

    device = resolve_device(args.device)
    print(f"Device: {device}")
    lpips_model = maybe_init_lpips(device, args.compute_lpips)

    # Initialize FL system
    print("Initializing FL System...")
    fl_system = FederatedLearningSystem(
        num_clients=args.num_clients,
        device=device,
        batch_size=args.batch_size,
        data_subset=args.data_subset,
        client_lr=args.client_lr,
        client_momentum=args.client_momentum,
        local_epochs=args.local_epochs,
        augment=not args.no_augment,
    )
    
    denorm_mean = getattr(fl_system, 'channel_mean', DEFAULT_DENORM)
    denorm_std = getattr(fl_system, 'channel_std', DEFAULT_DENORM)

    # Train for several rounds
    print("\nPhase 1: Training FL Model...")
    captured_data = None
    num_rounds = args.num_rounds

    for round_num in range(num_rounds):
        # Capture from selected client in the selected round
        if args.capture_round is not None:
            capture_client = args.capture_client if round_num == args.capture_round else None
        else:
            capture_client = args.capture_client if round_num == (num_rounds - 1) else None
        
        print(f"Starting Round {round_num}...")
        round_data = fl_system.train_round(
            round_num,
            capture_from_client=capture_client,
            capture_mode=args.attack_source,
        )
        
        if round_data is not None:
            captured_data = round_data
            src = round_data.get('source', args.attack_source)
            if src == 'agg_update':
                print("  -> Captured aggregated update for this round.")
            else:
                victim = capture_client if capture_client is not None else 'unknown'
                print(f"  -> Captured {src} from client {victim}!")
    
    print("\n" + "="*60)
    print("Phase 2: Gradient Inversion Attack...")
    print("="*60)
    
    if captured_data is None:
        print("ERROR: No gradients captured!")
        return
    
    # Perform attack
    attacker = GradientInversionAttack(
        fl_system.global_model, device=device, num_classes=fl_system.num_classes
    )

    print("Reconstructing image from captured signal...")
    source = captured_data.get('source', args.attack_source)
    param_names = None
    if source == 'one_step_update':
        grads = attacker.gradients_from_one_step_update(
            captured_data['first_update'], captured_data['opt_lr']
        )
    elif source == 'agg_update':
        grads = attacker.gradients_from_avg_update(
            captured_data['avg_update'], captured_data['opt_lr']
        )
        param_names = captured_data.get('param_names')
    else:
        grads = captured_data['gradients']
    # Fallback param names from model
    if param_names is None:
        try:
            param_names = [n for (n, _) in fl_system.global_model.named_parameters()]
        except Exception:
            param_names = None

    # ============================================================
    # Apply Privacy Defenses (DP and/or HE)
    # ============================================================
    if args.dp_epsilon is not None:
        print(f"\n[DEFENSE] Applying Differential Privacy (ε={args.dp_epsilon}, δ={args.dp_delta})")
        sigma = gaussian_sigma_for_dp(args.dp_epsilon, args.dp_delta, sensitivity=args.dp_max_norm)
        print(f"  -> Clipping gradients to L2 norm ≤ {args.dp_max_norm}")
        grads = clip_gradients(grads, args.dp_max_norm)
        print(f"  -> Adding Gaussian noise with σ={sigma:.4f}")
        grads = add_gaussian_noise(grads, sigma)
        print(f"  -> DP protection applied!")

    if args.use_he:
        print(f"\n[DEFENSE] Applying Homomorphic Encryption (bits={args.he_bits})")
        total_elements = sum(g.numel() for g in grads)
        print(f"  -> Total gradient elements: {total_elements:,}")

        if total_elements <= HE_SAMPLE_LIMIT:
            # Small model: actual HE encrypt/decrypt
            print("  -> Using actual HE encryption")
            he = HomomorphicEncryptor(bits=args.he_bits, precision=args.he_precision)
            protected_grads = []
            for g in grads:
                flat = g.flatten().tolist()
                encrypted = he.encrypt_vector(flat)
                encrypted_noisy = he.add_noise_encrypted(encrypted, scale=0.01)
                decrypted = he.decrypt_vector(encrypted_noisy)
                protected_grads.append(
                    torch.tensor(decrypted, dtype=g.dtype, device=g.device).view_as(g)
                )
            grads = protected_grads
            print("  -> HE round-trip with noise applied!")
        else:
            # Large model: simulate HE effect (quantization + noise)
            print("  -> Using fast HE simulation (large model)")
            precision = args.he_precision
            noise_scale = 0.01

            protected_grads = []
            for g in grads:
                quantized = torch.round(g * precision) / precision
                laplace_noise = torch.distributions.Laplace(0, noise_scale).sample(g.shape).to(g.device)
                protected_grads.append((quantized + laplace_noise).to(g.dtype))
            grads = protected_grads
            print("  -> HE simulation applied!")

    label_strategy = args.label_strategy
    if label_strategy == 'auto':
        if args.attack_batch == 1 and source != 'agg_update':
            label_strategy = 'idlg'
        else:
            label_strategy = 'optimize'

    reconstructed, inferred_labels = attacker.reconstruct_best_of_restarts(
        grads,
        restarts=args.attack_restarts,
        base_seed=args.seed,
        num_iterations=args.attack_iterations,
        lr=args.attack_lr,
        tv_weight=args.tv_weight,
        optimizer_type=args.attack_optimizer,
        batch_size=args.attack_batch,
        label_strategy=label_strategy,
        param_names=param_names,
        # Attack enhancements
        lr_schedule=args.lr_schedule,
        early_stop=args.early_stop,
        patience=args.patience,
        min_delta=args.min_delta,
        fft_init=args.fft_init,
        preset=args.preset,
        match_metric=args.match_metric,
        l2_weight=args.l2_weight,
        cos_weight=args.cos_weight,
        use_layers=args.use_layers,
        select_by_name=args.select_by_name,
        layer_weights=_parse_layer_weights(args.layer_weights),
    )
    
    # Visualize results
    # Prepare output directory
    out_dir = args.out_dir
    if out_dir is None:
        ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = os.path.join('results', f'baseline_{ts}')
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, 'baseline_attack_result.png')
    print(f"Saving result to '{out_path}'...")

    recon_batch = reconstructed.detach().cpu()
    raw_true_data = captured_data.get('true_data')
    true_batch_tensor = None
    viz_true_batch = None
    if raw_true_data is not None:
        true_batch_tensor = raw_true_data[:recon_batch.size(0)]
        viz_true_batch = true_batch_tensor.detach().cpu()
    true_labels = captured_data.get('true_label')
    if true_labels is not None:
        true_labels = true_labels[:recon_batch.size(0)]
        true_labels_cpu = true_labels.detach().cpu()
    else:
        true_labels_cpu = None
    num_show = min(recon_batch.size(0), max(1, min(args.attack_batch, 4)))
    pred_labels_cpu = inferred_labels.detach().cpu()

    show_heatmap = (viz_true_batch is not None) and (not args.no_heatmap)
    if viz_true_batch is not None:
        if show_heatmap:
            # Three rows: Original, Reconstruction, |Diff| heatmap
            fig, axes = plt.subplots(3, num_show, figsize=(num_show * 3, 9))
        else:
            fig, axes = plt.subplots(2, num_show, figsize=(num_show * 3, 6))
    else:
        fig, axes = plt.subplots(1, num_show, figsize=(num_show * 3, 3))

    axes = np.array(axes).reshape(-1)

    for idx in range(num_show):
        recon_chw = denormalize_tensor(recon_batch[idx], denorm_mean, denorm_std)
        if viz_true_batch is not None:
            true_chw = denormalize_tensor(viz_true_batch[idx], denorm_mean, denorm_std)
            axes[idx].imshow(true_chw.permute(1, 2, 0).clamp(0, 1))
            if true_labels_cpu is not None:
                label_text = true_labels_cpu[idx].item()
            else:
                label_text = 'N/A'
            axes[idx].set_title(f"Original #{idx}\nLabel: {label_text}")
            axes[idx].axis('off')
            axes[num_show + idx].imshow(recon_chw.permute(1, 2, 0).clamp(0, 1))
            axes[num_show + idx].set_title(
                f"Recon #{idx}\nPred: {pred_labels_cpu[idx].item()}"
            )
            axes[num_show + idx].axis('off')
            if show_heatmap:
                # Difference heatmap (per-pixel mean abs diff over channels)
                diff_map = (recon_chw - true_chw).abs().mean(dim=0).clamp(0, 1).cpu()
                axes[2 * num_show + idx].imshow(diff_map, cmap='magma', vmin=0.0, vmax=1.0)
                axes[2 * num_show + idx].set_title(f"|Diff| #{idx}")
                axes[2 * num_show + idx].axis('off')
        else:
            axes[idx].imshow(recon_chw.permute(1, 2, 0).clamp(0, 1))
            axes[idx].set_title(f"Recon #{idx}\nPred: {pred_labels_cpu[idx].item()}")
            axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print("Result saved.")
    
    metrics_txt = os.path.join(out_dir, 'metrics.txt')
    print("\nAttack Results:")
    label_match = None
    if true_batch_tensor is not None:
        target_tensor = true_batch_tensor.to(reconstructed.device)
        metrics = compute_metrics(
            reconstructed, target_tensor,
            compute_lpips=args.compute_lpips,
            lpips_model=lpips_model,
            denorm_mean=denorm_mean,
            denorm_std=denorm_std,
        )
        if true_labels is not None:
            compare_true = true_labels
            pred_labels = inferred_labels[:compare_true.size(0)]
            label_match = (pred_labels.cpu() == compare_true.cpu()).float().mean().item()
            metrics['LabelMatch'] = label_match
    else:
        metrics = {'info': 'No ground truth available to compute metrics.'}

    with open(metrics_txt, 'w') as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")

    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"  Saved metrics: {metrics_txt}")


# ---------------------------------------------------------------------------
# CLI Argument Parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="FL Gradient Inversion Attack Experiment Runner",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # General options
    g = p.add_argument_group("General")
    g.add_argument('--device', type=str, default=None, help='Device (cuda|mps|cpu, auto if None)')
    g.add_argument('--seed', type=int, default=42, help='Random seed')
    g.add_argument('--out-dir', type=str, default=None, help='Output directory')
    g.add_argument('--save-config', action='store_true', help='Save config.json')

    # FL system configuration
    g = p.add_argument_group("Federated Learning")
    g.add_argument('--num-clients', type=int, default=10, help='Number of FL clients')
    g.add_argument('--batch-size', type=int, default=1, help='Client batch size')
    g.add_argument('--data-subset', type=int, default=200, help='Samples per client')
    g.add_argument('--num-rounds', type=int, default=5, help='FL training rounds')
    g.add_argument('--local-epochs', type=int, default=1, help='Local epochs per round')
    g.add_argument('--client-lr', type=float, default=0.01, help='Client learning rate')
    g.add_argument('--client-momentum', type=float, default=0.9, help='Client SGD momentum')
    g.add_argument('--no-augment', action='store_true', help='Disable data augmentation')

    # Capture configuration
    g = p.add_argument_group("Gradient Capture")
    g.add_argument('--capture-client', type=int, default=0, help='Client to capture from')
    g.add_argument('--capture-round', type=int, default=None, help='Round to capture (default: last)')
    g.add_argument('--attack-source', type=str, default='gradients',
                   choices=['gradients', 'one_step_update', 'agg_update'],
                   help='Source of gradients for attack')

    # Attack configuration
    g = p.add_argument_group("Attack Parameters")
    g.add_argument('--attack-iterations', type=int, default=2000, help='Optimization iterations')
    g.add_argument('--attack-lr', type=float, default=0.1, help='Attack learning rate')
    g.add_argument('--tv-weight', type=float, default=0.001, help='Total variation weight')
    g.add_argument('--attack-optimizer', type=str, default='adam',
                   choices=['adam', 'lbfgs', 'sgd', 'adamw'], help='Optimizer')
    g.add_argument('--attack-restarts', type=int, default=1, help='Random restarts')
    g.add_argument('--attack-batch', type=int, default=1, help='Batch size to reconstruct')
    g.add_argument('--label-strategy', type=str, default='auto',
                   choices=['auto', 'idlg', 'optimize'], help='Label inference strategy')
    g.add_argument('--compute-lpips', action='store_true', help='Compute LPIPS metric')

    # Attack enhancements
    g = p.add_argument_group("Attack Enhancements")
    g.add_argument('--lr-schedule', type=str, default='none',
                   choices=['none', 'cosine'], help='Learning rate schedule')
    g.add_argument('--early-stop', action='store_true', help='Enable early stopping')
    g.add_argument('--patience', type=int, default=500, help='Early stop patience')
    g.add_argument('--min-delta', type=float, default=1e-4, help='Min improvement delta')
    g.add_argument('--fft-init', action='store_true', help='Use FFT initialization')
    g.add_argument('--preset', type=str, default=None, help='Attack preset (soft/tight/none)')
    g.add_argument('--match-metric', type=str, default='l2',
                   choices=['l2', 'cosine', 'both', 'sim'], help='Gradient matching metric')
    g.add_argument('--l2-weight', type=float, default=1.0, help='L2 loss weight')
    g.add_argument('--cos-weight', type=float, default=1.0, help='Cosine loss weight')
    g.add_argument('--use-layers', type=int, nargs='+', default=None, help='Layer indices to match')
    g.add_argument('--select-by-name', type=str, nargs='+', default=None, help='Layer name patterns')
    g.add_argument('--layer-weights', type=str, default=None, help='Layer weighting mode or values')
    g.add_argument('--no-heatmap', action='store_true', help='Disable difference heatmap')

    # Privacy defenses
    g = p.add_argument_group("Privacy Defenses")
    g.add_argument('--dp-epsilon', type=float, default=None, help='DP epsilon (enables DP)')
    g.add_argument('--dp-delta', type=float, default=1e-5, help='DP delta')
    g.add_argument('--dp-max-norm', type=float, default=1.0, help='DP clipping norm')
    g.add_argument('--use-he', action='store_true', help='Enable Homomorphic Encryption')
    g.add_argument('--he-bits', type=int, default=512, help='HE key size (bits)')
    g.add_argument('--he-precision', type=int, default=1000000, help='HE precision')

    return p.parse_args()


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    # Normalize out-dir and optionally save config before the run
    if args.out_dir is None:
        ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        args.out_dir = os.path.join('results', f'baseline_{ts}')
    os.makedirs(args.out_dir, exist_ok=True)

    if args.save_config:
        config_path = os.path.join(args.out_dir, 'config.json')
        with open(config_path, 'w') as f:
            json.dump(vars(args), f, indent=2)

    run_baseline_experiment(args)
