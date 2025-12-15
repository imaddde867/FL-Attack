import argparse
import datetime as _dt
import os
from typing import Optional

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from device_utils import resolve_device

from fl_system import FederatedLearningSystem
from gradient_attack import GradientInversionAttack


def denormalize_tensor(tensor, mean, std):
    """Generic denormalization helper."""
    mean_t = tensor.new_tensor(mean).view(-1, 1, 1)
    std_t = tensor.new_tensor(std).view(-1, 1, 1)
    return tensor * std_t + mean_t


def compute_simple_ssim(pred, target, data_range=4.0):
    dims = list(range(1, pred.ndim))
    mu_x = pred.mean(dim=dims)
    mu_y = target.mean(dim=dims)
    sigma_x = pred.var(dim=dims, unbiased=False)
    sigma_y = target.var(dim=dims, unbiased=False)
    sigma_xy = ((pred - mu_x.view(-1, *([1] * (pred.ndim - 1)))) *
                (target - mu_y.view(-1, *([1] * (target.ndim - 1))))).mean(dim=dims)
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2
    ssim = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2))
    return ssim.mean().item()


def maybe_init_lpips(device, enabled):
    if not enabled:
        return None
    try:
        import lpips  # type: ignore
    except ImportError:
        print("[WARN] LPIPS requested but package is not installed. Skipping LPIPS metric.")
        return None
    model = lpips.LPIPS(net='alex').to(device)
    model.eval()
    return model


def compute_metrics(pred, target, compute_lpips=False, lpips_model=None, denorm_mean=None, denorm_std=None):
    metrics = {}
    mse = F.mse_loss(pred, target).item()
    metrics['MSE'] = mse
    max_range = 4.0  # approx data range used for normalized tensors
    metrics['PSNR'] = 10 * np.log10((max_range ** 2) / max(mse, 1e-12))
    metrics['SSIM'] = compute_simple_ssim(pred, target, data_range=max_range)
    if compute_lpips:
        if lpips_model is None:
            metrics['LPIPS'] = float('nan')
        else:
            with torch.no_grad():
                if denorm_mean is not None and denorm_std is not None:
                    # Convert to pixel space [0,1], then to [-1,1] for LPIPS
                    pred_pix = torch.clamp(denormalize_tensor(pred, denorm_mean, denorm_std), 0, 1)
                    target_pix = torch.clamp(denormalize_tensor(target, denorm_mean, denorm_std), 0, 1)
                    pred_lp = pred_pix * 2.0 - 1.0
                    target_lp = target_pix * 2.0 - 1.0
                else:
                    # Fallback: assume roughly [-1,1] already
                    pred_lp = torch.clamp(pred, -1, 1)
                    target_lp = torch.clamp(target, -1, 1)
                score = lpips_model(pred_lp, target_lp)
                metrics['LPIPS'] = score.mean().item()
    return metrics

def _parse_layer_weights(arg_val):
    if arg_val is None:
        return None
    val = str(arg_val).strip()
    if val.lower() == 'auto':
        return 'auto'
    try:
        parts = [p for p in val.replace(';', ',').split(',') if p]
        return [float(p) for p in parts]
    except Exception:
        print(f"[WARN] Could not parse --layer-weights='{arg_val}', using uniform.")
        return None

def run_baseline_experiment(args):
    """
    Experiment 1: Baseline FL Training + Gradient Inversion Attack
    """
    print("-"*60)
    print("EXPERIMENT 1: BASELINE (No Privacy Protection - without DP or HE)")
    print("-"*60)
    
    device = resolve_device(args.device)
    print(f"Using device: {device}")
    lpips_model = maybe_init_lpips(device, args.compute_lpips)
    
    # Initialize FL system
    # Use batch_size=1 to showcase a successful DLG attack
    # Use data_subset=200 to allow the experiment to run quickly
    print("Initializing FL System (configurable via CLI)...")
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
    # Use CelebA normalization defaults if attributes are missing
    denorm_mean = getattr(fl_system, 'channel_mean', (0.5, 0.5, 0.5))
    denorm_std = getattr(fl_system, 'channel_std', (0.5, 0.5, 0.5))
    
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

def _parse_args():
    p = argparse.ArgumentParser(description="Baseline FL + Gradient Attack R&D Runner")
    # General
    p.add_argument('--device', type=str, default=None, help='cuda|mps|cpu (auto if None)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', type=str, default=None, help='Output directory')
    p.add_argument('--save-config', action='store_true', help='Save args as config.json into out-dir')

    # FL system config
    p.add_argument('--num-clients', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--data-subset', type=int, default=200)
    p.add_argument('--num-rounds', type=int, default=5)
    p.add_argument('--local-epochs', type=int, default=1)
    p.add_argument('--client-lr', type=float, default=0.01)
    p.add_argument('--client-momentum', type=float, default=0.9)
    p.add_argument('--no-augment', action='store_true', help='Disable train-time augmentation')

    # Capture config
    p.add_argument('--capture-client', type=int, default=0)
    p.add_argument('--capture-round', type=int, default=None, help='If None, captures last round')
    p.add_argument('--attack-source', type=str, default='gradients',
                   choices=['gradients','one_step_update','agg_update'])

    # Attack config
    p.add_argument('--attack-iterations', type=int, default=2000)
    p.add_argument('--attack-lr', type=float, default=0.1)
    p.add_argument('--tv-weight', type=float, default=0.001)
    p.add_argument('--attack-optimizer', type=str, default='adam', choices=['adam','lbfgs'])
    p.add_argument('--attack-restarts', type=int, default=1)
    p.add_argument('--attack-batch', type=int, default=1)
    p.add_argument('--label-strategy', type=str, default='auto', choices=['auto','idlg','optimize'])
    p.add_argument('--compute-lpips', action='store_true')
    # Attack enhancements
    p.add_argument('--lr-schedule', type=str, default='none', choices=['none','cosine'])
    p.add_argument('--early-stop', action='store_true')
    p.add_argument('--patience', type=int, default=500)
    p.add_argument('--min-delta', type=float, default=1e-4)
    p.add_argument('--fft-init', action='store_true')
    p.add_argument('--preset', type=str, default=None)
    p.add_argument('--match-metric', type=str, default='l2', choices=['l2','cosine','both'])
    p.add_argument('--l2-weight', type=float, default=1.0)
    p.add_argument('--cos-weight', type=float, default=1.0)
    p.add_argument('--use-layers', type=int, nargs='+', default=None, help='Indices of layers to match')
    p.add_argument('--select-by-name', type=str, nargs='+', default=None, help='Substring patterns to select layers by name')
    p.add_argument('--layer-weights', type=str, default=None, help="'auto' or comma-separated floats")
    # Visualization
    p.add_argument('--no-heatmap', action='store_true', help='Disable difference heatmap row')

    return p.parse_args()


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
        import json
        with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
    run_baseline_experiment(args)
