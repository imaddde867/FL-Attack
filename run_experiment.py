import argparse
import datetime as _dt
import os

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from fl_system import FederatedLearningSystem
from gradient_attack import GradientInversionAttack

def denormalize_cifar10(tensor):
    """Denormalize CIFAR-10 image for visualization"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    # Tensor is (C, H, W)
    return tensor * std + mean

def run_baseline_experiment(args):
    """
    Experiment 1: Baseline FL Training + Gradient Inversion Attack
    """
    print("-"*60)
    print("EXPERIMENT 1: BASELINE (No Privacy Protection - without DP or HE)")
    print("-"*60)
    
    # Determine device
    if args.device:
        device = args.device
    else:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")
    
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
            print(f"  -> Gradients captured from client {capture_client}!")
    
    print("\n" + "="*60)
    print("Phase 2: Gradient Inversion Attack...")
    print("="*60)
    
    if captured_data is None:
        print("ERROR: No gradients captured!")
        return
    
    # Perform attack
    attacker = GradientInversionAttack(fl_system.global_model, device=device)

    print("Reconstructing image from captured signal...")
    if captured_data.get('source') == 'one_step_update' or args.attack_source == 'one_step_update':
        grads = attacker.gradients_from_one_step_update(
            captured_data['first_update'], captured_data['opt_lr']
        )
    else:
        grads = captured_data['gradients']

    reconstructed, inferred_label = attacker.reconstruct_best_of_restarts(
        grads,
        restarts=args.attack_restarts,
        base_seed=args.seed,
        num_iterations=args.attack_iterations,
        lr=args.attack_lr,
        tv_weight=args.tv_weight,
        optimizer_type=args.attack_optimizer,
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
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # True image
    # Move to cpu for visualization
    true_img_tensor = captured_data['true_data'][0].cpu()
    true_img = denormalize_cifar10(true_img_tensor)
    axes[0].imshow(true_img.permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title(f"Original Image\nTrue Label: {captured_data['true_label'].item()}")
    axes[0].axis('off')
    
    # Reconstructed image
    recon_img_tensor = reconstructed[0].cpu()
    recon_img = denormalize_cifar10(recon_img_tensor)
    axes[1].imshow(recon_img.permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title(f"Reconstructed Image\nInferred Label: {inferred_label.item()}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print("Result saved.")
    
    # Compute similarity metrics
    # Calculate MSE on normalized tensors
    mse = F.mse_loss(reconstructed, captured_data['true_data']).item()
    
    # Range of normalized data is approx [-2, 2], so span is 4. MAX^2 = 16.
    psnr = 10 * np.log10(16 / mse)
    
    results_txt = os.path.join(out_dir, 'metrics.txt')
    label_match = inferred_label.item() == captured_data['true_label'].item()
    with open(results_txt, 'w') as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"PSNR: {psnr:.4f} dB\n")
        f.write(f"LabelMatch: {label_match}\n")
    print(f"\nAttack Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Label Match: {label_match}")
    print(f"  Saved metrics: {results_txt}")

def _parse_args():
    p = argparse.ArgumentParser(description="Baseline FL + Gradient Attack R&D Runner")
    # General
    p.add_argument('--device', type=str, default=None, help='cuda|mps|cpu (auto if None)')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', type=str, default=None, help='Output directory')

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
    p.add_argument('--attack-source', type=str, default='gradients', choices=['gradients','one_step_update'])

    # Attack config
    p.add_argument('--attack-iterations', type=int, default=2000)
    p.add_argument('--attack-lr', type=float, default=0.1)
    p.add_argument('--tv-weight', type=float, default=0.001)
    p.add_argument('--attack-optimizer', type=str, default='adam', choices=['adam','lbfgs'])
    p.add_argument('--attack-restarts', type=int, default=1)

    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_baseline_experiment(args)
