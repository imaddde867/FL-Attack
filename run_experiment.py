import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import os
from fl_system import FederatedLearningSystem
from gradient_attack import GradientInversionAttack

def denormalize_cifar10(tensor):
    """Denormalize CIFAR-10 image for visualization"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    # Tensor is (C, H, W)
    return tensor * std + mean

def run_baseline_experiment():
    """
    Experiment 1: Baseline FL Training + Gradient Inversion Attack
    """
    print("-"*60)
    print("EXPERIMENT 1: BASELINE (No Privacy Protection - without DP or HE)")
    print("-"*60)
    
    # Determine device
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
    print("Initializing FL System with batch_size=1 and data_subset=200 for R&D demo...")
    fl_system = FederatedLearningSystem(
        num_clients=10, 
        device=device,
        batch_size=1,  # Critical for simple DLG attack to work well
        data_subset=200 # Faster execution
    )
    
    # Train for several rounds
    print("\nPhase 1: Training FL Model...")
    captured_data = None
    num_rounds = 5  # Reduced rounds for quick demo
    
    for round_num in range(num_rounds):
        # Capture gradients from client 0 in the last round
        capture_client = 0 if round_num == (num_rounds - 1) else None
        
        print(f"Starting Round {round_num}...")
        round_data = fl_system.train_round(round_num, capture_from_client=capture_client)
        
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
    
    print("Reconstructing image from gradients...")
    reconstructed, inferred_label = attacker.reconstruct_with_label_inference(
        captured_data['gradients'],
        num_iterations=2000, # Reduced iterations slightly for speed
        lr=0.1
    )
    
    # Visualize results
    print("Saving result to 'baseline_attack_result.png'...")
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
    plt.savefig('baseline_attack_result.png', dpi=150, bbox_inches='tight')
    print("Result saved.")
    
    # Compute similarity metrics
    # Calculate MSE on normalized tensors
    mse = F.mse_loss(reconstructed, captured_data['true_data']).item()
    
    # Range of normalized data is approx [-2, 2], so span is 4. MAX^2 = 16.
    psnr = 10 * np.log10(16 / mse)
    
    print(f"\nAttack Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Label Match: {inferred_label.item() == captured_data['true_label'].item()}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_baseline_experiment()
