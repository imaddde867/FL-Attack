import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np
from fl_system import FederatedLearningSystem
from gradient_attack import GradientInversionAttack

def denormalize_cifar10(tensor):
    """Denormalize CIFAR-10 image for visualization"""
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
    std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
    return tensor * std + mean

def run_baseline_experiment():
    """
    Experiment 1: Baseline FL Training + Gradient Inversion Attack
    """
    print("-"*60)
    print("EXPERIMENT 1: BASELINE (No Privacy Protection - without DP or HE)")
    print("-"*60)
    
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Initialize FL system
    fl_system = FederatedLearningSystem(num_clients=10, device=device)
    
    # Train for several rounds to get decent accuracy
    print("\nPhase 1: Training FL Model...")
    captured_data = None
    for round_num in range(20):  # 20 rounds should give ~60-70% accuracy
        # Capture gradients from client 0 in round 10
        capture_client = 0 if round_num == 10 else None
        round_data = fl_system.train_round(round_num, capture_from_client=capture_client)
        
        if round_data is not None:
            captured_data = round_data
    
    print("\n" + "="*60)
    print("Phase 2: Gradient Inversion Attack...")
    print("="*60)
    
    if captured_data is None:
        print("ERROR: No gradients captured!")
        return
    
    # Perform attack
    attacker = GradientInversionAttack(fl_system.global_model, device=device)
    
    reconstructed, inferred_label = attacker.reconstruct_with_label_inference(
        captured_data['gradients'],
        num_iterations=3000,
        lr=0.1
    )
    
    # Visualize results
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # True image
    true_img = denormalize_cifar10(captured_data['true_data'][0].cpu())
    axes[0].imshow(true_img.permute(1, 2, 0).clamp(0, 1))
    axes[0].set_title(f"Original Image\nTrue Label: {captured_data['true_label'].item()}")
    axes[0].axis('off')
    
    # Reconstructed image
    recon_img = denormalize_cifar10(reconstructed[0].cpu())
    axes[1].imshow(recon_img.permute(1, 2, 0).clamp(0, 1))
    axes[1].set_title(f"Reconstructed Image\nInferred Label: {inferred_label.item()}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('baseline_attack_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Compute similarity metrics
    mse = F.mse_loss(reconstructed, captured_data['true_data']).item()
    psnr = 10 * np.log10(4 / mse)  # Range is [-2, 2] so max squared diff is 16
    
    print(f"\nAttack Results:")
    print(f"  MSE: {mse:.4f}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  Label Match: {inferred_label.item() == captured_data['true_label'].item()}")

if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    
    run_baseline_experiment()