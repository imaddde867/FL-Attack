import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

class GradientInversionAttack:
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        
    def reconstruct_image(self, captured_gradients, num_iterations=5000, lr=0.1):
        """
        Reconstruct image from gradients using optimization
        
        This is a simplified version of the DLG/iDLG attack
        """
        # Initialize dummy data and label
        dummy_data = torch.randn(1, 3, 32, 32, requires_grad=True, device=self.device)
        dummy_label = torch.randint(0, 10, (1,), device=self.device)
        
        # Optimizer for dummy data
        optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        
        criterion = nn.CrossEntropyLoss()
        
        history = []
        
        for iteration in range(num_iterations):
            def closure():
                optimizer.zero_grad()
                
                # Forward pass with dummy data
                self.model.zero_grad()
                output = self.model(dummy_data)
                loss = criterion(output, dummy_label)
                
                # Compute gradients
                dummy_gradients = torch.autograd.grad(
                    loss, self.model.parameters(), create_graph=True
                )
                
                # Match gradients
                grad_diff = 0
                for dg, tg in zip(dummy_gradients, captured_gradients):
                    grad_diff += ((dg - tg) ** 2).sum()
                
                grad_diff.backward()
                return grad_diff
            
            loss = optimizer.step(closure)
            
            if iteration % 500 == 0:
                current_loss = loss.item() if torch.is_tensor(loss) else loss
                print(f"Iteration {iteration}: Loss = {current_loss:.4f}")
                history.append(current_loss)
        
        return dummy_data.detach(), history
    
    def reconstruct_with_label_inference(
        self,
        captured_gradients,
        num_classes=10,
        num_iterations=3000,
        lr=0.1,
        tv_weight=0.001,
        clamp_min=-2.0,
        clamp_max=2.0,
        optimizer_type='adam',
        seed=None,
    ):
        """
        Enhanced attack with label inference (iDLG approach), with configurable
        TV regularization, optimizer choice, and clamping.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Infer label from last layer gradients
        last_layer_grad = captured_gradients[-2]  # Weights of last layer

        # The true label corresponds to the most negative gradient
        inferred_label = torch.argmin(torch.sum(last_layer_grad, dim=1))
        print(f"Inferred label: {inferred_label.item()}")

        # Initialize dummy data
        dummy_data = torch.randn(1, 3, 32, 32, requires_grad=True, device=self.device)

        # Use inferred label
        dummy_label = inferred_label.unsqueeze(0)

        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam([dummy_data], lr=lr)
        elif optimizer_type.lower() == 'lbfgs':
            optimizer = torch.optim.LBFGS([dummy_data], lr=lr)
        else:
            optimizer = torch.optim.Adam([dummy_data], lr=lr)

        criterion = nn.CrossEntropyLoss()

        best_loss = float('inf')
        best_image = None

        def step_once():
            optimizer.zero_grad()
            # Compute gradients for dummy data
            self.model.zero_grad()
            output = self.model(dummy_data)
            loss = criterion(output, dummy_label)

            dummy_gradients = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )

            # Gradient matching loss
            grad_diff = sum(
                ((dg - tg) ** 2).sum()
                for dg, tg in zip(dummy_gradients, captured_gradients)
            )

            # Total variation regularization for smoothness
            tv_loss = total_variation(dummy_data)

            total_loss = grad_diff + tv_weight * tv_loss
            total_loss.backward()
            return total_loss

        for iteration in range(num_iterations):
            if isinstance(optimizer, torch.optim.LBFGS):
                loss = optimizer.step(step_once)
                cur_loss = loss.item() if torch.is_tensor(loss) else float(loss)
            else:
                cur_loss = step_once().item()
                optimizer.step()

            # Clamp to valid image range
            with torch.no_grad():
                dummy_data.data = torch.clamp(dummy_data.data, clamp_min, clamp_max)

            if cur_loss < best_loss:
                best_loss = cur_loss
                best_image = dummy_data.detach().clone()

            if iteration % 500 == 0:
                print(f"Iteration {iteration}: Loss = {cur_loss:.4f}")

        return best_image, inferred_label

    def reconstruct_best_of_restarts(
        self,
        captured_gradients,
        restarts=1,
        base_seed=123,
        **kwargs,
    ):
        """Run multiple random restarts and return the best image by loss."""
        best_img, best_lbl, best_loss = None, None, float('inf')
        for r in range(restarts):
            seed = base_seed + r if base_seed is not None else None
            img, lbl = self.reconstruct_with_label_inference(
                captured_gradients, seed=seed, **kwargs
            )
            # Compute final loss proxy by re-evaluating one pass
            dummy = img.clone().detach().requires_grad_(True).to(self.device)
            self.model.zero_grad()
            out = self.model(dummy)
            loss = F.cross_entropy(out, lbl.unsqueeze(0))
            grads = torch.autograd.grad(loss, self.model.parameters())
            grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(grads, captured_gradients))
            score = grad_diff.item()
            if score < best_loss:
                best_loss = score
                best_img, best_lbl = img, lbl
        return best_img, best_lbl

    @staticmethod
    def gradients_from_one_step_update(first_update, opt_lr):
        """Approximate per-parameter gradients from a one-step SGD update (no momentum).

        grad ≈ - delta / lr
        """
        return [(-1.0 / opt_lr) * du for du in first_update]

def total_variation(x):
    """Total variation regularization for smoothness"""
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    return dx.sum() + dy.sum()
