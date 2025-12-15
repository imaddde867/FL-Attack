import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from device_utils import resolve_device

class GradientInversionAttack:
    def __init__(self, model, device=None, num_classes=None):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        if num_classes is None:
            num_classes = getattr(model, 'num_classes', None)
        if num_classes is None:
            for module in reversed(list(model.modules())):
                if isinstance(module, nn.Linear):
                    num_classes = module.out_features
                    break
        if num_classes is None:
            raise ValueError("num_classes must be provided or inferable from model")
        self.num_classes = num_classes
        
    def reconstruct_image(self, captured_gradients, num_iterations=5000, lr=0.1):
        """
        Reconstruct image from gradients using optimization
        
        This is a simplified version of the DLG/iDLG attack
        """
        # Initialize dummy data and label
        dummy_data = torch.randn(1, 3, 64, 64, requires_grad=True, device=self.device)
        dummy_label = torch.randint(0, self.num_classes, (1,), device=self.device)
        
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

        inferred_label = self.infer_label_from_gradients(captured_gradients)
        print(f"Inferred label: {inferred_label.item()}")

        # Initialize dummy data
        dummy_data = torch.randn(1, 3, 64, 64, requires_grad=True, device=self.device)

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
        batch_size=1,
        label_strategy='idlg',
        **kwargs,
    ):
        """Run multiple random restarts and return the best image by loss."""
        best_img, best_lbl, best_loss = None, None, float('inf')
        label_strategy = label_strategy or 'idlg'
        for r in range(restarts):
            seed = base_seed + r if base_seed is not None else None
            if batch_size == 1 and label_strategy == 'idlg':
                img, lbl = self.reconstruct_with_label_inference(
                    captured_gradients, seed=seed, **kwargs
                )
                label_tensor = lbl.unsqueeze(0)
            else:
                img, label_tensor = self.reconstruct_batch_optimize_labels(
                    captured_gradients,
                    batch_size=batch_size,
                    seed=seed,
                    label_strategy=label_strategy,
                    **kwargs,
                )
                lbl = label_tensor
            # Compute final loss proxy by re-evaluating one pass
            dummy = img.clone().detach().requires_grad_(True).to(self.device)
            self.model.zero_grad()
            out = self.model(dummy)
            loss = F.cross_entropy(out, label_tensor)
            grads = torch.autograd.grad(loss, self.model.parameters())
            grad_diff = sum(((dg - tg) ** 2).sum() for dg, tg in zip(grads, captured_gradients))
            score = grad_diff.item()
            if score < best_loss:
                best_loss = score
                best_img, best_lbl = img, label_tensor
        return best_img, best_lbl

    @staticmethod
    def gradients_from_one_step_update(first_update, opt_lr):
        """Approximate per-parameter gradients from a one-step SGD update (no momentum).

        grad ≈ - delta / lr
        """
        return [(-1.0 / opt_lr) * du for du in first_update]

    @staticmethod
    def gradients_from_avg_update(avg_update, opt_lr):
        """Approximate average gradients from FedAvg delta under 1-step SGD."""
        return [(-1.0 / opt_lr) * du for du in avg_update]

    def infer_label_from_gradients(self, captured_gradients):
        last_layer_grad = captured_gradients[-2]
        inferred_label = torch.argmin(torch.sum(last_layer_grad, dim=1))
        return inferred_label

    def reconstruct_batch_optimize_labels(
        self,
        captured_gradients,
        batch_size=2,
        num_iterations=3000,
        lr=0.1,
        tv_weight=0.001,
        clamp_min=-2.0,
        clamp_max=2.0,
        label_strategy='optimize',
        optimizer_type='adam',
        seed=None,
    ):
        """Reconstruct a batch of dummy samples while optimizing soft labels (DLG style)."""
        if seed is not None:
            torch.manual_seed(seed)

        dummy_data = torch.randn(batch_size, 3, 64, 64, requires_grad=True, device=self.device)
        params = [dummy_data]

        if label_strategy == 'idlg' and batch_size == 1:
            inferred_label = self.infer_label_from_gradients(captured_gradients)
            dummy_label_tensor = inferred_label.unsqueeze(0)
            optimize_labels = False
        else:
            optimize_labels = True
            dummy_label_logits = torch.zeros(
                batch_size, self.num_classes, device=self.device, requires_grad=True
            )
            params.append(dummy_label_logits)

        if optimizer_type.lower() == 'adam':
            optimizer = torch.optim.Adam(params, lr=lr)
        else:
            if optimizer_type.lower() != 'adam':
                print("[WARN] LBFGS not supported for multi-batch attack, using Adam instead.")
            optimizer = torch.optim.Adam(params, lr=lr)

        best_loss = float('inf')
        best_image = None
        best_labels = None

        def step_once():
            optimizer.zero_grad()
            self.model.zero_grad()
            output = self.model(dummy_data)
            if optimize_labels:
                soft_targets = dummy_label_logits.softmax(dim=-1)
                ce_loss = -(soft_targets * F.log_softmax(output, dim=1)).sum(dim=1).mean()
                label_tensor = torch.argmax(soft_targets.detach(), dim=1)
            else:
                ce_loss = F.cross_entropy(output, dummy_label_tensor)
                label_tensor = dummy_label_tensor

            dummy_gradients = torch.autograd.grad(
                ce_loss, self.model.parameters(), create_graph=True
            )

            grad_diff = sum(
                ((dg - tg) ** 2).sum()
                for dg, tg in zip(dummy_gradients, captured_gradients)
            )

            tv_loss = total_variation(dummy_data)
            total_loss = grad_diff + tv_weight * tv_loss
            total_loss.backward()
            return total_loss, label_tensor

        for iteration in range(num_iterations):
            total_loss, label_tensor = step_once()
            optimizer.step()

            with torch.no_grad():
                dummy_data.data = torch.clamp(dummy_data.data, clamp_min, clamp_max)

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_image = dummy_data.detach().clone()
                best_labels = label_tensor.detach().clone()

            if iteration % 500 == 0:
                print(f"Iteration {iteration}: Loss = {total_loss.item():.4f}")

        return best_image, best_labels

def total_variation(x):
    """Total variation regularization for smoothness"""
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    return dx.sum() + dy.sum()
