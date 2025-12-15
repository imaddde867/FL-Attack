import math
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
        lr_schedule='none',
        early_stop=False,
        patience=500,
        min_delta=1e-4,
        fft_init=False,
        preset=None,
        # Layer selection/weighting and loss metric
        use_layers=None,
        select_by_name=None,
        param_names=None,
        layer_weights=None,
        match_metric='l2',
        l2_weight=1.0,
        cos_weight=1.0,
    ):
        """
        Enhanced attack with label inference (iDLG approach), with configurable
        TV regularization, optimizer choice, and clamping. Adds cosine LR
        schedule, early stopping, optional FFT initialization, and TV/clamp presets.
        """
        if seed is not None:
            torch.manual_seed(seed)

        # Apply preset overrides if provided
        tv_weight, clamp_min, clamp_max = _apply_preset(tv_weight, clamp_min, clamp_max, preset)

        inferred_label = self.infer_label_from_gradients(captured_gradients)
        print(f"Inferred label: {inferred_label.item()}")

        # Initialize dummy data
        if fft_init:
            init = fourier_init((1, 3, 64, 64), device=self.device)
        else:
            init = torch.randn(1, 3, 64, 64, device=self.device)
        dummy_data = init.clone().detach().requires_grad_(True)

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
        no_improve_steps = 0

        def step_once():
            optimizer.zero_grad()
            # Compute gradients for dummy data
            self.model.zero_grad()
            output = self.model(dummy_data)
            loss = criterion(output, dummy_label)

            dummy_gradients = torch.autograd.grad(
                loss, self.model.parameters(), create_graph=True
            )

            # Gradient matching loss (supports layer selection/weighting and cosine similarity)
            grad_match = gradient_matching_loss(
                dummy_gradients,
                captured_gradients,
                use_layers=use_layers,
                select_by_name=select_by_name,
                param_names=param_names,
                layer_weights=layer_weights,
                metric=match_metric,
                l2_weight=l2_weight,
                cos_weight=cos_weight,
            )

            # Total variation regularization for smoothness
            tv_loss = total_variation(dummy_data)

            total_loss = grad_match + tv_weight * tv_loss
            total_loss.backward()
            return total_loss

        for iteration in range(num_iterations):
            if isinstance(optimizer, torch.optim.LBFGS):
                loss = optimizer.step(step_once)
                cur_loss = loss.item() if torch.is_tensor(loss) else float(loss)
            else:
                cur_loss = step_once().item()
                optimizer.step()

            # Cosine LR schedule (for Adam)
            if lr_schedule and lr_schedule.lower() == 'cosine' and not isinstance(optimizer, torch.optim.LBFGS):
                _set_lr_cosine(optimizer, base_lr=lr, t=iteration + 1, T=num_iterations)

            # Clamp to valid image range
            with torch.no_grad():
                dummy_data.data = torch.clamp(dummy_data.data, clamp_min, clamp_max)

            if cur_loss < best_loss - min_delta:
                best_loss = cur_loss
                best_image = dummy_data.detach().clone()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if early_stop and no_improve_steps >= patience:
                print(f"Early stopping at iter {iteration} (best loss {best_loss:.4f})")
                break

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
            # Use same matching settings for scoring if provided in kwargs
            score_tensor = gradient_matching_loss(
                grads,
                captured_gradients,
                use_layers=kwargs.get('use_layers'),
                select_by_name=kwargs.get('select_by_name'),
                param_names=kwargs.get('param_names'),
                layer_weights=kwargs.get('layer_weights'),
                metric=kwargs.get('match_metric', 'l2'),
                l2_weight=kwargs.get('l2_weight', 1.0),
                cos_weight=kwargs.get('cos_weight', 1.0),
            )
            score = float(score_tensor.item())
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
        lr_schedule='none',
        early_stop=False,
        patience=500,
        min_delta=1e-4,
        fft_init=False,
        preset=None,
        # Layer selection/weighting and loss metric
        use_layers=None,
        select_by_name=None,
        param_names=None,
        layer_weights=None,
        match_metric='l2',
        l2_weight=1.0,
        cos_weight=1.0,
    ):
        """Reconstruct a batch of dummy samples while optimizing soft labels (DLG style)."""
        if seed is not None:
            torch.manual_seed(seed)

        # Apply preset overrides if provided
        tv_weight, clamp_min, clamp_max = _apply_preset(tv_weight, clamp_min, clamp_max, preset)

        if fft_init:
            init = fourier_init((batch_size, 3, 64, 64), device=self.device)
        else:
            init = torch.randn(batch_size, 3, 64, 64, device=self.device)
        dummy_data = init.clone().detach().requires_grad_(True)
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
        no_improve_steps = 0

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

            grad_match = gradient_matching_loss(
                dummy_gradients,
                captured_gradients,
                use_layers=use_layers,
                select_by_name=select_by_name,
                param_names=param_names,
                layer_weights=layer_weights,
                metric=match_metric,
                l2_weight=l2_weight,
                cos_weight=cos_weight,
            )

            tv_loss = total_variation(dummy_data)
            total_loss = grad_match + tv_weight * tv_loss
            total_loss.backward()
            return total_loss, label_tensor

        for iteration in range(num_iterations):
            total_loss, label_tensor = step_once()
            optimizer.step()

            # Cosine LR schedule (for Adam)
            if lr_schedule and lr_schedule.lower() == 'cosine':
                _set_lr_cosine(optimizer, base_lr=lr, t=iteration + 1, T=num_iterations)

            with torch.no_grad():
                dummy_data.data = torch.clamp(dummy_data.data, clamp_min, clamp_max)

            cur = total_loss.item()
            if cur < best_loss - min_delta:
                best_loss = cur
                best_image = dummy_data.detach().clone()
                best_labels = label_tensor.detach().clone()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            if early_stop and no_improve_steps >= patience:
                print(f"Early stopping at iter {iteration} (best loss {best_loss:.4f})")
                break

            if iteration % 500 == 0:
                print(f"Iteration {iteration}: Loss = {cur:.4f}")

        return best_image, best_labels

def total_variation(x):
    """Total variation regularization for smoothness"""
    dx = torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:])
    dy = torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :])
    return dx.sum() + dy.sum()


# ---- Layer selection and gradient matching utilities ----

def _resolve_layer_indices(total_layers, use_layers=None, select_by_name=None, param_names=None):
    if use_layers is not None:
        idxs = [i for i in use_layers if 0 <= i < total_layers]
        return idxs
    if select_by_name and param_names:
        patterns = select_by_name if isinstance(select_by_name, (list, tuple)) else [select_by_name]
        sel = []
        for i, name in enumerate(param_names):
            if any(pat in name for pat in patterns):
                sel.append(i)
        if sel:
            return sel
    return list(range(total_layers))


def _prepare_layer_weights(indices, layer_weights, target_grads):
    n = len(indices)
    if layer_weights is None:
        return [1.0] * n
    if isinstance(layer_weights, (list, tuple)):
        if len(layer_weights) == n:
            return list(layer_weights)
        if len(layer_weights) == len(target_grads):
            return [float(layer_weights[i]) for i in indices]
        print("[WARN] layer_weights length mismatch, falling back to uniform.")
        return [1.0] * n
    mode = str(layer_weights).lower()
    if mode in ('auto', 'auto_norm', 'inv_norm'):
        eps = 1e-8
        ws = []
        for i in indices:
            g = target_grads[i]
            w = 1.0 / (g.norm().item() + eps)
            ws.append(w)
        s = sum(ws) + eps
        ws = [w * (n / s) for w in ws]
        return ws
    return [1.0] * n


def _cosine_loss(a, b, eps=1e-8):
    a_flat = a.view(-1)
    b_flat = b.view(-1)
    an = a_flat.norm() + eps
    bn = b_flat.norm() + eps
    return 1.0 - torch.dot(a_flat, b_flat) / (an * bn)


def gradient_matching_loss(dummy_grads, target_grads,
                           use_layers=None, select_by_name=None, param_names=None,
                           layer_weights=None, metric='l2', l2_weight=1.0, cos_weight=1.0):
    L = len(target_grads)
    idxs = _resolve_layer_indices(L, use_layers, select_by_name, param_names)
    ws = _prepare_layer_weights(idxs, layer_weights, target_grads)
    total = None
    for w, i in zip(ws, idxs):
        dg = dummy_grads[i]
        tg = target_grads[i]
        if metric == 'cosine':
            loss_i = _cosine_loss(dg, tg)
        elif metric == 'both':
            loss_i = l2_weight * ((dg - tg) ** 2).sum() + cos_weight * _cosine_loss(dg, tg)
        else:
            loss_i = ((dg - tg) ** 2).sum()
        total = w * loss_i if total is None else total + w * loss_i
    return total


# ---- Utilities for presets, scheduling, and initialization ----

def _apply_preset(tv_weight, clamp_min, clamp_max, preset):
    if not preset:
        return tv_weight, clamp_min, clamp_max
    presets = {
        'default': {'tv': 1e-3, 'clamp': (-2.0, 2.0)},
        'soft': {'tv': 3e-4, 'clamp': (-3.0, 3.0)},
        'tight': {'tv': 1e-2, 'clamp': (-1.5, 1.5)},
        'none': {'tv': 0.0, 'clamp': (-1e9, 1e9)},
    }
    cfg = presets.get(str(preset).lower())
    if cfg is None:
        return tv_weight, clamp_min, clamp_max
    tmin, tmax = cfg['clamp']
    return cfg['tv'], tmin, tmax


def _set_lr_cosine(optimizer, base_lr, t, T):
    lr = base_lr * 0.5 * (1 + math.cos(math.pi * min(t, T) / T))
    for pg in optimizer.param_groups:
        pg['lr'] = lr


def fourier_init(shape, device=None, decay_power=1.5, std=0.1):
    """FFT-based image initialization with 1/f^p spectrum.

    shape: (B, C, H, W)
    Returns a tensor on device.
    """
    device = device or 'cpu'
    b, c, h, w = shape
    fy = torch.fft.fftfreq(h, d=1.0, device=device).view(h, 1).abs()
    fx = torch.fft.rfftfreq(w, d=1.0, device=device).view(1, w // 2 + 1).abs()
    f = torch.sqrt(fx**2 + fy**2)
    scale = (1.0 / (f + 1e-6) ** decay_power)
    scale = scale / scale.max()
    real = torch.randn(b, c, h, w // 2 + 1, device=device)
    imag = torch.randn(b, c, h, w // 2 + 1, device=device)
    spectrum = (real + 1j * imag) * scale
    img = torch.fft.irfftn(spectrum, s=(h, w), dim=(-2, -1))
    img = img / img.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    img = img * std
    return img
