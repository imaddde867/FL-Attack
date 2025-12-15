import copy
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CelebA

from device_utils import resolve_device
from Differential_privacy import gaussian_sigma_for_dp, aggregate_clipped_noisy

from homomorphic_encryptor import HomomorphicEncryptor  # your HE module


class SimpleCNN(nn.Module):
    """LeNet-style CNN sized for 64x64 CelebA crops."""
    def __init__(self, num_classes=8192):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),   # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),   # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(2),   # 16 -> 8
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class FederatedLearningSystem:
    def __init__(
        self,
        num_clients=10,
        num_classes=8192,
        batch_size=64,
        data_subset=None,
        device=None,
        client_lr=0.01,
        client_momentum=0.9,
        local_epochs=1,
        augment=True,
        use_he=False,
        he_bits=256,
        he_precision=1_000_000,
    ):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.data_subset = data_subset
        self.device = resolve_device(device)
        self.client_lr = client_lr
        self.client_momentum = client_momentum
        self.local_epochs = local_epochs
        self.augment = augment

        self.use_he = use_he
        self.encryptor = None
        if self.use_he:
            self.encryptor = HomomorphicEncryptor(bits=he_bits, precision=he_precision)

        self.global_model = SimpleCNN(num_classes).to(self.device)

        self.train_data, self.test_data = self._load_celeba()
        self.client_loaders = self._partition_data()
        self.test_loader = DataLoader(self.test_data, batch_size=128, shuffle=False)

    def _load_celeba(self):
        if self.augment:
            transform_train = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)),
            ])
        else:
            transform_train = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5)),
            ])

        transform_test = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)),
        ])

        train_data = CelebA(
            root="./data",
            split="train",
            target_type="identity",
            transform=transform_train,
            download=True,
        )
        test_data = CelebA(
            root="./data",
            split="valid",
            target_type="identity",
            transform=transform_test,
            download=True,
        )

        if self.data_subset:
            train_indices = list(range(min(len(train_data), self.data_subset)))
            test_indices = list(range(min(len(test_data), self.data_subset // 5)))
            train_data = Subset(train_data, train_indices)
            test_data = Subset(test_data, test_indices)

        return train_data, test_data

    def _partition_data(self):
        """Simple IID partition of data across clients"""
        data_per_client = len(self.train_data) // self.num_clients
        client_loaders = []

        for i in range(self.num_clients):
            start_idx = i * data_per_client
            end_idx = start_idx + data_per_client
            indices = list(range(start_idx, end_idx))

            subset = Subset(self.train_data, indices)
            loader = DataLoader(subset, batch_size=self.batch_size, shuffle=True)
            client_loaders.append(loader)

        return client_loaders

    def train_client(self, client_id, epochs=None, capture=False, capture_mode='gradients'):
        """Train a single client and optionally capture attack source."""
        local_model = copy.deepcopy(self.global_model)
        local_model.train()

        optimizer = torch.optim.SGD(
            local_model.parameters(), lr=self.client_lr, momentum=self.client_momentum
        )
        criterion = nn.CrossEntropyLoss()

        captured_data = None
        train_epochs = self.local_epochs if epochs is None else epochs

        for epoch in range(train_epochs):
            for batch_idx, (data, target) in enumerate(self.client_loaders[client_id]):
                data, target = data.to(self.device), target.to(self.device)

                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()

                if capture and batch_idx == 0 and epoch == 0:
                    if capture_mode == 'gradients':
                        captured_data = {
                            'source': 'gradients',
                            'gradients': [p.grad.clone() for p in local_model.parameters()],
                            'true_data': data.detach().cpu().clone(),
                            'true_label': target.detach().cpu().clone(),
                        }
                    elif capture_mode == 'one_step_update':
                        before = [p.detach().clone() for p in local_model.parameters()]
                        optimizer.step()
                        after = [p.detach().clone() for p in local_model.parameters()]
                        deltas = [a - b for a, b in zip(after, before)]
                        captured_data = {
                            'source': 'one_step_update',
                            'first_update': deltas,
                            'opt_lr': self.client_lr,
                            'true_data': data.detach().cpu().clone(),
                            'true_label': target.detach().cpu().clone(),
                        }
                        continue
                    else:
                        if capture_mode == 'metadata':
                            captured_data = {
                                'source': 'metadata',
                                'true_data': data.detach().cpu().clone(),
                                'true_label': target.detach().cpu().clone(),
                            }
                        else:
                            raise ValueError(f"Unknown capture_mode: {capture_mode}")

                optimizer.step()

        update = OrderedDict()
        for (name, param), global_param in zip(
            local_model.named_parameters(),
            self.global_model.parameters()
        ):
            update[name] = param.data - global_param.data

        return update, captured_data

    def aggregate_updates(self, updates, return_update=False):
        """FedAvg aggregation.

        If use_he is True: homomorphic sum on flattened updates (no DP).
        Else: central DP via Gaussian mechanism on flattened updates.
        """
        param_names = list(updates[0].keys())
        flat_updates = []

        for upd in updates:
            flat_updates.append(
                torch.cat([upd[name].view(-1) for name in param_names]).cpu().numpy()
            )

        if self.use_he:
            enc_vectors = []
            for vec in flat_updates:
                vals = vec.astype(np.float64).tolist()
                enc_vec = self.encryptor.encrypt_vector(vals)
                enc_vectors.append(enc_vec)

            enc_sum = self.encryptor.aggregate_encrypted_vectors(enc_vectors)
            sum_vals = self.encryptor.decrypt_vector(enc_sum)
            flat_avg = (np.array(sum_vals, dtype=np.float32) / len(flat_updates))
        else:
            clip_norm = 1.0
            eps, delta = 1.0, 1e-5
            sigma_sum = gaussian_sigma_for_dp(clip_norm, eps, delta)
            flat_avg = aggregate_clipped_noisy(
                flat_updates, clip_norm=clip_norm, sigma_sum=sigma_sum, rng_seed=None
            )

        flat_tensor = torch.from_numpy(flat_avg).to(self.device)
        avg_update = OrderedDict()
        offset = 0
        for name, param in self.global_model.named_parameters():
            numel = param.numel()
            avg_update[name] = flat_tensor[offset:offset + numel].view_as(param)
            offset += numel

        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.data += avg_update[name]

        if return_update:
            return avg_update

    def evaluate(self):
        """Evaluate global model on test set"""
        self.global_model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.global_model(data)
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        accuracy = 100. * correct / total
        return accuracy

    def train_round(self, round_num, capture_from_client=None, capture_mode='gradients'):
        """Execute one federated learning round"""
        updates = []
        captured_data = None
        captured_metadata = None

        num_participants = max(1, self.num_clients // 2)
        participants = np.random.choice(self.num_clients, num_participants, replace=False)

        for client_id in participants:
            capture = (capture_from_client is not None and client_id == capture_from_client)
            current_mode = capture_mode
            if capture_mode == 'agg_update':
                current_mode = 'metadata'

            update, client_data = self.train_client(
                client_id, epochs=None, capture=capture, capture_mode=current_mode
            )
            updates.append(update)

            if client_data is not None:
                if client_data.get('source') == 'metadata':
                    captured_metadata = client_data
                else:
                    captured_data = client_data

        return_update = (capture_mode == 'agg_update')
        avg_update = self.aggregate_updates(updates, return_update=return_update)

        if capture_mode == 'agg_update' and avg_update is not None:
            captured_data = {
                'source': 'agg_update',
                'avg_update': [avg_update[key].clone() for key in avg_update.keys()],
                'param_names': list(avg_update.keys()),
                'opt_lr': self.client_lr,
            }
            if captured_metadata is not None:
                captured_data['true_data'] = captured_metadata['true_data']
                captured_data['true_label'] = captured_metadata['true_label']

        accuracy = self.evaluate()
        print(f"Round {round_num}: Accuracy = {accuracy:.2f}%")

        return captured_data
