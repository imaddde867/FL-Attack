"""
Federated Learning System with CelebA dataset.

Provides FL simulation infrastructure including:
- CelebA attribute classification dataset
- Simple CNN model
- FedAvg-style training with gradient capture for attack research
"""

import copy
import csv
import os
from collections import OrderedDict
from typing import Optional, Dict, List, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
import torchvision.transforms as transforms

from device_utils import resolve_device

# Default normalization parameters
CELEBA_MEAN = (0.5, 0.5, 0.5)
CELEBA_STD = (0.5, 0.5, 0.5)
IMAGE_SIZE = 64


class CelebAAttributeDataset(Dataset):
    """
    CelebA dataset loader for binary attribute classification.
    
    Loads images and labels from CSV metadata files.
    """
    SPLIT_MAP = {"train": 0, "valid": 1, "test": 2}

    def __init__(
        self, 
        root: str = "./data", 
        split: str = "train", 
        transform: Optional[transforms.Compose] = None, 
        target_attr: str = "Male"
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_attr = target_attr
        
        # Handle nested directory structure
        base_img_dir = os.path.join(root, "img_align_celeba")
        nested_dir = os.path.join(base_img_dir, "img_align_celeba")
        self.img_dir = nested_dir if os.path.isdir(nested_dir) else base_img_dir
        
        self.attr_path = os.path.join(root, "list_attr_celeba.csv")
        self.partition_path = os.path.join(root, "list_eval_partition.csv")
        
        # Validate paths
        if not os.path.isdir(self.img_dir):
            raise RuntimeError(f"CelebA images not found at '{self.img_dir}'")
        for path in (self.attr_path, self.partition_path):
            if not os.path.exists(path):
                raise RuntimeError(f"Required metadata file '{path}' not found")
        
        self.labels = self._load_labels()
        
        if split not in self.SPLIT_MAP:
            raise ValueError(f"Invalid split '{split}'. Use: {list(self.SPLIT_MAP.keys())}")
        self.filenames = self._load_split_indices(self.SPLIT_MAP[split])

    def _load_labels(self) -> Dict[str, int]:
        """Load attribute labels from CSV."""
        labels = {}
        with open(self.attr_path, newline="") as f:
            for row in csv.DictReader(f):
                labels[row["image_id"]] = 1 if row[self.target_attr].strip() == "1" else 0
        return labels

    def _load_split_indices(self, split_id: int) -> List[str]:
        """Load filenames for the specified split."""
        filenames = []
        with open(self.partition_path, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["partition"]) == split_id:
                    fname = row["image_id"]
                    if fname in self.labels:
                        filenames.append(fname)
        return filenames

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        fname = self.filenames[index]
        image = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[fname]


class SimpleCNN(nn.Module):
    """LeNet-style CNN for 64×64 CelebA classification."""
    
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True), nn.AvgPool2d(2),    # 64→32
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True), nn.AvgPool2d(2),  # 32→16
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True), nn.AvgPool2d(2), # 16→8
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x.view(x.size(0), -1))


class FederatedLearningSystem:
    """
    Federated Learning simulation system.
    
    Implements FedAvg-style training with gradient capture capability
    for gradient inversion attack research.
    """
    
    def __init__(
        self,
        num_clients: int = 10,
        num_classes: int = 2,
        batch_size: int = 64,
        data_subset: Optional[int] = None,
        device: Optional[str] = None,
        client_lr: float = 0.01,
        client_momentum: float = 0.9,
        local_epochs: int = 1,
        augment: bool = True,
        target_attr: str = "Male",
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
        self.target_attr = target_attr
        self.channel_mean = CELEBA_MEAN
        self.channel_std = CELEBA_STD

        self.global_model = SimpleCNN(num_classes).to(self.device)
        self.train_data, self.test_data = self._load_celeba()
        self.client_loaders = self._partition_data()
        self.test_loader = DataLoader(self.test_data, batch_size=128, shuffle=False)

    def _make_transform(self, augment: bool = False) -> transforms.Compose:
        """Create image transform pipeline."""
        ops = [
            transforms.Resize(IMAGE_SIZE),
            transforms.CenterCrop(IMAGE_SIZE),
        ]
        if augment:
            ops.append(transforms.RandomHorizontalFlip())
        ops.extend([
            transforms.ToTensor(),
            transforms.Normalize(CELEBA_MEAN, CELEBA_STD),
        ])
        return transforms.Compose(ops)

    def _load_celeba(self) -> Tuple[Dataset, Dataset]:
        """Load CelebA train and validation datasets."""
        train_data = CelebAAttributeDataset(
            root="./data",
            split="train",
            transform=self._make_transform(augment=self.augment),
            target_attr=self.target_attr,
        )
        test_data = CelebAAttributeDataset(
            root="./data",
            split="valid",
            transform=self._make_transform(augment=False),
            target_attr=self.target_attr,
        )

        if self.data_subset:
            train_data = Subset(train_data, list(range(min(len(train_data), self.data_subset))))
            test_data = Subset(test_data, list(range(min(len(test_data), self.data_subset // 5))))

        return train_data, test_data

    def _partition_data(self) -> List[DataLoader]:
        """Partition data IID across clients."""
        data_per_client = len(self.train_data) // self.num_clients
        loaders = []
        for i in range(self.num_clients):
            indices = list(range(i * data_per_client, (i + 1) * data_per_client))
            subset = Subset(self.train_data, indices)
            loaders.append(DataLoader(subset, batch_size=self.batch_size, shuffle=True))
        return loaders

    def train_client(
        self, 
        client_id: int, 
        epochs: Optional[int] = None, 
        capture: bool = False, 
        capture_mode: str = "gradients"
    ) -> Tuple[OrderedDict, Optional[Dict[str, Any]]]:
        """
        Train a single client and optionally capture gradients for attack.
        
        Args:
            client_id: Client index
            epochs: Local training epochs (defaults to self.local_epochs)
            capture: Whether to capture gradients
            capture_mode: 'gradients', 'one_step_update', or 'metadata'
        
        Returns:
            (model_update, captured_data)
        """
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        optimizer = torch.optim.SGD(
            local_model.parameters(), lr=self.client_lr, momentum=self.client_momentum
        )
        criterion = nn.CrossEntropyLoss()
        captured_data = None
        train_epochs = epochs or self.local_epochs

        for epoch in range(train_epochs):
            for batch_idx, (data, target) in enumerate(self.client_loaders[client_id]):
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                loss = criterion(local_model(data), target)
                loss.backward()

                # Capture on first batch of first epoch
                if capture and batch_idx == 0 and epoch == 0:
                    captured_data = self._capture_batch(
                        local_model, optimizer, data, target, capture_mode
                    )
                    if capture_mode == "one_step_update":
                        continue  # Already stepped

                optimizer.step()

        # Compute update delta
        update = OrderedDict()
        for (name, param), global_param in zip(
            local_model.named_parameters(), self.global_model.parameters()
        ):
            update[name] = param.data - global_param.data

        return update, captured_data

    def _capture_batch(
        self, 
        model: nn.Module, 
        optimizer: torch.optim.Optimizer,
        data: torch.Tensor, 
        target: torch.Tensor, 
        mode: str
    ) -> Dict[str, Any]:
        """Capture gradient or update information for attack."""
        base = {
            "true_data": data.detach().cpu().clone(),
            "true_label": target.detach().cpu().clone(),
        }
        
        if mode == "gradients":
            return {"source": "gradients", **base,
                    "gradients": [p.grad.clone() for p in model.parameters()]}
        
        elif mode == "one_step_update":
            before = [p.detach().clone() for p in model.parameters()]
            optimizer.step()
            after = [p.detach().clone() for p in model.parameters()]
            return {"source": "one_step_update", **base,
                    "first_update": [a - b for a, b in zip(after, before)],
                    "opt_lr": self.client_lr}
        
        else:  # metadata
            return {"source": "metadata", **base}

    def aggregate_updates(
        self, 
        updates: List[OrderedDict], 
        return_update: bool = False
    ) -> Optional[OrderedDict]:
        """FedAvg aggregation - average client updates and apply to global model."""
        avg_update = OrderedDict()
        for key in updates[0].keys():
            avg_update[key] = torch.stack([u[key] for u in updates]).mean(0)

        with torch.no_grad():
            for (name, param) in self.global_model.named_parameters():
                param.data += avg_update[name]

        return avg_update if return_update else None

    def evaluate(self) -> float:
        """Evaluate global model accuracy on test set."""
        self.global_model.eval()
        correct = total = 0
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = self.global_model(data).argmax(1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        return 100.0 * correct / total

    def train_round(
        self, 
        round_num: int, 
        capture_from_client: Optional[int] = None, 
        capture_mode: str = "gradients"
    ) -> Optional[Dict[str, Any]]:
        """
        Execute one federated learning round.
        
        Args:
            round_num: Current round number
            capture_from_client: Client ID to capture gradients from (None = no capture)
            capture_mode: Type of data to capture
        
        Returns:
            Captured data dict if capture_from_client is specified
        """
        num_participants = max(1, self.num_clients // 2)
        participants = np.random.choice(self.num_clients, num_participants, replace=False)
        
        # Ensure capture client participates
        if capture_from_client is not None and capture_from_client not in participants:
            participants = np.append(participants[:-1], capture_from_client)

        updates = []
        captured_data = None
        captured_metadata = None

        for client_id in participants:
            capture = (capture_from_client is not None and client_id == capture_from_client)
            mode = "metadata" if capture_mode == "agg_update" else capture_mode
            
            update, client_data = self.train_client(client_id, capture=capture, capture_mode=mode)
            updates.append(update)

            if client_data:
                if client_data.get("source") == "metadata":
                    captured_metadata = client_data
                else:
                    captured_data = client_data

        return_update = (capture_mode == "agg_update")
        avg_update = self.aggregate_updates(updates, return_update=return_update)

        # Package aggregated update for attack
        if capture_mode == "agg_update" and avg_update:
            captured_data = {
                "source": "agg_update",
                "avg_update": [avg_update[k].clone() for k in avg_update.keys()],
                "param_names": list(avg_update.keys()),
                "opt_lr": self.client_lr,
            }
            if captured_metadata:
                captured_data.update({
                    "true_data": captured_metadata["true_data"],
                    "true_label": captured_metadata["true_label"],
                })

        accuracy = self.evaluate()
        print(f"Round {round_num}: Accuracy = {accuracy:.2f}%")
        return captured_data
