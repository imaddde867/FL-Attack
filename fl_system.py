import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict
import numpy as np
import copy

class SimpleCNN(nn.Module):
    """Lightweight CNN for CIFAR-10"""
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # Use AvgPool2d for gradient inversion compatibility
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(2),  # Use AvgPool2d for gradient inversion compatibility
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class FederatedLearningSystem:
    def __init__(self, num_clients=10, num_classes=10, device='cuda'):
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.device = device
        
        # Initialize global model
        self.global_model = SimpleCNN(num_classes).to(device)
        
        # Load and partition data
        self.train_data, self.test_data = self._load_cifar10()
        self.client_loaders = self._partition_data()
        self.test_loader = DataLoader(self.test_data, batch_size=128, shuffle=False)
        
    def _load_cifar10(self):
        """Load CIFAR-10 with standard normalization"""
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2470, 0.2435, 0.2616))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2470, 0.2435, 0.2616))
        ])
        
        train_data = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
        test_data = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
        
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
            loader = DataLoader(subset, batch_size=64, shuffle=True)
            client_loaders.append(loader)
        
        return client_loaders
    
    def train_client(self, client_id, epochs=1, capture_gradients=False):
        """Train a single client and optionally capture gradients"""
        # Clone global model for local training
        local_model = copy.deepcopy(self.global_model)
        local_model.train()
        
        optimizer = torch.optim.SGD(local_model.parameters(), lr=0.01, momentum=0.9)
        criterion = nn.CrossEntropyLoss()
        
        captured_data = None
        
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.client_loaders[client_id]):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = local_model(data)
                loss = criterion(output, target)
                loss.backward()
                
                # Capture first batch gradients for attack
                if capture_gradients and batch_idx == 0 and epoch == 0:
                    captured_data = {
                        'gradients': [p.grad.clone() for p in local_model.parameters()],
                        'true_data': data[0:1].clone(),  # First image
                        'true_label': target[0:1].clone()
                    }
                
                optimizer.step()
        
        # Compute model update (delta)
        update = OrderedDict()
        for (name, param), global_param in zip(
            local_model.named_parameters(), 
            self.global_model.parameters()
        ):
            update[name] = param.data - global_param.data
        
        return update, captured_data
    
    def aggregate_updates(self, updates):
        """FedAvg aggregation"""
        # Average all updates
        avg_update = OrderedDict()
        for key in updates[0].keys():
            avg_update[key] = torch.stack([u[key] for u in updates]).mean(0)
        
        # Apply to global model
        with torch.no_grad():
            for (name, param), key in zip(
                self.global_model.named_parameters(), 
                avg_update.keys()
            ):
                param.data += avg_update[key]
    
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
    
    def train_round(self, round_num, capture_from_client=None):
        """Execute one federated learning round"""
        updates = []
        captured_data = None
        
        # Train subset of clients (e.g., 5 out of 10)
        num_participants = max(1, self.num_clients // 2)
        participants = np.random.choice(self.num_clients, num_participants, replace=False)
        
        for client_id in participants:
            capture = (capture_from_client is not None and client_id == capture_from_client)
            update, client_data = self.train_client(client_id, epochs=1, capture_gradients=capture)
            updates.append(update)
            
            if client_data is not None:
                captured_data = client_data
        
        # Aggregate updates
        self.aggregate_updates(updates)
        
        # Evaluate
        accuracy = self.evaluate()
        print(f"Round {round_num}: Accuracy = {accuracy:.2f}%")
        
        return captured_data