# federated/client.py
"""
Client-side implementation for MoE-FL
Handles local training with activation tracking
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Tuple, Dict
import numpy as np
import copy


# federated/client.py
# Fix the __init__ method and device checks

class FederatedClient:
    """
    Federated learning client with MoE model
    Implements Algorithm 1 - ClientUpdate function
    """

    def __init__(
            self,
            client_id: int,
            model: nn.Module,
            train_loader: DataLoader,
            test_loader: DataLoader,
            config,
            device: str = "cuda"
    ):
        self.client_id = client_id
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.config = config

        # Convert device string to torch.device
        self.device = torch.device(device)
        self.model = model.to(self.device)

        # Activation tracking
        self.activation_counts = np.zeros(config.model.num_experts)
        self.total_samples = 0

        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        self.criterion = nn.CrossEntropyLoss()

    def _create_optimizer(self):
        """Create optimizer based on config"""
        if self.config.federated.optimizer == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.federated.learning_rate,
                momentum=self.config.federated.momentum,
                weight_decay=self.config.federated.weight_decay
            )
        elif self.config.federated.optimizer == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.federated.learning_rate,
                weight_decay=self.config.federated.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.federated.optimizer}")

    def update_model(self, global_model_state: Dict):
        """
        Update local model with global model state
        Algorithm 1, Line 17: Initialize local parameters
        """
        self.model.load_state_dict(global_model_state)
        self.activation_counts = np.zeros(self.config.model.num_experts)
        self.total_samples = 0

    # federated/client.py
    # Update the train method to be more memory efficient

    def train(self, current_round: int) -> Tuple[Dict, Dict, np.ndarray]:
        """
        Perform local training with memory management
        Algorithm 1, Lines 17-31: ClientUpdate function

        Returns:
            model_state: Updated model parameters (on CPU)
            metrics: Training metrics
            activation_frequencies: Expert activation frequencies
        """
        self.model.train()

        total_loss = 0.0
        total_balance_loss = 0.0
        total_task_loss = 0.0
        correct = 0
        total = 0

        # Algorithm 1, Line 19: Loop over epochs
        for epoch in range(self.config.federated.local_epochs):
            epoch_loss = 0.0

            # Algorithm 1, Line 20: Loop over batches
            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()

                # Algorithm 1, Lines 21-24: Forward pass with routing
                output, balance_loss, routing_info = self.model(
                    data,
                    return_routing_info=True
                )

                # Algorithm 1, Line 25: Update activation counters
                if routing_info:
                    selected_experts = routing_info['selected_experts'].cpu().numpy()
                    for expert_id in selected_experts.flatten():
                        self.activation_counts[expert_id] += 1
                    self.total_samples += data.size(0)

                # Algorithm 1, Lines 27-29: Compute losses
                task_loss = self.criterion(output, target)
                total_loss_batch = task_loss + balance_loss

                # Algorithm 1, Lines 31-32: Backward pass and update
                total_loss_batch.backward()

                # Gradient clipping for Shakespeare
                if self.config.model.dataset == "Shakespeare":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                self.optimizer.step()

                # Track metrics
                epoch_loss += total_loss_batch.item()
                total_task_loss += task_loss.item()
                total_balance_loss += balance_loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Clear cache periodically
                if batch_idx % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            if self.config.experiment.verbose and epoch % 2 == 0:
                print(f"Client {self.client_id}, Round {current_round}, "
                      f"Epoch {epoch}, Loss: {epoch_loss / len(self.train_loader):.4f}")

        # Algorithm 1, Line 34: Normalize activation frequencies
        activation_frequencies = self.activation_counts / (self.total_samples + 1e-10)

        # Prepare metrics
        metrics = {
            'train_loss': total_task_loss / (self.config.federated.local_epochs * len(self.train_loader)),
            'train_accuracy': 100. * correct / total,
            'balance_loss': total_balance_loss / (self.config.federated.local_epochs * len(self.train_loader)),
            'num_samples': len(self.train_loader.dataset)
        }

        # Move model state to CPU before returning to save GPU memory
        model_state_cpu = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Clear GPU cache
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Algorithm 1, Line 35: Return updated parameters
        return model_state_cpu, metrics, activation_frequencies

    def evaluate(self) -> Dict:
        """
        Evaluate model on local test set

        Returns:
            metrics: Evaluation metrics
        """
        self.model.eval()

        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output, balance_loss, _ = self.model(data, return_routing_info=False)

                loss = self.criterion(output, target)
                test_loss += loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        metrics = {
            'test_loss': test_loss / len(self.test_loader),
            'test_accuracy': 100. * correct / total,
            'num_samples': total
        }

        return metrics

    def add_privacy_noise(
            self,
            activation_frequencies: np.ndarray,
            epsilon: float
    ) -> np.ndarray:
        """
        Add Laplace noise for differential privacy
        Algorithm 1, Lines 10-12: Privacy-preserving activation reporting

        Args:
            activation_frequencies: True activation frequencies
            epsilon: Privacy budget

        Returns:
            noised_frequencies: Activation frequencies with added noise
        """
        # Sensitivity Delta = 1 / |D_i|
        sensitivity = 1.0 / max(self.total_samples, 1)

        # Scale parameter for Laplace distribution
        scale = sensitivity / epsilon

        # Add Laplace noise
        noise = np.random.laplace(0, scale, size=activation_frequencies.shape)
        noised_frequencies = activation_frequencies + noise

        # Clip to valid probability range
        noised_frequencies = np.clip(noised_frequencies, 0, 1)

        # Renormalize
        noised_frequencies = noised_frequencies / (noised_frequencies.sum() + 1e-10)

        return noised_frequencies


def create_client(client_id: int,
        model: nn.Module,
        train_dataset,
        test_dataset,
        config,
        device: str = "cuda") -> FederatedClient:
    """
    Factory function to create a federated client
    
    Args:
        client_id: Unique client identifier
        model: MoE model instance
        train_dataset: Training dataset for this client
        test_dataset: Test dataset for this client
        config: Configuration object
        device: Device to run on

    Returns:
        client: FederatedClient instance
    """
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.federated.batch_size,
        shuffle=True,
        num_workers=config.experiment.num_workers,
        pin_memory=config.experiment.pin_memory
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.federated.batch_size,
        shuffle=False,
        num_workers=config.experiment.num_workers,
        pin_memory=config.experiment.pin_memory
    )

    return FederatedClient(
        client_id=client_id,
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        config=config,
        device=device
    )
