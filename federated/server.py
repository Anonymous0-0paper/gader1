# federated/server.py
"""
Server-side implementation for MoE-FL
Handles aggregation with activation-weighted expert updates
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Tuple
import copy
from collections import defaultdict


class FederatedServer:
    """
    Federated learning server for MoE-FL
    Implements Algorithm 1 - Server-side operations
    """

    def __init__(
            self,
            model: nn.Module,
            config,
            device: str = "cuda"
    ):
        # Convert device string to torch.device
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.config = config
        self.num_experts = config.model.num_experts

        # Track global statistics
        self.round_metrics = []
        self.client_weights = {}  # p_i values

    def select_clients(self, round_num: int, num_clients: int) -> List[int]:
        """
        Select clients for the current round
        Algorithm 1, Line 4: Sample subset of clients

        Args:
            round_num: Current round number
            num_clients: Total number of clients

        Returns:
            selected_clients: List of selected client IDs
        """
        np.random.seed(round_num)  # For reproducibility
        num_selected = self.config.federated.clients_per_round
        selected_clients = np.random.choice(
            num_clients,
            size=min(num_selected, num_clients),
            replace=False
        ).tolist()

        return selected_clients

    def get_global_model_state(self) -> Dict:
        """
        Get current global model state
        Algorithm 1, Line 5: Send global model to clients

        Returns:
            model_state: Global model parameters
        """
        return copy.deepcopy(self.model.state_dict())

    # federated/server.py
    # Better version - only aggregate trainable parameters

    # federated/server.py
    # Replace the aggregate method with a memory-efficient version

    def aggregate(
            self,
            client_updates: List[Dict],
            client_metrics: List[Dict],
            activation_frequencies: List[np.ndarray]
    ) -> Dict:
        """
        Memory-efficient aggregation with activation-weighted expert updates
        Moves tensors to CPU during aggregation to avoid GPU OOM
        """
        num_clients = len(client_updates)

        # Compute client weights p_i (data proportion)
        total_samples = sum(m['num_samples'] for m in client_metrics)
        client_weights = [m['num_samples'] / total_samples for m in client_metrics]

        # Move model to CPU for aggregation to save GPU memory
        original_device = next(self.model.parameters()).device
        self.model.cpu()

        # Client updates are already on CPU from client.train()
        # No need to move them again

        # Clear GPU cache
        if original_device.type == 'cuda':
            torch.cuda.empty_cache()

        # Get current global state
        global_state = self.model.state_dict()

        # Get list of trainable parameter names (exclude buffers)
        trainable_params = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.add(name)

        # Initialize aggregated state
        aggregated_state = {}

        # Algorithm 1, Lines 14-16: Expert-wise aggregation
        for expert_id in range(self.num_experts):
            # Compute activation weights
            activation_weights = []
            for i, (p_i, a_i) in enumerate(zip(client_weights, activation_frequencies)):
                activation_weights.append(p_i * a_i[expert_id])

            # Normalize weights
            weight_sum = sum(activation_weights) + 1e-10
            activation_weights = [w / weight_sum for w in activation_weights]

            # Get expert parameters (only trainable ones)
            expert_params = self._get_expert_params(expert_id, global_state)
            trainable_expert_params = [p for p in expert_params if p in trainable_params]

            # Aggregate each parameter separately to minimize memory usage
            for param_name in trainable_expert_params:
                # Initialize on CPU
                aggregated_param = torch.zeros_like(global_state[param_name])

                # Aggregate from clients
                for client_idx, client_state in enumerate(client_updates):
                    weight = activation_weights[client_idx]
                    aggregated_param.add_(client_state[param_name], alpha=weight)

                aggregated_state[param_name] = aggregated_param

        # Algorithm 1, Line 18: Aggregate gating network (only trainable parameters)
        gating_params = self._get_gating_params(global_state)
        trainable_gating_params = [p for p in gating_params if p in trainable_params]

        for param_name in trainable_gating_params:
            # Initialize on CPU
            aggregated_param = torch.zeros_like(global_state[param_name])

            # Aggregate from clients
            for client_idx, client_state in enumerate(client_updates):
                weight = client_weights[client_idx]
                aggregated_param.add_(client_state[param_name], alpha=weight)

            aggregated_state[param_name] = aggregated_param

        # Copy non-trainable parameters (buffers) from global state
        for name in global_state.keys():
            if name not in aggregated_state:
                aggregated_state[name] = global_state[name]

        # Load aggregated state
        self.model.load_state_dict(aggregated_state)

        # Move model back to original device
        self.model.to(original_device)

        # Clear memory
        if original_device.type == 'cuda':
            torch.cuda.empty_cache()

        # Compute aggregation metrics
        aggregation_metrics = {
            'num_clients': num_clients,
            'avg_train_loss': np.mean([m['train_loss'] for m in client_metrics]),
            'avg_train_accuracy': np.mean([m['train_accuracy'] for m in client_metrics]),
            'avg_balance_loss': np.mean([m['balance_loss'] for m in client_metrics]),
            'expert_utilization': self._compute_expert_utilization(activation_frequencies),
            'routing_entropy': self._compute_routing_entropy(activation_frequencies)
        }

        return aggregation_metrics

    def _get_expert_params(self, expert_id: int, state_dict: Dict) -> List[str]:
        """Get parameter names for specific expert"""
        expert_params = []
        prefix = f"experts.{expert_id}."

        for key in state_dict.keys():
            if key.startswith(prefix):
                expert_params.append(key)

        return expert_params

    def _get_gating_params(self, state_dict: Dict) -> List[str]:
        """Get parameter names for gating network"""
        gating_params = []
        prefix = "gating_network."

        for key in state_dict.keys():
            if key.startswith(prefix):
                gating_params.append(key)

        return gating_params

    def _compute_expert_utilization(self, activation_frequencies: List[np.ndarray]) -> float:
        """
        Compute fraction of experts receiving >5% of activations
        Used in ablation studies (Table 5)
        """
        avg_frequencies = np.mean(activation_frequencies, axis=0)
        utilization = (avg_frequencies > 0.05).sum() / self.num_experts
        return float(utilization)

    def _compute_routing_entropy(self, activation_frequencies: List[np.ndarray]) -> float:
        """
        Compute average routing entropy across clients
        H_i = -sum_n a_{i,n} log(a_{i,n})
        Normalized by log(N) for comparison
        """
        entropies = []

        for a_i in activation_frequencies:
            # Add small epsilon to avoid log(0)
            a_i = a_i + 1e-10
            entropy = -np.sum(a_i * np.log(a_i))
            # Normalize by maximum entropy
            normalized_entropy = entropy / np.log(self.num_experts)
            entropies.append(normalized_entropy)

        return float(np.mean(entropies))

    def evaluate_global(self, test_loader) -> Dict:
        """
        Evaluate global model on centralized test set

        Args:
            test_loader: DataLoader with test data

        Returns:
            metrics: Global evaluation metrics
        """
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total = 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)

                output, _, _ = self.model(data, return_routing_info=False)

                loss = criterion(output, target)
                total_loss += loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        metrics = {
            'global_test_loss': total_loss / len(test_loader),
            'global_test_accuracy': 100. * correct / total
        }

        return metrics

    def save_checkpoint(self, round_num: int, save_path: str):
        """Save model checkpoint"""
        checkpoint = {
            'round': round_num,
            'model_state_dict': self.model.state_dict(),
            'config': self.config.to_dict(),
            'round_metrics': self.round_metrics
        }
        torch.save(checkpoint, save_path)

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.round_metrics = checkpoint['round_metrics']
        return checkpoint['round']


class AggregationStrategy:
    """
    Base class for different aggregation strategies
    Allows easy extension to other aggregation methods
    """

    @staticmethod
    def fedavg(client_updates: List[Dict], client_weights: List[float]) -> Dict:
        """Standard FedAvg aggregation"""
        aggregated_state = {}

        for key in client_updates[0].keys():
            aggregated_param = torch.zeros_like(client_updates[0][key])

            for client_state, weight in zip(client_updates, client_weights):
                aggregated_param += weight * client_state[key]

            aggregated_state[key] = aggregated_param

        return aggregated_state

    @staticmethod
    def activation_weighted(
            client_updates: List[Dict],
            client_weights: List[float],
            activation_frequencies: List[np.ndarray],
            num_experts: int,
            state_template: Dict
    ) -> Dict:
        """
        Activation-weighted aggregation for MoE-FL
        Algorithm 1, Lines 14-18
        """
        aggregated_state = {}

        # Aggregate expert parameters with activation weighting
        for expert_id in range(num_experts):
            # Compute activation weights
            activation_weights = []
            for p_i, a_i in zip(client_weights, activation_frequencies):
                activation_weights.append(p_i * a_i[expert_id])

            weight_sum = sum(activation_weights) + 1e-10
            activation_weights = [w / weight_sum for w in activation_weights]

            # Get expert parameter names
            expert_prefix = f"experts.{expert_id}."
            expert_params = [k for k in state_template.keys() if k.startswith(expert_prefix)]

            # Aggregate
            for param_name in expert_params:
                aggregated_param = torch.zeros_like(state_template[param_name])
                for client_state, weight in zip(client_updates, activation_weights):
                    aggregated_param += weight * client_state[param_name]
                aggregated_state[param_name] = aggregated_param

        # Aggregate gating network with data proportion weighting
        gating_prefix = "gating_network."
        gating_params = [k for k in state_template.keys() if k.startswith(gating_prefix)]

        for param_name in gating_params:
            aggregated_param = torch.zeros_like(state_template[param_name])
            for client_state, weight in zip(client_updates, client_weights):
                aggregated_param += weight * client_state[param_name]
            aggregated_state[param_name] = aggregated_param

        return aggregated_state
