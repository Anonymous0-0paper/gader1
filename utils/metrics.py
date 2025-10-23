# utils/metrics.py
"""
Metrics computation and tracking for experiments
"""

import numpy as np
import pandas as pd
from typing import Dict, List
import os


class MetricsTracker:
    """
    Track and compute metrics throughout training
    """

    def __init__(self, save_dir: str):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        # Storage for metrics
        self.round_metrics = []
        self.client_metrics = {}

    def update_round(self, round_num: int, metrics: Dict):
        """Update metrics for current round"""
        metrics['round'] = round_num
        self.round_metrics.append(metrics)

    def update_client(self, client_id: int, round_num: int, metrics: Dict):
        """Update metrics for specific client"""
        if client_id not in self.client_metrics:
            self.client_metrics[client_id] = []

        metrics['round'] = round_num
        self.client_metrics[client_id].append(metrics)

    def get_personalized_accuracy(self, round_num: int = None) -> float:
        """
        Compute average personalized accuracy across all clients
        Table 4 metric: Pers. Acc.
        """
        if round_num is None:
            # Get latest round
            round_num = max([m['round'] for metrics in self.client_metrics.values()
                             for m in metrics])

        accuracies = []
        for client_id, metrics_list in self.client_metrics.items():
            round_metrics = [m for m in metrics_list if m['round'] == round_num]
            if round_metrics:
                accuracies.append(round_metrics[-1]['test_accuracy'])

        return np.mean(accuracies) if accuracies else 0.0

    def get_global_accuracy(self, round_num: int = None) -> float:
        """
        Get global test accuracy
        Table 4 metric: Global Acc.
        """
        if round_num is None:
            round_num = max([m['round'] for m in self.round_metrics])

        round_data = [m for m in self.round_metrics if m['round'] == round_num]
        if round_data and 'global_test_accuracy' in round_data[-1]:
            return round_data[-1]['global_test_accuracy']
        return 0.0

    def get_routing_entropy(self, round_num: int = None) -> float:
        """
        Get average routing entropy
        Table 5 metric: Routing Entropy
        """
        if round_num is None:
            round_num = max([m['round'] for m in self.round_metrics])

        round_data = [m for m in self.round_metrics if m['round'] == round_num]
        if round_data and 'routing_entropy' in round_data[-1]:
            return round_data[-1]['routing_entropy']
        return 0.0

    def get_expert_utilization(self, round_num: int = None) -> float:
        """
        Get expert utilization
        Table 5 metric: Expert Utilization
        """
        if round_num is None:
            round_num = max([m['round'] for m in self.round_metrics])

        round_data = [m for m in self.round_metrics if m['round'] == round_num]
        if round_data and 'expert_utilization' in round_data[-1]:
            return round_data[-1]['expert_utilization']
        return 0.0

    def save_to_csv(self, filename: str = "metrics.csv"):
        """Save metrics to CSV file"""
        # Save round-level metrics
        if self.round_metrics:
            df_rounds = pd.DataFrame(self.round_metrics)
            round_path = os.path.join(self.save_dir, f"round_{filename}")
            df_rounds.to_csv(round_path, index=False)
            print(f"Saved round metrics to {round_path}")

        # Save client-level metrics
        all_client_data = []
        for client_id, metrics_list in self.client_metrics.items():
            for metrics in metrics_list:
                data = {'client_id': client_id}
                data.update(metrics)
                all_client_data.append(data)

        if all_client_data:
            df_clients = pd.DataFrame(all_client_data)
            client_path = os.path.join(self.save_dir, f"client_{filename}")
            df_clients.to_csv(client_path, index=False)
            print(f"Saved client metrics to {client_path}")

    def get_summary(self) -> Dict:
        """Get summary statistics"""
        final_round = max([m['round'] for m in self.round_metrics]) if self.round_metrics else 0

        summary = {
            'final_round': final_round,
            'personalized_accuracy': self.get_personalized_accuracy(final_round),
            'global_accuracy': self.get_global_accuracy(final_round),
            'routing_entropy': self.get_routing_entropy(final_round),
            'expert_utilization': self.get_expert_utilization(final_round)
        }

        return summary


def compute_communication_cost(
        num_parameters: int,
        num_rounds: int,
        clients_per_round: int,
        bytes_per_param: int = 4
) -> float:
    """
    Compute total communication cost in MB
    Table 6 metric: Comm. Cost (MB)

    Args:
        num_parameters: Total number of model parameters
        num_rounds: Number of communication rounds
        clients_per_round: Clients participating per round
        bytes_per_param: Bytes per parameter (4 for float32)

    Returns:
        communication_cost: Total communication in MB
    """
    # Download: server sends model to clients
    download_per_round = num_parameters * clients_per_round * bytes_per_param

    # Upload: clients send updates to server
    upload_per_round = num_parameters * clients_per_round * bytes_per_param

    # Total over all rounds
    total_bytes = (download_per_round + upload_per_round) * num_rounds

    # Convert to MB
    total_mb = total_bytes / (1024 ** 2)

    return total_mb


def compute_convergence_round(
        metrics_list: List[Dict],
        target_accuracy: float,
        metric_name: str = 'test_accuracy'
) -> int:
    """
    Find round number when target accuracy is first reached
    Used for Table 6: communication cost to reach target

    Args:
        metrics_list: List of metrics dictionaries
        target_accuracy: Target accuracy threshold
        metric_name: Name of metric to check

    Returns:
        round_num: Round when target is reached, or -1 if never reached
    """
    for metrics in metrics_list:
        if metrics.get(metric_name, 0) >= target_accuracy:
            return metrics['round']

    return -1  # Target not reached