# config.py
"""
Configuration file for MoE-FL: Mixture-of-Experts Federated Learning
"""

import torch
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Dataset specific
    dataset: str = "CIFAR10"  # CIFAR10, CIFAR100, FEMNIST, Shakespeare
    num_classes: int = 10
    input_channels: int = 3

    # MoE Architecture
    num_experts: int = 8
    top_k: int = 2
    expert_architecture: str = "ResNet18"  # ResNet18, CNN2Layer, LSTM2Layer
    gating_hidden_dims: List[int] = None  # Will be set based on dataset
    gating_activation: str = "relu"

    # Load balancing
    balance_loss_weight: float = 0.01
    balance_type: str = "cv"  # coefficient of variation

    # Expert initialization
    expert_init_diversity: bool = True
    routing_noise_std: float = 1e-2

    def __post_init__(self):
        if self.gating_hidden_dims is None:
            if self.dataset in ["CIFAR10", "CIFAR100", "Shakespeare"]:
                self.gating_hidden_dims = [512, 256]
            elif self.dataset == "FEMNIST":
                self.gating_hidden_dims = [256, 128]


@dataclass
class FederatedConfig:
    """Federated learning configuration"""
    # Federation setup
    num_clients: int = 50
    clients_per_round: int = 5
    participation_rate: float = 0.1

    # Training
    num_rounds: int = 200
    local_epochs: int = 10
    batch_size: int = 16

    # Optimization
    optimizer: str = "sgd"  # sgd, adam
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_scheduler: str = "cosine"  # cosine, step
    warmup_rounds: int = 10

    # Privacy
    use_privacy: bool = True
    privacy_epsilon: float = 5.0

    # Aggregation
    aggregation_type: str = "activation_weighted"  # uniform, activation_weighted

    # Communication
    communication_timeout: int = 300  # seconds
    use_compression: bool = True


@dataclass
class DataConfig:
    """Data configuration"""
    # Data paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    checkpoint_dir: str = "./checkpoints"

    # Data preprocessing
    normalization: dict = None
    augmentation: bool = True

    # Heterogeneity
    heterogeneity_type: str = "label_skew"  # label_skew, feature_skew, natural
    dirichlet_alpha: float = 0.5

    # Splits
    test_split: float = 0.2
    validation_split: float = 0.1
    min_samples_per_client: int = 50

    def __post_init__(self):
        if self.normalization is None:
            self.normalization = {
                "CIFAR10": {
                    "mean": [0.4914, 0.4822, 0.4465],
                    "std": [0.2470, 0.2435, 0.2616]
                },
                "CIFAR100": {
                    "mean": [0.5071, 0.4867, 0.4408],
                    "std": [0.2675, 0.2565, 0.2761]
                },
                "FEMNIST": {
                    "mean": [0.9637],
                    "std": [0.1597]
                }
            }


@dataclass
class ExperimentConfig:
    """Experiment configuration"""
    # Reproducibility
    random_seed: int = 42
    num_runs: int = 5
    seeds: List[int] = None

    # Logging
    log_interval: int = 5
    eval_interval: int = 5
    checkpoint_interval: int = 10
    verbose: bool = True

    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    pin_memory: bool = True
    use_mixed_precision: bool = True

    # Evaluation
    save_results: bool = True
    save_plots: bool = True
    plot_formats: List[str] = None

    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [42, 123, 456, 789, 1024]
        if self.plot_formats is None:
            self.plot_formats = ["png", "pdf"]


@dataclass
class MoEFLConfig:
    """Complete MoE-FL configuration"""
    model: ModelConfig = None
    federated: FederatedConfig = None
    data: DataConfig = None
    experiment: ExperimentConfig = None

    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.federated is None:
            self.federated = FederatedConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.experiment is None:
            self.experiment = ExperimentConfig()

    @classmethod
    def from_dataset(cls, dataset: str):
        """Create configuration for specific dataset"""
        configs = {
            "CIFAR10": cls(
                model=ModelConfig(
                    dataset="CIFAR10",
                    num_classes=10,
                    num_experts=8,
                    top_k=2,
                    expert_architecture="ResNet18"
                ),
                federated=FederatedConfig(
                    num_rounds=200,
                    local_epochs=5,
                    batch_size=64,
                    learning_rate=0.01
                ),
                data=DataConfig(
                    dirichlet_alpha=0.5
                )
            ),
            "CIFAR100": cls(
                model=ModelConfig(
                    dataset="CIFAR100",
                    num_classes=100,
                    num_experts=12,
                    top_k=3,
                    expert_architecture="ResNet18"
                ),
                federated=FederatedConfig(
                    num_rounds=300,
                    local_epochs=5,
                    batch_size=64,
                    learning_rate=0.01
                ),
                data=DataConfig(
                    dirichlet_alpha=0.1
                )
            ),
            "FEMNIST": cls(
                model=ModelConfig(
                    dataset="FEMNIST",
                    num_classes=62,
                    num_experts=8,
                    top_k=2,
                    expert_architecture="CNN2Layer",
                    input_channels=1
                ),
                federated=FederatedConfig(
                    num_rounds=150,
                    local_epochs=3,
                    batch_size=32,
                    learning_rate=0.02
                ),
                data=DataConfig(
                    heterogeneity_type="natural"
                )
            ),
            "Shakespeare": cls(
                model=ModelConfig(
                    dataset="Shakespeare",
                    num_classes=80,
                    num_experts=6,
                    top_k=2,
                    expert_architecture="LSTM2Layer"
                ),
                federated=FederatedConfig(
                    num_rounds=100,
                    local_epochs=2,
                    batch_size=32,
                    learning_rate=0.015,
                    optimizer="adam"
                ),
                data=DataConfig(
                    heterogeneity_type="natural"
                )
            )
        }

        if dataset not in configs:
            raise ValueError(f"Unknown dataset: {dataset}. Choose from {list(configs.keys())}")

        return configs[dataset]

    def to_dict(self):
        """Convert config to dictionary"""
        return {
            "model": self.model.__dict__,
            "federated": self.federated.__dict__,
            "data": self.data.__dict__,
            "experiment": self.experiment.__dict__
        }


# Preset configurations for paper experiments
CIFAR10_CONFIG = MoEFLConfig.from_dataset("CIFAR10")
CIFAR100_CONFIG = MoEFLConfig.from_dataset("CIFAR100")
FEMNIST_CONFIG = MoEFLConfig.from_dataset("FEMNIST")
SHAKESPEARE_CONFIG = MoEFLConfig.from_dataset("Shakespeare")
