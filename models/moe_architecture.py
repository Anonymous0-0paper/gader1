# models/moe_architecture.py
"""
Mixture-of-Experts architecture for MoE-FL
Following Shazeer et al. (2017) sparse gating with GShard load balancing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List
import numpy as np


class GatingNetwork(nn.Module):
    """
    Gating network for routing inputs to experts
    Implements softmax-based routing with noisy top-k selection
    """

    def __init__(
            self,
            input_dim: int,
            num_experts: int,
            hidden_dims: List[int] = [512, 256],
            activation: str = "relu",
            noise_std: float = 1e-2
    ):
        super(GatingNetwork, self).__init__()

        self.num_experts = num_experts
        self.noise_std = noise_std

        # Build MLP layers
        layers = []
        in_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU() if activation == "relu" else nn.GELU()
            ])
            in_dim = hidden_dim

        # Output layer for expert scores
        layers.append(nn.Linear(in_dim, num_experts))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, input_dim]
            training: Whether in training mode (adds noise)

        Returns:
            gating_scores: Softmax probabilities [batch_size, num_experts]
        """
        # Get gating logits
        logits = self.network(x)

        # Add noise during training (Shazeer et al., 2017)
        if training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise

        # Softmax to get probabilities
        gating_scores = F.softmax(logits, dim=-1)

        return gating_scores


def create_batchnorm_free_resnet18(num_classes: int, input_channels: int = 3):
    """
    Create ResNet-18 with GroupNorm instead of BatchNorm
    This avoids issues with small batch sizes in federated learning
    """
    from torchvision.models import resnet18

    model = resnet18(weights=None, norm_layer=lambda channels: nn.GroupNorm(
        num_groups=min(32, channels // 16) if channels >= 16 else 1,
        num_channels=channels
    ))

    # Modify first conv if needed
    if input_channels != 3:
        model.conv1 = nn.Conv2d(
            input_channels, 64, kernel_size=7,
            stride=2, padding=3, bias=False
        )

    # Modify final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model

class ExpertNetwork(nn.Module):
    """
    Single expert network
    Can be ResNet, CNN, or LSTM based on task
    """

    def __init__(
            self,
            architecture: str,
            num_classes: int,
            input_channels: int = 3,
            expert_id: int = 0
    ):
        super(ExpertNetwork, self).__init__()

        self.architecture = architecture
        self.expert_id = expert_id

        if architecture == "ResNet18":
            self.network = self._build_resnet18(input_channels, num_classes)
        elif architecture == "CNN2Layer":
            self.network = self._build_cnn2layer(input_channels, num_classes)
        elif architecture == "LSTM2Layer":
            self.network = self._build_lstm2layer(num_classes)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def _build_resnet18(self, input_channels: int, num_classes: int):
        """Build ResNet-18 expert with GroupNorm for stable training"""
        return create_batchnorm_free_resnet18(num_classes, input_channels)

    def _build_cnn2layer(self, input_channels: int, num_classes: int):
        """Build simple 2-layer CNN expert for FEMNIST"""
        return nn.Sequential(
            # Conv layer 1
            nn.Conv2d(input_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Conv layer 2
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Flatten and FC
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def _build_lstm2layer(self, num_classes: int, embedding_dim: int = 128, hidden_dim: int = 256):
        """Build 2-layer LSTM expert for Shakespeare"""
        return nn.ModuleDict({
            'embedding': nn.Embedding(num_classes, embedding_dim),
            'lstm': nn.LSTM(embedding_dim, hidden_dim, num_layers=2,
                            batch_first=True, dropout=0.2),
            'fc': nn.Linear(hidden_dim, num_classes)
        })

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through expert"""
        if self.architecture == "LSTM2Layer":
            # Special handling for LSTM
            x = self.network['embedding'](x)
            lstm_out, _ = self.network['lstm'](x)
            x = self.network['fc'](lstm_out[:, -1, :])
        else:
            x = self.network(x)

        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get intermediate features for gating network"""
        if self.architecture == "ResNet18":
            # Get features before final FC layer
            x = self.network.conv1(x)
            x = self.network.bn1(x)
            x = self.network.relu(x)
            x = self.network.maxpool(x)

            x = self.network.layer1(x)
            x = self.network.layer2(x)
            x = self.network.layer3(x)
            x = self.network.layer4(x)

            x = self.network.avgpool(x)
            x = torch.flatten(x, 1)

            return x
        elif self.architecture == "CNN2Layer":
            # Get features after conv layers
            for layer in self.network[:8]:  # Up to flatten
                x = layer(x)
            return x
        else:
            # For LSTM, use embedding output
            return self.network['embedding'](x).mean(dim=1)


class MixtureOfExpertsModel(nn.Module):
    """
    Complete Mixture-of-Experts model for federated learning
    Implements sparse top-k routing with load balancing
    """

    def __init__(
            self,
            num_experts: int,
            top_k: int,
            expert_architecture: str,
            num_classes: int,
            input_channels: int = 3,
            gating_hidden_dims: List[int] = [512, 256],
            balance_loss_weight: float = 0.01,
            noise_std: float = 1e-2
    ):
        super(MixtureOfExpertsModel, self).__init__()

        self.num_experts = num_experts
        self.top_k = top_k
        self.balance_loss_weight = balance_loss_weight
        self.expert_architecture = expert_architecture

        # Create experts with diverse initialization
        self.experts = nn.ModuleList([
            ExpertNetwork(
                architecture=expert_architecture,
                num_classes=num_classes,
                input_channels=input_channels,
                expert_id=i
            )
            for i in range(num_experts)
        ])

        # Initialize experts with different seeds for diversity
        for i, expert in enumerate(self.experts):
            torch.manual_seed(42 + i)
            for param in expert.parameters():
                if param.dim() > 1:
                    nn.init.kaiming_normal_(param)

        # Determine gating input dimension using batch_size=2 for BatchNorm
        with torch.no_grad():
            dummy_input = self._create_dummy_input(expert_architecture, input_channels)
            feature_sample = self.experts[0].get_features(dummy_input)
            feature_dim = feature_sample.shape[-1]

        # Create gating network
        self.gating_network = GatingNetwork(
            input_dim=feature_dim,
            num_experts=num_experts,
            hidden_dims=gating_hidden_dims,
            noise_std=noise_std
        )

        # Track activation statistics for load balancing
        self.register_buffer('expert_counts', torch.zeros(num_experts))
        self.register_buffer('total_samples', torch.tensor(0))

    def _create_dummy_input(self, architecture: str, input_channels: int):
        """Create dummy input for dimension inference - use batch_size=2 for BatchNorm"""
        if architecture in ["ResNet18", "CNN2Layer"]:
            return torch.randn(2, input_channels, 32, 32)  # batch_size=2
        else:  # LSTM
            return torch.randint(0, 80, (2, 80))  # batch_size=2

    def forward(
            self,
            x: torch.Tensor,
            return_routing_info: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Forward pass with sparse expert selection

        Args:
            x: Input tensor [batch_size, ...]
            return_routing_info: Whether to return routing statistics

        Returns:
            output: Model output [batch_size, num_classes]
            balance_loss: Load balancing auxiliary loss
            routing_info: Dictionary with routing statistics (if requested)
        """
        batch_size = x.shape[0]

        # Get features for gating
        features = self.experts[0].get_features(x)

        # Get gating scores
        gating_scores = self.gating_network(features, training=self.training)

        # Top-k expert selection
        topk_scores, topk_indices = torch.topk(gating_scores, self.top_k, dim=-1)

        # Renormalize top-k scores
        topk_scores = topk_scores / topk_scores.sum(dim=-1, keepdim=True)

        # Get output dimension from first expert
        with torch.no_grad():
            if self.expert_architecture in ["ResNet18", "CNN2Layer"]:
                dummy_out = self.experts[0](x[:1])
            else:  # LSTM
                dummy_out = self.experts[0](x[:1])
            output_dim = dummy_out.shape[-1]

        # Initialize output
        output = torch.zeros(batch_size, output_dim, device=x.device, dtype=x.dtype)

        # Process each sample individually to avoid BatchNorm issues with single samples
        # When batch_size is large enough, we can batch process
        if batch_size >= 2 * self.num_experts:
            # Batch processing: group samples by expert
            for expert_id in range(self.num_experts):
                # Find all samples that should use this expert
                expert_mask = (topk_indices == expert_id).any(dim=1)

                if expert_mask.sum() > 1:  # Only process if at least 2 samples (for BatchNorm)
                    # Get indices and weights for this expert
                    sample_indices = torch.where(expert_mask)[0]

                    # Get expert output for these samples
                    expert_input = x[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)

                    # Weight and accumulate
                    for idx, sample_idx in enumerate(sample_indices):
                        # Find weight for this expert for this sample
                        expert_positions = (topk_indices[sample_idx] == expert_id).nonzero(as_tuple=True)[0]
                        if len(expert_positions) > 0:
                            weight = topk_scores[sample_idx, expert_positions[0]]
                            output[sample_idx] += weight * expert_output[idx]
                elif expert_mask.sum() == 1:  # Single sample - use eval mode
                    sample_idx = torch.where(expert_mask)[0][0]
                    expert_input = x[sample_idx:sample_idx + 1]

                    # Temporarily switch to eval mode for single sample
                    was_training = self.experts[expert_id].training
                    self.experts[expert_id].eval()
                    with torch.no_grad():
                        expert_output = self.experts[expert_id](expert_input)
                    if was_training:
                        self.experts[expert_id].train()

                    # Find weight
                    expert_positions = (topk_indices[sample_idx] == expert_id).nonzero(as_tuple=True)[0]
                    if len(expert_positions) > 0:
                        weight = topk_scores[sample_idx, expert_positions[0]]
                        output[sample_idx] += weight * expert_output[0]
        else:
            # Small batch: process each sample individually in eval mode
            for i in range(batch_size):
                sample_experts = topk_indices[i]
                sample_weights = topk_scores[i]

                for j, expert_id in enumerate(sample_experts):
                    # Set expert to eval mode temporarily
                    was_training = self.experts[expert_id].training
                    self.experts[expert_id].eval()

                    with torch.no_grad():
                        expert_output = self.experts[expert_id](x[i:i + 1])

                    # Restore training mode
                    if was_training:
                        self.experts[expert_id].train()

                    output[i] += sample_weights[j] * expert_output[0]

        # Update activation counts for load balancing
        if self.training:
            unique_experts, counts = torch.unique(topk_indices, return_counts=True)
            for exp_id, count in zip(unique_experts, counts):
                self.expert_counts[exp_id] += count
            self.total_samples += batch_size

        # Compute load balancing loss (CV-based from GShard)
        balance_loss = self._compute_balance_loss(topk_indices)

        # Prepare routing info
        routing_info = {}
        if return_routing_info:
            routing_info = {
                'gating_scores': gating_scores.detach(),
                'selected_experts': topk_indices.detach(),
                'expert_weights': topk_scores.detach(),
                'activation_frequencies': self._get_activation_frequencies()
            }

        return output, balance_loss, routing_info

    def _compute_balance_loss(self, selected_experts: torch.Tensor) -> torch.Tensor:
        """
        Compute coefficient of variation (CV) based load balancing loss
        Following GShard (Lepikhin et al., 2021)
        """
        if not self.training:
            return torch.tensor(0.0).to(selected_experts.device)

        # Count expert usage in current batch
        batch_counts = torch.bincount(
            selected_experts.flatten(),
            minlength=self.num_experts
        ).float()

        # Normalize to get frequencies
        frequencies = batch_counts / (batch_counts.sum() + 1e-10)

        # Compute coefficient of variation
        mean_freq = frequencies.mean()
        std_freq = frequencies.std()
        cv = std_freq / (mean_freq + 1e-10)

        return self.balance_loss_weight * cv

    def _get_activation_frequencies(self) -> torch.Tensor:
        """Get normalized activation frequencies for each expert"""
        if self.total_samples == 0:
            return torch.ones(self.num_experts) / self.num_experts

        return self.expert_counts / self.total_samples

    def reset_activation_stats(self):
        """Reset activation statistics"""
        self.expert_counts.zero_()
        self.total_samples.zero_()

    def get_expert_parameters(self, expert_id: int) -> List[torch.Tensor]:
        """Get parameters of specific expert"""
        return list(self.experts[expert_id].parameters())

    def get_gating_parameters(self) -> List[torch.Tensor]:
        """Get gating network parameters"""
        return list(self.gating_network.parameters())


def create_moe_model(config) -> MixtureOfExpertsModel:
    """Factory function to create MoE model from config"""
    model = MixtureOfExpertsModel(
        num_experts=config.model.num_experts,
        top_k=config.model.top_k,
        expert_architecture=config.model.expert_architecture,
        num_classes=config.model.num_classes,
        input_channels=config.model.input_channels,
        gating_hidden_dims=config.model.gating_hidden_dims,
        balance_loss_weight=config.model.balance_loss_weight,
        noise_std=config.model.routing_noise_std
    )

    return model