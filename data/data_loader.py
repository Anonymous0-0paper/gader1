# data/data_loader.py
"""
Data loading and partitioning for federated learning
Implements Dirichlet distribution partitioning and natural splits
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import List, Tuple, Dict
import os


class FederatedDataset:
    """
    Manages federated data partitioning
    """

    def __init__(self, config):
        self.config = config
        self.dataset_name = config.model.dataset

        # Load dataset
        self.train_dataset, self.test_dataset = self._load_dataset()

        # Partition data
        if config.data.heterogeneity_type == "label_skew":
            self.client_datasets = self._partition_dirichlet()
        elif config.data.heterogeneity_type == "natural":
            self.client_datasets = self._partition_natural()
        else:
            raise ValueError(f"Unknown heterogeneity type: {config.data.heterogeneity_type}")

    def _load_dataset(self):
        """Load dataset based on config"""
        data_dir = self.config.data.data_dir

        if self.dataset_name == "CIFAR10":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.data.normalization["CIFAR10"]["mean"],
                    std=self.config.data.normalization["CIFAR10"]["std"]
                )
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.data.normalization["CIFAR10"]["mean"],
                    std=self.config.data.normalization["CIFAR10"]["std"]
                )
            ])

            train_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform_test
            )

        elif self.dataset_name == "CIFAR100":
            transform_train = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.data.normalization["CIFAR100"]["mean"],
                    std=self.config.data.normalization["CIFAR100"]["std"]
                )
            ])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.data.normalization["CIFAR100"]["mean"],
                    std=self.config.data.normalization["CIFAR100"]["std"]
                )
            ])

            train_dataset = torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=transform_train
            )
            test_dataset = torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=transform_test
            )

        elif self.dataset_name == "FEMNIST":
            # Note: FEMNIST requires LEAF dataset
            # For now, using EMNIST as placeholder
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.config.data.normalization["FEMNIST"]["mean"],
                    std=self.config.data.normalization["FEMNIST"]["std"]
                )
            ])

            train_dataset = torchvision.datasets.EMNIST(
                root=data_dir, split='byclass', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.EMNIST(
                root=data_dir, split='byclass', train=False, download=True, transform=transform
            )

        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return train_dataset, test_dataset

    def _partition_dirichlet(self) -> List[Tuple[Subset, Subset]]:
        """
        Partition data using Dirichlet distribution for label skew
        Following Hsu et al. (2019) methodology
        """
        num_clients = self.config.federated.num_clients
        alpha = self.config.data.dirichlet_alpha
        min_size = self.config.data.min_samples_per_client

        # Get labels
        if hasattr(self.train_dataset, 'targets'):
            labels = np.array(self.train_dataset.targets)
        else:
            labels = np.array([y for _, y in self.train_dataset])

        num_classes = len(np.unique(labels))
        num_samples = len(labels)

        # Initialize client data indices
        client_indices = [[] for _ in range(num_clients)]

        # For each class, distribute samples to clients using Dirichlet
        for c in range(num_classes):
            class_indices = np.where(labels == c)[0]
            np.random.shuffle(class_indices)

            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(alpha * np.ones(num_clients))

            # Allocate samples proportionally
            proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
            client_class_indices = np.split(class_indices, proportions)

            # Assign to clients
            for client_id, indices in enumerate(client_class_indices):
                client_indices[client_id].extend(indices.tolist())

        # Ensure minimum samples per client
        for client_id in range(num_clients):
            if len(client_indices[client_id]) < min_size:
                # Sample additional data from other clients
                deficit = min_size - len(client_indices[client_id])
                # Find clients with excess data
                excess_clients = [i for i in range(num_clients)
                                  if len(client_indices[i]) > min_size + deficit]
                if excess_clients:
                    donor = np.random.choice(excess_clients)
                    # Transfer samples
                    transfer = client_indices[donor][:deficit]
                    client_indices[client_id].extend(transfer)
                    client_indices[donor] = client_indices[donor][deficit:]

        # Create train/test splits for each client
        client_datasets = []
        test_split = self.config.data.test_split

        for indices in client_indices:
            np.random.shuffle(indices)
            split_point = int(len(indices) * (1 - test_split))

            train_indices = indices[:split_point]
            test_indices = indices[split_point:]

            train_subset = Subset(self.train_dataset, train_indices)
            test_subset = Subset(self.train_dataset, test_indices)

            client_datasets.append((train_subset, test_subset))

        # Print statistics
        self._print_partition_stats(client_indices, labels, num_classes)

        return client_datasets

    def _partition_natural(self) -> List[Tuple[Subset, Subset]]:
        """
        Natural partitioning based on user identity (for FEMNIST/Shakespeare)
        """
        # For EMNIST (FEMNIST placeholder), partition by writer
        # In real FEMNIST, this would use LEAF benchmark's natural splits

        num_clients = self.config.federated.num_clients
        num_samples = len(self.train_dataset)

        # Simple partitioning: divide data into chunks
        samples_per_client = num_samples // num_clients

        client_datasets = []
        for i in range(num_clients):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client if i < num_clients - 1 else num_samples

            indices = list(range(start_idx, end_idx))
            np.random.shuffle(indices)

            split_point = int(len(indices) * (1 - self.config.data.test_split))
            train_indices = indices[:split_point]
            test_indices = indices[split_point:]

            train_subset = Subset(self.train_dataset, train_indices)
            test_subset = Subset(self.train_dataset, test_indices)

            client_datasets.append((train_subset, test_subset))

        return client_datasets

    def _print_partition_stats(self, client_indices: List, labels: np.ndarray, num_classes: int):
        """Print statistics about data partitioning"""
        print("\n" + "=" * 50)
        print("DATA PARTITIONING STATISTICS")
        print("=" * 50)

        # Samples per client
        samples_per_client = [len(indices) for indices in client_indices]
        print(f"Samples per client - Mean: {np.mean(samples_per_client):.1f}, "
              f"Std: {np.std(samples_per_client):.1f}")
        print(f"Min: {np.min(samples_per_client)}, Max: {np.max(samples_per_client)}")

        # Classes per client
        classes_per_client = []
        for indices in client_indices:
            client_labels = labels[indices]
            unique_classes = len(np.unique(client_labels))
            classes_per_client.append(unique_classes)

        print(f"\nClasses per client - Mean: {np.mean(classes_per_client):.1f}, "
              f"Std: {np.std(classes_per_client):.1f}")

        # Class imbalance
        max_imbalance = 0
        for indices in client_indices:
            client_labels = labels[indices]
            class_counts = [np.sum(client_labels == c) for c in range(num_classes)]
            class_counts = [c for c in class_counts if c > 0]
            if len(class_counts) > 1:
                imbalance = max(class_counts) / (min(class_counts) + 1e-10)
                max_imbalance = max(max_imbalance, imbalance)

        print(f"Max class imbalance ratio: {max_imbalance:.1f}")
        print("=" * 50 + "\n")

    def get_client_data(self, client_id: int) -> Tuple[Subset, Subset]:
        """Get train and test data for specific client"""
        return self.client_datasets[client_id]

    def get_global_test_loader(self, batch_size: int = None) -> DataLoader:
        """Get centralized test loader for global evaluation"""
        if batch_size is None:
            batch_size = self.config.federated.batch_size

        return DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.experiment.num_workers,
            pin_memory=self.config.experiment.pin_memory
        )