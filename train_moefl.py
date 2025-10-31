# train_moefl.py
"""
Main training script for MoE-FL
Implements complete federated learning pipeline
"""

import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
from datetime import datetime

from config import MoEFLConfig, CIFAR10_CONFIG, CIFAR100_CONFIG
from models.moe_architecture import create_moe_model
from federated.server import FederatedServer
from federated.client import create_client
from data.data_loader import FederatedDataset
from utils.metrics import MetricsTracker, compute_communication_cost
from utils.visualization import PublicationPlotter, create_summary_table



def set_random_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_federated(config: MoEFLConfig, seed: int, run_id: int):
    """
    Main federated training loop
    Implements Algorithm 1 from the paper

    Args:
        config: Configuration object
        seed: Random seed for this run
        run_id: Run identifier

    Returns:
        final_metrics: Dictionary with final metrics
    """
    # Set seed
    set_random_seed(seed)

    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(
        config.data.output_dir,
        f"{config.model.dataset}_run{run_id}_seed{seed}_{timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)

    plots_dir = os.path.join(run_dir, "plots")
    csv_dir = os.path.join(run_dir, "csv")
    checkpoint_dir = os.path.join(run_dir, "checkpoints")
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Starting MoE-FL Training - Run {run_id}, Seed {seed}")
    print(f"Dataset: {config.model.dataset}")
    print(f"Experts: {config.model.num_experts}, Top-k: {config.model.top_k}")
    print(f"Output directory: {run_dir}")
    print(f"{'=' * 70}\n")

    # Save configuration
    with open(os.path.join(run_dir, "config.json"), 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(csv_dir)
    plotter = PublicationPlotter(plots_dir, formats=config.experiment.plot_formats)

    # Load and partition data
    print("Loading and partitioning data...")
    federated_data = FederatedDataset(config)
    global_test_loader = federated_data.get_global_test_loader()

    # Create global model
    print("Creating MoE model...")
    global_model = create_moe_model(config)
    num_parameters = sum(p.numel() for p in global_model.parameters())
    print(f"Total parameters: {num_parameters:,}")

    # Create server
    server = FederatedServer(
        model=global_model,
        config=config,
        device=config.experiment.device
    )

    # Create clients
    print(f"Creating {config.federated.num_clients} clients...")
    clients = []
    for client_id in range(config.federated.num_clients):
        train_data, test_data = federated_data.get_client_data(client_id)
        client_model = create_moe_model(config)

        client = create_client(
            client_id=client_id,
            model=client_model,
            train_dataset=train_data,
            test_dataset=test_data,
            config=config,
            device=config.experiment.device
        )
        clients.append(client)

    print(f"Setup complete. Starting training for {config.federated.num_rounds} rounds...\n")

    # Algorithm 1, Lines 1-2: Server initialization
    # Training loop - Algorithm 1, Line 3: for round t
    for round_num in tqdm(range(config.federated.num_rounds), desc="Training"):
        # Algorithm 1, Line 4: Sample clients
        selected_client_ids = server.select_clients(round_num, len(clients))

        # Algorithm 1, Line 5: Broadcast global model
        global_model_state = server.get_global_model_state()

        # Collect client updates
        client_updates = []
        client_metrics = []
        activation_frequencies = []

        # Algorithm 1, Lines 6-13: Client updates
        for client_id in selected_client_ids:
            client = clients[client_id]

            # Algorithm 1, Line 17: Update client model
            client.update_model(global_model_state)

            # Algorithm 1, Lines 18-34: Local training
            model_state, train_metrics, activations = client.train(round_num)

            # Algorithm 1, Lines 10-12: Add privacy noise
            if config.federated.use_privacy:
                activations = client.add_privacy_noise(
                    activations,
                    config.federated.privacy_epsilon
                )

            # Collect updates (already on CPU from client.train())
            client_updates.append(model_state)
            client_metrics.append(train_metrics)
            activation_frequencies.append(activations)

            # Clear cache after each client to free memory
            if config.experiment.device == 'cuda':
                torch.cuda.empty_cache()

        # Algorithm 1, Lines 14-18: Server aggregation
        agg_metrics = server.aggregate(
            client_updates,
            client_metrics,
            activation_frequencies
        )

        # Clear client updates from memory
        del client_updates
        if config.experiment.device == 'cuda':
            torch.cuda.empty_cache()

        # Update metrics
        metrics_tracker.update_round(round_num, agg_metrics)

        # Evaluate clients periodically
        if (round_num + 1) % config.experiment.eval_interval == 0 or round_num == 0:
            print(f"\nRound {round_num + 1}/{config.federated.num_rounds}")
            print(f"  Train Loss: {agg_metrics['avg_train_loss']:.4f}")
            print(f"  Train Acc: {agg_metrics['avg_train_accuracy']:.2f}%")
            print(f"  Balance Loss: {agg_metrics['avg_balance_loss']:.4f}")
            print(f"  Expert Util: {agg_metrics['expert_utilization']:.3f}")
            print(f"  Routing Entropy: {agg_metrics['routing_entropy']:.3f}")

            # Evaluate all clients
            for client_id in range(len(clients)):
                client = clients[client_id]
                client.update_model(server.get_global_model_state())
                eval_metrics = client.evaluate()
                metrics_tracker.update_client(client_id, round_num, eval_metrics)

            # Global evaluation
            global_metrics = server.evaluate_global(global_test_loader)
            agg_metrics.update(global_metrics)
            metrics_tracker.update_round(round_num, global_metrics)

            print(f"  Global Test Acc: {global_metrics['global_test_accuracy']:.2f}%")

            # Personalized accuracy
            pers_acc = metrics_tracker.get_personalized_accuracy(round_num)
            print(f"  Personalized Test Acc: {pers_acc:.2f}%")

        # Save checkpoint periodically
        if (round_num + 1) % config.experiment.checkpoint_interval == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir,
                f"checkpoint_round_{round_num + 1}.pt"
            )
            server.save_checkpoint(round_num, checkpoint_path)

    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70 + "\n")

    # Final evaluation
    print("Performing final evaluation...")
    final_global_metrics = server.evaluate_global(global_test_loader)

    # Evaluate all clients for final personalized accuracy
    for client_id in range(len(clients)):
        client = clients[client_id]
        client.update_model(server.get_global_model_state())
        eval_metrics = client.evaluate()
        metrics_tracker.update_client(
            client_id,
            config.federated.num_rounds - 1,
            eval_metrics
        )

    final_metrics = metrics_tracker.get_summary()
    final_metrics.update(final_global_metrics)

    # Compute communication cost
    comm_cost = compute_communication_cost(
        num_parameters=num_parameters,
        num_rounds=config.federated.num_rounds,
        clients_per_round=config.federated.clients_per_round
    )
    final_metrics['communication_cost_mb'] = comm_cost

    print("\nFinal Results:")
    print(f"  Personalized Accuracy: {final_metrics['personalized_accuracy']:.2f}%")
    print(f"  Global Accuracy: {final_metrics['global_accuracy']:.2f}%")
    print(f"  Routing Entropy: {final_metrics['routing_entropy']:.3f}")
    print(f"  Expert Utilization: {final_metrics['expert_utilization']:.3f}")
    print(f"  Communication Cost: {comm_cost:.2f} MB")

    # Save metrics
    print("\nSaving results...")
    metrics_tracker.save_to_csv("metrics.csv")

    # Save final metrics
    with open(os.path.join(run_dir, "final_metrics.json"), 'w') as f:
        json.dump(final_metrics, f, indent=2)

    # Generate plots
    if config.experiment.save_plots:
        print("Generating plots...")

        # Extract data for plotting
        round_metrics = metrics_tracker.round_metrics
        rounds = [m['round'] for m in round_metrics if 'global_test_accuracy' in m]
        global_accs = [m['global_test_accuracy'] for m in round_metrics if 'global_test_accuracy' in m]
        entropies = [m['routing_entropy'] for m in round_metrics if 'routing_entropy' in m]
        train_losses = [m['avg_train_loss'] for m in round_metrics]
        balance_losses = [m['avg_balance_loss'] for m in round_metrics]

        # Plot accuracy over rounds
        if rounds:
            plotter.plot_accuracy_vs_rounds(
                {'MoE-FL': [{'round': r, 'test_accuracy': a}
                            for r, a in zip(rounds, global_accs)]},
                title=f"{config.model.dataset}: Test Accuracy vs Rounds",
                filename=f"accuracy_vs_rounds_{config.model.dataset.lower()}"
            )

        # Plot routing entropy
        if entropies:
            plotter.plot_routing_entropy(
                rounds=list(range(len(entropies))),
                entropy_values=entropies,
                num_experts=config.model.num_experts,
                filename=f"routing_entropy_{config.model.dataset.lower()}"
            )

        # Plot loss landscape
        if train_losses:
            plotter.plot_loss_landscape(
                rounds=list(range(len(train_losses))),
                train_losses=train_losses,
                balance_losses=balance_losses,
                filename=f"loss_landscape_{config.model.dataset.lower()}"
            )

        # Get expert utilization heatmap data
        # Collect activation patterns from last evaluation
        activation_matrix = []
        for client_id in range(min(50, len(clients))):  # Sample 50 clients for visualization
            client = clients[client_id]
            if hasattr(client, 'activation_counts') and client.total_samples > 0:
                freq = client.activation_counts / client.total_samples
                activation_matrix.append(freq)

        if activation_matrix:
            activation_matrix = np.array(activation_matrix)
            plotter.plot_expert_utilization_heatmap(
                activation_matrix=activation_matrix,
                filename=f"expert_utilization_{config.model.dataset.lower()}"
            )

    print(f"\nAll results saved to: {run_dir}\n")

    return final_metrics


def run_multiple_seeds(config: MoEFLConfig):
    """
    Run experiment with multiple random seeds

    Args:
        config: Configuration object

    Returns:
        aggregated_results: Results aggregated across seeds
    """
    all_results = []

    for run_id, seed in enumerate(config.experiment.seeds):
        print(f"\n{'#' * 70}")
        print(f"# RUN {run_id + 1}/{len(config.experiment.seeds)} - SEED {seed}")
        print(f"{'#' * 70}")

        results = train_federated(config, seed, run_id)
        all_results.append(results)

    # Aggregate results across seeds
    aggregated = {}
    for key in all_results[0].keys():
        values = [r[key] for r in all_results if key in r]
        if values and isinstance(values[0], (int, float)):
            aggregated[f"{key}_mean"] = np.mean(values)
            aggregated[f"{key}_std"] = np.std(values)

    # Save aggregated results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_dir = os.path.join(
        config.data.output_dir,
        f"{config.model.dataset}_aggregated_{timestamp}"
    )
    os.makedirs(agg_dir, exist_ok=True)

    with open(os.path.join(agg_dir, "aggregated_results.json"), 'w') as f:
        json.dump(aggregated, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATED RESULTS ACROSS ALL RUNS")
    print("=" * 70)
    print(
        f"Personalized Accuracy: {aggregated['personalized_accuracy_mean']:.2f} ± {aggregated['personalized_accuracy_std']:.2f}%")
    print(f"Global Accuracy: {aggregated['global_accuracy_mean']:.2f} ± {aggregated['global_accuracy_std']:.2f}%")
    print(f"Routing Entropy: {aggregated['routing_entropy_mean']:.3f} ± {aggregated['routing_entropy_std']:.3f}")
    print(
        f"Expert Utilization: {aggregated['expert_utilization_mean']:.3f} ± {aggregated['expert_utilization_std']:.3f}")
    print(
        f"Communication Cost: {aggregated['communication_cost_mb_mean']:.2f} ± {aggregated['communication_cost_mb_std']:.2f} MB")
    print("=" * 70 + "\n")

    return aggregated


def main():
    parser = argparse.ArgumentParser(description="Train MoE-FL model")
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100', 'FEMNIST', 'Shakespeare'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--num_rounds',
        type=int,
        default=None,
        help='Number of training rounds (overrides config)'
    )

    parser.add_argument(
        '--grad_checkpoint',
        action='store_true',
        help='Use gradient checkpointing to save memory'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='Batch size (reduce if OOM)'
    )

    parser.add_argument(
        '--num_experts',
        type=int,
        default=None,
        help='Number of experts (overrides config)'
    )
    parser.add_argument(
        '--top_k',
        type=int,
        default=None,
        help='Top-k expert selection (overrides config)'
    )
    parser.add_argument(
        '--clients_per_round',
        type=int,
        default=None,
        help='Clients per round (overrides config)'
    )
    parser.add_argument(
        '--single_run',
        action='store_true',
        help='Run single experiment (no multiple seeds)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for single run'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='device to use'
    )

    args = parser.parse_args()

    # Load configuration for dataset
    config = MoEFLConfig.from_dataset(args.dataset)

    # Override with command line arguments
    if args.num_rounds is not None:
        config.federated.num_rounds = args.num_rounds
    if args.num_experts is not None:
        config.model.num_experts = args.num_experts
    if args.top_k is not None:
        config.model.top_k = args.top_k
    if args.clients_per_round is not None:
        config.federated.clients_per_round = args.clients_per_round

    config.data.output_dir = args.output_dir

    # Run experiment
    if args.single_run:
        config.experiment.seeds = [args.seed]
        results = train_federated(config, args.seed, run_id=0)
    else:
        results = run_multiple_seeds(config)

    print("\nExperiment completed successfully!")


if __name__ == "__main__":
    main()
