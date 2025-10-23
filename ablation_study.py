# ablation_study.py
"""
Ablation study for MoE-FL
Generates Table 5 from the paper
"""

import torch
import numpy as np
import pandas as pd
import os
import argparse
from typing import List, Dict

from config import MoEFLConfig
from train_moefl import train_federated
from utils.visualization import PublicationPlotter


def ablation_num_experts(config: MoEFLConfig, seed: int, output_dir: str):
    """
    Ablation study: varying number of experts
    """
    print("\n" + "=" * 70)
    print("ABLATION: NUMBER OF EXPERTS")
    print("=" * 70)

    expert_counts = [4, 8, 12, 16]
    results = []

    for N in expert_counts:
        print(f"\nTesting N={N} experts...")
        config.model.num_experts = N
        config.model.top_k = min(2, N // 2)  # Adjust top-k proportionally

        metrics = train_federated(config, seed, run_id=len(results))
        results.append({
            'N': N,
            'k': config.model.top_k,
            'personalized_accuracy': metrics['personalized_accuracy'],
            'routing_entropy': metrics['routing_entropy'],
            'expert_utilization': metrics['expert_utilization'],
            'communication_cost_mb': metrics['communication_cost_mb']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "ablation_num_experts.csv"), index=False)

    # Plot
    plotter = PublicationPlotter(
        output_dir=os.path.join(output_dir, "plots"),
        formats=['png', 'pdf']
    )

    plotter.plot_ablation_study(
        configurations=[f"N={r['N']}" for r in results],
        accuracies=[r['personalized_accuracy'] for r in results],
        baseline_idx=1,  # N=8 is default
        filename="ablation_num_experts"
    )

    return results


def ablation_top_k(config: MoEFLConfig, seed: int, output_dir: str):
    """
    Ablation study: varying top-k selection
    """
    print("\n" + "=" * 70)
    print("ABLATION: TOP-K ROUTING")
    print("=" * 70)

    N = config.model.num_experts
    k_values = [1, 2, 4, N]  # Including dense (k=N)
    results = []

    for k in k_values:
        print(f"\nTesting k={k} (top-{k} routing)...")
        config.model.top_k = k

        metrics = train_federated(config, seed, run_id=len(results))
        results.append({
            'k': k,
            'personalized_accuracy': metrics['personalized_accuracy'],
            'routing_entropy': metrics['routing_entropy'],
            'expert_utilization': metrics['expert_utilization'],
            'communication_cost_mb': metrics['communication_cost_mb']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "ablation_top_k.csv"), index=False)

    # Plot
    plotter = PublicationPlotter(
        output_dir=os.path.join(output_dir, "plots"),
        formats=['png', 'pdf']
    )

    plotter.plot_ablation_study(
        configurations=[f"k={r['k']}" for r in results],
        accuracies=[r['personalized_accuracy'] for r in results],
        baseline_idx=1,  # k=2 is default
        filename="ablation_top_k"
    )

    return results


def ablation_aggregation(config: MoEFLConfig, seed: int, output_dir: str):
    """
    Ablation study: aggregation strategy
    """
    print("\n" + "=" * 70)
    print("ABLATION: AGGREGATION STRATEGY")
    print("=" * 70)

    strategies = ['activation_weighted', 'uniform']
    results = []

    for strategy in strategies:
        print(f"\nTesting {strategy} aggregation...")
        config.federated.aggregation_type = strategy

        metrics = train_federated(config, seed, run_id=len(results))
        results.append({
            'strategy': strategy,
            'personalized_accuracy': metrics['personalized_accuracy'],
            'routing_entropy': metrics['routing_entropy']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "ablation_aggregation.csv"), index=False)

    return results


def ablation_load_balancing(config: MoEFLConfig, seed: int, output_dir: str):
    """
    Ablation study: load balancing loss
    """
    print("\n" + "=" * 70)
    print("ABLATION: LOAD BALANCING")
    print("=" * 70)

    alpha_values = [0.0, 0.005, 0.01, 0.05]
    results = []

    for alpha in alpha_values:
        print(f"\nTesting alpha={alpha}...")
        config.model.balance_loss_weight = alpha

        metrics = train_federated(config, seed, run_id=len(results))
        results.append({
            'alpha': alpha,
            'personalized_accuracy': metrics['personalized_accuracy'],
            'expert_utilization': metrics['expert_utilization']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "ablation_load_balancing.csv"), index=False)

    # Plot
    plotter = PublicationPlotter(
        output_dir=os.path.join(output_dir, "plots"),
        formats=['png', 'pdf']
    )

    plotter.plot_ablation_study(
        configurations=[f"α={r['alpha']}" for r in results],
        accuracies=[r['personalized_accuracy'] for r in results],
        baseline_idx=2,  # alpha=0.01 is default
        filename="ablation_load_balancing"
    )

    return results


def ablation_privacy(config: MoEFLConfig, seed: int, output_dir: str):
    """
    Ablation study: privacy budget
    """
    print("\n" + "=" * 70)
    print("ABLATION: PRIVACY BUDGET")
    print("=" * 70)

    epsilon_values = [1.0, 2.0, 5.0, 10.0, float('inf')]
    results = []

    for epsilon in epsilon_values:
        print(f"\nTesting epsilon={epsilon}...")
        config.federated.privacy_epsilon = epsilon
        config.federated.use_privacy = (epsilon != float('inf'))

        metrics = train_federated(config, seed, run_id=len(results))
        results.append({
            'epsilon': epsilon,
            'personalized_accuracy': metrics['personalized_accuracy']
        })

    # Save results
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "ablation_privacy.csv"), index=False)

    # Plot
    plotter = PublicationPlotter(
        output_dir=os.path.join(output_dir, "plots"),
        formats=['png', 'pdf']
    )

    plotter.plot_privacy_utility_tradeoff(
        epsilon_values=[r['epsilon'] for r in results if r['epsilon'] != float('inf')],
        accuracies=[r['personalized_accuracy'] for r in results if r['epsilon'] != float('inf')],
        filename="privacy_utility_tradeoff"
    )

    return results


def run_all_ablations(dataset: str, output_dir: str = "./outputs/ablation"):
    """
    Run all ablation studies

    Args:
        dataset: Dataset name
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load base configuration
    config = MoEFLConfig.from_dataset(dataset)
    config.data.output_dir = output_dir

    # Reduce rounds for faster ablation
    config.federated.num_rounds = min(50, config.federated.num_rounds)

    seed = 42

    # Run ablations
    print(f"\n{'#' * 70}")
    print(f"# ABLATION STUDIES FOR {dataset}")
    print(f"{'#' * 70}")

    all_results = {}

    all_results['num_experts'] = ablation_num_experts(config, seed, output_dir)

    # Reset config
    config = MoEFLConfig.from_dataset(dataset)
    config.federated.num_rounds = min(50, config.federated.num_rounds)
    all_results['top_k'] = ablation_top_k(config, seed, output_dir)

    # Reset config
    config = MoEFLConfig.from_dataset(dataset)
    config.federated.num_rounds = min(50, config.federated.num_rounds)
    all_results['aggregation'] = ablation_aggregation(config, seed, output_dir)

    # Reset config
    config = MoEFLConfig.from_dataset(dataset)
    config.federated.num_rounds = min(50, config.federated.num_rounds)
    all_results['load_balancing'] = ablation_load_balancing(config, seed, output_dir)

    # Reset config
    config = MoEFLConfig.from_dataset(dataset)
    config.federated.num_rounds = min(50, config.federated.num_rounds)
    all_results['privacy'] = ablation_privacy(config, seed, output_dir)

    print(f"\n{'=' * 70}")
    print("ALL ABLATION STUDIES COMPLETED")
    print(f"Results saved to: {output_dir}")
    print("=" * 70 + "\n")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies for MoE-FL")
    parser.add_argument(
        '--dataset',
        type=str,
        default='CIFAR10',
        choices=['CIFAR10', 'CIFAR100', 'FEMNIST', 'Shakespeare'],
        help='Dataset to use'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/ablation',
        help='Output directory'
    )
    parser.add_argument(
        '--study',
        type=str,
        default='all',
        choices=['all', 'num_experts', 'top_k', 'aggregation', 'load_balancing', 'privacy'],
        help='Which ablation study to run'
    )

    args = parser.parse_args()

    if args.study == 'all':
        run_all_ablations(args.dataset, args.output_dir)
    else:
        config = MoEFLConfig.from_dataset(args.dataset)
        config.federated.num_rounds = min(50, config.federated.num_rounds)
        seed = 42

        if args.study == 'num_experts':
            ablation_num_experts(config, seed, args.output_dir)
        elif args.study == 'top_k':
            ablation_top_k(config, seed, args.output_dir)
        elif args.study == 'aggregation':
            ablation_aggregation(config, seed, args.output_dir)
        elif args.study == 'load_balancing':
            ablation_load_balancing(config, seed, args.output_dir)
        elif args.study == 'privacy':
            ablation_privacy(config, seed, args.output_dir)


if __name__ == "__main__":
    main()