# compare_baselines.py
"""
Compare MoE-FL with baseline methods
Generates Table 4 from the paper
"""

import torch
import numpy as np
import pandas as pd
import os
from typing import Dict, List
import argparse

from config import MoEFLConfig
from train_moefl import train_federated
from utils.visualization import PublicationPlotter, create_summary_table


def run_baseline_fedavg(config: MoEFLConfig, seed: int):
    """
    Run FedAvg baseline (single model, no MoE)
    """
    print("\nRunning FedAvg baseline...")
    # Modify config to use single model
    baseline_config = config
    baseline_config.model.num_experts = 1
    baseline_config.model.top_k = 1

    results = train_federated(baseline_config, seed, run_id=0)
    return results


def compare_methods(dataset: str, output_dir: str = "./outputs/comparison"):
    """
    Compare MoE-FL with multiple baseline methods

    Args:
        dataset: Dataset name
        output_dir: Output directory for comparison results
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load configuration
    config = MoEFLConfig.from_dataset(dataset)
    config.data.output_dir = output_dir

    # For faster comparison, reduce rounds
    config.federated.num_rounds = min(100, config.federated.num_rounds)

    print(f"\n{'=' * 70}")
    print(f"COMPARING METHODS ON {dataset}")
    print(f"{'=' * 70}\n")

    results = {}
    seed = 42

    # Run MoE-FL
    print("\n" + "=" * 70)
    print("RUNNING MoE-FL")
    print("=" * 70)
    moefl_config = config
    results['MoE-FL'] = train_federated(moefl_config, seed, run_id=0)

    # Run FedAvg
    print("\n" + "=" * 70)
    print("RUNNING FedAvg BASELINE")
    print("=" * 70)
    fedavg_config = MoEFLConfig.from_dataset(dataset)
    fedavg_config.model.num_experts = 1
    fedavg_config.model.top_k = 1
    fedavg_config.federated.num_rounds = config.federated.num_rounds
    results['FedAvg'] = train_federated(fedavg_config, seed, run_id=1)

    # Note: For full comparison, you would implement other baselines
    # (FedProx, CFL, IFCA, FedPer, Ditto, Per-FedAvg)
    # Here we show the structure

    # Generate comparison plots and tables
    print("\n" + "=" * 70)
    print("GENERATING COMPARISON RESULTS")
    print("=" * 70)

    methods = list(results.keys())
    pers_accs = [results[m]['personalized_accuracy'] for m in methods]
    global_accs = [results[m]['global_accuracy'] for m in methods]
    comm_costs = [results[m]['communication_cost_mb'] for m in methods]

    # Create summary table
    summary_df = create_summary_table(
        methods=methods,
        pers_accuracies=pers_accs,
        global_accuracies=global_accs,
        comm_costs=comm_costs,
        output_path=os.path.join(output_dir, f"comparison_table_{dataset}.csv")
    )

    print("\nComparison Table:")
    print(summary_df)

    # Create plots
    plotter = PublicationPlotter(
        output_dir=os.path.join(output_dir, "plots"),
        formats=['png', 'pdf']
    )

    plotter.plot_personalized_vs_global(
        methods=methods,
        pers_accuracies=pers_accs,
        global_accuracies=global_accs,
        filename=f"comparison_bar_{dataset}"
    )

    plotter.plot_communication_cost(
        methods=methods,
        comm_costs=comm_costs,
        target_accuracy=min(pers_accs),
        filename=f"communication_comparison_{dataset}"
    )

    print(f"\nComparison results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Compare MoE-FL with baselines")
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
        default='./outputs/comparison',
        help='Output directory'
    )

    args = parser.parse_args()

    compare_methods(args.dataset, args.output_dir)


if __name__ == "__main__":
    main()