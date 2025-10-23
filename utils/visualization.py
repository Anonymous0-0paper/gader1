# utils/visualization.py
"""
Publication-grade visualization utilities
Generates plots for paper figures"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os

# Set publication-quality defaults
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.pad_inches'] = 0.05

# Color palette for consistent styling
COLORS = {
    'moefl': '#2E86AB',      # Blue for MoE-FL
    'fedavg': '#A23B72',     # Purple for FedAvg
    'fedprox': '#F18F01',    # Orange for FedProx
    'cfl': '#C73E1D',        # Red for CFL
    'ifca': '#6A994E',       # Green for IFCA
    'fedper': '#BC4B51',     # Dark red for FedPer
    'ditto': '#8B5A3C',      # Brown for Ditto
    'perfedavg': '#5E548E',  # Dark purple for Per-FedAvg
    'baseline': '#999999'    # Gray for baselines
}

MARKERS = {
    'moefl': 'o',
    'fedavg': 's',
    'fedprox': '^',
    'cfl': 'v',
    'ifca': 'D',
    'fedper': 'p',
    'ditto': '*',
    'perfedavg': 'h'
}


class PublicationPlotter:
    """
    Create publication-grade plots for MoE-FL paper
    """

    def __init__(self, output_dir: str, formats: List[str] = ['png', 'pdf']):
        self.output_dir = output_dir
        self.formats = formats
        os.makedirs(output_dir, exist_ok=True)

    def plot_accuracy_vs_rounds(
        self,
        results: Dict[str, List[Dict]],
        title: str = "Test Accuracy vs Communication Rounds",
        filename: str = "accuracy_vs_rounds"
    ):
        """
        Plot test accuracy over communication rounds
        Figure for main results (Table 4 visualization)

        Args:
            results: Dict mapping method name to list of round metrics
            title: Plot title
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        for method_name, metrics_list in results.items():
            rounds = [m['round'] for m in metrics_list]
            accuracies = [m.get('test_accuracy', 0) for m in metrics_list]

            color = COLORS.get(method_name.lower(), COLORS['baseline'])
            marker = MARKERS.get(method_name.lower(), 'o')

            # Plot with confidence interval if available
            if 'test_accuracy_std' in metrics_list[0]:
                stds = [m['test_accuracy_std'] for m in metrics_list]
                ax.fill_between(
                    rounds,
                    np.array(accuracies) - np.array(stds),
                    np.array(accuracies) + np.array(stds),
                    alpha=0.2,
                    color=color
                )

            ax.plot(
                rounds,
                accuracies,
                label=method_name,
                color=color,
                marker=marker,
                markevery=max(1, len(rounds) // 10),
                linewidth=2,
                markersize=6
            )

        ax.set_xlabel('Communication Rounds')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title(title)
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        self._save_figure(fig, filename)
        plt.close()

    def plot_personalized_vs_global(
        self,
        methods: List[str],
        pers_accuracies: List[float],
        global_accuracies: List[float],
        pers_stds: List[float] = None,
        global_stds: List[float] = None,
        filename: str = "personalized_vs_global"
    ):
        """
        Bar plot comparing personalized vs global accuracy
        Visualization of Table 4 results

        Args:
            methods: List of method names
            pers_accuracies: Personalized accuracies
            global_accuracies: Global accuracies
            pers_stds: Standard deviations for personalized (optional)
            global_stds: Standard deviations for global (optional)
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        x = np.arange(len(methods))
        width = 0.35

        # Plot bars
        bars1 = ax.bar(
            x - width/2,
            pers_accuracies,
            width,
            label='Personalized',
            color='#2E86AB',
            yerr=pers_stds if pers_stds else None,
            capsize=3,
            edgecolor='black',
            linewidth=0.5
        )

        bars2 = ax.bar(
            x + width/2,
            global_accuracies,
            width,
            label='Global',
            color='#F18F01',
            yerr=global_stds if global_stds else None,
            capsize=3,
            edgecolor='black',
            linewidth=0.5
        )

        ax.set_xlabel('Methods')
        ax.set_ylabel('Test Accuracy (%)')
        ax.set_title('Personalized vs Global Test Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )

        self._save_figure(fig, filename)
        plt.close()

    def plot_ablation_study(
        self,
        configurations: List[str],
        accuracies: List[float],
        stds: List[float] = None,
        baseline_idx: int = None,
        filename: str = "ablation_study"
    ):
        """
        Plot ablation study results
        Visualization of Table 5

        Args:
            configurations: List of configuration names
            accuracies: Accuracy for each configuration
            stds: Standard deviations (optional)
            baseline_idx: Index of baseline configuration to highlight
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = ['#2E86AB' if i == baseline_idx else '#999999'
                 for i in range(len(configurations))]

        bars = ax.bar(
            range(len(configurations)),
            accuracies,
            color=colors,
            yerr=stds if stds else None,
            capsize=3,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )

        # Highlight baseline
        if baseline_idx is not None:
            bars[baseline_idx].set_edgecolor('#2E86AB')
            bars[baseline_idx].set_linewidth(2.5)

        ax.set_xlabel('Configuration')
        ax.set_ylabel('Personalized Accuracy (%)')
        ax.set_title('Ablation Study: Impact of Design Choices')
        ax.set_xticks(range(len(configurations)))
        ax.set_xticklabels(configurations, rotation=45, ha='right')
        ax.grid(True, axis='y', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            height = bar.get_height()
            label = f'{acc:.1f}'
            if baseline_idx is not None and i == baseline_idx:
                label += '\n(default)'
            ax.text(
                bar.get_x() + bar.get_width()/2.,
                height,
                label,
                ha='center',
                va='bottom',
                fontsize=8,
                fontweight='bold' if i == baseline_idx else 'normal'
            )

        self._save_figure(fig, filename)
        plt.close()

    def plot_communication_cost(
        self,
        methods: List[str],
        comm_costs: List[float],
        target_accuracy: float,
        filename: str = "communication_cost"
    ):
        """
        Plot communication cost comparison
        Visualization of Table 6

        Args:
            methods: List of method names
            comm_costs: Communication costs in MB
            target_accuracy: Target accuracy threshold
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        colors = [COLORS.get(m.lower(), COLORS['baseline']) for m in methods]

        bars = ax.barh(
            range(len(methods)),
            comm_costs,
            color=colors,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )

        # Highlight MoE-FL
        moefl_idx = next((i for i, m in enumerate(methods) if m.lower() == 'moefl'), None)
        if moefl_idx is not None:
            bars[moefl_idx].set_edgecolor('#2E86AB')
            bars[moefl_idx].set_linewidth(2.5)

        ax.set_xlabel('Communication Cost (MB)')
        ax.set_ylabel('Methods')
        ax.set_title(f'Communication Cost to Reach {target_accuracy}% Accuracy')
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (bar, cost) in enumerate(zip(bars, comm_costs)):
            width = bar.get_width()
            label = f'{cost:.0f} MB'
            if i == moefl_idx:
                # Calculate savings
                avg_baseline = np.mean([c for j, c in enumerate(comm_costs) if j != moefl_idx])
                savings = (avg_baseline - cost) / avg_baseline * 100
                label += f'\n(-{savings:.1f}%)'

            ax.text(
                width,
                bar.get_y() + bar.get_height()/2.,
                label,
                ha='left',
                va='center',
                fontsize=8,
                fontweight='bold' if i == moefl_idx else 'normal'
            )

        self._save_figure(fig, filename)
        plt.close()

    def plot_routing_entropy(
        self,
        rounds: List[int],
        entropy_values: List[float],
        num_experts: int,
        filename: str = "routing_entropy"
    ):
        """
        Plot routing entropy over training
        Shows expert specialization dynamics

        Args:
            rounds: Communication rounds
            entropy_values: Normalized entropy values
            num_experts: Number of experts
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.plot(
            rounds,
            entropy_values,
            color=COLORS['moefl'],
            linewidth=2,
            marker='o',
            markevery=max(1, len(rounds) // 10),
            markersize=6
        )

        # Add reference lines
        ax.axhline(
            y=1.0,
            color='red',
            linestyle='--',
            linewidth=1.5,
            alpha=0.5,
            label='Max Entropy (uniform routing)'
        )
        ax.axhline(
            y=0.0,
            color='green',
            linestyle='--',
            linewidth=1.5,
            alpha=0.5,
            label='Min Entropy (full specialization)'
        )

        ax.set_xlabel('Communication Rounds')
        ax.set_ylabel('Normalized Routing Entropy')
        ax.set_title(f'Routing Entropy Evolution (N={num_experts} experts)')
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(left=0)
        ax.set_ylim([-0.05, 1.05])

        self._save_figure(fig, filename)
        plt.close()

    def plot_expert_utilization_heatmap(
        self,
        activation_matrix: np.ndarray,
        client_labels: List[str] = None,
        expert_labels: List[str] = None,
        filename: str = "expert_utilization_heatmap"
    ):
        """
        Heatmap showing which clients use which experts
        Demonstrates emergent clustering

        Args:
            activation_matrix: [num_clients, num_experts] activation frequencies
            client_labels: Labels for clients (optional)
            expert_labels: Labels for experts (optional)
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Normalize rows to show relative usage
        activation_normalized = activation_matrix / (activation_matrix.sum(axis=1, keepdims=True) + 1e-10)

        # Create heatmap
        im = ax.imshow(
            activation_normalized,
            cmap='YlOrRd',
            aspect='auto',
            interpolation='nearest'
        )

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Activation Frequency', rotation=270, labelpad=20)

        # Set ticks
        if expert_labels is None:
            expert_labels = [f'E{i+1}' for i in range(activation_matrix.shape[1])]
        if client_labels is None:
            # Show subset of clients
            num_clients = activation_matrix.shape[0]
            step = max(1, num_clients // 20)
            client_labels = [f'C{i}' if i % step == 0 else ''
                           for i in range(num_clients)]

        ax.set_xticks(range(len(expert_labels)))
        ax.set_xticklabels(expert_labels)
        ax.set_yticks(range(0, len(client_labels), max(1, len(client_labels) // 20)))
        ax.set_yticklabels([client_labels[i] for i in range(0, len(client_labels),
                                                             max(1, len(client_labels) // 20))])

        ax.set_xlabel('Experts')
        ax.set_ylabel('Clients')
        ax.set_title('Expert Activation Patterns Across Clients')

        self._save_figure(fig, filename)
        plt.close()

    def plot_privacy_utility_tradeoff(
        self,
        epsilon_values: List[float],
        accuracies: List[float],
        stds: List[float] = None,
        filename: str = "privacy_utility_tradeoff"
    ):
        """
        Plot privacy-utility tradeoff
        Visualization of Table 7

        Args:
            epsilon_values: Privacy budget values
            accuracies: Corresponding accuracies
            stds: Standard deviations (optional)
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(6, 4))

        if stds:
            ax.errorbar(
                epsilon_values,
                accuracies,
                yerr=stds,
                color=COLORS['moefl'],
                linewidth=2,
                marker='o',
                markersize=8,
                capsize=5,
                capthick=2
            )
        else:
            ax.plot(
                epsilon_values,
                accuracies,
                color=COLORS['moefl'],
                linewidth=2,
                marker='o',
                markersize=8
            )

        # Add reference line for no privacy
        if len(epsilon_values) > 0 and max(epsilon_values) < 100:
            ax.axhline(
                y=accuracies[0] if epsilon_values[0] > 100 else max(accuracies),
                color='gray',
                linestyle='--',
                linewidth=1.5,
                alpha=0.5,
                label='No Privacy (ε=∞)'
            )

        ax.set_xlabel('Privacy Budget (ε)')
        ax.set_ylabel('Personalized Accuracy (%)')
        ax.set_title('Privacy-Utility Tradeoff')
        ax.set_xscale('log')
        ax.legend(loc='best', frameon=True, fancybox=False, edgecolor='black')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Annotate key points
        for i, (eps, acc) in enumerate(zip(epsilon_values, accuracies)):
            if eps in [1, 2, 5, 10]:
                ax.annotate(
                    f'ε={eps}\n{acc:.1f}%',
                    xy=(eps, acc),
                    xytext=(10, 10),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
                )

        self._save_figure(fig, filename)
        plt.close()

    def plot_convergence_comparison(
        self,
        results: Dict[str, List[Tuple[int, float]]],
        target_accuracy: float,
        filename: str = "convergence_comparison"
    ):
        """
        Plot convergence speed comparison
        Shows rounds to reach target accuracy

        Args:
            results: Dict mapping method to list of (round, accuracy) tuples
            target_accuracy: Target accuracy threshold
            filename: Output filename
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        methods = []
        rounds_to_converge = []

        for method_name, data in results.items():
            rounds, accuracies = zip(*data)

            # Find first round reaching target
            converged = False
            for r, acc in zip(rounds, accuracies):
                if acc >= target_accuracy:
                    methods.append(method_name)
                    rounds_to_converge.append(r)
                    converged = True
                    break

            if not converged:
                methods.append(method_name)
                rounds_to_converge.append(max(rounds))  # Did not converge

        # Sort by rounds
        sorted_indices = np.argsort(rounds_to_converge)
        methods = [methods[i] for i in sorted_indices]
        rounds_to_converge = [rounds_to_converge[i] for i in sorted_indices]

        colors = [COLORS.get(m.lower(), COLORS['baseline']) for m in methods]

        bars = ax.barh(
            range(len(methods)),
            rounds_to_converge,
            color=colors,
            edgecolor='black',
            linewidth=0.5,
            alpha=0.8
        )

        # Highlight MoE-FL
        moefl_idx = next((i for i, m in enumerate(methods) if m.lower() == 'moefl'), None)
        if moefl_idx is not None:
            bars[moefl_idx].set_edgecolor('#2E86AB')
            bars[moefl_idx].set_linewidth(2.5)

        ax.set_xlabel('Rounds to Convergence')
        ax.set_ylabel('Methods')
        ax.set_title(f'Convergence Speed (Target: {target_accuracy}% Accuracy)')
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels(methods)
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (bar, rounds) in enumerate(zip(bars, rounds_to_converge)):
            ax.text(
                rounds,
                bar.get_y() + bar.get_height()/2.,
                f'  {rounds} rounds',
                ha='left',
                va='center',
                fontsize=8,
                fontweight='bold' if i == moefl_idx else 'normal'
            )

        self._save_figure(fig, filename)
        plt.close()

    def plot_loss_landscape(
        self,
        rounds: List[int],
        train_losses: List[float],
        balance_losses: List[float],
        filename: str = "loss_landscape"
    ):
        """
        Plot training and balance loss over rounds
        Shows optimization dynamics

        Args:
            rounds: Communication rounds
            train_losses: Training task losses
            balance_losses: Load balancing losses
            filename: Output filename
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

        # Task loss
        ax1.plot(
            rounds,
            train_losses,
            color='#2E86AB',
            linewidth=2,
            label='Task Loss'
        )
        ax1.set_ylabel('Task Loss')
        ax1.set_title('Training Dynamics')
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Balance loss
        ax2.plot(
            rounds,
            balance_losses,
            color='#F18F01',
            linewidth=2,
            label='Balance Loss'
        )
        ax2.set_xlabel('Communication Rounds')
        ax2.set_ylabel('Balance Loss')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3, linestyle='--')

        plt.tight_layout()
        self._save_figure(fig, filename)
        plt.close()

    def _save_figure(self, fig, filename: str):
        """Save figure in multiple formats"""
        for fmt in self.formats:
            filepath = os.path.join(self.output_dir, f"{filename}.{fmt}")
            fig.savefig(filepath, format=fmt, bbox_inches='tight', dpi=300)
        print(f"Saved plot: {filename}")


def create_summary_table(
    methods: List[str],
    pers_accuracies: List[float],
    global_accuracies: List[float],
    comm_costs: List[float],
    output_path: str
):
    """
    Create summary table similar to Table 4 in paper

    Args:
        methods: List of method names
        pers_accuracies: Personalized accuracies
        global_accuracies: Global accuracies
        comm_costs: Communication costs
        output_path: Path to save CSV
    """
    df = pd.DataFrame({
        'Method': methods,
        'Personalized Acc. (%)': pers_accuracies,
        'Global Acc. (%)': global_accuracies,
        'Comm. Cost (MB)': comm_costs
    })

    # Add improvement row for MoE-FL
    if 'MoE-FL' in methods:
        moefl_idx = methods.index('MoE-FL')
        baseline_pers = np.max([acc for i, acc in enumerate(pers_accuracies) if i != moefl_idx])
        baseline_global = np.max([acc for i, acc in enumerate(global_accuracies) if i != moefl_idx])
        baseline_comm = np.min([cost for i, cost in enumerate(comm_costs) if i != moefl_idx])

        improvement = {
            'Method': 'Improvement',
            'Personalized Acc. (%)': f"+{pers_accuracies[moefl_idx] - baseline_pers:.1f}",
            'Global Acc. (%)': f"+{global_accuracies[moefl_idx] - baseline_global:.1f}",
            'Comm. Cost (MB)': f"-{((baseline_comm - comm_costs[moefl_idx]) / baseline_comm * 100):.1f}%"
        }
        df = pd.concat([df, pd.DataFrame([improvement])], ignore_index=True)

    df.to_csv(output_path, index=False)
    print(f"Saved summary table to {output_path}")

    return df