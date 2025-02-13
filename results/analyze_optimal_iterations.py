#!/usr/bin/env python3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from gidnet.qubitreuse import GidNET
from gidnet.utils import safe_eval, find_minimum_width_sets, iteration_score, create_qiskit_and_qnet_GRCS_circuits
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Experiment parameters
circuit_sizes = ["10x11", "11x11", "11x12", "12x12"]  # Circuit sizes
cycle_num = 13  # Depth of circuits

# Get the working directory
repo_path = Path.cwd()
directory_path = repo_path / "data" / "GRSC"

# Define iteration settings
iteration_labels = ["n", "n/2", "n/4", "log(n)", "log(n/2)", "log(n/4)"]

def compute_min_width_probability(circuit, iter_count):
    """
    Runs GidNET multiple times to determine the minimum width found.

    Parameters:
    - circuit (QuantumCircuit): The input quantum circuit.
    - iter_count (int): Number of iterations.

    Returns:
    - min_width (int): The smallest circuit width found.
    - probability (float): Probability of achieving this width.
    """
    gidnet = GidNET(circuit)
    gidnet.compile_to_dynamic_circuit(iter_count)
    
    list_of_computed_reuse_sequences = gidnet.list_of_computed_reuse_sequences
    reuse_sequences_with_min_width = find_minimum_width_sets(list_of_computed_reuse_sequences)

    if not reuse_sequences_with_min_width:
        return None, 0  # No valid width found

    min_width = len(reuse_sequences_with_min_width[0])
    probability = len(reuse_sequences_with_min_width) / iter_count
    return min_width, probability

def run_iteration_analysis():
    """
    Runs the optimal iteration analysis by testing different iteration counts.

    Saves:
    - CSV file containing computed scores for different iteration settings.
    - A bar chart visualizing the results.
    """
    gidnet_iteration_scores = []

    for circuit_size in circuit_sizes:
        logging.info(f"Processing Circuit: {circuit_size}")

        # Convert circuit size (e.g., "10x11") to integer representation
        n = safe_eval(circuit_size)
        iteration_counts = [n, n/2, n/4, np.log(n), np.log(n/2), np.log(n/4)]
        gidnet_results = []

        # Load the circuit
        qiskit_circuit, _ = create_qiskit_and_qnet_GRCS_circuits(circuit_size, cycle_num, directory_path)

        for iter_count in iteration_counts:
            iter_count = int(np.ceil(iter_count))  # Ensure integer iteration count
            min_width, probability = compute_min_width_probability(qiskit_circuit, iter_count)

            if min_width is not None:
                gidnet_results.append((min_width, probability))
            else:
                gidnet_results.append((float('inf'), 0))  # Handle edge case where no min width is found

        # Compute scores
        scores = iteration_score(gidnet_results)
        gidnet_iteration_scores.append(scores)

    # Convert results into DataFrame
    df_scores = pd.DataFrame(gidnet_iteration_scores, columns=iteration_labels, index=circuit_sizes)

    # Save results to CSV
    output_path = Path("Optimal_iterations")
    output_path.mkdir(parents=True, exist_ok=True)
    df_scores.to_csv(output_path / "gidnet_iteration_analysis.csv", index=True)
    logging.info("Iteration analysis completed and saved.")

    # Generate the bar chart
    plot_iteration_scores(df_scores, output_path)

def plot_iteration_scores(df_scores, output_path):
    """
    Generates a bar chart to visualize the effect of iteration count on optimal circuit width discovery.

    Parameters:
    - df_scores (pd.DataFrame): The computed iteration scores.
    - output_path (Path): The directory to save the plot.
    """
    plt.figure(figsize=(8, 6), dpi=400)
    x = np.arange(len(iteration_labels))  # X-axis positions

    for i, (circuit_size, scores) in enumerate(df_scores.iterrows()):
        plt.bar(x + i * 0.15, scores, width=0.15, label=f"Circuit {circuit_size}")

    # Formatting
    plt.xlabel("Iteration Count", fontsize=14, fontweight="bold")
    plt.ylabel("Score", fontsize=14, fontweight="bold")
    plt.xticks(x + 0.15, iteration_labels, fontsize=12, fontweight="bold")
    plt.yticks(fontsize=12, fontweight="bold")
    plt.title("Effect of Iterations on Finding the Optimal Circuit Width", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10, loc="upper right")
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Save plot
    plt.savefig(output_path / "gidnet_iteration_vs_optimal_width_score.pdf", format="pdf", bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    run_iteration_analysis()

