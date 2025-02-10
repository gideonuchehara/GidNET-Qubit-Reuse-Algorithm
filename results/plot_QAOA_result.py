#!/usr/bin/env python3

import numpy as np
import pandas as pd
import ast
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
from matplotlib.lines import Line2D

# Define paths
data_path = Path("QAOA_result")  # Folder containing experiment results
# plot_path = Path("plots_qaoa_cct")  # Folder to save plots
plot_path = Path("QAOA_result/plots_qaoa_cct")
plot_path.mkdir(parents=True, exist_ok=True)

# Define the layers used in the experiment
experiment_layers = [1, 2]  # Modify this list if more layers are added


def process_experiment_data(layer: int) -> pd.DataFrame:
    """
    Processes the raw QAOA experiment data for a given layer.

    Args:
        layer (int): The QAOA layer number.

    Returns:
        pd.DataFrame: Processed dataframe containing mean, std, min, max values.
    """
    file_name = f"width_and_runtime_QAOA_cct_P_{layer}.csv"
    file_path = data_path / file_name

    if not file_path.exists():
        print(f"Warning: Data file for layer {layer} not found: {file_name}")
        return None

    df_raw = pd.read_csv(file_path)

    df_processed = pd.DataFrame({
        "Circuit Sizes": df_raw["Circuit Size"],
        "GidNET Mean Runtime": df_raw["GidNET Mean Runtime"].apply(ast.literal_eval).apply(np.mean),
        "GidNET Stdev Runtime": df_raw["GidNET Stdev Runtime"].apply(ast.literal_eval).apply(np.std),
        "GidNET Width": df_raw["GidNET Min Width"].apply(ast.literal_eval),
        "QNET Mean Runtime": df_raw["QNET Mean Runtime"].apply(ast.literal_eval).apply(np.mean),
        "QNET Stdev Runtime": df_raw["QNET Stdev Runtime"].apply(ast.literal_eval).apply(np.std),
        "QNET Width": df_raw["QNET Min Width"].apply(ast.literal_eval),
    })

    # Process Qiskit data only if it exists (for P=1)
    if "Qiskit Mean Runtime" in df_raw.columns:
        df_processed["Qiskit Mean Runtime"] = df_raw["Qiskit Mean Runtime"].apply(ast.literal_eval).apply(np.mean)
        df_processed["Qiskit Stdev Runtime"] = df_raw["Qiskit Stdev Runtime"].apply(ast.literal_eval).apply(np.std)
        df_processed["Qiskit Width"] = df_raw["Qiskit Min Width"].apply(ast.literal_eval)

    # Save the processed data
    processed_file = plot_path / f"processed_QAOA_results_P_{layer}.csv"
    df_processed.to_csv(processed_file, index=False)
    print(f"Processed data for P={layer} saved to {processed_file}")

    return df_processed


# Define polynomial model functions for runtime fitting
def model_gidnet(n, c):
    return c * n**3

def model_qnet(n, c):
    return c * n**5


def plot_runtime_analysis(df: pd.DataFrame, layer: int):
    """
    Plots runtime comparison for GidNET, QNET, and Qiskit (if available).

    Args:
        df (pd.DataFrame): Processed DataFrame.
        layer (int): QAOA layer.
    """
    plt.figure(figsize=(8, 6), dpi=350)

    # Fit the runtime data
    x = df["Circuit Sizes"].values.astype(float)
    y_gidnet = df["GidNET Mean Runtime"].values.astype(float)
    yerr_gidnet = df["GidNET Stdev Runtime"].values.astype(float)
    y_qnet = df["QNET Mean Runtime"].values.astype(float)
    yerr_qnet = df["QNET Stdev Runtime"].values.astype(float)

    popt_gidnet, _ = curve_fit(model_gidnet, x, y_gidnet)
    popt_qnet, _ = curve_fit(model_qnet, x, y_qnet)

    # Plot error bars and fitted models
    plt.errorbar(x, y_gidnet, yerr=yerr_gidnet, label="GidNET Experimental", marker="o", color="green", fmt="-o", capsize=5)
    plt.plot(x, model_gidnet(x, *popt_gidnet), label=r'GidNET Fit: \mathbf{c \cdot n^3}', color="darkgreen", linestyle="--")

    plt.errorbar(x, y_qnet, yerr=yerr_qnet, label="QNET Experimental", marker="^", color="blue", fmt="-^", capsize=5)
    plt.plot(x, model_qnet(x, *popt_qnet), label=r'QNET Fit: \mathbf{c \cdot n^5}', color="navy", linestyle="--")

    if "Qiskit Mean Runtime" in df.columns:
        plt.errorbar(df["Circuit Sizes"], df["Qiskit Mean Runtime"], yerr=df["Qiskit Stdev Runtime"], label="Qiskit Experimental", marker="x", color="red", fmt="-x", capsize=5)

    plt.xlabel("Initial Circuit Width (n)", fontsize=14, fontweight="bold")
    plt.ylabel("Average Runtime (s)", fontsize=14, fontweight="bold")
    plt.xticks(fontweight="bold", fontsize=14)
    plt.yticks(fontweight="bold", fontsize=14)
    plt.legend(prop={"size": 14, "weight": "bold"})
    plt.yscale("log")

    output_file = plot_path / f"QAOA_runtime_analysis_P_{layer}.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved runtime analysis plot: {output_file}")


def plot_width_boxplots(df: pd.DataFrame, layer: int):
    """
    Generates a boxplot for the final circuit widths of GidNET, QNET, and Qiskit (if available).

    Args:
        df (pd.DataFrame): Processed DataFrame.
        layer (int): QAOA layer.
    """
    # Convert string lists to actual lists if necessary
    if isinstance(df["GidNET Width"].iloc[0], str):
        df["GidNET Width"] = df["GidNET Width"].apply(ast.literal_eval)
    
    if isinstance(df["QNET Width"].iloc[0], str):
        df["QNET Width"] = df["QNET Width"].apply(ast.literal_eval)

    if "Qiskit Width" in df.columns and isinstance(df["Qiskit Width"].iloc[0], str):
        df["Qiskit Width"] = df["Qiskit Width"].apply(ast.literal_eval)

    # Sort circuit sizes for proper ordering in the plot
    circuit_sizes_sorted = sorted(df["Circuit Sizes"].unique())

    # Prepare data for boxplot
    gidnet_widths = [df["GidNET Width"].iloc[i] for i in range(len(df))]
    qnet_widths = [df["QNET Width"].iloc[i] for i in range(len(df))]
    qiskit_widths = [df["Qiskit Width"].iloc[i] for i in range(len(df))] if "Qiskit Width" in df.columns else None

    fig, ax = plt.subplots(figsize=(8, 6), dpi=400)

    positions_gidnet = [x - 0.2 for x in range(len(circuit_sizes_sorted))]
    positions_qnet = [x for x in range(len(circuit_sizes_sorted))]
    positions_qiskit = [x + 0.2 for x in range(len(circuit_sizes_sorted))] if qiskit_widths else None

    # Plot boxplots for each algorithm
    box_gidnet = ax.boxplot(gidnet_widths, positions=positions_gidnet, widths=0.2, patch_artist=True, boxprops=dict(facecolor="lightgreen"))
    box_qnet = ax.boxplot(qnet_widths, positions=positions_qnet, widths=0.2, patch_artist=True, boxprops=dict(facecolor="lightblue"))

    if qiskit_widths:
        box_qiskit = ax.boxplot(qiskit_widths, positions=positions_qiskit, widths=0.2, patch_artist=True, boxprops=dict(facecolor="salmon"))
        ax.legend([box_gidnet["boxes"][0], box_qnet["boxes"][0], box_qiskit["boxes"][0]], ["GidNET", "QNET", "Qiskit"], loc="upper left", prop={"weight": "bold"})
    else:
        ax.legend([box_gidnet["boxes"][0], box_qnet["boxes"][0]], ["GidNET", "QNET"], loc="upper left", prop={"weight": "bold"})

    # Formatting
    ax.set_xlabel("Initial Circuit Width", fontsize=14, fontweight="bold")
    ax.set_ylabel("Final Circuit Width", fontsize=14, fontweight="bold")
    plt.xticks(range(len(circuit_sizes_sorted)), labels=circuit_sizes_sorted, fontweight="bold", fontsize=14)
    plt.yticks(fontweight="bold", fontsize=14)

    # Save the plot
    output_file = plot_path / f"QAOA_width_boxplot_P_{layer}.pdf"
    plt.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.show()
    print(f"Saved width boxplot: {output_file}")



if __name__ == "__main__":
    for layer in experiment_layers:
        df = process_experiment_data(layer)
        if df is not None:
            plot_runtime_analysis(df, layer)
            plot_width_boxplots(df, layer)

