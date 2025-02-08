#!/usr/bin/env python3

"""
This script runs QAOA circuit experiments for GidNET, QNET, and Qiskit algorithms.
It generates and saves runtime and width results for various circuit sizes and QAOA layers.

Usage:
    Run this script from the command line to execute QAOA experiments for predefined circuit sizes and layers.

Outputs:
    - CSV files containing runtime statistics and circuit widths for different algorithms.
"""

import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from benchmarks.biadu_qnet_qubit_reuse.baidu_qnet_qr import (
    compute_qnet_qubit_reuse_list,
    from_qiskit_to_qnet
)
from gidnet.qubitreuse import GidNET
from gidnet.utils import safe_eval, apply_qiskit_qubit_reuse, create_qiskit_and_qnet_QAOA_circuits
from typing import List, Union

# Configure logging to track experiment progress
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def group_data_by_column(data: List[List[Union[int, float]]], columns: List[str]) -> pd.DataFrame:
    """
    Groups data by the first column and aggregates the remaining columns into lists.

    Args:
        data (List[List[Union[int, float]]]): List of data rows where each row contains numerical values.
        columns (List[str]): List of column names for the output DataFrame.

    Returns:
        pd.DataFrame: Data grouped by the first column with aggregated lists for the remaining columns.
    """
    if not data or not columns:
        raise ValueError("Data and columns must not be empty.")
    
    grouped_data = {}
    for dataset in data:
        for row in dataset:
            group_key = row[0]
            if group_key not in grouped_data:
                grouped_data[group_key] = {col: [] for col in columns[1:]}
            for i, col in enumerate(columns[1:], start=1):
                grouped_data[group_key][col].append(row[i])
    
    return pd.DataFrame([{columns[0]: k, **v} for k, v in grouped_data.items()])

# Define experiment parameters
circuit_sizes = [6, 10]  # List of circuit sizes to test
seed_num = 20  # Number of random graphs per qubit number
iterations = 10  # Number of iterations for optimization
experiment_layers = {1: [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50],
                     2: [6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46, 50]}  # QAOA layers and corresponding circuit sizes

def run_qaoa_experiment(layers: int, circuit_sizes: List[int]):
    """
    Runs the QAOA experiment for the specified circuit sizes and layers.

    Args:
        layers (int): The number of QAOA unitary layers.
        circuit_sizes (List[int]): The circuit sizes to be tested.
    """
    gidnet_results, qnet_results, qiskit_results = [], [], []
    
    for i, circuit_size in enumerate(circuit_sizes):
        logging.info(f"Running circuit {i+1} with {circuit_size} qubits for P={layers}")
        tmp_gidnet, tmp_qnet, tmp_qiskit = [], [], []
        
        circuits, _ = create_qiskit_and_qnet_QAOA_circuits(circuit_size, seed_num, layers)
        
        for circuit in circuits:
            # GidNET Experiment
            gidnet = GidNET(circuit)
            runtimes, min_width = [], int(1e10)
            for _ in range(iterations):
                start_time = time.time()
                gidnet.compile_to_dynamic_circuit(iterations)
                runtimes.append(time.time() - start_time)
                min_width = min(min_width, gidnet.dynamic_circuit_width)
            tmp_gidnet.append([circuit_size, np.mean(runtimes), np.std(runtimes), min_width])
            
            # QNET Experiment
            qnet_circuit = from_qiskit_to_qnet(circuit)
            runtimes, min_width = [], int(1e10)
            for _ in range(iterations):
                start_time = time.time()
                qnet_result = compute_qnet_qubit_reuse_list(qnet_circuit, method="random", shots=iterations)
                runtimes.append(time.time() - start_time)
                min_width = min(min_width, len(qnet_result))
            tmp_qnet.append([circuit_size, np.mean(runtimes), np.std(runtimes), min_width])
            
            # Qiskit Experiment (only for P=1)
            if layers == 1:
                runtimes, min_width = [], int(1e10)
                for _ in range(iterations):
                    start_time = time.time()
                    compiled_qiskit_circuit = apply_qiskit_qubit_reuse(circuit)
                    runtimes.append(time.time() - start_time)
                    min_width = min(min_width, compiled_qiskit_circuit.num_qubits)
                tmp_qiskit.append([circuit_size, np.mean(runtimes), np.std(runtimes), min_width])
        
        gidnet_results.append(tmp_gidnet)
        qnet_results.append(tmp_qnet)
        if layers == 1:
            qiskit_results.append(tmp_qiskit)
    
    df_gidnet = group_data_by_column(gidnet_results, ["Circuit Size", "GidNET Mean Runtime", "GidNET Stdev Runtime", "GidNET Min Width"])
    df_qnet = group_data_by_column(qnet_results, ["Circuit Size", "QNET Mean Runtime", "QNET Stdev Runtime", "QNET Min Width"])
    df = pd.merge(df_gidnet, df_qnet, on="Circuit Size")
    
    if layers == 1:
        df_qiskit = group_data_by_column(qiskit_results, ["Circuit Size", "Qiskit Mean Runtime", "Qiskit Stdev Runtime", "Qiskit Min Width"])
        df = pd.merge(df, df_qiskit, on="Circuit Size")
    
    output_path = Path("QAOA_result")
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / f"width_and_runtime_QAOA_cct_P_{layers}.csv", index=False)
    logging.info(f"Experiment for P = {layers} completed and data saved.")

if __name__ == "__main__":
    for layers, sizes in experiment_layers.items():
        run_qaoa_experiment(layers, sizes)
