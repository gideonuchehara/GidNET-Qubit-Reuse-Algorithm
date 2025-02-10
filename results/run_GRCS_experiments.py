import numpy as np
import pandas as pd
import logging
import time
from pathlib import Path
from benchmarks.biadu_qnet_qubit_reuse.baidu_qnet_qr import (
    compute_qnet_qubit_reuse_list,
    from_qiskit_to_qnet,
    compute_qnet_qubit_reuse_list_timing
)
from gidnet.qubitreuse import GidNET
from gidnet.utils import safe_eval, apply_qiskit_qubit_reuse, create_qiskit_and_qnet_GRCS_circuits
from benchmarks.qcg.helper_functions.benchmarks import generate_circ
from qiskit import QuantumCircuit

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="ANTLR")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the current working directory (repo root if running from the repo)
repo_path = Path.cwd()

# Define the data directory path
directory_path = repo_path / "data" / "GRSC"

# Define parameters
experiment_depths = {11: [16, 20, 25, 30, 36, 42, 49, 56, 64, 72, 81, 90, 100, 110, 121, 132, 144],
                     12: ["4x4", "4x5", "5x5", "5x6", "6x6", "6x7", "7x7", "7x8", "8x8", "8x9",
                          "9x9", "9x10", "10x10", "10x11", "11x11", "11x12", "12x12"],
                     15: ["4x4", "4x5", "5x5", "5x6", "6x6", "6x7", "7x7", "7x8", "8x8", "8x9",
                          "9x9", "9x10", "10x10", "10x11", "11x11", "11x12", "12x12"]}


iterations = 10  # Number of iterations for qubit reuse algorithms

def run_grcs_experiment(depth, circuit_sizes):
    gidnet_results = []
    qnet_results = []
    qiskit_results = []
    
    for circuit_size in circuit_sizes:
        logging.info(f"Running experiments for circuit size: {circuit_size} at depth {depth}")
        
        if depth == 11:
            circuit = generate_circ(
                num_qubits=circuit_size,
                depth=1,
                circuit_type="supremacy",
                reg_name="q",
                connected_only=True,
                seed=None,
            )
            circuit.measure_all()
        else:
            circuit, qnet_circuit = create_qiskit_and_qnet_GRCS_circuits(circuit_size, depth, directory_path)
            
            
        # GidNET Experiment
        gidnet = GidNET(circuit)
        runtimes = []
        min_width = int(1e10)
        for _ in range(iterations):
            start_time = time.time()
            gidnet.compile_to_dynamic_circuit(iterations)
            runtimes.append(time.time() - start_time)
            min_width = min(min_width, gidnet.dynamic_circuit_width)
        if depth != 11:
            gidnet_results.append([safe_eval(circuit_size), np.mean(runtimes), np.std(runtimes), min_width])
        else:
            gidnet_results.append([circuit_size, np.mean(runtimes), np.std(runtimes), min_width])
        
        # QNET Experiment
        qnet_circuit = from_qiskit_to_qnet(circuit)
        runtimes = []
        min_width = int(1e10)
        for _ in range(iterations):
            start_time = time.time()
            qnet_result = compute_qnet_qubit_reuse_list(qnet_circuit, method="random", shots=iterations)
            runtimes.append(time.time() - start_time)
            min_width = min(min_width, len(qnet_result))
        if depth != 11:
            qnet_results.append([safe_eval(circuit_size), np.mean(runtimes), np.std(runtimes), min_width])
        else:
            qnet_results.append([circuit_size, np.mean(runtimes), np.std(runtimes), min_width])
        
        # Qiskit Experiment
        runtimes = []
        min_width = int(1e10)
        for _ in range(iterations):
            start_time = time.time()
            compiled_qiskit_circuit = apply_qiskit_qubit_reuse(circuit)
            runtimes.append(time.time() - start_time)
            min_width = min(min_width, compiled_qiskit_circuit.num_qubits)
        if depth != 11:
            qiskit_results.append([safe_eval(circuit_size), np.mean(runtimes), np.std(runtimes), min_width])
        else:
            qiskit_results.append([circuit_size, np.mean(runtimes), np.std(runtimes), min_width])
    
    df_gidnet = pd.DataFrame(gidnet_results, columns=["Circuit Size", "GidNET Mean Runtime", "GidNET Stdev Runtime", "GidNET Min Width"])
    df_qnet = pd.DataFrame(qnet_results, columns=["Circuit Size", "QNET Mean Runtime", "QNET Stdev Runtime", "QNET Min Width"])
    df_qiskit = pd.DataFrame(qiskit_results, columns=["Circuit Size", "Qiskit Mean Runtime", "Qiskit Stdev Runtime", "Qiskit Min Width"])
    
    df = pd.merge(df_gidnet, df_qnet, on="Circuit Size")
    df = pd.merge(df, df_qiskit, on="Circuit Size")
    
    output_path = Path("GRCS_result")
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path / f"width_and_runtime_GRCS_cct_depth_{depth}.csv", index=False)
    logging.info(f"Experiment for depth {depth} completed and data saved.")

if __name__ == "__main__":
    for depth, sizes in experiment_depths.items():
        run_grcs_experiment(depth, sizes)
