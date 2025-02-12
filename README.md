# GidNET: Graph-Based Identification of Qubit Network for Qubit Reuse

## Overview
GidNET (Graph-Based Identification of Qubit Network) is a **qubit reuse algorithm** designed to optimize quantum circuits by minimizing the number of qubits required while preserving computational integrity. The algorithm applies a **graph-theoretic approach** to dynamically reassign logical qubits, leveraging structural properties of quantum circuits to enable efficient execution on quantum hardware with limited qubit resources.

This repository provides an **implementation of GidNET**, along with benchmark experiments comparing it to other qubit reuse techniques such as **Qiskit qubit reuse** and **QNET random qubit reuse**. The repository also includes scripts for reproducing experimental results, analyzing runtime trade-offs, and evaluating optimal iteration counts.

## Key Features
- **Graph-Theoretic Qubit Reuse**: GidNET identifies reuse opportunities using graph-based techniques.
- **Comparison with Existing Approaches**: Benchmarks against **Qiskit** and **QNET** reuse methods.
- **Flexible Experimentation Framework**: Supports various quantum circuit types, including **QAOA** and **GRCS (Google Random Circuit Sampling)**.
- **Iteration Analysis**: Evaluates the optimal number of iterations to maximize qubit reuse efficiency.
- **Robust Data Processing & Visualization**: Generates plots and tables for analysis.

## Reference Papers
The algorithm and its theoretical foundation are described in the following papers:
- **Published IEEE Version**: [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10821360?casa_token=F2Zpmr1CPiMAAAAA:mu8Zo15ZlD9sAoOst3680nRpIaIB5Tu_HXSiKofl6KUnf69q6yf__uJrVKdnaSuw0sP3q1MxdQ)
- **Preprint on ArXiv**: [arXiv 2410.08817](https://arxiv.org/abs/2410.08817)

## Repository Structure
This repository is organized as follows:

### **1. Core Algorithm (gidnet/)**
- **`gidnet/qubitreuse.py`** - Implementation of GidNET’s qubit reuse algorithm.
- **`gidnet/utils.py`** - Helper functions, including circuit transformations and analysis tools.
- **`gidnet/__init__.py`** - Package initialization.

### **2. Experimental Results (results/)**
Contains data, scripts, and plots generated from benchmark experiments:

- **GRCS_result/** - Results from Google Random Circuit Sampling experiments.
- **QAOA_result/** - Results from Quantum Approximate Optimization Algorithm (QAOA) circuits.
- **Optimal_iterations/** - Analysis of iteration count needed to achieve optimal qubit reuse.
- **data/** - Raw datasets used in experiments.
- **`run_GRCS_experiments.py`** - Script for running GRCS circuit experiments.
- **`run_QAOA_experiments.py`** - Script for running QAOA circuit experiments.
- **`plot_GRCS_result.py`** - Script for visualizing GRCS results.
- **`plot_QAOA_result.py`** - Script for visualizing QAOA results.
- **`gidnet_iteration_analysis.py`** - Computes the optimal number of iterations for GidNET.

### **3. Documentation (docs/)**
Contains explanatory materials and theoretical insights:
- **`docs/GidNET_Iteration_Analysis.md`** - Explanation of how optimal iterations for GidNET are determined.
- **`docs/Qubit_Reuse_Methods.md`** - Comparison of GidNET with other qubit reuse strategies.

## Installation
### **Requirements**
This project requires Python 3.8+ and the following dependencies:
```bash
pip install -r requirements.txt
```
Ensure that you have **Qiskit** installed to run quantum circuit simulations.

### **Usage**
#### **Running Qubit Reuse Experiments**
1. **GRCS Circuit Experiments**:
   ```bash
   python results/run_GRCS_experiments.py
   ```
2. **QAOA Circuit Experiments**:
   ```bash
   python results/run_QAOA_experiments.py
   ```
3. **Optimal Iterations Analysis**:
   ```bash
   python results/gidnet_iteration_analysis.py
   ```

#### **Plotting Results**
To visualize the experiment outputs, use:
```bash
python results/plot_GRCS_result.py
python results/plot_QAOA_result.py
```

## Optimal Number of Iterations Analysis
To determine the best number of iterations for GidNET, we analyze the **probability of finding the least-width circuit** under different iteration settings:
- **n** (number of qubits in the original circuit)
- **n/2, n/4**
- **log(n), log(n/2), log(n/4)**

We compute a **score** based on:
\[ \text{Score} = \text{Probability} \times \left( \frac{\text{Min Width Found}}{\text{Observed Width}} \right) \]
This helps balance the trade-off between computational cost and effectiveness of qubit reuse.

To run the analysis and generate plots:
```bash
python results/gidnet_iteration_analysis.py
```

## Contributing
Contributions to GidNET are welcome! Feel free to open **issues** or submit **pull requests** for bug fixes, enhancements, or new features.

## License
This project is licensed under the **MIT License**.

---
For more details, check the full documentation in the `docs/` directory or refer to the referenced papers.


