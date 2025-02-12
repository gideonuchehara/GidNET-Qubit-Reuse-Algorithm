# GidNET: Graph-Based Identification of Qubit Network for Qubit Reuse

## Overview
GidNET (Graph-Based Identification of Qubit Network) is a **qubit reuse algorithm** designed to optimize quantum circuits by minimizing the number of qubits required while preserving computational integrity. The algorithm applies a **graph-theoretic approach** to dynamically reassign logical qubits, leveraging structural properties of quantum circuits to enable efficient execution on quantum hardware with limited qubit resources.

This repository provides an **implementation of GidNET**, along with benchmark experiments comparing it to other qubit reuse techniques such as **Qiskit qubit reuse** and **QNET random qubit reuse**. The repository also includes scripts for reproducing experimental results, analyzing runtime trade-offs, and evaluating optimal iteration counts.

## Key Features
- **Graph-Based Qubit Reuse**: GidNET identifies reuse opportunities using graph-based techniques.
- **Comparison with Existing Approaches**: Benchmarks against **Qiskit** and **QNET** reuse methods.
- **Flexible Experimentation Framework**: Supports various quantum circuit types, including **QAOA** and **GRCS (Google Random Circuit Sampling)**.
- **Iteration Analysis**: Evaluates the optimal number of iterations to maximize qubit reuse efficiency.
- **Robust Data Processing & Visualization**: Generates plots and tables for analysis.

## Repository Structure
This repository is organized as follows:

---

## 📂 Repository Structure

```
GidNET-Qubit-Reuse-Algorithm/
│── gidnet/                     # Implementation of GidNET
│   ├── __init__.py             # Package initialization
│   ├── qubitreuse.py           # GidNET algorithm
│   ├── utils.py                # Utility functions
│
│── results/                    # Experimental results and plots
│   ├── GRCS_result/            # GidNET results for GRCS circuits
│   ├── QAOA_result/            # GidNET results for QAOA circuits
│   ├── Optimal_iterations/      # Iteration analysis for optimal qubit reuse
│   ├── gidnet_iteration_analysis.py # Script for analyzing optimal iterations
│   ├── plot_GRCS_result.py     # Plot script for GRCS results
│   ├── plot_QAOA_result.py     # Plot script for QAOA results
│   ├── run_GRCS_experiments.py # Script to run GRCS circuit experiments
│   ├── run_QAOA_experiments.py # Script to run QAOA circuit experiments
│
│── docs/                       # Documentation
│   ├── iteration_analysis.md   # Detailed explanation of iteration analysis
│
│── notebook.ipynb              # Jupyter notebook for explaining how GidNET is used
│── README.md                   # This readme file
```

---

<!--

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
- **`docs/optimal_iterations_analysis.md`** - Explanation of how optimal iterations for GidNET are determined.
 -->
 
 ---

## 🔧 Installation

To use GidNET, first clone the repository and install dependencies:

```bash
git clone https://github.com/gideonuchehara/GidNET-Qubit-Reuse-Algorithm.git
cd GidNET-Qubit-Reuse-Algorithm
pip install -r requirements.txt
```

GidNET relies on Qiskit, NumPy, and Matplotlib for quantum circuit generation, analysis, and visualization.

---

## 🚀 Running Experiments

### 1️⃣ **Running GRCS Circuit Experiments**

To run the experiments on Google Random Circuit Sampling (GRCS) circuits:

```bash
python results/run_GRCS_experiments.py
```

Results will be saved in `results/GRCS_result/`.

### 2️⃣ **Running QAOA Circuit Experiments**

To run the experiments on QAOA circuits:

```bash
python results/run_QAOA_experiments.py
```

Results will be saved in `results/QAOA_result/`.

---

## 📊 Analyzing and Visualizing Results

### **Plotting GRCS Circuit Results**
```bash
python results/plot_GRCS_result.py
```
This generates plots comparing GidNET, QNET, and Qiskit in terms of circuit width reduction and runtime.

### **Plotting QAOA Circuit Results**
```bash
python results/plot_QAOA_result.py
```
This generates plots for the QAOA experiment results.

### **Iteration Analysis**
To determine the optimal number of iterations for GidNET to find the smallest qubit width:
```bash
python results/gidnet_iteration_analysis.py
```
Results will be stored in `results/Optimal_iterations/`.

---

## 📖 Understanding the Iteration Analysis

Since GidNET is a probabilistic algorithm, multiple iterations are performed to ensure the best qubit reuse outcome. However, running too many iterations increases runtime without significant improvement in width reduction. We analyze optimal iterations using:

| Iteration Setting  | Description |
|--------------------|-------------|
| **n**             | Total number of qubits in the circuit |
| **n/2**           | Half the total qubits |
| **n/4**           | A quarter of the total qubits |
| **log(n)**        | Logarithmic scaling of qubits |
| **log(n/2)**      | Logarithmic scaling of half the qubits |
| **log(n/4)**      | Logarithmic scaling of a quarter of the qubits |

We score each iteration count based on its effectiveness in finding the minimal circuit width:

```
score = probability × (min_width / obtained_width)
```

A higher score indicates a better iteration setting.

---

## Contributing
Contributions to GidNET are welcome! Feel free to open **issues** or submit **pull requests** for bug fixes, enhancements, or new features.

## License
This project is licensed under the **MIT License**.

## 📜 References

📄 **Official Paper:** [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10821360?casa_token=F2Zpmr1CPiMAAAAA:mu8Zo15ZlD9sAoOst3680nRpIaIB5Tu_HXSiKofl6KUnf69q6yf__uJrVKdnaSuw0sP3q1MxdQ)  
📄 **Preprint Version:** [arXiv](https://arxiv.org/abs/2410.08817)

---

## 👨‍💻 Author

**Gideon Uchehara**  
Email: [gideonuchehara@gmail.com](mailto:gideonuchehara@gmail.com)  
GitHub: [@gideonuchehara](https://github.com/gideonuchehara)


