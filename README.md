# GidNET: Qubit Reuse Algorithm

[![arXiv](https://img.shields.io/badge/arXiv-2410.08817-B31B1B.svg)](https://arxiv.org/abs/2410.08817)

This repository contains the implementation of **GidNET**, a qubit reuse algorithm introduced in the paper [*Techniques for Optimized Quantum Circuit Cutting, Scalable Qubit Reuse, and High-Fidelity Generalized Measurements*](https://arxiv.org/abs/2410.08817). GidNET enables efficient quantum circuit decomposition by reusing qubits in a structured manner, optimizing hardware resource allocation and reducing quantum circuit width.

## 🔥 Key Features
- **Qubit Reuse Optimization**: Efficiently reduces the number of physical qubits required for a given circuit.
- **Circuit Cutting Integration**: Works alongside standard circuit cutting techniques for scalable quantum computation.
- **High-Fidelity Execution**: Mitigates errors introduced by qubit reuse while maintaining algorithmic accuracy.
- **Benchmark Comparisons**: Implements comparisons with Qiskit’s circuit cutting techniques and other quantum algorithms.

## 🏗️ Installation
To run the implementation, first clone this repository:
```bash
git clone https://github.com/gideonuchehara/GidNET-Qubit-Reuse-Algorithm.git
cd GidNET-Qubit-Reuse-Algorithm
```
Then install dependencies:
```bash
pip install -r requirements.txt
```
Ensure you have a working installation of [Qiskit](https://qiskit.org/) and Python (>= 3.8).

## 🚀 Usage
To test the GidNET qubit reuse algorithm on example circuits, run:
```bash
python run_experiments.py
```
To visualize results, use:
```bash
python plot_results.py
```
For in-depth implementation details, refer to `gidnet_qubit_reuse.py`.

## 📜 Paper Abstract
In this work, we propose **GidNET**, a structured qubit reuse algorithm that allows logical qubits to be dynamically mapped onto a minimal set of physical qubits while maintaining computational integrity. By strategically reusing qubits in complex quantum circuits, GidNET optimizes resource allocation, enabling deeper and more efficient quantum computations. We benchmark our approach against existing methods and demonstrate its effectiveness in reducing quantum circuit width while preserving high-fidelity results.

For full details, see our paper: [arXiv:2410.08817](https://arxiv.org/abs/2410.08817).

## 🛠️ Code Structure
```
📂 GidNET-Qubit-Reuse-Algorithm
│── 📄 gidnet_qubit_reuse.py   # Implementation of GidNET algorithm
│── 📄 run_experiments.py      # Runs benchmarks and example circuits
│── 📄 plot_results.py         # Generates plots for result visualization
│── 📄 README.md               # This file
│── 📄 requirements.txt        # Dependencies
│── 📂 data                    # Sample quantum circuit data
│── 📂 results                 # Output and benchmarking results
```

## 🔬 Benchmarking & Comparisons
The GidNET algorithm is tested against:
- **Qiskit Circuit Cutting** 
- **Standard Qubit Allocation Heuristics**
- **Other Quantum Hardware Optimization Techniques**

Results can be reproduced using:
```bash
python run_experiments.py --benchmark
```

## 📢 Citation
If you find this work useful, please consider citing:
```bibtex
@article{uchehara2024gidnet,
  author = {Uchehara, Gideon},
  title = {Techniques for Optimized Quantum Circuit Cutting, Scalable Qubit Reuse, and High-Fidelity Generalized Measurements},
  journal = {arXiv preprint arXiv:2410.08817},
  year = {2024}
}
```

## 🤝 Contributing
Contributions are welcome! Feel free to:
- Open an **issue** for bug reports and feature requests.
- Submit a **pull request** with improvements or additional benchmarks.

## 📩 Contact
For questions or collaborations, reach out via:
- **Email**: [gideonuchehara@example.com](mailto:gideonuchehara@example.com)
- **LinkedIn**: [Gideon Uchehara](https://www.linkedin.com/in/gideonuchehara/)
- **Twitter**: [@Gid_Uchehara](https://twitter.com/Gid_Uchehara)
