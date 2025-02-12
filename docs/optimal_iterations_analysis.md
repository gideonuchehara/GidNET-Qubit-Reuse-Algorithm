# **Analysis of Optimal Iterations for GidNET**

## **Introduction**
GidNET is a qubit reuse algorithm that aims to minimize the number of physical qubits required for a quantum circuit. However, since GidNET employs a **randomized approach**, it needs to be executed multiple times to determine the best qubit reuse sequence. This analysis explores the optimal number of iterations required to find the **smallest possible width** while balancing runtime efficiency.

## **Iteration Strategy**
To determine the optimal number of iterations, we analyze different iteration settings based on the original number of qubits **(n)** in the quantum circuit:

- **n** (Total number of qubits)
- **n/2** (Half the number of qubits)
- **n/4** (One-quarter the number of qubits)
- **log(n)** (Logarithm of the number of qubits)
- **log(n/2)** (Logarithm of half the number of qubits)
- **log(n/4)** (Logarithm of one-quarter the number of qubits)

For each setting, GidNET is run multiple times, and the smallest circuit width found is recorded along with the probability of obtaining that width.

## **Scoring Function**
The scoring function evaluates how effective each iteration setting is at finding the smallest possible width. The score is calculated as:

\[ \text{Score} = \text{Probability of finding min width} \times \left( \frac{\text{Smallest width found}}{\text{Width at this iteration setting}} \right) \]

Where:
- **Probability of finding min width**: Fraction of times the smallest width was found out of the total runs.
- **Smallest width found**: The minimum width achieved across all runs.
- **Width at this iteration setting**: The width obtained at a given iteration count.

This formula ensures that iteration settings with **higher probabilities** of finding the optimal width receive **higher scores**, while settings that result in **larger widths** receive lower scores.

## **Results & Interpretation**
### **Bar Chart Analysis**
A **bar chart** is generated with:
- **X-axis**: The iteration settings (`n, n/2, n/4, log(n), log(n/2), log(n/4)`).
- **Y-axis**: The computed **score** for each setting.

This visualization helps to determine which iteration setting offers the best trade-off between **runtime efficiency** and **probability of finding the optimal width**.

### **Key Observations**
- Higher values of **n (full iterations)** generally find the optimal width but may be computationally expensive.
- **log(n) and log(n/2)** often provide an efficient balance between **runtime and accuracy**.
- Smaller values like **n/4 or log(n/4)** may lead to suboptimal results due to insufficient exploration of reuse sequences.

## **Conclusion**
This analysis helps to determine a practical iteration setting based on circuit size. The best choice depends on the application:
- **For high accuracy**, running **n** or **n/2** iterations is ideal.
- **For a trade-off between speed and accuracy**, **log(n) or log(n/2)** is a reasonable choice.
- **For quick estimations**, **n/4 or log(n/4)** may be sufficient but could lead to suboptimal widths.

By leveraging this approach, one can efficiently run GidNET to maximize qubit reuse while minimizing computational cost.

---

*For further details, refer to the implementation in `gidnet_iteration_analysis.py`.*


