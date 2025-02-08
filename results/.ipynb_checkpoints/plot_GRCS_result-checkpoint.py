import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from pathlib import Path
from matplotlib.lines import Line2D

# Load data for each depth
data_path = Path("GRCS_result")
df11 = pd.read_csv(data_path / "width_and_runtime_GRCS_cct_depth_11.csv")
df12 = pd.read_csv(data_path / "width_and_runtime_GRCS_cct_depth_12.csv")
df15 = pd.read_csv(data_path / "width_and_runtime_GRCS_cct_depth_15.csv")

# Adjust the figure size and set DPI for higher quality output
plt.figure(figsize=(8, 6), dpi=400)

# Define distinct linestyles
styles = ['-', ':', '-.']
depth_labels = ['Depth 11', 'Depth 12', 'Depth 15']

# Define colors and markers
colors = {'GidNET': 'green', 'QNET': 'blue', 'Qiskit': 'red'}
markers = {'GidNET': 'o', 'QNET': '^', 'Qiskit': 'x'}
algorithm_labels = list(colors.keys())

# Plot for each depth (GidNET, QNET, Qiskit)
for i, df in enumerate([df11, df12, df15]):
    plt.plot(df["Circuit Size"], df["GidNET Min Width"], marker=markers['GidNET'], color=colors['GidNET'], linestyle=styles[i], markersize=4)
    plt.plot(df["Circuit Size"], df["QNET Min Width"], marker=markers['QNET'], color=colors['QNET'], linestyle=styles[i], markersize=4)
    plt.plot(df["Circuit Size"], df["Qiskit Min Width"], marker=markers['Qiskit'], color=colors['Qiskit'], linestyle=styles[i], markersize=4)

# Adding titles and labels
plt.xlabel('Initial Circuit Width', fontsize=14, fontweight='bold')
plt.ylabel('Final Circuit Width', fontsize=14, fontweight='bold')

# Create custom legends
algorithm_handles = [Line2D([0], [0], color=colors[key], marker=markers[key], linestyle='None', markersize=8) for key in colors]
depth_handles = [Line2D([0], [0], color='black', lw=4, linestyle=style) for style in styles]
spacer = Line2D([0], [0], color='none', marker='None', linestyle='None')
combined_handles = algorithm_handles + [spacer] + depth_handles
combined_labels = algorithm_labels + [" "] + depth_labels

plt.legend(prop={'size': 12, 'weight': 'bold'}, handles=combined_handles, labels=combined_labels, loc='upper left', ncol=2, handlelength=3, fontsize=12, frameon=True, borderpad=1, labelspacing=1.2, columnspacing=4)
plt.xticks(fontweight='bold', fontsize=14)
plt.yticks(fontweight='bold', fontsize=14)
plt.savefig(data_path / 'GidNET_QNET_Qiskit_width_reduction.pdf', format='pdf', bbox_inches='tight')
plt.show()

# Define polynomial model functions
def model_gidnet(n, c):
    return c * n**3

def model_qnet(n, c):
    return c * n**5

plt.figure(figsize=(10, 8), dpi=600)
for i, df in enumerate([df11, df12, df15]):
    popt_gidnet, _ = curve_fit(model_gidnet, df["Circuit Size"], df["GidNET Mean Runtime"])
    popt_qnet, _ = curve_fit(model_qnet, df["Circuit Size"], df["QNET Mean Runtime"])
    
    plt.errorbar(df["Circuit Size"], df["GidNET Mean Runtime"], yerr=df["GidNET Stdev Runtime"],
                 fmt=styles[i], marker=markers['GidNET'], color=colors['GidNET'], capsize=5, label=f'GidNET {depth_labels[i]}')
    plt.errorbar(df["Circuit Size"], df["QNET Mean Runtime"], yerr=df["QNET Stdev Runtime"],
                 fmt=styles[i], marker=markers['QNET'], color=colors['QNET'], capsize=5, label=f'QNET {depth_labels[i]}')
    plt.errorbar(df["Circuit Size"], df["Qiskit Mean Runtime"], yerr=df["Qiskit Stdev Runtime"],
                 fmt=styles[i], marker=markers['Qiskit'], color=colors['Qiskit'], capsize=5, label=f'Qiskit {depth_labels[i]}')
    
if 1 in range(len([df11, df12, df15])):
    plt.plot(df12["Circuit Size"], model_gidnet(df12["Circuit Size"], *popt_gidnet), color='#cc79a7', linestyle="--",
             label=r'GidNET Fit: c⋅n3c \cdot n^3')
    plt.plot(df12["Circuit Size"], model_qnet(df12["Circuit Size"], *popt_qnet), color="orange", linestyle="--",
             label=r'QNET Fit: c⋅n5c \cdot n^5')

plt.xlabel('Initial Circuit Width', fontsize=16, fontweight='bold')
plt.ylabel('Average Runtime (s)', fontsize=16, fontweight='bold')
plt.yscale('log')
plt.xticks(fontweight='bold', fontsize=16)
plt.yticks(fontweight='bold', fontsize=16)

legend_handles = algorithm_handles + depth_handles + [spacer] + [Line2D([0], [0], color='#cc79a7', linestyle="--", lw=3), Line2D([0], [0], color="orange", linestyle="--", lw=3)]
legend_labels = algorithm_labels + depth_labels + [" "] + [r'GidNET Fit: c⋅n3c \cdot n^3', r'QNET Fit: c⋅n5c \cdot n^5']

plt.legend(prop={'size': 12, 'weight': 'bold'}, handles=legend_handles, labels=legend_labels, loc='upper left', ncol=3, handlelength=3, fontsize=12, frameon=True, borderpad=1, labelspacing=1.2, columnspacing=1.1)
plt.savefig(data_path / 'GidNET_QNET_Qiskit_runtime_comparison.pdf', format='pdf', bbox_inches='tight')
plt.show()
