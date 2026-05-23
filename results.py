import matplotlib.pyplot as plt
import numpy as np

# --- Deployment Data ---
# Replace these with your exact final numbers
models = ['Baseline (FP32)', 'Edge Deploy (INT8)']
size_kb = [245.5, 62.1]        # Model file size
accuracy = [94.8, 93.6]        # Validation Accuracy
latency_ms = [45.0, 3.2]       # CPU vs Microcontroller Latency

# --- IEEE Formatting Settings ---
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 300 # Publication quality

# --- Plot 1: Quantization Impact (Size vs Accuracy) ---
fig, ax1 = plt.subplots(figsize=(3.5, 2.5)) # IEEE single-column width is ~3.5 inches

color1 = '#2c3e50'
color2 = '#e74c3c'

# Bar settings
x = np.arange(len(models))
width = 0.35

rects1 = ax1.bar(x - width/2, size_kb, width, label='Size (KB)', color=color1)
ax1.set_ylabel('Model Size (KB)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(x)
ax1.set_xticklabels(models)

# Create a twin axis to plot accuracy on the same chart
ax2 = ax1.twinx()
rects2 = ax2.bar(x + width/2, accuracy, width, label='Accuracy (%)', color=color2)
ax2.set_ylabel('Accuracy (%)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)
ax2.set_ylim(85, 100) # Zoom in on the relevant accuracy range

# Combine legends
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

plt.title('Content-Aware PTQ Impact')
plt.tight_layout()
plt.savefig('quantization_results_ieee.png', bbox_inches='tight')
print("Saved quantization_results_ieee.png")