import matplotlib.pyplot as plt
import numpy as np
import os

# Data
metrics = ['MRR', 'Hits@1', 'Hits@3', 'Hits@10']
scores = [18.49, 11.98, 20.02, 31.57]

# Create figure
fig, ax = plt.subplots(figsize=(8, 6))

# Define colors
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

# Create bars
bars = ax.bar(metrics, scores, color=colors, width=0.6)

# Add value labels on top of the bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
            f'{height:.2f}%',
            ha='center', va='bottom', fontweight='bold')

# Customize axes and title
ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
ax.set_title('FinDKG-full Link Prediction Results (Figure 4 Replication)', fontsize=14, fontweight='bold')
ax.set_ylim(0, max(scores) + 5)
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Make it look nice
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Save the figure
output_path = r'C:\Users\mohit\.gemini\antigravity-ide\brain\fa86d4ab-bae7-4a9f-b18d-7987604433ed\scratch\figure4_replication.png'
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to: {output_path}")
