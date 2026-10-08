import matplotlib.pyplot as plt
import numpy as np
import os

# Data from paper + our replicated scores
metrics = ['MRR', 'Hits@3', 'Hits@10']
models = ['KGTransformer w/o node type (Replicated)', 'KGTransformer (Replicated)']

# Scores for each model across the 3 metrics (MRR, Hits@3, Hits@10)
# Multiplied by 100 for percentage scale
scores = {
    'KGTransformer w/o node type (Replicated)': [11.67, 12.87, 19.85],
    'KGTransformer (Replicated)': [12.04, 13.21, 20.32] 
}

x = np.arange(len(metrics))  # the label locations
width = 0.35  # wider bars since there are only 2 models

fig, ax = plt.subplots(figsize=(10, 6))

# Beautiful gradient blue colors for our 2 models
colors = ['#4dbaf5', '#b3e5fc']

# Plot bars
rects_list = []
for i, model in enumerate(models):
    offset = (i - 0.5) * width
    rects = ax.bar(x + offset, scores[model], width, label=model, color=colors[i], edgecolor='white')
    rects_list.append(rects)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Metric Value', fontsize=12)
ax.set_title('Performance comparison of models on FinDKG (Figure 4)', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(loc='upper left')

# Attach a text label above each bar, displaying its height.
for rects in rects_list:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

ax.set_ylim(0, 25)

plt.tight_layout()

# Save the figure
output_path = os.path.join(os.path.dirname(__file__), 'FinDKG', 'outputs', 'figure4_replication.png')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Graph successfully saved to: {output_path}")
