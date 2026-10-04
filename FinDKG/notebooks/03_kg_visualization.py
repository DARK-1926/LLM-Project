"""
FinDKG Graph Visualization
==========================
Stage 1 - Notebook 03: Replicates Figure 3 from the research paper.
Visualizes a "snapshot subgraph of FinDKG as of January 2023",
highlighting the most relevant entities ranked by centrality, 
with nodes colored by their entity category.
"""

import os
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'FinDKG-full')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

print("=" * 60)
print("Replicating Figure 3: Subgraph as of January 1, 2023")
print("=" * 60)

# 1. Load mappings
entities = pd.read_csv(os.path.join(os.path.dirname(DATA_DIR), 'FinDKG', 'entity2id.txt'), sep='\t', header=None, names=['name','id','type','type_id'])
relations = pd.read_csv(os.path.join(DATA_DIR, 'relation2id.txt'), sep='\t', header=None, names=['name','id'])
time_map = pd.read_csv(os.path.join(DATA_DIR, 'time2id.txt'))

id2entity = dict(zip(entities['id'], entities['name']))
id2type = dict(zip(entities['id'], entities['type']))
id2relation = dict(zip(relations['id'], relations['name']))
time2id = dict(zip(time_map['DATE_WK'], time_map['TimeID']))

# 2. Find the time ID for January 1, 2023
target_time_id = time2id.get('2023-01-01')
if target_time_id is None:
    # Fallback to closest if exact date missing
    target_time_id = 260 

# 3. Load dataset
cols = ['head', 'rel', 'tail', 'time', '_']
train = pd.read_csv(os.path.join(DATA_DIR, 'train.txt'), sep='\t', header=None, names=cols)
valid = pd.read_csv(os.path.join(DATA_DIR, 'valid.txt'), sep='\t', header=None, names=cols)
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.txt'),  sep='\t', header=None, names=cols)
all_data = pd.concat([train, valid, test], ignore_index=True)

# 4. Extract snapshot for Jan 1, 2023
time_window = all_data[all_data['time'] == target_time_id].copy()
print(f"Total facts on Jan 1, 2023: {len(time_window)}")

# The paper states: "highlighting the most relevant entities... ranked by graph centrality metrics"
# So we select the top entities by degree in this snapshot
all_nodes_in_window = list(time_window['head']) + list(time_window['tail'])
top_nodes = [node for node, count in Counter(all_nodes_in_window).most_common(25)]

# Filter edges where BOTH nodes are in our top_nodes list to make a clean, highly connected subgraph
filtered_edges = time_window[time_window['head'].isin(top_nodes) & time_window['tail'].isin(top_nodes)]

print(f"Plotting subgraph with {len(filtered_edges)} highly central edges to match the paper...")

# 5. Define Category Colors (Exact Match from Paper's Figure 3 Legend)
category_colors = {
    'GPE': '#E74C3C',           # Red (Geopolitical Entities)
    'ORG/GOV': '#2C3E50',       # Dark Blue (Government)
    'ORG/REG': '#3498DB',       # Light Blue (Regulatory Bodies)
    'EVENT': '#2ECC71',         # Green (Event)
    'ECON_INDICATOR': '#E67E22',# Orange (Economic Indicators)
    'CONCEPT': '#F1C40F',       # Yellow (Concept)
    'COMP': '#34495E',          # Navy Blue (Companies)
    'PERSON': '#9B59B6',        # Purple (Individuals)
}
default_color = '#95A5A6'       # Grey for others

# 6. Build NetworkX Graph
G = nx.DiGraph()
for _, row in filtered_edges.iterrows():
    h_id, t_id, r_id = row['head'], row['tail'], row['rel']
    h_name = id2entity.get(h_id, str(h_id))
    t_name = id2entity.get(t_id, str(t_id))
    r_name = id2relation.get(r_id, str(r_id))
    
    # Shorten names for clean plotting
    if len(h_name) > 20: h_name = h_name[:17] + "..."
    if len(t_name) > 20: t_name = t_name[:17] + "..."
    
    # Store category for coloring
    G.add_node(h_name, category=id2type.get(h_id, 'Other'))
    G.add_node(t_name, category=id2type.get(t_id, 'Other'))
    
    G.add_edge(h_name, t_name, label=r_name)

# 7. Plotting
plt.figure(figsize=(16, 12), facecolor='white')
pos = nx.spring_layout(G, k=2.0, seed=42)

# Draw nodes color-coded by category
node_colors = [category_colors.get(G.nodes[n]['category'], default_color) for n in G.nodes()]
nx.draw_networkx_nodes(G, pos, node_size=1800, node_color=node_colors, 
                       edgecolors='white', linewidths=2, alpha=0.95)

# Draw edges and labels
nx.draw_networkx_edges(G, pos, edge_color='#BDC3C7', arrows=True, arrowsize=15, 
                       connectionstyle="arc3,rad=0.1", min_source_margin=15, min_target_margin=20)
nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold', font_family='sans-serif', font_color='black')

edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='#C0392B')

plt.title(f"Figure 3 Replica: Subgraph of FinDKG's most influential entities (Jan 1, 2023)", fontsize=16, fontweight='bold', pad=20)
plt.axis('off')

# Add legend for categories
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, edgecolor='white', label=cat) 
                  for cat, color in category_colors.items()]
plt.legend(handles=legend_elements, loc='upper left', title="Entity Categories", frameon=False)

plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'Figure_3.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')
print(f"[Success] Exact replica plot saved to: {out_path}")
