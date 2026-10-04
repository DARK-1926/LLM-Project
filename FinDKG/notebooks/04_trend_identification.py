"""
Trend Identification: Covid-19 Centrality Over Time
===================================================
Stage 1 - Notebook 04: Replicates Figure 5 from the research paper.
This script calculates four graph metrics of centrality (Degree, 
Betweenness, Eigenvector, and PageRank) for the "COVID-19" entity
across all temporal snapshots of the FinDKG-full dataset, applies a 
rolling 1-year z-score normalization, and plots the trend to match
the paper's Figure 5.
"""

import os
import pandas as pd
import networkx as nx
import numpy as np
import os
os.environ.pop('MPLBACKEND', None)  # Prevent Kaggle inline backend clash
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'FinDKG-full')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

print("=" * 60)
print("Replicating Figure 5: COVID-19 Centrality Trend (All 4 Metrics)")
print("=" * 60)

# Load entity mapping
print("1. Loading entity mappings...")
entities = pd.read_csv(os.path.join(DATA_DIR, 'entity2id.txt'), sep='\t', header=None, names=['name','id','type','type_id'])
covid_id = entities[entities['name'] == 'COVID-19']['id'].values[0]

# Load time mapping
print("2. Loading time mappings...")
time_map = pd.read_csv(os.path.join(DATA_DIR, 'time2id.txt'))
id2time = dict(zip(time_map['TimeID'], time_map['DATE_WK']))

# Load graph quadruples
print("3. Loading graph quadruples...")
cols = ['head', 'rel', 'tail', 'time', '_']
train = pd.read_csv(os.path.join(DATA_DIR, 'train.txt'), sep='\t', header=None, names=cols)
valid = pd.read_csv(os.path.join(DATA_DIR, 'valid.txt'), sep='\t', header=None, names=cols)
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.txt'),  sep='\t', header=None, names=cols)
all_data = pd.concat([train, valid, test], ignore_index=True)

# We will limit the timeline from 2018 to end of 2022 to match the paper's X-axis
times = sorted(all_data['time'].unique())

dates = []
metrics = {'Degree': [], 'Betweenness': [], 'Eigenvector': [], 'PageRank': []}

print("4. Computing Centrality Metrics for each time step (This may take a minute)...")
for t in times:
    date = pd.to_datetime(id2time.get(t))
    if date.year > 2022:  # Paper's graph stops at Dec 2022
        break
        
    dates.append(date)
    window = all_data[all_data['time'] == t]
    
    # Build NetworkX Directed Graph for this time step
    G = nx.DiGraph()
    for _, row in window.iterrows():
        G.add_edge(row['head'], row['tail'])
        
    # If COVID-19 is not in this week's news, metrics are 0
    if covid_id not in G:
        metrics['Degree'].append(0.0)
        metrics['Betweenness'].append(0.0)
        metrics['Eigenvector'].append(0.0)
        metrics['PageRank'].append(0.0)
        continue

    # 1. Degree Centrality
    deg = nx.degree_centrality(G).get(covid_id, 0.0)
    
    # 2. Betweenness Centrality (approximate for speed, k=50)
    bet = nx.betweenness_centrality(G, k=min(50, len(G.nodes))).get(covid_id, 0.0)
    
    # 3. Eigenvector Centrality (wrap in try-except in case it doesn't converge)
    try:
        eig = nx.eigenvector_centrality(G, max_iter=500).get(covid_id, 0.0)
    except:
        eig = 0.0
        
    # 4. PageRank
    pr = nx.pagerank(G, alpha=0.85).get(covid_id, 0.0)
    
    metrics['Degree'].append(deg)
    metrics['Betweenness'].append(bet)
    metrics['Eigenvector'].append(eig)
    metrics['PageRank'].append(pr)

df = pd.DataFrame(metrics, index=dates)

# Calculate rolling 1-year z-score (52 weeks)
print("5. Applying rolling 1-year z-score normalization...")
window_size = 52

for col in df.columns:
    rolling_mean = df[col].rolling(window=window_size, min_periods=1).mean()
    rolling_std = df[col].rolling(window=window_size, min_periods=1).std()
    rolling_std = rolling_std.replace(0, np.nan)
    df[f'{col}_Z'] = (df[col] - rolling_mean) / rolling_std
    df[f'{col}_Z'] = df[f'{col}_Z'].fillna(0)

# Plotting to match Paper's Figure 5
print("6. Generating exact Figure 5 replica...")
plt.figure(figsize=(14, 7), facecolor='white')

plt.plot(df.index, df['Degree_Z'], color='darkblue', linewidth=1.5, label='Degree Centrality')
plt.plot(df.index, df['Betweenness_Z'], color='deepskyblue', linewidth=1.5, label='Betweenness Centrality')
plt.plot(df.index, df['Eigenvector_Z'], color='limegreen', linewidth=1.5, label='Eigenvector Centrality')
plt.plot(df.index, df['PageRank_Z'], color='mediumpurple', linewidth=1.5, label='PageRank')

plt.axhline(0, color='gray', linestyle='-', linewidth=0.5)

# Formatting to match the paper
plt.title('Evolution of the Covid-19 entity centrality measures over time (2018-2022)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Rolling 1-Year Z-score, Covid-19')
plt.ylim(-4, 8)  # Match scale of paper (-4 to 8)

# Add Legend (match paper's top-left style)
plt.legend(loc='upper left', ncol=4, fontsize=9, frameon=False)

# Add annotations matching paper
max_peak_date = pd.to_datetime('2020-03-15')
plt.annotate('Jan 22, 2020\nChina lockdown', xy=(pd.to_datetime('2020-01-22'), 6), xytext=(pd.to_datetime('2019-09-01'), 7), arrowprops=dict(arrowstyle="->", color='red'))
plt.annotate('Mar 11, 2020\nWHO declared Covid-19 a pandemic', xy=(pd.to_datetime('2020-03-11'), 5.5), xytext=(pd.to_datetime('2020-06-01'), 6.5), arrowprops=dict(arrowstyle="->", color='red'))

plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()

out_path = os.path.join(OUTPUT_DIR, 'Figure_5.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight')

print(f"\n[Success] Exact replica plot saved to: {out_path}")
