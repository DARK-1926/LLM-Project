"""
FinDKG Dataset Exploration
==========================
Stage 1 - Notebook 01: Understand the dataset structure, entity types,
relation types, time splits, and basic statistics.

This script can be run directly or converted to a Jupyter notebook.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
from collections import Counter

# ── Paths ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'FinDKG')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================================
# 1. Load Dataset Files
# =====================================================================
print("=" * 60)
print("1. LOADING DATASET FILES")
print("=" * 60)

cols = ['head', 'rel', 'tail', 'time', '_']
train = pd.read_csv(os.path.join(DATA_DIR, 'train.txt'), sep='\t', header=None, names=cols)
valid = pd.read_csv(os.path.join(DATA_DIR, 'valid.txt'), sep='\t', header=None, names=cols)
test  = pd.read_csv(os.path.join(DATA_DIR, 'test.txt'),  sep='\t', header=None, names=cols)

stat = pd.read_csv(os.path.join(DATA_DIR, 'stat.txt'), sep='\t', header=None,
                   names=['num_entities', 'num_relations', '_'])

print(f"stat.txt → Entities: {stat['num_entities'].item()}, Relations: {stat['num_relations'].item()}")
print(f"\nSplit sizes:")
print(f"  Train:      {len(train):,} quadruples")
print(f"  Validation: {len(valid):,} quadruples")
print(f"  Test:       {len(test):,} quadruples")
print(f"  Total:      {len(train) + len(valid) + len(test):,} quadruples")


# =====================================================================
# 2. Entity Mapping (entity2id.txt)
# =====================================================================
print("\n" + "=" * 60)
print("2. ENTITY MAPPING")
print("=" * 60)

entities = pd.read_csv(os.path.join(DATA_DIR, 'entity2id.txt'), sep='\t', header=None,
                       names=['name', 'id', 'type', 'type_id'])
print(f"Total entities: {len(entities):,}")
print(f"\nEntity type distribution:")
entity_type_counts = entities['type'].value_counts()
for etype, count in entity_type_counts.items():
    print(f"  {etype:20s} → {count:5d} entities ({100*count/len(entities):.1f}%)")

print(f"\nSample entities (first 20):")
print(entities.head(20).to_string(index=False))

# Dump Table 2
table_2 = pd.DataFrame(list(entity_type_counts.items()), columns=['Category', 'Count'])
table_2.to_csv(os.path.join(OUTPUT_DIR, 'Table_2.csv'), index=False)
print("[Saved] Table_2.csv")
# Dump Table 2
table_2 = pd.DataFrame(list(entity_type_counts.items()), columns=['Category', 'Count'])
table_2.to_csv(os.path.join(OUTPUT_DIR, 'Table_2.csv'), index=False)
print("[Saved] Table_2.csv")


# =====================================================================
# 3. Relation Mapping (relation2id.txt)
# =====================================================================
print("\n" + "=" * 60)
print("3. RELATION TYPES (15 total)")
print("=" * 60)

relations = pd.read_csv(os.path.join(DATA_DIR, 'relation2id.txt'), sep='\t', header=None,
                        names=['name', 'id'])
for _, row in relations.iterrows():
    print(f"  ID {row['id']:2d} → {row['name']}")

# Dump Table 1
relations[['name', 'id']].to_csv(os.path.join(OUTPUT_DIR, 'Table_1.csv'), index=False, header=['Relation', 'ID'])
print("[Saved] Table_1.csv")

# Build lookup dicts
id2entity = dict(zip(entities['id'], entities['name']))
id2relation = dict(zip(relations['id'], relations['name']))
id2entity_type = dict(zip(entities['id'], entities['type']))


# =====================================================================
# 4. Quadruple Format Explanation
# =====================================================================
print("\n" + "=" * 60)
print("4. QUADRUPLE FORMAT")
print("=" * 60)
print("Each line in train/valid/test.txt is a quadruple:")
print("  head_id  relation_id  tail_id  time_id  _")
print("\nDecoded examples from training set:")
for i, row in train.head(10).iterrows():
    h_name = id2entity.get(row['head'], f"entity_{row['head']}")
    r_name = id2relation.get(row['rel'], f"rel_{row['rel']}")
    t_name = id2entity.get(row['tail'], f"entity_{row['tail']}")
    print(f"  ({h_name}, {r_name}, {t_name}, t={row['time']})")


# =====================================================================
# 5. Temporal Distribution
# =====================================================================
print("\n" + "=" * 60)
print("5. TEMPORAL DISTRIBUTION")
print("=" * 60)

all_data = pd.concat([train, valid, test], ignore_index=True)
time_counts = all_data['time'].value_counts().sort_index()

print(f"Total unique time steps: {len(time_counts)}")
print(f"Train time range:      {train['time'].min()} → {train['time'].max()} ({train['time'].nunique()} steps)")
print(f"Validation time range: {valid['time'].min()} → {valid['time'].max()} ({valid['time'].nunique()} steps)")
print(f"Test time range:       {test['time'].min()} → {test['time'].max()} ({test['time'].nunique()} steps)")

# Plot temporal distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Quadruples per time step
axes[0].bar(time_counts.index, time_counts.values, color='steelblue', alpha=0.7)
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Number of Quadruples')
axes[0].set_title('Quadruples per Time Step')

# Cumulative distribution
cumul = time_counts.cumsum()
axes[1].plot(cumul.index, cumul.values, color='darkred', linewidth=2)
axes[1].set_xlabel('Time Step')
axes[1].set_ylabel('Cumulative Quadruples')
axes[1].set_title('Cumulative Knowledge Graph Growth')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'temporal_distribution.png'), dpi=150)
print(f"\n[Saved] temporal_distribution.png")


# =====================================================================
# 6. Relation Type Distribution
# =====================================================================
print("\n" + "=" * 60)
print("6. RELATION TYPE DISTRIBUTION IN TRAINING SET")
print("=" * 60)

rel_counts = train['rel'].value_counts().sort_values(ascending=True)
rel_names = [id2relation.get(r, str(r)) for r in rel_counts.index]

for rname, count in zip(rel_names, rel_counts.values):
    bar = '█' * int(50 * count / rel_counts.max())
    print(f"  {rname:25s} {count:6d}  {bar}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(rel_names, rel_counts.values, color='teal', alpha=0.8)
ax.set_xlabel('Count')
ax.set_title('Relation Type Distribution (Training Set)')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'relation_distribution.png'), dpi=150)
print(f"[Saved] relation_distribution.png")


# =====================================================================
# 7. Most Connected Entities (Degree Analysis)
# =====================================================================
print("\n" + "=" * 60)
print("7. TOP 20 MOST CONNECTED ENTITIES (Training Set)")
print("=" * 60)

all_entities_in_train = list(train['head']) + list(train['tail'])
entity_degree = Counter(all_entities_in_train).most_common(20)

print(f"  {'Rank':<5} {'Entity':<40} {'Type':<15} {'Degree':<8}")
print(f"  {'─'*5} {'─'*40} {'─'*15} {'─'*8}")
for rank, (eid, degree) in enumerate(entity_degree, 1):
    ename = id2entity.get(eid, f"entity_{eid}")[:40]
    etype = id2entity_type.get(eid, "UNK")
    print(f"  {rank:<5} {ename:<40} {etype:<15} {degree:<8}")


# =====================================================================
# 8. Summary Statistics
# =====================================================================
print("\n" + "=" * 60)
print("8. SUMMARY — KEY NUMBERS FOR YOUR REPORT")
print("=" * 60)

summary = {
    "Total entities": stat['num_entities'].item(),
    "Total relations": stat['num_relations'].item(),
    "Entity types": 12,
    "Training quadruples": len(train),
    "Validation quadruples": len(valid),
    "Test quadruples": len(test),
    "Total quadruples": len(train) + len(valid) + len(test),
    "Train time steps": train['time'].nunique(),
    "Val time steps": valid['time'].nunique(),
    "Test time steps": test['time'].nunique(),
    "Total time steps": all_data['time'].nunique(),
    "Avg quadruples/time step": round(len(all_data) / all_data['time'].nunique(), 1),
}

for key, val in summary.items():
    print(f"  {key:<30} {val}")

# Dump Table 3 (FinDKG row)
table_3 = pd.DataFrame([{
    'Dataset': 'FinDKG',
    'N_train': len(train),
    'N_val': len(valid),
    'N_test': len(test),
    '|E|': stat['num_entities'].item(),
    '|R|': stat['num_relations'].item(),
    '|C_E|': 12
}])
table_3.to_csv(os.path.join(OUTPUT_DIR, 'Table_3.csv'), index=False)
print("[Saved] Table_3.csv")

print("\n✅ Data exploration complete. All tables and plots saved to outputs/")
