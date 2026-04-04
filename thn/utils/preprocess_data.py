"""
THGN Data Preprocessing Module
==============================

This module converts raw temporal hypergraph data (CSV format) into the format expected by THGN.
It handles node reindexing, feature preparation, and file format conversion for hypergraphs.

Key Differences from TGN:
-------------------------
1. Hyperedges instead of edges: Each interaction can involve any number of nodes (≥1)
2. Variable-size inputs: Hyperedge sizes vary (unlike fixed pairs in TGN)
3. Set representation: Hyperedges stored as sets/lists of nodes instead of (u, i) pairs

Key Concepts:
-------------
1. 1-Indexing: THGN reserves node ID 0 for padding/missing nodes. All valid nodes start from 1.
   - Hyperedge features: feat[0] is padding, feat[hyperedge_idx] corresponds to hyperedge with idx=hyperedge_idx
   - Node features: rand_feat[node_id] for node with id=node_id (1-indexed)

2. Hyperedge Representation: Hyperedges are stored as comma-separated node lists
   - Input: "50,60,70" (comma-separated string of node IDs)
   - Output: "1,2,3" (reindexed, 1-indexed, comma-separated string)
   - Allows variable-size hyperedges in CSV format

3. Node Reindexing: All nodes from all hyperedges are collected and reindexed to 1-indexed
   continuous space. Unlike TGN's bipartite handling, hypergraphs typically use a unified
   node space (all nodes treated equally).

4. Feature Padding: Adds dummy feature vector at index 0 for both hyperedge and node features
   to support 1-indexing and handle missing/padding cases.

Functions:
----------
1. preprocess(data_name)
   - Reads raw CSV file with format: nodes, ts, label, [feat1, feat2, ...]
   - Parses each line into components (node list, timestamp, label, optional features)
   - If no features provided, creates 1-dimensional zero features [0.0] for each hyperedge
   - Returns: DataFrame with columns (nodes, ts, label, idx) and hyperedge features array
   - Input: CSV file path
   - Output: (DataFrame, numpy array of hyperedge features)
   - nodes column: Comma-separated string of node IDs (e.g., "50,60,70")

2. reindex(df)
   - Reindexes all node IDs to 1-indexed continuous space
   - Extracts all unique nodes from all hyperedges
   - Shifts all node IDs by +1 (1-indexing)
   - Updates node IDs in all hyperedge node lists
   - Also shifts hyperedge indices (idx) by +1 for consistency
   - Returns: Reindexed DataFrame

3. run(data_name)
   - Main preprocessing pipeline that orchestrates the full conversion
   - Steps:
     a) Load raw data using preprocess()
     b) Reindex nodes using reindex()
     c) Add padding feature at index 0 for hyperedge features
     d) Create zero-filled node features (172 dimensions) if not provided
     e) Save outputs:
        - ml_{data_name}.csv: Reindexed hyperedge list
        - ml_{data_name}.npy: Hyperedge features (with padding)
        - ml_{data_name}_node.npy: Node features (zeros if not provided)

Input Format:
-------------
Raw CSV file: nodes, ts, label, [feat1, feat2, feat3, ...]
  - nodes: Comma-separated string of node IDs (0-indexed), e.g., "50,60,70"
  - ts: timestamp of the hyperedge
  - label: task label
  - feat1, feat2, ...: hyperedge features (OPTIONAL - if missing, 1-dim zero features will be created)

Example input lines:
  With features: "50,60,70", 100, 1, 0.5, 0.3, 0.2
  Without features: "50,60,70", 100, 1

Output Format:
--------------
1. ml_{data_name}.csv: Reindexed hyperedge list
   - Columns: nodes, ts, label, idx
   - nodes: Comma-separated string of reindexed node IDs (1-indexed), e.g., "1,2,3"
   - All node IDs are 1-indexed
   - idx: hyperedge index (1-indexed, corresponds to hyperedge features array)

2. ml_{data_name}.npy: Hyperedge features array
   - Shape: [num_hyperedges + 1, feat_dim]
   - Index 0: padding (zeros)
   - Index i: features for hyperedge with idx=i

3. ml_{data_name}_node.npy: Node features array
   - Shape: [max_node_id + 1, 172]
   - All zeros if dataset doesn't provide node features
   - Index i: features for node with id=i

Example Transformation:
-----------------------
Input CSV:
  nodes, ts, label, feat1, feat2
  "0,1", 100, 1, 0.5, 0.3
  "0,1,2", 200, 1, 0.2, 0.4
  "1,2", 300, 0, 0.1, 0.6

After preprocessing:
  - Nodes: All unique nodes from hyperedges reindexed to [1, 2, 3, ...]
  - Hyperedge features: feat[0] = [0, 0] (padding), feat[1] = [0.5, 0.3], ...
  - Node features: zeros for all nodes (if not provided)
  - Output CSV:
    nodes, ts, label, idx
    "1,2", 100, 1, 0
    "1,2,3", 200, 1, 1
    "2,3", 300, 0, 2
"""

import json
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
  """
  Preprocess raw hypergraph CSV data.
  
  Expected format: nodes, ts, label, feat1, feat2, ...
  where nodes is a comma-separated string of node IDs.
  
  Returns:
    - DataFrame with columns: nodes, ts, label, idx
    - numpy array of hyperedge features
  """
  nodes_list, ts_list, label_list = [], [], []
  feat_l = []
  idx_list = []

  with open(data_name, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    for idx, e in enumerate(reader):
      # First field is nodes (comma-separated string, may be quoted)
      nodes_str = e[0].strip().strip('"').strip("'")
      nodes = [int(x.strip()) for x in nodes_str.split(',')]
      
      # Parse remaining fields
      ts = float(e[1])
      label = float(e[2])
      
      # Features start from index 3 (optional - if missing, create 1-dim zero feature)
      if len(e) > 3:
        feat = np.array([float(x) for x in e[3:]])
      else:
        # No features provided - create 1-dimensional zero feature
        feat = np.array([0.0])
      
      # Store as comma-separated string for CSV compatibility
      nodes_str_stored = ','.join(map(str, nodes))
      
      nodes_list.append(nodes_str_stored)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)
      
      feat_l.append(feat)
  
  return pd.DataFrame({
    'nodes': nodes_list,
    'ts': ts_list,
    'label': label_list,
    'idx': idx_list
  }), np.array(feat_l)


def reindex(df):
  """
  Reindex all node IDs to 1-indexed continuous space.
  
  Extracts all unique nodes from all hyperedges and reindexes them.
  Updates node IDs in all hyperedge node lists.
  
  Args:
    df: DataFrame with 'nodes' column containing comma-separated node ID strings
    
  Returns:
    Reindexed DataFrame with 1-indexed node IDs
  """
  new_df = df.copy()
  
  # Extract all unique nodes from all hyperedges
  all_nodes = set()
  for nodes_str in df['nodes']:
    nodes = [int(x) for x in nodes_str.split(',')]
    all_nodes.update(nodes)
  
  # Create mapping from old node ID to new node ID (1-indexed)
  unique_nodes = sorted(all_nodes)
  node_mapping = {old_id: new_id + 1 for new_id, old_id in enumerate(unique_nodes)}
  
  # Update node IDs in all hyperedges
  new_nodes_list = []
  for nodes_str in df['nodes']:
    nodes = [int(x) for x in nodes_str.split(',')]
    new_nodes = [node_mapping[node_id] for node_id in nodes]
    new_nodes_str = ','.join(map(str, new_nodes))
    new_nodes_list.append(new_nodes_str)
  
  new_df['nodes'] = new_nodes_list
  new_df['idx'] = df['idx'] + 1  # Shift indices by +1 for consistency
  
  return new_df


def run(data_name):
  """
  Main preprocessing pipeline for temporal hypergraph data.
  
  Args:
    data_name: Name of the dataset (without .csv extension)
  """
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df)

  # Add padding feature at index 0
  # Handle case where features might be empty (shouldn't happen after fix above, but safety check)
  if len(feat) == 0 or feat.shape[0] == 0:
    # All hyperedges have no features - create 1-dim zero features for all
    feat = np.array([[0.0]] * len(new_df))
    feat_dim = 1
  else:
    feat_dim = feat.shape[1]
  
  empty = np.zeros(feat_dim)[np.newaxis, :]
  feat = np.vstack([empty, feat])

  # Find maximum node ID across all hyperedges
  max_node_id = 0
  for nodes_str in new_df['nodes']:
    nodes = [int(x) for x in nodes_str.split(',')]
    max_node_id = max(max_node_id, max(nodes))
  
  # Create zero-filled node features
  rand_feat = np.zeros((max_node_id + 1, 172))

  # Save outputs
  # Use QUOTE_NONNUMERIC to ensure nodes column (string) is always quoted
  new_df.to_csv(OUT_DF, index=False, quoting=csv.QUOTE_NONNUMERIC)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)
  
  print(f"Preprocessing complete!")
  print(f"  - Processed {len(new_df)} hyperedges")
  print(f"  - Max node ID: {max_node_id}")
  print(f"  - Feature dimension: {feat.shape[1]}")
  print(f"  - Output files:")
  print(f"    * {OUT_DF}")
  print(f"    * {OUT_FEAT}")
  print(f"    * {OUT_NODE_FEAT}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser('Interface for THGN data preprocessing')
  parser.add_argument('--data', type=str, help='Dataset name (without .csv extension)',
                      default='wikipedia')
  
  args = parser.parse_args()
  run(args.data)
