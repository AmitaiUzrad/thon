"""
TGN Data Preprocessing Module
==============================

This module converts raw temporal graph data (CSV format) into the format expected by TGN.
It handles node reindexing, feature preparation, and file format conversion.

Key Concepts:
-------------
1. 1-Indexing: TGN reserves node ID 0 for padding/missing nodes. All valid nodes start from 1.
   - Edge features: feat[0] is padding, feat[edge_idx] corresponds to edge with idx=edge_idx
   - Node features: rand_feat[node_id] for node with id=node_id (1-indexed)

2. Bipartite Graph Handling: For bipartite graphs (e.g., user-item), source and destination
   nodes are in separate ID spaces. This module merges them into one continuous space.
   - Original: u ∈ [0, N-1], i ∈ [0, M-1] (separate spaces)
   - After: u ∈ [1, N], i ∈ [N+1, N+M] (merged, 1-indexed)
   - Formula: new_i = old_i + max_u + 1, then shift all by +1

3. Feature Padding: Adds dummy feature vector at index 0 for both edge and node features
   to support 1-indexing and handle missing/padding cases.

Functions:
----------
1. preprocess(data_name)
   - Reads raw CSV file with format: u, i, ts, label, feat1, feat2, ...
   - Parses each line into components (source, destination, timestamp, label, features)
   - Returns: DataFrame with columns (u, i, ts, label, idx) and edge features array
   - Input: CSV file path
   - Output: (DataFrame, numpy array of edge features)

2. reindex(df, bipartite=True)
   - Reindexes node IDs to ensure no overlap and 1-indexing
   - For bipartite: Merges separate source/destination ID spaces
     - Asserts that u and i are in contiguous ranges
     - Shifts destination nodes: new_i = old_i + max_u + 1
     - Then shifts all nodes by +1 (1-indexing)
   - For non-bipartite: Simply shifts all node IDs by +1
   - Also shifts edge indices (idx) by +1 for consistency
   - Returns: Reindexed DataFrame

3. run(data_name, bipartite=True)
   - Main preprocessing pipeline that orchestrates the full conversion
   - Steps:
     a) Load raw data using preprocess()
     b) Reindex nodes using reindex()
     c) Add padding feature at index 0 for edge features
     d) Create zero-filled node features (172 dimensions) if not provided
     e) Save outputs:
        - ml_{data_name}.csv: Reindexed edge list
        - ml_{data_name}.npy: Edge features (with padding)
        - ml_{data_name}_node.npy: Node features (zeros if not provided)

Input Format:
-------------
Raw CSV file: u, i, ts, label, feat1, feat2, feat3, ...
  - u: source node ID (0-indexed)
  - i: destination node ID (0-indexed)
  - ts: timestamp of the edge
  - label: task label
  - feat1, feat2, ...: edge features (variable number)

Output Format:
--------------
1. ml_{data_name}.csv: Reindexed edge list
   - Columns: u, i, ts, label, idx
   - All IDs are 1-indexed
   - idx: edge index (1-indexed, corresponds to edge features array)

2. ml_{data_name}.npy: Edge features array
   - Shape: [num_edges + 1, feat_dim]
   - Index 0: padding (zeros)
   - Index i: features for edge with idx=i

3. ml_{data_name}_node.npy: Node features array
   - Shape: [max_node_id + 1, 172]
   - All zeros if dataset doesn't provide node features
   - Index i: features for node with id=i

Example Transformation:
-----------------------
Input CSV:
  u, i, ts, label, feat1, feat2
  0, 0, 100, 1, 0.5, 0.3
  0, 1, 200, 1, 0.2, 0.4
  1, 0, 300, 0, 0.1, 0.6

After preprocessing (bipartite=True):
  - Nodes: u ∈ [1, 2], i ∈ [3, 4] (merged and shifted)
  - Edge features: feat[0] = [0, 0] (padding), feat[1] = [0.5, 0.3], ...
  - Node features: zeros for all nodes (if not provided)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse


def preprocess(data_name):
  u_list, i_list, ts_list, label_list = [], [], [], []
  feat_l = []
  idx_list = []

  with open(data_name) as f:
    s = next(f)
    for idx, line in enumerate(f):
      e = line.strip().split(',')
      u = int(e[0])
      i = int(e[1])

      ts = float(e[2])
      label = float(e[3])  # int(e[3])

      feat = np.array([float(x) for x in e[4:]])

      u_list.append(u)
      i_list.append(i)
      ts_list.append(ts)
      label_list.append(label)
      idx_list.append(idx)

      feat_l.append(feat)
  return pd.DataFrame({'u': u_list,
                       'i': i_list,
                       'ts': ts_list,
                       'label': label_list,
                       'idx': idx_list}), np.array(feat_l)


def reindex(df, bipartite=True):
  new_df = df.copy()
  if bipartite:
    assert (df.u.max() - df.u.min() + 1 == len(df.u.unique()))
    assert (df.i.max() - df.i.min() + 1 == len(df.i.unique()))

    upper_u = df.u.max() + 1
    new_i = df.i + upper_u

    new_df.i = new_i
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1
  else:
    new_df.u += 1
    new_df.i += 1
    new_df.idx += 1

  return new_df


def run(data_name, bipartite=True):
  Path("data/").mkdir(parents=True, exist_ok=True)
  PATH = './data/{}.csv'.format(data_name)
  OUT_DF = './data/ml_{}.csv'.format(data_name)
  OUT_FEAT = './data/ml_{}.npy'.format(data_name)
  OUT_NODE_FEAT = './data/ml_{}_node.npy'.format(data_name)

  df, feat = preprocess(PATH)
  new_df = reindex(df, bipartite)

  empty = np.zeros(feat.shape[1])[np.newaxis, :]
  feat = np.vstack([empty, feat])

  max_idx = max(new_df.u.max(), new_df.i.max())
  rand_feat = np.zeros((max_idx + 1, 172))

  new_df.to_csv(OUT_DF)
  np.save(OUT_FEAT, feat)
  np.save(OUT_NODE_FEAT, rand_feat)

parser = argparse.ArgumentParser('Interface for TGN data preprocessing')
parser.add_argument('--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bipartite', action='store_true', help='Whether the graph is bipartite')

args = parser.parse_args()

run(args.data, bipartite=args.bipartite)