"""
THGN Data Processing Module
===========================

This module handles loading and preprocessing temporal hypergraph data for THGN (Temporal Hypergraph Networks).
It provides functions for temporal train/val/test splits and inductive evaluation setups.

Key Differences from TGN:
-------------------------
1. Hyperedges instead of edges: Each interaction involves variable-size sets of nodes (≥1 node)
2. No source/destination distinction: All nodes in a hyperedge are treated equally
3. Unified time statistics: Single per-node time tracking (no separate source/destination stats)

Key Concepts:
-------------
1. Temporal Splits: Unlike static graphs, temporal hypergraphs are split by TIME, not randomly.
   - Train: hyperedges with timestamps <= 70th percentile
   - Val: hyperedges with timestamps between 70th-85th percentile  
   - Test: hyperedges with timestamps > 85th percentile
   This prevents data leakage (future hyperedges can't influence past predictions).

2. Inductive Evaluation: Tests model's ability to handle nodes it never saw during training.
   - Transductive: Predict on nodes seen in training (simpler)
   - Inductive: Predict on completely new nodes (more challenging, tests generalization)

3. Time Statistics: Computes mean/std of time differences between consecutive interactions.
   Used by THGN to normalize time deltas in the model. Uses unified per-node tracking.

Classes:
--------
- Data: Container class for temporal hypergraph data
  - hyperedges: list of lists/sets of node IDs for each hyperedge (variable size)
  - timestamps: when each hyperedge occurred
  - edge_idxs: indices into hyperedge features array
  - labels: task labels (hyperedge prediction or node classification)
  - unique_nodes: set of all nodes in the hypergraph

Functions:
----------
1. get_data_node_classification(dataset_name, use_validation=False)
   - Loads data for node classification task
   - Simple temporal split (no inductive evaluation)
   - All test nodes appear in training (transductive setting)
   - Returns: full_data, node_features, edge_features, train_data, val_data, test_data

2. get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False)
   - Loads data for hyperedge prediction task
   - Temporal split + inductive evaluation setup
   
   New Node Sampling Process (for inductive evaluation):
   -----------------------------------------------------
   Step 1: Find all nodes that "appear at test time"
          - A node "appears at test time" if it appears in at least one hyperedge
            with timestamp > val_time (i.e., participates in test period hyperedges)
          - Note: These nodes might ALSO have appeared in training (not necessarily new)
   
   Step 2: Sample 10% of total unique nodes from test-time nodes
          - Randomly selects 10% of all unique nodes from those appearing at test time
          - These become "new test nodes" for inductive evaluation
          - Example: If dataset has 1000 nodes, samples 100 nodes from test-time nodes
   
   Step 3: Remove all training hyperedges involving sampled nodes
          - All hyperedges where ANY node is a sampled "new test node" are
            removed from training data
          - This artificially creates an inductive setting: even if a node appeared in
            training, removing its training hyperedges makes it "new" for evaluation
          - Example: Node 50 appeared in hyperedges at [100, 200, 300, 1500]. If sampled as "new",
            hyperedges [100, 200, 300] are removed from training, leaving only test hyperedge [1500]
   
   Step 4: Create separate evaluation sets
          - Regular val/test: All hyperedges in validation/test period
          - New node val/test: Only hyperedges involving at least one "new test node"
          - This allows measuring performance on truly unseen nodes
   
   Why sampling instead of using naturally new nodes?
   - Ensures controlled 10% of nodes for evaluation (might be few truly new nodes)
   - Tests if model can generalize when training history is hidden
   - More challenging evaluation scenario
   
   - Returns: node_features, edge_features, full_data, train_data, val_data, test_data,
             new_node_val_data, new_node_test_data

3. compute_time_statistics(hyperedges, timestamps)
   - Computes mean and std of time differences between consecutive interactions
   - Unified statistics: tracks last timestamp per node (no source/destination distinction)
   - Used by THGN to normalize time deltas: (time_diff - mean) / std
   - Returns: mean_time_shift, std_time_shift

Data Format:
------------
Input CSV format: (nodes, ts, label, idx)
  - nodes: Comma-separated string of node IDs, e.g., "1,2,3" or "50"
  - ts: timestamp of the hyperedge
  - label: task label
  - idx: index into hyperedge features array

Edge features: numpy array loaded from .npy file
Node features: numpy array loaded from .npy file
"""

import numpy as np
import random
import pandas as pd
import csv


class Data:
  def __init__(self, hyperedges, timestamps, edge_idxs, labels, negative_hyperedges=None, rows_df=None):
    self.hyperedges = hyperedges  # List of lists/sets of node IDs
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.negative_hyperedges = negative_hyperedges
    self.rows_df = rows_df
    self.n_interactions = len(hyperedges)
    # Extract all unique nodes from all hyperedges
    self.unique_nodes = set(node for hyperedge in hyperedges for node in hyperedge)
    self.n_unique_nodes = len(self.unique_nodes)


def _parse_nodes_column(nodes_str):
  """Parse comma-separated string of node IDs into list of integers."""
  return [int(x.strip()) for x in nodes_str.split(',')]


def _hyperedge_contains_any_node(hyperedge, node_set):
  """Check if hyperedge contains any node from node_set."""
  return any(node in node_set for node in hyperedge)


def _has_precomputed_inductive_columns(graph_df):
  required_cols = {"split", "split_inductive", "is_new_node_val", "is_new_node_test"}
  return required_cols.issubset(set(graph_df.columns))


def _parse_neg_nodes_column(neg_nodes_str):
  if pd.isna(neg_nodes_str):
    return []
  value = str(neg_nodes_str).strip()
  if value == "":
    return []
  return [int(x.strip()) for x in value.split(',') if x.strip()]


def _load_graph_df_from_splits_or_full(dataset_name):
  base = "./data/ml_{}.csv".format(dataset_name)
  train_p = "./data/ml_{}_train.csv".format(dataset_name)
  val_p = "./data/ml_{}_val.csv".format(dataset_name)
  test_p = "./data/ml_{}_test.csv".format(dataset_name)
  try:
    train_df = pd.read_csv(train_p)
    val_df = pd.read_csv(val_p)
    test_df = pd.read_csv(test_p)
    return pd.concat([train_df, val_df, test_df], ignore_index=True)
  except FileNotFoundError:
    return pd.read_csv(base)


def get_data_node_classification(dataset_name, use_validation=False):
  ### Load data and train val test split
  graph_df = _load_graph_df_from_splits_or_full(dataset_name)
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name))

  # Parse nodes column (comma-separated strings) into lists of node IDs
  hyperedges = [_parse_nodes_column(nodes_str) for nodes_str in graph_df.nodes.values]
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  random.seed(2020)

  if "split" in graph_df.columns:
    split_values = graph_df["split"].values
    train_mask = (split_values == "train") if use_validation else np.logical_or(split_values == "train", split_values == "val")
    test_mask = split_values == "test"
    val_mask = (split_values == "val") if use_validation else test_mask
  else:
    val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))
    train_mask = timestamps <= val_time if use_validation else timestamps <= test_time
    test_mask = timestamps > test_time
    val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time) if use_validation else test_mask

  full_data = Data(hyperedges, timestamps, edge_idxs, labels)

  train_data = Data([hyperedges[i] for i in np.where(train_mask)[0]],
                    timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  val_data = Data([hyperedges[i] for i in np.where(val_mask)[0]],
                  timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data([hyperedges[i] for i in np.where(test_mask)[0]],
                   timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = _load_graph_df_from_splits_or_full(dataset_name)
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 
    
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  # Parse nodes column (comma-separated strings) into lists of node IDs
  hyperedges = [_parse_nodes_column(nodes_str) for nodes_str in graph_df.nodes.values]
  negative_hyperedges = [_parse_neg_nodes_column(x) for x in graph_df.neg_nodes.values] if "neg_nodes" in graph_df.columns else None
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(hyperedges, timestamps, edge_idxs, labels)

  if _has_precomputed_inductive_columns(graph_df):
    split_inductive = graph_df["split_inductive"].values
    split_temporal = graph_df["split"].values
    new_node_val_mask = graph_df["is_new_node_val"].astype(int).values == 1
    new_node_test_mask = graph_df["is_new_node_test"].astype(int).values == 1

    train_mask = split_inductive == "train"
    val_mask = split_temporal == "val"
    test_mask = split_temporal == "test"

    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    new_val_indices = np.where(new_node_val_mask)[0]
    new_test_indices = np.where(new_node_test_mask)[0]

    train_data = Data([hyperedges[i] for i in train_indices],
                      timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask],
                      [negative_hyperedges[i] for i in train_indices] if negative_hyperedges is not None else None,
                      graph_df.iloc[train_indices].copy())
    val_data = Data([hyperedges[i] for i in val_indices],
                    timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask],
                    [negative_hyperedges[i] for i in val_indices] if negative_hyperedges is not None else None,
                    graph_df.iloc[val_indices].copy())
    test_data = Data([hyperedges[i] for i in test_indices],
                     timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask],
                     [negative_hyperedges[i] for i in test_indices] if negative_hyperedges is not None else None,
                     graph_df.iloc[test_indices].copy())
    new_node_val_data = Data([hyperedges[i] for i in new_val_indices],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask],
                             [negative_hyperedges[i] for i in new_val_indices] if negative_hyperedges is not None else None,
                             graph_df.iloc[new_val_indices].copy())
    new_node_test_data = Data([hyperedges[i] for i in new_test_indices],
                              timestamps[new_node_test_mask],
                              edge_idxs[new_node_test_mask], labels[new_node_test_mask],
                              [negative_hyperedges[i] for i in new_test_indices] if negative_hyperedges is not None else None,
                              graph_df.iloc[new_test_indices].copy())

    print("Using precomputed split/new-node columns from preprocessing.")
    print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                        full_data.n_unique_nodes))
    print("The training dataset has {} interactions, involving {} different nodes".format(
      train_data.n_interactions, train_data.n_unique_nodes))
    print("The validation dataset has {} interactions, involving {} different nodes".format(
      val_data.n_interactions, val_data.n_unique_nodes))
    print("The test dataset has {} interactions, involving {} different nodes".format(
      test_data.n_interactions, test_data.n_unique_nodes))
    print("The new node validation dataset has {} interactions, involving {} different nodes".format(
      new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
    print("The new node test dataset has {} interactions, involving {} different nodes".format(
      new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))

    return node_features, edge_features, full_data, train_data, val_data, test_data, \
           new_node_val_data, new_node_test_data

  val_time, test_time = list(np.quantile(graph_df.ts, [0.70, 0.85]))

  random.seed(2020)

  # Extract all unique nodes from all hyperedges
  node_set = set(node for hyperedge in hyperedges for node in hyperedge)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time (in hyperedges with timestamp > val_time)
  test_node_set = set()
  for i, hyperedge in enumerate(hyperedges):
    if timestamps[i] > val_time:
      test_node_set.update(hyperedge)
  
  # Sample nodes which we keep as new nodes (to test inductiveness), so that we have to remove all
  # their hyperedges from training
  n_new_nodes_to_sample = min(int(0.1 * n_total_unique_nodes), len(test_node_set))
  if n_new_nodes_to_sample > 0:
    new_test_node_set = set(random.sample(list(test_node_set), n_new_nodes_to_sample))
  else:
    new_test_node_set = set()

  # Mask which is true for hyperedges that do NOT contain any new test node
  # (because we want to remove all hyperedges involving any new test node)
  observed_hyperedges_mask = np.array([
    not _hyperedge_contains_any_node(hyperedge, new_test_node_set)
    for hyperedge in hyperedges
  ])

  # For train we keep hyperedges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_hyperedges_mask)

  train_data = Data([hyperedges[i] for i in np.where(train_mask)[0]],
                    timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask])

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(node for hyperedge in train_data.hyperedges for node in hyperedge)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array([
      _hyperedge_contains_any_node(hyperedge, val_new_node_set)
      for hyperedge in hyperedges
    ])
    edge_contains_new_test_node_mask = np.array([
      _hyperedge_contains_any_node(hyperedge, test_new_node_set)
      for hyperedge in hyperedges
    ])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)

  else:
    edge_contains_new_node_mask = np.array([
      _hyperedge_contains_any_node(hyperedge, new_node_set)
      for hyperedge in hyperedges
    ])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # validation and test with all hyperedges
  val_data = Data([hyperedges[i] for i in np.where(val_mask)[0]],
                  timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask])

  test_data = Data([hyperedges[i] for i in np.where(test_mask)[0]],
                   timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask])

  # validation and test with hyperedges that have at least one new node (not in training set)
  new_node_val_data = Data([hyperedges[i] for i in np.where(new_node_val_mask)[0]],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask])

  new_node_test_data = Data([hyperedges[i] for i in np.where(new_node_test_mask)[0]],
                            timestamps[new_node_test_mask],
                            edge_idxs[new_node_test_mask], labels[new_node_test_mask])

  print("The dataset has {} interactions, involving {} different nodes".format(full_data.n_interactions,
                                                                      full_data.n_unique_nodes))
  print("The training dataset has {} interactions, involving {} different nodes".format(
    train_data.n_interactions, train_data.n_unique_nodes))
  print("The validation dataset has {} interactions, involving {} different nodes".format(
    val_data.n_interactions, val_data.n_unique_nodes))
  print("The test dataset has {} interactions, involving {} different nodes".format(
    test_data.n_interactions, test_data.n_unique_nodes))
  print("The new node validation dataset has {} interactions, involving {} different nodes".format(
    new_node_val_data.n_interactions, new_node_val_data.n_unique_nodes))
  print("The new node test dataset has {} interactions, involving {} different nodes".format(
    new_node_test_data.n_interactions, new_node_test_data.n_unique_nodes))
  print("{} nodes were used for the inductive testing, i.e. are never seen during training".format(
    len(new_test_node_set)))

  return node_features, edge_features, full_data, train_data, val_data, test_data, \
         new_node_val_data, new_node_test_data


def compute_time_statistics(hyperedges, timestamps):
  """
  Compute unified time statistics for all nodes (no source/destination distinction).
  
  For each hyperedge, updates the last timestamp for all nodes in that hyperedge.
  Returns mean and std of time differences across all node interactions.
  """
  last_timestamp_nodes = dict()
  all_timediffs = []
  
  for k in range(len(hyperedges)):
    hyperedge = hyperedges[k]
    c_timestamp = timestamps[k]
    
    # For each node in the hyperedge, compute time difference from its last interaction
    for node_id in hyperedge:
      if node_id not in last_timestamp_nodes.keys():
        last_timestamp_nodes[node_id] = 0
      all_timediffs.append(c_timestamp - last_timestamp_nodes[node_id])
      last_timestamp_nodes[node_id] = c_timestamp
  
  assert len(all_timediffs) == sum(len(hyperedge) for hyperedge in hyperedges)
  
  mean_time_shift = np.mean(all_timediffs)
  std_time_shift = np.std(all_timediffs)

  return mean_time_shift, std_time_shift
