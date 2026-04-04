"""
TGN Data Processing Module
==========================

This module handles loading and preprocessing temporal graph data for TGN (Temporal Graph Networks).
It provides functions for temporal train/val/test splits and inductive evaluation setups.

Key Concepts:
-------------
1. Temporal Splits: Unlike static graphs, temporal graphs are split by TIME, not randomly.
   - Train: edges with timestamps <= 70th percentile
   - Val: edges with timestamps between 70th-85th percentile  
   - Test: edges with timestamps > 85th percentile
   This prevents data leakage (future edges can't influence past predictions).

2. Inductive Evaluation: Tests model's ability to handle nodes it never saw during training.
   - Transductive: Predict on nodes seen in training (simpler)
   - Inductive: Predict on completely new nodes (more challenging, tests generalization)

3. Time Statistics: Computes mean/std of time differences between consecutive interactions.
   Used by TGN to normalize time deltas in the model.

Classes:
--------
- Data: Container class for temporal graph data
  - sources: source node IDs for each edge
  - destinations: destination node IDs for each edge
  - timestamps: when each edge occurred
  - edge_idxs: indices into edge features array
  - labels: task labels (link prediction or node classification)
  - unique_nodes: set of all nodes in the graph

Functions:
----------
1. get_data_node_classification(dataset_name, use_validation=False)
   - Loads data for node classification task
   - Simple temporal split (no inductive evaluation)
   - All test nodes appear in training (transductive setting)
   - Returns: full_data, node_features, edge_features, train_data, val_data, test_data

2. get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False)
   - Loads data for link prediction task
   - Temporal split + inductive evaluation setup
   
   New Node Sampling Process (for inductive evaluation):
   -----------------------------------------------------
   Step 1: Find all nodes that "appear at test time"
          - A node "appears at test time" if it is a source OR destination of at least
            one edge with timestamp > val_time (i.e., participates in test period edges)
          - Note: These nodes might ALSO have appeared in training (not necessarily new)
   
   Step 2: Sample 10% of total unique nodes from test-time nodes
          - Randomly selects 10% of all unique nodes from those appearing at test time
          - These become "new test nodes" for inductive evaluation
          - Example: If dataset has 1000 nodes, samples 100 nodes from test-time nodes
   
   Step 3: Remove all training edges involving sampled nodes
          - All edges where source OR destination is a sampled "new test node" are
            removed from training data
          - This artificially creates an inductive setting: even if a node appeared in
            training, removing its training edges makes it "new" for evaluation
          - Example: Node 50 had edges at [100, 200, 300, 1500]. If sampled as "new",
            edges [100, 200, 300] are removed from training, leaving only test edge [1500]
   
   Step 4: Create separate evaluation sets
          - Regular val/test: All edges in validation/test period
          - New node val/test: Only edges involving at least one "new test node"
          - This allows measuring performance on truly unseen nodes
   
   Why sampling instead of using naturally new nodes?
   - Ensures controlled 10% of nodes for evaluation (might be few truly new nodes)
   - Tests if model can generalize when training history is hidden
   - More challenging evaluation scenario
   
   - Returns: node_features, edge_features, full_data, train_data, val_data, test_data,
             new_node_val_data, new_node_test_data

3. compute_time_statistics(sources, destinations, timestamps)
   - Computes mean and std of time differences between consecutive interactions
   - Separate statistics for source and destination nodes
   - Used by TGN to normalize time deltas: (time_diff - mean) / std
   - Returns: mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst

Data Format:
------------
Input CSV format: (u, i, ts, label, idx)
  - u: source node ID
  - i: destination node ID  
  - ts: timestamp of the edge
  - label: task label
  - idx: index into edge features array

Edge features: numpy array loaded from .npy file
Node features: numpy array loaded from .npy file
"""

import numpy as np
import random
import pandas as pd


class Data:
  def __init__(self, sources, destinations, timestamps, edge_idxs, labels,
               negative_destinations=None, rows_df=None, interaction_ids=None):
    self.sources = sources
    self.destinations = destinations
    self.timestamps = timestamps
    self.edge_idxs = edge_idxs
    self.labels = labels
    self.negative_destinations = negative_destinations
    self.rows_df = rows_df
    self.interaction_ids = interaction_ids
    self.n_interactions = len(sources)
    self.unique_nodes = set(sources) | set(destinations)
    self.n_unique_nodes = len(self.unique_nodes)


def _has_precomputed_inductive_columns(graph_df):
  required_cols = {"split", "split_inductive", "is_new_node_val", "is_new_node_test"}
  return required_cols.issubset(set(graph_df.columns))


def _parse_neg_dst_column(value):
  if pd.isna(value):
    return -1
  s = str(value).strip()
  if s == "":
    return -1
  return int(float(s))


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

  sources = graph_df.u.values
  destinations = graph_df.i.values
  interaction_ids = graph_df.interaction_id.values if "interaction_id" in graph_df.columns else None
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

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels, interaction_ids=interaction_ids)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask],
                    interaction_ids=interaction_ids[train_mask] if interaction_ids is not None else None)

  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask],
                  interaction_ids=interaction_ids[val_mask] if interaction_ids is not None else None)

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask],
                   interaction_ids=interaction_ids[test_mask] if interaction_ids is not None else None)

  return full_data, node_features, edge_features, train_data, val_data, test_data


def get_data(dataset_name, different_new_nodes_between_val_and_test=False, randomize_features=False):
  ### Load data and train val test split
  graph_df = _load_graph_df_from_splits_or_full(dataset_name)
  edge_features = np.load('./data/ml_{}.npy'.format(dataset_name))
  node_features = np.load('./data/ml_{}_node.npy'.format(dataset_name)) 
    
  if randomize_features:
    node_features = np.random.rand(node_features.shape[0], node_features.shape[1])

  sources = graph_df.u.values
  destinations = graph_df.i.values
  interaction_ids = graph_df.interaction_id.values if "interaction_id" in graph_df.columns else None
  neg_destinations = np.array([_parse_neg_dst_column(x) for x in graph_df.neg_i.values]) if "neg_i" in graph_df.columns else None
  edge_idxs = graph_df.idx.values
  labels = graph_df.label.values
  timestamps = graph_df.ts.values

  full_data = Data(sources, destinations, timestamps, edge_idxs, labels, interaction_ids=interaction_ids)

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

    train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                      edge_idxs[train_mask], labels[train_mask],
                      neg_destinations[train_mask] if neg_destinations is not None else None,
                      graph_df.iloc[train_indices].copy(),
                      interaction_ids[train_mask] if interaction_ids is not None else None)
    val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                    edge_idxs[val_mask], labels[val_mask],
                    neg_destinations[val_mask] if neg_destinations is not None else None,
                    graph_df.iloc[val_indices].copy(),
                    interaction_ids[val_mask] if interaction_ids is not None else None)
    test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                     edge_idxs[test_mask], labels[test_mask],
                     neg_destinations[test_mask] if neg_destinations is not None else None,
                     graph_df.iloc[test_indices].copy(),
                     interaction_ids[test_mask] if interaction_ids is not None else None)
    new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                             timestamps[new_node_val_mask],
                             edge_idxs[new_node_val_mask], labels[new_node_val_mask],
                             neg_destinations[new_node_val_mask] if neg_destinations is not None else None,
                             graph_df.iloc[new_val_indices].copy(),
                             interaction_ids[new_node_val_mask] if interaction_ids is not None else None)
    new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                              timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                              labels[new_node_test_mask],
                              neg_destinations[new_node_test_mask] if neg_destinations is not None else None,
                              graph_df.iloc[new_test_indices].copy(),
                              interaction_ids[new_node_test_mask] if interaction_ids is not None else None)

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

  node_set = set(sources) | set(destinations)
  n_total_unique_nodes = len(node_set)

  # Compute nodes which appear at test time
  test_node_set = set(sources[timestamps > val_time]).union(
    set(destinations[timestamps > val_time]))
  # Sample nodes which we keep as new nodes (to test inductiveness), so than we have to remove all
  # their edges from training
  new_test_node_set = set(random.sample(test_node_set, int(0.1 * n_total_unique_nodes)))

  # Mask saying for each source and destination whether they are new test nodes
  new_test_source_mask = graph_df.u.map(lambda x: x in new_test_node_set).values
  new_test_destination_mask = graph_df.i.map(lambda x: x in new_test_node_set).values

  # Mask which is true for edges with both destination and source not being new test nodes (because
  # we want to remove all edges involving any new test node)
  observed_edges_mask = np.logical_and(~new_test_source_mask, ~new_test_destination_mask)

  # For train we keep edges happening before the validation time which do not involve any new node
  # used for inductiveness
  train_mask = np.logical_and(timestamps <= val_time, observed_edges_mask)

  train_data = Data(sources[train_mask], destinations[train_mask], timestamps[train_mask],
                    edge_idxs[train_mask], labels[train_mask],
                    interaction_ids=interaction_ids[train_mask] if interaction_ids is not None else None)

  # define the new nodes sets for testing inductiveness of the model
  train_node_set = set(train_data.sources).union(train_data.destinations)
  assert len(train_node_set & new_test_node_set) == 0
  new_node_set = node_set - train_node_set

  val_mask = np.logical_and(timestamps <= test_time, timestamps > val_time)
  test_mask = timestamps > test_time

  if different_new_nodes_between_val_and_test:
    n_new_nodes = len(new_test_node_set) // 2
    val_new_node_set = set(list(new_test_node_set)[:n_new_nodes])
    test_new_node_set = set(list(new_test_node_set)[n_new_nodes:])

    edge_contains_new_val_node_mask = np.array(
      [(a in val_new_node_set or b in val_new_node_set) for a, b in zip(sources, destinations)])
    edge_contains_new_test_node_mask = np.array(
      [(a in test_new_node_set or b in test_new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_val_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_test_node_mask)


  else:
    edge_contains_new_node_mask = np.array(
      [(a in new_node_set or b in new_node_set) for a, b in zip(sources, destinations)])
    new_node_val_mask = np.logical_and(val_mask, edge_contains_new_node_mask)
    new_node_test_mask = np.logical_and(test_mask, edge_contains_new_node_mask)

  # validation and test with all edges
  val_data = Data(sources[val_mask], destinations[val_mask], timestamps[val_mask],
                  edge_idxs[val_mask], labels[val_mask],
                  interaction_ids=interaction_ids[val_mask] if interaction_ids is not None else None)

  test_data = Data(sources[test_mask], destinations[test_mask], timestamps[test_mask],
                   edge_idxs[test_mask], labels[test_mask],
                   interaction_ids=interaction_ids[test_mask] if interaction_ids is not None else None)

  # validation and test with edges that at least has one new node (not in training set)
  new_node_val_data = Data(sources[new_node_val_mask], destinations[new_node_val_mask],
                           timestamps[new_node_val_mask],
                           edge_idxs[new_node_val_mask], labels[new_node_val_mask],
                           interaction_ids=interaction_ids[new_node_val_mask] if interaction_ids is not None else None)

  new_node_test_data = Data(sources[new_node_test_mask], destinations[new_node_test_mask],
                            timestamps[new_node_test_mask], edge_idxs[new_node_test_mask],
                            labels[new_node_test_mask],
                            interaction_ids=interaction_ids[new_node_test_mask] if interaction_ids is not None else None)

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


def compute_time_statistics(sources, destinations, timestamps):
  last_timestamp_sources = dict()
  last_timestamp_dst = dict()
  all_timediffs_src = []
  all_timediffs_dst = []
  for k in range(len(sources)):
    source_id = sources[k]
    dest_id = destinations[k]
    c_timestamp = timestamps[k]
    if source_id not in last_timestamp_sources.keys():
      last_timestamp_sources[source_id] = 0
    if dest_id not in last_timestamp_dst.keys():
      last_timestamp_dst[dest_id] = 0
    all_timediffs_src.append(c_timestamp - last_timestamp_sources[source_id])
    all_timediffs_dst.append(c_timestamp - last_timestamp_dst[dest_id])
    last_timestamp_sources[source_id] = c_timestamp
    last_timestamp_dst[dest_id] = c_timestamp
  assert len(all_timediffs_src) == len(sources)
  assert len(all_timediffs_dst) == len(sources)
  mean_time_shift_src = np.mean(all_timediffs_src)
  std_time_shift_src = np.std(all_timediffs_src)
  mean_time_shift_dst = np.mean(all_timediffs_dst)
  std_time_shift_dst = np.std(all_timediffs_dst)

  return mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst
