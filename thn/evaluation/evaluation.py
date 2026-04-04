"""
THGN Evaluation Module
======================

This module provides evaluation functions for THGN (Temporal Hypergraph Network) model.
It includes functions for hyperedge prediction evaluation and node classification evaluation.

Functions:
----------
1. eval_hyperedge_prediction: Evaluates hyperedge prediction performance using AP and AUC-ROC
2. eval_node_classification: Evaluates node classification performance using AUC-ROC

Key Changes from TGN:
---------------------
1. **Data structure**: Uses `data.hyperedges` (list of lists) instead of `data.sources`/`data.destinations`
2. **Negative sampling**: `negative_hyperedge_sampler.sample(hyperedge_sizes)` takes sizes and returns list of lists
3. **Model method**: Calls `compute_hyperedge_probabilities` instead of `compute_edge_probabilities`
4. **Variable-size handling**: Handles hyperedges of different sizes in batches
"""

import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from pathlib import Path


def eval_hyperedge_prediction(model, negative_hyperedge_sampler, data, n_neighbors, batch_size=200,
                              dataset_name=None, split_name=None):
  """
  Evaluate hyperedge prediction performance using Average Precision (AP) and AUC-ROC.
  
  For each test hyperedge, samples a negative hyperedge of the same size and computes
  probabilities for both. Compares predictions against ground truth (positive=1, negative=0).
  
  Key Changes from TGN:
  ---------------------
  1. Uses `data.hyperedges` (list of lists) instead of `data.sources`/`data.destinations`
  2. Computes hyperedge sizes and passes to sampler: `sampler.sample(hyperedge_sizes)`
  3. Calls `model.compute_hyperedge_probabilities` instead of `compute_edge_probabilities`
  
  Parameters:
  -----------
  model: THGN
    The THGN model to evaluate
  negative_hyperedge_sampler: RandHyperedgeSampler
    Sampler for negative hyperedges (must have seed set for reproducibility)
  data: Data
    Test data object with hyperedges, timestamps, edge_idxs, labels
  n_neighbors: int
    Number of hyperedges to sample per node for temporal attention
  batch_size: int
    Batch size for evaluation (default 200)
  
  Returns:
  --------
  mean_ap: float
    Mean Average Precision across all batches
  mean_auc: float
    Mean AUC-ROC across all batches
  """
  use_precomputed_negatives = (
    hasattr(data, "negative_hyperedges") and data.negative_hyperedges is not None and
    len(data.negative_hyperedges) == len(data.hyperedges)
  )

  # Fallback to sampler-based negatives when precomputed negatives are unavailable.
  if not use_precomputed_negatives:
    assert negative_hyperedge_sampler.seed is not None
    negative_hyperedge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  # For runtime dumps
  pos_store = np.full(len(data.hyperedges), np.nan, dtype=np.float32)
  neg_store = np.full(len(data.hyperedges), np.nan, dtype=np.float32)
  neg_nodes_store = np.empty(len(data.hyperedges), dtype=object)

  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.hyperedges)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      hyperedges_batch = data.hyperedges[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(hyperedges_batch)
      if use_precomputed_negatives:
        negative_hyperedges = data.negative_hyperedges[s_idx:e_idx]
      else:
        hyperedge_sizes = [len(h) for h in hyperedges_batch]
        negative_hyperedges = negative_hyperedge_sampler.sample(hyperedge_sizes)

      # Capture the exact negatives used for this batch (string-joined for CSV)
      neg_nodes_store[s_idx:e_idx] = [",".join(map(str, h)) for h in negative_hyperedges]

      pos_prob, neg_prob = model.compute_hyperedge_probabilities(hyperedges_batch, 
                                                                  negative_hyperedges,
                                                                  timestamps_batch,
                                                                  edge_idxs_batch, 
                                                                  n_neighbors)

      pos_np = (pos_prob).detach().cpu().numpy().reshape(-1)
      neg_np = (neg_prob).detach().cpu().numpy().reshape(-1)
      # store for runtime dump
      pos_store[s_idx:e_idx] = pos_np
      neg_store[s_idx:e_idx] = neg_np

      pred_score = np.concatenate([pos_np, neg_np])
      true_label = np.concatenate([np.ones(size), np.zeros(size)])

      val_ap.append(average_precision_score(true_label, pred_score))
      val_auc.append(roc_auc_score(true_label, pred_score))

  # Runtime dump augmentation for this split (streaming per-batch and per-call)
  if dataset_name and split_name and getattr(data, "rows_df", None) is not None:
    dump_dir = Path("runtime_split_dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)
    stream_path = dump_dir / f"ml_{dataset_name}_{split_name}_runtime_stream.csv"
    # write header once per eval call
    try:
      with open(stream_path, "w", newline="") as f:
        f.write("interaction_id,nodes,ts,label,idx,split,split_inductive,contains_new_node,"
                "is_new_node_val,is_new_node_test,pos_prob,neg_prob,neg_nodes\n")
      # append all rows from this eval call in batch order
      # replay batches to preserve order
      TEST_BATCH_SIZE = batch_size
      num_instance = len(data.hyperedges)
      num_batch = math.ceil(num_instance / TEST_BATCH_SIZE)
      for k in range(num_batch):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_instance, s_idx + TEST_BATCH_SIZE)
        batch_rows = data.rows_df.iloc[s_idx:e_idx].copy().reset_index(drop=True)
        batch_rows["pos_prob"] = pos_store[s_idx:e_idx]
        batch_rows["neg_prob"] = neg_store[s_idx:e_idx]
        batch_rows["neg_nodes"] = neg_nodes_store[s_idx:e_idx]
        cols = ["interaction_id","nodes","ts","label","idx","split","split_inductive",
                "contains_new_node","is_new_node_val","is_new_node_test","pos_prob","neg_prob","neg_nodes"]
        batch_rows[cols].to_csv(stream_path, mode="a", index=False, header=False)
    except Exception:
      pass

  return np.mean(val_ap), np.mean(val_auc)


def eval_node_classification(thgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  """
  Evaluate node classification performance using AUC-ROC.
  
  For each hyperedge in the test set, extracts node embeddings and uses the decoder
  to predict node labels. Currently uses the first node in each hyperedge for classification.
  
  Key Changes from TGN:
  ---------------------
  1. Uses `data.hyperedges` (list of lists) instead of `data.sources`/`data.destinations`
  2. Extracts first node from each hyperedge for classification
  3. Calls `thgn.compute_temporal_embeddings` which returns (node_embeddings, mappings, ...)
  4. Extracts embeddings for the first node of each hyperedge using the mappings
  
  Note: This implementation uses the first node in each hyperedge. Alternative approaches
  could aggregate embeddings from all nodes in each hyperedge, or classify all nodes.
  
  Parameters:
  -----------
  thgn: THGN
    The THGN model to use for computing embeddings
  decoder: torch.nn.Module
    Node classifier (MLP) that takes node embeddings and outputs class probabilities
  data: Data
    Test data object with hyperedges, timestamps, edge_idxs, labels
  edge_idxs: np.ndarray
    Edge indices for the test set
  batch_size: int
    Batch size for evaluation
  n_neighbors: int
    Number of hyperedges to sample per node for temporal attention
  
  Returns:
  --------
  auc_roc: float
    AUC-ROC score for node classification
  """
  pred_prob = np.zeros(len(data.hyperedges))
  num_instance = len(data.hyperedges)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    thgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      hyperedges_batch = data.hyperedges[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      # For node classification, we need to extract embeddings for specific nodes
      # We'll use the first node in each hyperedge (could be adapted to use all nodes)
      # Create dummy negative hyperedges (same as positive for this task)
      negative_hyperedges = [[h[0]] for h in hyperedges_batch]  # Single-node "negatives" (not used for classification)
      
      # Compute embeddings for all nodes in hyperedges
      node_embeddings, hyperedge_to_node_indices, _ = thgn.compute_temporal_embeddings(
        hyperedges_batch,
        negative_hyperedges,
        timestamps_batch,
        edge_idxs_batch,
        n_neighbors)
      
      # Extract embeddings for the first node of each hyperedge
      first_node_embeddings = []
      for i, hyperedge in enumerate(hyperedges_batch):
        if len(hyperedge) > 0:
          node_indices = hyperedge_to_node_indices[i]
          first_node_embedding = node_embeddings[node_indices[0]]  # First node in hyperedge
          first_node_embeddings.append(first_node_embedding)
        else:
          # Edge case: empty hyperedge (shouldn't happen, but handle gracefully)
          first_node_embeddings.append(torch.zeros(node_embeddings.shape[1], device=node_embeddings.device))
      
      first_node_embeddings = torch.stack(first_node_embeddings)  # [batch_size, embed_dim]
      
      pred_prob_batch = decoder(first_node_embeddings).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
