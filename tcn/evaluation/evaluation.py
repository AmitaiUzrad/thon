import math

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
import pandas as pd
from pathlib import Path


def eval_edge_prediction(model, negative_edge_sampler, data, n_neighbors, batch_size=200,
                         dataset_name=None, split_name=None):
  use_precomputed_negatives = (
    hasattr(data, "negative_destinations") and data.negative_destinations is not None and
    len(data.negative_destinations) == len(data.sources)
  )
  use_interaction_level_eval = (
    hasattr(data, "interaction_ids") and data.interaction_ids is not None and
    len(data.interaction_ids) == len(data.sources)
  )
  if not use_precomputed_negatives:
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

  val_ap, val_auc = [], []
  # Stores for runtime dump
  pair_pos_store = np.full(len(data.sources), np.nan, dtype=np.float32)
  pair_neg_store = np.full(len(data.sources), np.nan, dtype=np.float32)
  interaction_ids_all = np.full(len(data.sources), -1, dtype=np.int64) if hasattr(data, "interaction_ids") and data.interaction_ids is not None else None
  interaction_pos_scores = {}
  interaction_neg_scores = {}
  with torch.no_grad():
    model = model.eval()
    # While usually the test batch size is as big as it fits in memory, here we keep it the same
    # size as the training batch size, since it allows the memory to be updated more frequently,
    # and later test batches to access information from interactions in previous test batches
    # through the memory
    TEST_BATCH_SIZE = batch_size
    num_test_instance = len(data.sources)
    num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

    for k in range(num_test_batch):
      s_idx = k * TEST_BATCH_SIZE
      e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)
      sources_batch = data.sources[s_idx:e_idx]
      destinations_batch = data.destinations[s_idx:e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = data.edge_idxs[s_idx: e_idx]

      size = len(sources_batch)
      if use_precomputed_negatives:
        negative_samples = data.negative_destinations[s_idx:e_idx]
        if np.any(negative_samples < 0):
          raise ValueError("Precomputed negative destinations contain invalid values for evaluation.")
      else:
        _, negative_samples = negative_edge_sampler.sample(size)

      pos_prob, neg_prob = model.compute_edge_probabilities(sources_batch, destinations_batch,
                                                            negative_samples, timestamps_batch,
                                                            edge_idxs_batch, n_neighbors)

      pos_prob_np = (pos_prob).detach().cpu().numpy().reshape(-1)
      neg_prob_np = (neg_prob).detach().cpu().numpy().reshape(-1)
      # store for dump
      pair_pos_store[s_idx:e_idx] = pos_prob_np
      pair_neg_store[s_idx:e_idx] = neg_prob_np
      if interaction_ids_all is not None:
        interaction_ids_all[s_idx:e_idx] = data.interaction_ids[s_idx:e_idx]

      if use_interaction_level_eval:
        interaction_ids_batch = data.interaction_ids[s_idx:e_idx]
        for i, interaction_id in enumerate(interaction_ids_batch):
          interaction_id = int(interaction_id)
          interaction_pos_scores.setdefault(interaction_id, []).append(float(pos_prob_np[i]))
          interaction_neg_scores.setdefault(interaction_id, []).append(float(neg_prob_np[i]))
      else:
        pred_score = np.concatenate([pos_prob_np, neg_prob_np])
        true_label = np.concatenate([np.ones(size), np.zeros(size)])
        val_ap.append(average_precision_score(true_label, pred_score))
        val_auc.append(roc_auc_score(true_label, pred_score))

  if use_interaction_level_eval:
    all_pos = []
    all_neg = []
    for interaction_id, pos_scores in interaction_pos_scores.items():
      if interaction_id not in interaction_neg_scores or len(interaction_neg_scores[interaction_id]) == 0:
        continue
      all_pos.append(float(np.mean(pos_scores)))
      all_neg.append(float(np.mean(interaction_neg_scores[interaction_id])))

    pred_score = np.concatenate([np.array(all_pos), np.array(all_neg)])
    true_label = np.concatenate([np.ones(len(all_pos)), np.zeros(len(all_neg))])
    ap = average_precision_score(true_label, pred_score)
    auc = roc_auc_score(true_label, pred_score)
  else:
    ap = np.mean(val_ap)
    auc = np.mean(val_auc)

  # Runtime dump augmentation for this split (streaming per-batch and per-call)
  if dataset_name and split_name and getattr(data, "rows_df", None) is not None:
    dump_dir = Path("runtime_split_dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)
    stream_path = dump_dir / f"ml_{dataset_name}_{split_name}_runtime_stream.csv"
    try:
      # header once per eval call
      with open(stream_path, "w", newline="") as f:
        f.write("interaction_id,u,i,ts,label,idx,split,split_inductive,contains_new_node,is_new_node_val,"
                "is_new_node_test,neg_u,neg_i,pair_pos_prob,pair_neg_prob\n")
      # append rows per batch in order
      TEST_BATCH_SIZE = batch_size
      num_instance = len(data.sources)
      num_batch = math.ceil(num_instance / TEST_BATCH_SIZE)
      for k in range(num_batch):
        s_idx = k * TEST_BATCH_SIZE
        e_idx = min(num_instance, s_idx + TEST_BATCH_SIZE)
        batch_rows = data.rows_df.iloc[s_idx:e_idx].copy().reset_index(drop=True)
        batch_rows["pair_pos_prob"] = pair_pos_store[s_idx:e_idx]
        batch_rows["pair_neg_prob"] = pair_neg_store[s_idx:e_idx]
        cols = ["interaction_id","u","i","ts","label","idx","split","split_inductive",
                "contains_new_node","is_new_node_val","is_new_node_test",
                "neg_u","neg_i","pair_pos_prob","pair_neg_prob"]
        # dtype align for neg_u/neg_i
        try:
          if "u" in batch_rows.columns and "neg_u" in batch_rows.columns:
            if pd.api.types.is_integer_dtype(batch_rows["u"].dtype):
              batch_rows["neg_u"] = pd.to_numeric(batch_rows["neg_u"], errors="coerce").astype("Int64")
            else:
              batch_rows["neg_u"] = batch_rows["neg_u"].astype(str)
          if "i" in batch_rows.columns and "neg_i" in batch_rows.columns:
            if pd.api.types.is_integer_dtype(batch_rows["i"].dtype):
              batch_rows["neg_i"] = pd.to_numeric(batch_rows["neg_i"], errors="coerce").astype("Int64")
            else:
              batch_rows["neg_i"] = batch_rows["neg_i"].astype(str)
        except Exception:
          pass
        batch_rows[cols].to_csv(stream_path, mode="a", index=False, header=False)
    except Exception:
      pass

  return ap, auc


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
  pred_prob = np.zeros(len(data.sources))
  num_instance = len(data.sources)
  num_batch = math.ceil(num_instance / batch_size)

  with torch.no_grad():
    decoder.eval()
    tgn.eval()
    for k in range(num_batch):
      s_idx = k * batch_size
      e_idx = min(num_instance, s_idx + batch_size)

      sources_batch = data.sources[s_idx: e_idx]
      destinations_batch = data.destinations[s_idx: e_idx]
      timestamps_batch = data.timestamps[s_idx:e_idx]
      edge_idxs_batch = edge_idxs[s_idx: e_idx]

      source_embedding, destination_embedding, _ = tgn.compute_temporal_embeddings(sources_batch,
                                                                                   destinations_batch,
                                                                                   destinations_batch,
                                                                                   timestamps_batch,
                                                                                   edge_idxs_batch,
                                                                                   n_neighbors)
      pred_prob_batch = decoder(source_embedding).sigmoid()
      pred_prob[s_idx: e_idx] = pred_prob_batch.cpu().numpy()

  auc_roc = roc_auc_score(data.labels, pred_prob)
  return auc_roc
