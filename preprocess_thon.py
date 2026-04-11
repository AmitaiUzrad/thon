#!/usr/bin/env python3
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd


def parse_hyperedge_nodes(nodes_str: str) -> List[int]:
  return [int(x.strip()) for x in str(nodes_str).strip().strip('"').strip("'").split(",") if x.strip()]


def load_raw_hypergraph_csv(csv_path: Path) -> Tuple[pd.DataFrame, np.ndarray]:
  nodes_list, ts_list, label_list, feat_list = [], [], [], []
  with open(csv_path, "r", newline="") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
      nodes_list.append(parse_hyperedge_nodes(row[0]))
      ts_list.append(float(row[1]))
      label_list.append(float(row[2]))
      if len(row) > 3:
        feat_list.append(np.array([float(x) for x in row[3:]], dtype=float))
      else:
        feat_list.append(np.array([0.0], dtype=float))
  return pd.DataFrame({"nodes_raw": nodes_list, "ts": ts_list, "label": label_list}), np.array(feat_list, dtype=float)


def reindex_nodes_contiguous_1_based(all_node_lists: List[List[int]]) -> Dict[int, int]:
  unique_nodes = sorted({n for nodes in all_node_lists for n in nodes})
  return {old: new + 1 for new, old in enumerate(unique_nodes)}


def assign_temporal_splits(timestamps: np.ndarray) -> np.ndarray:
  val_time, test_time = np.quantile(timestamps, [0.70, 0.85])
  return np.where(timestamps <= val_time, "train", np.where(timestamps <= test_time, "val", "test")).astype(object)


def _format_distribution(size_counts: Dict[int, int]) -> List[str]:
  return [f"size {size}: {count}" for size, count in sorted(size_counts.items())]


def write_dataset_stats(raw_df: pd.DataFrame, stats_out_path: Path) -> None:
  all_sizes = np.array([len(nodes) for nodes in raw_df["nodes_raw"].tolist()], dtype=float)
  non_singleton_sizes = all_sizes[all_sizes >= 2]
  raw_distribution: Dict[int, int] = {}
  for size in all_sizes.astype(int).tolist():
    raw_distribution[size] = raw_distribution.get(size, 0) + 1

  lines = [
    "HOTN dataset statistics",
    "",
    f"- unique_nodes: {len({n for nodes in raw_df['nodes_raw'].tolist() for n in nodes})}",
    f"- interactions_total: {len(all_sizes)}",
    f"- interactions_non_singleton: {len(non_singleton_sizes)}",
    "",
    "Distribution by interaction size (all interactions):",
  ]
  lines.extend(_format_distribution(raw_distribution) if raw_distribution else ["none"])
  lines.append("")
  stats_out_path.parent.mkdir(parents=True, exist_ok=True)
  stats_out_path.write_text("\n".join(lines), encoding="utf-8")


def compute_inductive_labels(
  nodes_list: List[List[int]],
  timestamps: np.ndarray,
  splits: np.ndarray,
  new_node_ratio: float,
  seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Set[int], Dict[str, float]]:
  rng = np.random.RandomState(seed)
  node_set = set(node for nodes in nodes_list for node in nodes)
  n_total_unique_nodes = len(node_set)
  val_time, test_time = np.quantile(timestamps, [0.70, 0.85])

  test_time_nodes = set()
  for i, nodes in enumerate(nodes_list):
    if timestamps[i] > val_time:
      test_time_nodes.update(nodes)

  n_sample = min(int(new_node_ratio * n_total_unique_nodes), len(test_time_nodes))
  new_test_node_set = set(int(x) for x in rng.choice(sorted(list(test_time_nodes)), size=n_sample, replace=False).tolist()) if n_sample > 0 else set()
  contains_new = np.array([1 if any(n in new_test_node_set for n in nodes) else 0 for nodes in nodes_list], dtype=int)

  split_inductive = splits.copy().astype(object)
  split_inductive[np.logical_and(splits == "train", contains_new == 1)] = "drop"
  is_new_node_val = np.logical_and(splits == "val", contains_new == 1).astype(int)
  is_new_node_test = np.logical_and(splits == "test", contains_new == 1).astype(int)

  meta = {
    "val_time": float(val_time),
    "test_time": float(test_time),
    "new_node_ratio": float(new_node_ratio),
    "inductive_seed": int(seed),
    "n_total_unique_nodes": int(n_total_unique_nodes),
    "n_new_test_nodes": int(len(new_test_node_set)),
  }
  return split_inductive, contains_new, is_new_node_val, is_new_node_test, new_test_node_set, meta


def sample_negative_hyperedge(
  pos_nodes: List[int],
  all_nodes: Set[int],
  overlap_ratio: float,
  rng: np.random.RandomState,
) -> List[int]:
  k = len(pos_nodes)
  keep_n = int(np.floor(overlap_ratio * k))
  keep_n = min(max(0, keep_n), k)
  replace_n = k - keep_n
  pos_set = set(pos_nodes)

  if keep_n > 0:
    kept_nodes = rng.choice(sorted(pos_nodes), size=keep_n, replace=False).tolist()
  else:
    kept_nodes = []
  kept_set = set(kept_nodes)

  if replace_n > 0:
    pool = sorted(list(all_nodes - kept_set - pos_set))
    if len(pool) < replace_n:
      pool = sorted(list(all_nodes - kept_set))
    repl_nodes = rng.choice(pool, size=replace_n, replace=False).tolist()
  else:
    repl_nodes = []

  neg_nodes = kept_nodes + repl_nodes
  if set(neg_nodes) == pos_set and replace_n > 0:
    # Force at least one changed node if possible.
    candidate_pool = list(all_nodes - set(neg_nodes))
    if candidate_pool:
      neg_nodes[-1] = int(rng.choice(candidate_pool))
  return [int(x) for x in neg_nodes]


def build_thgn_dataframe(
  raw_df: pd.DataFrame,
  hyperedge_feat: np.ndarray,
  node_mapping: Dict[int, int],
  new_node_ratio: float,
  inductive_seed: int,
  neg_overlap_ratio: float,
  neg_seed: int,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, float], Set[int]]:
  kept_rows = [i for i, nodes in enumerate(raw_df["nodes_raw"].tolist()) if len(nodes) >= 2]
  rows = []
  feat_rows: List[np.ndarray] = []
  mapped_node_lists: List[List[int]] = []

  for interaction_id, row_id in enumerate(kept_rows, start=1):
    mapped_nodes = [node_mapping[n] for n in raw_df.iloc[row_id]["nodes_raw"]]
    mapped_node_lists.append(mapped_nodes)
    feat_rows.append(hyperedge_feat[row_id])
    rows.append({
      "interaction_id": interaction_id,
      "nodes_list": mapped_nodes,
      "nodes": ",".join(map(str, mapped_nodes)),
      "ts": float(raw_df.iloc[row_id]["ts"]),
      "label": float(raw_df.iloc[row_id]["label"]),
      "idx": interaction_id,
    })

  thgn_df = pd.DataFrame(rows)
  splits = assign_temporal_splits(thgn_df["ts"].values)
  split_inductive, contains_new, is_new_node_val, is_new_node_test, new_nodes, meta = compute_inductive_labels(
    thgn_df["nodes_list"].tolist(),
    thgn_df["ts"].values,
    splits,
    new_node_ratio,
    inductive_seed,
  )
  thgn_df["split"] = splits
  thgn_df["split_inductive"] = split_inductive
  thgn_df["contains_new_node"] = contains_new
  thgn_df["is_new_node_val"] = is_new_node_val
  thgn_df["is_new_node_test"] = is_new_node_test

  all_mapped_nodes = set(node_mapping.values())
  neg_rng = np.random.RandomState(neg_seed)
  neg_nodes_col = []
  for _, row in thgn_df.iterrows():
    if row["split"] in ("val", "test"):
      neg_nodes = sample_negative_hyperedge(row["nodes_list"], all_mapped_nodes, neg_overlap_ratio, neg_rng)
      neg_nodes_col.append(",".join(map(str, neg_nodes)))
    else:
      neg_nodes_col.append("")
  thgn_df["neg_nodes"] = neg_nodes_col

  feat_dim = hyperedge_feat.shape[1] if hyperedge_feat.size > 0 else 1
  thgn_feat = np.vstack([np.zeros((1, feat_dim), dtype=float), np.array(feat_rows, dtype=float)])
  return thgn_df, thgn_feat, meta, new_nodes


def build_tcen_dataframe(thgn_df: pd.DataFrame, node_mapping: Dict[int, int]) -> Tuple[pd.DataFrame, np.ndarray]:
  rows = []
  feat_rows: List[np.ndarray] = []
  edge_idx = 1
  for _, row in thgn_df.iterrows():
    pos_nodes = sorted(parse_hyperedge_nodes(row["nodes"]))
    pos_pairs = list(combinations(pos_nodes, 2))
    neg_pairs = []
    if row["split"] in ("val", "test") and str(row["neg_nodes"]).strip():
      neg_nodes = sorted(parse_hyperedge_nodes(row["neg_nodes"]))
      neg_pairs = list(combinations(neg_nodes, 2))
    for j, (u, i) in enumerate(pos_pairs):
      neg_u, neg_i = ("", "")
      if j < len(neg_pairs):
        neg_u, neg_i = neg_pairs[j]
      rows.append({
        "interaction_id": int(row["interaction_id"]),
        "u": int(u),
        "i": int(i),
        "ts": float(row["ts"]),
        "label": float(row["label"]),
        "idx": edge_idx,
        "split": row["split"],
        "split_inductive": row["split_inductive"],
        "contains_new_node": int(row["contains_new_node"]),
        "is_new_node_val": int(row["is_new_node_val"]),
        "is_new_node_test": int(row["is_new_node_test"]),
        "neg_u": neg_u,
        "neg_i": neg_i,
      })
      feat_rows.append(np.array([0.0]))  # Will be replaced by parent feature dim logic below.
      edge_idx += 1
  tcen_df = pd.DataFrame(rows)
  _ = node_mapping
  return tcen_df, np.array(feat_rows, dtype=float)


def write_split_csvs_only(df: pd.DataFrame, out_dir: Path, dataset_name: str) -> None:
  out_dir.mkdir(parents=True, exist_ok=True)
  for split in ["train", "val", "test"]:
    split_df = df[df["split"] == split].copy()
    split_df.to_csv(out_dir / f"ml_{dataset_name}_{split}.csv", index=False)


def save_metadata(stats_out_dir: Path, dataset_name: str, new_nodes: Set[int], inductive_meta: Dict[str, float], neg_overlap_ratio: float, neg_seed: int) -> None:
  stats_out_dir.mkdir(parents=True, exist_ok=True)
  (stats_out_dir / f"{dataset_name}_new_nodes.txt").write_text("\n".join(str(x) for x in sorted(list(new_nodes))) + "\n", encoding="utf-8")
  meta = dict(inductive_meta)
  meta["neg_overlap_ratio"] = float(neg_overlap_ratio)
  meta["neg_seed"] = int(neg_seed)
  (stats_out_dir / f"{dataset_name}_inductive_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


def main():
  parser = argparse.ArgumentParser("Shared HOTN preprocessing for THGN and TCEN")
  parser.add_argument("--data", type=str, required=True)
  parser.add_argument("--input-dir", type=str, default="data")
  parser.add_argument("--thgn-out", type=str, default="thgn/data")
  parser.add_argument("--tcen-out", type=str, default="tcen/data")
  parser.add_argument("--stats-out-dir", type=str, default="data")
  parser.add_argument("--new-node-ratio", type=float, default=0.1)
  parser.add_argument("--inductive-seed", type=int, default=2020)
  parser.add_argument("--neg-overlap-ratio", type=float, default=0.99)
  parser.add_argument("--neg-seed", type=int, default=3030)
  args = parser.parse_args()

  csv_path = Path(args.input_dir) / f"{args.data}.csv"
  raw_df, hyperedge_feat = load_raw_hypergraph_csv(csv_path)
  write_dataset_stats(raw_df, Path(args.stats_out_dir) / f"{args.data}_stats.txt")

  node_mapping = reindex_nodes_contiguous_1_based(raw_df["nodes_raw"].tolist())
  thgn_df, thgn_feat, inductive_meta, new_nodes = build_thgn_dataframe(
    raw_df,
    hyperedge_feat,
    node_mapping,
    args.new_node_ratio,
    args.inductive_seed,
    args.neg_overlap_ratio,
    args.neg_seed,
  )

  feat_dim = thgn_feat.shape[1]
  np.save(Path(args.thgn_out) / f"ml_{args.data}.npy", thgn_feat)
  np.save(Path(args.thgn_out) / f"ml_{args.data}_node.npy", np.zeros((max(node_mapping.values()) + 1, 172)))
  write_split_csvs_only(thgn_df.drop(columns=["nodes_list"]), Path(args.thgn_out), args.data)

  tcen_rows = []
  tcen_feat_rows = [np.zeros(feat_dim, dtype=float)]
  edge_idx = 1
  for _, row in thgn_df.iterrows():
    pos_nodes = sorted(row["nodes_list"])
    pos_pairs = list(combinations(pos_nodes, 2))
    neg_pairs = []
    if row["split"] in ("val", "test") and row["neg_nodes"]:
      neg_nodes = sorted(parse_hyperedge_nodes(row["neg_nodes"]))
      neg_pairs = list(combinations(neg_nodes, 2))
    parent_feat = thgn_feat[int(row["idx"])]
    for j, (u, i) in enumerate(pos_pairs):
      neg_u, neg_i = ("", "")
      if j < len(neg_pairs):
        neg_u, neg_i = neg_pairs[j]
      tcen_rows.append({
        "interaction_id": int(row["interaction_id"]),
        "u": int(u),
        "i": int(i),
        "ts": float(row["ts"]),
        "label": float(row["label"]),
        "idx": edge_idx,
        "split": row["split"],
        "split_inductive": row["split_inductive"],
        "contains_new_node": int(row["contains_new_node"]),
        "is_new_node_val": int(row["is_new_node_val"]),
        "is_new_node_test": int(row["is_new_node_test"]),
        "neg_u": neg_u,
        "neg_i": neg_i,
      })
      tcen_feat_rows.append(parent_feat.copy())
      edge_idx += 1

  tcen_df = pd.DataFrame(tcen_rows)
  np.save(Path(args.tcen_out) / f"ml_{args.data}.npy", np.array(tcen_feat_rows, dtype=float))
  np.save(Path(args.tcen_out) / f"ml_{args.data}_node.npy", np.zeros((max(node_mapping.values()) + 1, 172)))
  write_split_csvs_only(tcen_df, Path(args.tcen_out), args.data)

  save_metadata(Path(args.stats_out_dir), args.data, new_nodes, inductive_meta, args.neg_overlap_ratio, args.neg_seed)
  print("Preprocessing complete.")


if __name__ == "__main__":
  main()

