#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List


def _add_prefixed_training_args(parser: argparse.ArgumentParser, prefix: str, include_train_neg_overlap: bool = False) -> None:
  parser.add_argument(f"--{prefix}-bs", type=int, default=None)
  parser.add_argument(f"--{prefix}-prefix", type=str, default=None)
  parser.add_argument(f"--{prefix}-n-degree", type=int, default=None)
  parser.add_argument(f"--{prefix}-n-head", type=int, default=None)
  parser.add_argument(f"--{prefix}-n-epoch", type=int, default=None)
  parser.add_argument(f"--{prefix}-n-layer", type=int, default=None)
  parser.add_argument(f"--{prefix}-lr", type=float, default=None)
  parser.add_argument(f"--{prefix}-patience", type=int, default=None)
  parser.add_argument(f"--{prefix}-n-runs", type=int, default=None)
  parser.add_argument(f"--{prefix}-drop-out", type=float, default=None)
  parser.add_argument(f"--{prefix}-gpu", type=int, default=None)
  parser.add_argument(f"--{prefix}-node-dim", type=int, default=None)
  parser.add_argument(f"--{prefix}-time-dim", type=int, default=None)
  parser.add_argument(f"--{prefix}-backprop-every", type=int, default=None)
  parser.add_argument(f"--{prefix}-message-dim", type=int, default=None)
  parser.add_argument(f"--{prefix}-memory-dim", type=int, default=None)
  parser.add_argument(f"--{prefix}-embedding-module", type=str, default=None)
  parser.add_argument(f"--{prefix}-message-function", type=str, default=None)
  parser.add_argument(f"--{prefix}-memory-updater", type=str, default=None)
  parser.add_argument(f"--{prefix}-aggregator", type=str, default=None)

  parser.add_argument(f"--{prefix}-use-memory", action="store_true")
  parser.add_argument(f"--{prefix}-memory-update-at-end", action="store_true")
  parser.add_argument(f"--{prefix}-different-new-nodes", action="store_true")
  parser.add_argument(f"--{prefix}-uniform", action="store_true")
  parser.add_argument(f"--{prefix}-randomize-features", action="store_true")
  parser.add_argument(f"--{prefix}-dyrep", action="store_true")

  if prefix == "thn":
    parser.add_argument(f"--{prefix}-use-node-embedding-in-message", action="store_true")
  else:
    parser.add_argument(f"--{prefix}-use-destination-embedding-in-message", action="store_true")
    parser.add_argument(f"--{prefix}-use-source-embedding-in-message", action="store_true")

  if include_train_neg_overlap:
    parser.add_argument("--thn-train-neg-overlap-ratio", type=float, default=None)


def _append_if_set(cmd: List[str], cli_flag: str, value) -> None:
  if value is None:
    return
  cmd.extend([cli_flag, str(value)])


def _append_bool(cmd: List[str], cli_flag: str, enabled: bool) -> None:
  if enabled:
    cmd.append(cli_flag)


def _build_train_cmd(python_exec: str, train_script_path: Path, model: str, args: argparse.Namespace) -> List[str]:
  cmd = [python_exec, str(train_script_path), "--data", args.data]

  _append_if_set(cmd, "--bs", getattr(args, f"{model}_bs"))
  _append_if_set(cmd, "--prefix", getattr(args, f"{model}_prefix"))
  _append_if_set(cmd, "--n_degree", getattr(args, f"{model}_n_degree"))
  _append_if_set(cmd, "--n_head", getattr(args, f"{model}_n_head"))
  _append_if_set(cmd, "--n_epoch", getattr(args, f"{model}_n_epoch"))
  _append_if_set(cmd, "--n_layer", getattr(args, f"{model}_n_layer"))
  _append_if_set(cmd, "--lr", getattr(args, f"{model}_lr"))
  _append_if_set(cmd, "--patience", getattr(args, f"{model}_patience"))
  _append_if_set(cmd, "--n_runs", getattr(args, f"{model}_n_runs"))
  _append_if_set(cmd, "--drop_out", getattr(args, f"{model}_drop_out"))
  _append_if_set(cmd, "--gpu", getattr(args, f"{model}_gpu"))
  _append_if_set(cmd, "--node_dim", getattr(args, f"{model}_node_dim"))
  _append_if_set(cmd, "--time_dim", getattr(args, f"{model}_time_dim"))
  _append_if_set(cmd, "--backprop_every", getattr(args, f"{model}_backprop_every"))
  _append_if_set(cmd, "--message_dim", getattr(args, f"{model}_message_dim"))
  _append_if_set(cmd, "--memory_dim", getattr(args, f"{model}_memory_dim"))
  _append_if_set(cmd, "--embedding_module", getattr(args, f"{model}_embedding_module"))
  _append_if_set(cmd, "--message_function", getattr(args, f"{model}_message_function"))
  _append_if_set(cmd, "--memory_updater", getattr(args, f"{model}_memory_updater"))
  _append_if_set(cmd, "--aggregator", getattr(args, f"{model}_aggregator"))

  _append_bool(cmd, "--use_memory", getattr(args, f"{model}_use_memory"))
  _append_bool(cmd, "--memory_update_at_end", getattr(args, f"{model}_memory_update_at_end"))
  _append_bool(cmd, "--different_new_nodes", getattr(args, f"{model}_different_new_nodes"))
  _append_bool(cmd, "--uniform", getattr(args, f"{model}_uniform"))
  _append_bool(cmd, "--randomize_features", getattr(args, f"{model}_randomize_features"))
  _append_bool(cmd, "--dyrep", getattr(args, f"{model}_dyrep"))

  if model == "thn":
    _append_bool(cmd, "--use_node_embedding_in_message", args.thn_use_node_embedding_in_message)
    _append_if_set(cmd, "--train_neg_overlap_ratio", args.thn_train_neg_overlap_ratio)
    for item in args.thn_extra_arg:
      cmd.extend(shlex.split(item))
  else:
    _append_bool(cmd, "--use_destination_embedding_in_message", args.tcn_use_destination_embedding_in_message)
    _append_bool(cmd, "--use_source_embedding_in_message", args.tcn_use_source_embedding_in_message)
    for item in args.tcn_extra_arg:
      cmd.extend(shlex.split(item))

  return cmd


def main() -> None:
  parser = argparse.ArgumentParser("Run THN then TCN self-supervised training on shared preprocessed data")
  parser.add_argument("--data", type=str, required=True, help="Dataset name (without .csv)")
  parser.add_argument("--python", type=str, default=sys.executable, help="Python executable to use")
  parser.add_argument("--dry-run", action="store_true", help="Print commands only")
  parser.add_argument("--skip-thn", action="store_true")
  parser.add_argument("--skip-tcn", action="store_true")
  parser.add_argument("--continue-on-error", action="store_true", help="Run TCN even if THN fails")
  parser.add_argument("--thn-extra-arg", action="append", default=[], help='Extra raw arg(s) for THN, e.g. --thn-extra-arg "--foo 1"')
  parser.add_argument("--tcn-extra-arg", action="append", default=[], help='Extra raw arg(s) for TCN, e.g. --tcn-extra-arg "--bar 2"')

  _add_prefixed_training_args(parser, "thn", include_train_neg_overlap=True)
  _add_prefixed_training_args(parser, "tcn", include_train_neg_overlap=False)
  args = parser.parse_args()

  root = Path(__file__).resolve().parent
  thn_script = root / "thn" / "train_self_supervised.py"
  tcn_script = root / "tcn" / "train_self_supervised.py"

  if not args.skip_thn:
    thn_cmd = _build_train_cmd(args.python, thn_script, "thn", args)
    print("[run_hotn] THN command:")
    print(" ".join(shlex.quote(x) for x in thn_cmd))
    if not args.dry_run:
      thn_proc = subprocess.run(thn_cmd, cwd=str(root / "thn"))
      if thn_proc.returncode != 0 and not args.continue_on_error:
        raise SystemExit(thn_proc.returncode)

  if not args.skip_tcn:
    tcn_cmd = _build_train_cmd(args.python, tcn_script, "tcn", args)
    print("[run_hotn] TCN command:")
    print(" ".join(shlex.quote(x) for x in tcn_cmd))
    if not args.dry_run:
      tcn_proc = subprocess.run(tcn_cmd, cwd=str(root / "tcn"))
      if tcn_proc.returncode != 0:
        raise SystemExit(tcn_proc.returncode)


if __name__ == "__main__":
  main()

