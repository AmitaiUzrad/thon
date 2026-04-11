"""
THGN Self-Supervised Training Script for Temporal Hyperedge Prediction
=======================================================================

Purpose: Training script for Temporal Hypergraph Network (THGN) that performs temporal hyperedge
prediction. This script orchestrates the complete training pipeline: data loading, batch processing,
memory management, validation, testing, and model checkpointing.

Task: Temporal Hyperedge Prediction
------------------------------------
Given a temporal hypergraph with hyperedges (set of nodes, t), predict whether a hyperedge exists
at a specific timestamp. The model learns to distinguish positive hyperedges (that exist) from
negative hyperedges (randomly sampled non-hyperedges) at the same timestamp.

Key Differences from TGN:
-------------------------
1. **Data structure**: Uses hyperedges (list of lists) instead of separate source/destination arrays
2. **Variable-size interactions**: Each hyperedge can have any number of nodes (≥ 1)
3. **Negative sampling**: Samples complete hyperedges matching the size of positive hyperedges
4. **Unified time statistics**: Single mean/std for all nodes (no source/destination distinction)
5. **Model method**: Calls `compute_hyperedge_probabilities` instead of `compute_edge_probabilities`

"Self-supervised" meaning: Learns from graph structure itself (predicting hyperedges) rather than
external labels, but still uses supervised learning (positive vs negative hyperedges).

Complete Training Flow with Running Example:
--------------------------------------------
Dataset: 10,000 hyperedges, split: train=7,000, val=1,500, test=1,500
Batch size: 200, Epochs: 50, Memory: enabled, Memory dimension: 172

Epoch 1:
---------
Step 1: Initialize memory (all zeros) for all nodes
  Memory: {node_id: [0, 0, ..., 0]} (172 dims per node)

Step 2: Training Loop (35 batches: 7,000 hyperedges / 200)
  Batch 0: Hyperedges [[50, 60, 70], [80, 90], ..., [100, 110, 120, 130]] (200 hyperedges)
    - Sample negatives: [[23, 45, 89], [12, 67], ..., [34, 56, 78, 90]] (matching sizes)
    - Compute probabilities:
      * pos_prob = thgn.compute_hyperedge_probabilities(hyperedges, negative_hyperedges, ...)
      * Returns: pos_prob=[0.52, 0.48, ...], neg_prob=[0.31, 0.29, ...]
    - Loss = BCE(pos_prob, 1) + BCE(neg_prob, 0) = 0.65
    - Backprop → Update model parameters
    - Update memory: Memory now reflects interactions from batch 0
      * Memory[50] updated with message from hyperedge [50, 60, 70] at t=100
      * Memory[60] updated with message from hyperedge [50, 60, 70] at t=100
      * Memory[70] updated with message from hyperedge [50, 60, 70] at t=100
      * Similar for all nodes in batch
  
  Batch 1: Hyperedges [[140, 150], ..., [160, 170, 180]] (200 hyperedges)
    - Memory state: Includes updates from batch 0
    - Query time for each hyperedge: t=250, t=300 (uses memory before these times)
    - Compute probabilities → Loss → Backprop → Update memory
    - Memory now includes interactions from batches 0 and 1
  
  ... (continue for all 35 batches)
  
  After all batches: Detach memory (prevent backprop through time)

Step 3: Validation
  - Switch to full graph (includes val/test hyperedges for neighbor finding)
  - Backup training memory
  - Validate on regular nodes (val_data: 1,500 hyperedges)
    * Process in batches, compute AP and AUC
    * Memory updated during validation (for next batch)
  - Restore training memory
  - Validate on new nodes (new_node_val_data)
    * Uses training memory (new nodes have no memory yet)
  - Restore validation memory
  - Check early stopping (if no improvement for 5 epochs)

Step 4: Save checkpoint
  - Save model state_dict to checkpoint file

Epoch 2:
---------
Step 1: Reset memory to zeros (fresh start each epoch)
Step 2: Training loop (same as epoch 1, but model parameters updated)
Step 3: Validation (compare with previous best)
...

After Training:
---------------
- Load best model (from early stopping)
- Test on test_data (regular nodes)
- Test on new_node_test_data (inductive setting)
- Save final model

Key Components:
---------------
- Data Loading: Splits temporal hypergraph into train/val/test chronologically
- Neighbor Finders: train_ngh_finder (training hyperedges only) vs full_ngh_finder (all hyperedges)
- Negative Samplers: Randomly sample complete hyperedges matching positive sizes (seeded for reproducibility)
- Memory Management: Reset each epoch, backup/restore for different evaluation scenarios
- Early Stopping: Monitors validation AP, stops if no improvement for 'patience' epochs
- Gradient Accumulation: Can accumulate gradients over multiple batches (backprop_every)

Important Points:
-----------------
- Memory resets at start of each epoch: Ensures fair comparison across epochs
- Batches processed chronologically: Earlier batches update memory used by later batches
- Validation uses full graph: Can see val/test hyperedges as neighbors (but not for prediction)
- Memory backup/restore: Allows testing different scenarios (regular vs new nodes) with same memory state
- Within-batch temporal consistency: All hyperedges in batch use same memory state (from before batch)
- Negative sampling: Each positive hyperedge gets one negative hyperedge of same size at same timestamp
- Loss: Binary cross-entropy on hyperedge probabilities (positive=1, negative=0)
- Evaluation metrics: Average Precision (AP) and Area Under ROC Curve (AUC)
"""

import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path

from evaluation.evaluation import eval_hyperedge_prediction
from model.thgn import THGN
from utils.utils import EarlyStopMonitor, RandHyperedgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('THGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. tags-ask-ubuntu)',
                    default='tags-ask-ubuntu')
parser.add_argument('--bs', type=int, default=200, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of hyperedges to sample per node')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_node_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of nodes as part of the message (THGN unified version of TGN\'s use_source/destination_embedding_in_message)')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')
parser.add_argument('--train_neg_overlap_ratio', type=float, default=0.99,
                    help='THGN training negative overlap ratio with positives (0.0-1.0)')


try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

if args.train_neg_overlap_ratio < 0.0 or args.train_neg_overlap_ratio > 1.0:
  raise ValueError(f"--train_neg_overlap_ratio must be in [0, 1], got {args.train_neg_overlap_ratio}")

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data, new_node_val_data, \
new_node_test_data = get_data(DATA,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

# Runtime verification dump: exactly what was loaded for train/val/test.
train_runtime_negative_hyperedges_epoch0 = []

# Initialize training neighbor finder to retrieve temporal graph
# THGN: Uses hyperedge-level neighbor finding (samples n_neighbors hyperedges, extracts nodes)
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# THGN: RandHyperedgeSampler takes a list of all unique nodes (not separate source/destination lists)
#       and samples complete hyperedges matching the sizes of positive hyperedges
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
all_unique_nodes = list(full_data.unique_nodes)
train_rand_sampler = RandHyperedgeSampler(all_unique_nodes)
val_rand_sampler = RandHyperedgeSampler(all_unique_nodes, seed=0)
nn_val_rand_sampler = RandHyperedgeSampler(list(new_node_val_data.unique_nodes), seed=1)
test_rand_sampler = RandHyperedgeSampler(all_unique_nodes, seed=2)
nn_test_rand_sampler = RandHyperedgeSampler(list(new_node_test_data.unique_nodes), seed=3)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
# THGN: Unified time statistics (no source/destination distinction) since hypergraphs are undirected
mean_time_shift, std_time_shift = compute_time_statistics(full_data.hyperedges, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  # THGN: Uses unified time shift parameters (mean_time_shift, std_time_shift) instead of separate
  #       source/destination parameters. Also uses use_node_embedding_in_message instead of
  #       use_source/destination_embedding_in_message.
  thgn = THGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift=mean_time_shift, std_time_shift=std_time_shift,
            use_node_embedding_in_message=args.use_node_embedding_in_message,
            dyrep=args.dyrep)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(thgn.parameters(), lr=LEARNING_RATE)
  thgn = thgn.to(device)

  # THGN: Use hyperedges instead of sources/destinations
  num_instance = len(train_data.hyperedges)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  new_nodes_val_aps = []
  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)
  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      thgn.memory.__init_memory__()

    # Train using only training graph
    thgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    # Initialize streaming train dump only for epoch 0
    if epoch == 0:
      stream_dir = Path("runtime_split_dumps"); stream_dir.mkdir(parents=True, exist_ok=True)
      thgn_train_stream_path = stream_dir / f"ml_{DATA}_train_runtime_stream.csv"
      with open(thgn_train_stream_path, "w", newline="") as f:
        f.write("interaction_id,nodes,ts,label,idx,split,split_inductive,contains_new_node,"
                "is_new_node_val,is_new_node_test,runtime_train_neg_nodes\n")
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        # DEBUG: Only print around batch 26
        # if batch_idx >= 24 and batch_idx <= 28:
        #   print(f"\n[DEBUG Training] Batch {batch_idx}")

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        # THGN: Extract hyperedges (list of lists) instead of separate sources/destinations
        hyperedges_batch = train_data.hyperedges[start_idx:end_idx]
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        size = len(hyperedges_batch)
        # THGN: Sample overlap-controlled negatives for training.
        negative_hyperedges = train_rand_sampler.sample_with_overlap(
          hyperedges_batch, overlap_ratio=args.train_neg_overlap_ratio
        )
        if epoch == 0:
          train_runtime_negative_hyperedges_epoch0.extend(
            [",".join(map(str, neg_h)) for neg_h in negative_hyperedges]
          )

        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)

        thgn = thgn.train()
        # THGN: Call compute_hyperedge_probabilities with hyperedges (list of lists) and
        #       negative_hyperedges (list of lists) instead of separate source/destination arrays
        pos_prob, neg_prob = thgn.compute_hyperedge_probabilities(hyperedges_batch, negative_hyperedges,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

        # Stream append the exact batch rows used (epoch 0 only)
        if epoch == 0 and getattr(train_data, "rows_df", None) is not None:
          batch_rows = train_data.rows_df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
          batch_rows["runtime_train_neg_nodes"] = [",".join(map(str, n)) for n in negative_hyperedges]
          cols = ["interaction_id","nodes","ts","label","idx","split","split_inductive",
                  "contains_new_node","is_new_node_val","is_new_node_test","runtime_train_neg_nodes"]
          batch_rows[cols].to_csv(thgn_train_stream_path, mode="a", index=False, header=False)

      loss /= args.backprop_every

      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        thgn.memory.detach_memory()

      # Debug print after each batch
      batches_in_group = min(args.backprop_every, num_batch - k)
      logger.debug('Epoch {}, Processed batches {}-{}, Loss: {:.4f}'.format(
        epoch, k, min(k + batches_in_group - 1, num_batch - 1), loss.item()))

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    thgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = thgn.memory.backup_memory()

    # THGN: Use eval_hyperedge_prediction instead of eval_edge_prediction
    val_ap, val_auc = eval_hyperedge_prediction(model=thgn,
                                                            negative_hyperedge_sampler=val_rand_sampler,
                                                            data=val_data,
                                                            n_neighbors=NUM_NEIGHBORS,
                                                            dataset_name=DATA, split_name="val")
    if USE_MEMORY:
      val_memory_backup = thgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test hyperedges are
      # strictly later in time than validation hyperedges)
      thgn.memory.restore_memory(train_memory_backup)

    # Validate on unseen nodes
    nn_val_ap, nn_val_auc = eval_hyperedge_prediction(model=thgn,
                                                                        negative_hyperedge_sampler=nn_val_rand_sampler,
                                                                        data=new_node_val_data,
                                                                        n_neighbors=NUM_NEIGHBORS)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      thgn.memory.restore_memory(val_memory_backup)

    new_nodes_val_aps.append(nn_val_ap)
    val_aps.append(val_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "new_nodes_val_aps": new_nodes_val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'val auc: {}, new node val auc: {}'.format(val_auc, nn_val_auc))
    logger.info(
      'val ap: {}, new node val ap: {}'.format(val_ap, nn_val_ap))

    # (Removed non-stream train dump enrichment; streaming is handled per-batch above)

    # Early stopping
    if early_stopper.early_stop_check(val_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      thgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      thgn.eval()
      break
    else:
      torch.save(thgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation hyperedges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = thgn.memory.backup_memory()

  ### Test
  thgn.embedding_module.neighbor_finder = full_ngh_finder
  test_ap, test_auc = eval_hyperedge_prediction(model=thgn,
                                                              negative_hyperedge_sampler=test_rand_sampler,
                                                              data=test_data,
                                                              n_neighbors=NUM_NEIGHBORS,
                                                              dataset_name=DATA, split_name="test")

  if USE_MEMORY:
    thgn.memory.restore_memory(val_memory_backup)

  # Test on unseen nodes
  nn_test_ap, nn_test_auc = eval_hyperedge_prediction(model=thgn,
                                                                          negative_hyperedge_sampler=nn_test_rand_sampler,
                                                                          data=new_node_test_data,
                                                                          n_neighbors=NUM_NEIGHBORS)

  logger.info(
    'Test statistics: Old nodes -- auc: {}, ap: {}'.format(test_auc, test_ap))
  logger.info(
    'Test statistics: New nodes -- auc: {}, ap: {}'.format(nn_test_auc, nn_test_ap))
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "new_nodes_val_aps": new_nodes_val_aps,
    "test_ap": test_ap,
    "new_node_test_ap": nn_test_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving THGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    thgn.memory.restore_memory(val_memory_backup)
  torch.save(thgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('THGN model saved')
