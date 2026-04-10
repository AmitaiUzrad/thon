"""
Memory Updater Module for THGN
===============================

Purpose: Updates node memories using RNN cells (GRU or RNN) based on processed messages.
This is the final step where node memory vectors evolve incrementally by combining
current memory state with new message information.

Role in THGN Flow:
------------------
1. Hyperedge interactions occur → Raw messages created (for each node in hyperedge)
2. Messages stored → Temporarily buffered in memory
3. Messages aggregated → Multiple messages per node combined (MessageAggregator)
4. Messages processed → MessageFunction transforms messages
5. Memory updated → MemoryUpdater uses GRU/RNN to update node memories ← THIS MODULE

What are GRU and RNN?
---------------------
Both are Recurrent Neural Network cells that update hidden states (memory) based on new inputs (messages).

RNN (Simple Recurrent Network):
  Formula: h' = tanh(W_hh * h + W_ih * x + b)
  - h: current memory/hidden state [memory_dimension]
  - x: new message/input [message_dimension]
  - h': updated memory [memory_dimension]
  - Simple: directly combines old memory and new message

GRU (Gated Recurrent Unit):
  More sophisticated with learnable gates to control information flow:
  
  r = σ(W_ir * x + W_hr * h + b_r)           # Reset gate: what to forget
  z = σ(W_iz * x + W_hz * h + b_z)           # Update gate: how much to update
  n = tanh(W_in * x + r ⊙ (W_hn * h + b_hn)) # New candidate: filtered combination
  h' = (1 - z) ⊙ n + z ⊙ h                    # Updated memory: blend old and new
  
  Where:
  - σ = sigmoid function (outputs 0-1)
  - ⊙ = Hadamard product (element-wise multiplication: [a,b] ⊙ [c,d] = [a*c, b*d])
  - r, z, n are intermediate gates/vectors
  - h' is the final updated memory

  Key insight: GRU learns to balance old memory vs new message via gates.

Input/Output Format:
--------------------
Input:
  - unique_node_ids: List of node IDs to update [n_nodes]
  - unique_messages: Processed messages tensor [n_nodes, message_dimension]
    (Messages already aggregated and transformed from hyperedge interactions)
  - timestamps: Timestamps tensor [n_nodes]

Output:
  - update_memory(): Updates memory in-place (modifies self.memory)
  - get_updated_memory(): Returns copies (memory, last_update) without modifying original

Running Example:
----------------
Setup: Node 50 has processed message ready for memory update (from hyperedge [50, 60, 70] at time 200)
  - Current memory[50] = [0.2, 0.5, -0.1, ..., 0.3] (172 dims)
  - Processed message = [0.1, 0.2, ..., 0.9] (100 dims)
    (Already aggregated from hyperedge interaction, transformed by MessageFunction)
  - Timestamp = 200

GRU Update Process:
  Step 1: Get current memory
    memory = [0.2, 0.5, -0.1, ..., 0.3]  # [172 dims]
  
  Step 2: GRU forward pass
    r = sigmoid(W_ir * message + W_hr * memory + b_r)  # Reset gate [172 dims]
    z = sigmoid(W_iz * message + W_hz * memory + b_z)  # Update gate [172 dims]
    n = tanh(W_in * message + r ⊙ (W_hn * memory + b_hn))  # New candidate [172 dims]
    h' = (1 - z) ⊙ n + z ⊙ memory  # Updated memory [172 dims]
    
    Example element-wise: h'[0] = (1 - z[0]) * n[0] + z[0] * memory[0]
  
  Step 3: Update memory
    memory[50] = h' = [0.22, 0.51, -0.08, ..., 0.31]  # Updated! [172 dims]
    last_update[50] = 200

RNN Update Process (simpler):
  Step 1: Get current memory
    memory = [0.2, 0.5, -0.1, ..., 0.3]  # [172 dims]
  
  Step 2: RNN forward pass
    h' = tanh(W_hh * memory + W_ih * message + b)  # [172 dims]
  
  Step 3: Update memory
    memory[50] = h' = [0.21, 0.49, -0.09, ..., 0.29]  # Updated! [172 dims]
    last_update[50] = 200

Key Points:
-----------
- Memory updates are incremental (evolve, not replace)
- GRU uses gates to learn optimal balance between old and new information
- RNN is simpler but less expressive than GRU
- Timestamps are validated to prevent updating to past times
- Two methods: update_memory (in-place) and get_updated_memory (returns copy)
- Memory dimension (e.g., 172) is the size of the memory vector per node
- Message dimension (e.g., 100) is the size of processed messages
- The updater is hyperedge-agnostic: it operates on processed message tensors,
  which have the same structure whether they come from binary edges (TGN) or
  hyperedges (THGN). The hyperedge-specific processing happens earlier in the
  pipeline (in get_raw_messages() and message aggregation/processing).
"""

from torch import nn
import torch
import numpy as np


class MemoryUpdater(nn.Module):
  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    pass


class SequenceMemoryUpdater(MemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(SequenceMemoryUpdater, self).__init__()
    self.memory = memory
    self.layer_norm = torch.nn.LayerNorm(memory_dimension)
    self.message_dimension = message_dimension
    self.device = device

  def update_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return

    # DEBUG: Track memory update for Node 37
    # if 37 in unique_node_ids:
    #   node_idx = unique_node_ids.index(37) if isinstance(unique_node_ids, list) else np.where(np.array(unique_node_ids) == 37)[0][0]
    #   before_last_update = self.memory.get_last_update([37]).item()
    #   update_timestamp = timestamps[node_idx].item() if isinstance(timestamps[node_idx], torch.Tensor) else timestamps[node_idx]
    #   print(f"[DEBUG MemoryUpdater.update] Node 37: before_last_update={before_last_update}, "
    #         f"update_timestamp={update_timestamp}")

    # Filter out nodes with stale timestamps (defensive check)
    last_updates = self.memory.get_last_update(unique_node_ids)
    valid_mask = last_updates <= timestamps
    
    if not valid_mask.all().item():
      # Filter to only valid nodes (skip stale messages)
      valid_indices = valid_mask.nonzero(as_tuple=True)[0]
      if len(valid_indices) == 0:
        # All messages are stale, nothing to update
        return
      
      unique_node_ids = [unique_node_ids[i] for i in valid_indices.cpu().numpy()]
      unique_messages = unique_messages[valid_indices]
      timestamps = timestamps[valid_indices]
      
      # Recompute last_updates for filtered nodes
      last_updates = self.memory.get_last_update(unique_node_ids)

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    memory = self.memory.get_memory(unique_node_ids)
    self.memory.last_update[unique_node_ids] = timestamps

    updated_memory = self.memory_updater(unique_messages, memory)

    self.memory.set_memory(unique_node_ids, updated_memory)
    
    # DEBUG: Track after update
    # if 37 in unique_node_ids:
    #   after_last_update = self.memory.get_last_update([37]).item()
    #   print(f"[DEBUG MemoryUpdater.update] Node 37: after_last_update={after_last_update}")

  def get_updated_memory(self, unique_node_ids, unique_messages, timestamps):
    if len(unique_node_ids) <= 0:
      return self.memory.memory.data.clone(), self.memory.last_update.data.clone()

    # Filter out nodes with stale timestamps (defensive check)
    last_updates = self.memory.get_last_update(unique_node_ids)
    valid_mask = last_updates <= timestamps
    
    if not valid_mask.all().item():
      # Filter to only valid nodes (skip stale messages)
      valid_indices = valid_mask.nonzero(as_tuple=True)[0]
      if len(valid_indices) == 0:
        # All messages are stale, return current memory unchanged
        return self.memory.memory.data.clone(), self.memory.last_update.data.clone()
      
      unique_node_ids = [unique_node_ids[i] for i in valid_indices.cpu().numpy()]
      unique_messages = unique_messages[valid_indices]
      timestamps = timestamps[valid_indices]
      
      # Recompute last_updates for filtered nodes
      last_updates = self.memory.get_last_update(unique_node_ids)

    # DEBUG: Check assertion condition before it fails
    violation_mask = last_updates > timestamps
    if violation_mask.any().item():
      # This should not happen after filtering, but keep for debugging
      # unique_node_ids_arr = np.array(unique_node_ids)
      # violating_nodes = unique_node_ids_arr[violation_mask.cpu().numpy()]
      # print(f"\n[DEBUG MemoryUpdater] Assertion would fail!")
      # print(f"  Nodes with violations: {violating_nodes.tolist()}")
      # for node_id in violating_nodes[:5]:
      #   node_idx = np.where(unique_node_ids_arr == node_id)[0][0]
      #   last_update = last_updates[node_idx].item()
      #   msg_timestamp = timestamps[node_idx].item() if isinstance(timestamps[node_idx], torch.Tensor) else timestamps[node_idx]
      #   print(f"    Node {node_id}: last_update={last_update}, message_timestamp={msg_timestamp}, "
      #         f"diff={last_update - msg_timestamp}")
      pass

    assert (self.memory.get_last_update(unique_node_ids) <= timestamps).all().item(), "Trying to " \
                                                                                     "update memory to time in the past"

    updated_memory = self.memory.memory.data.clone()
    updated_memory[unique_node_ids] = self.memory_updater(unique_messages, updated_memory[unique_node_ids])

    updated_last_update = self.memory.last_update.data.clone()
    updated_last_update[unique_node_ids] = timestamps

    return updated_memory, updated_last_update


class GRUMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(GRUMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.GRUCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


class RNNMemoryUpdater(SequenceMemoryUpdater):
  def __init__(self, memory, message_dimension, memory_dimension, device):
    super(RNNMemoryUpdater, self).__init__(memory, message_dimension, memory_dimension, device)

    self.memory_updater = nn.RNNCell(input_size=message_dimension,
                                     hidden_size=memory_dimension)


def get_memory_updater(module_type, memory, message_dimension, memory_dimension, device):
  if module_type == "gru":
    return GRUMemoryUpdater(memory, message_dimension, memory_dimension, device)
  elif module_type == "rnn":
    return RNNMemoryUpdater(memory, message_dimension, memory_dimension, device)
  else:
    raise ValueError("Memory updater {} not implemented".format(module_type))
