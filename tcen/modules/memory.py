"""
Memory Module for TGN
=====================

Purpose: Maintains persistent per-node memory vectors that evolve over time based on
interactions. Each node has a single memory vector (not multiple stored updates).

Key Concept - Memory Dimension:
--------------------------------
memory_dimension = SIZE of the memory vector (like embedding dimension), NOT the number
of updates stored per node.

Example: memory_dimension = 172 means each node has a 172-dimensional vector.
- Node 50: memory[50] = [0.2, 0.5, -0.1, ..., 0.3]  (172 numbers)
- This vector is UPDATED (not appended to) as interactions occur

Data Structures:
----------------
1. self.memory: [n_nodes, memory_dimension] - One vector per node (persistent state)
2. self.last_update: [n_nodes] - Timestamp of last memory update per node
3. self.messages: dict - Temporary buffer storing messages before aggregation
   Format: {node_id: [(message_vector, timestamp), ...]}

Complete Flow with Running Example:
-----------------------------------
Setup: Nodes 50, 60, 70; memory_dimension = 172

Initial State:
  memory[50] = [0, 0, ..., 0]  (172 zeros)
  memory[60] = [0, 0, ..., 0]
  memory[70] = [0, 0, ..., 0]
  last_update[50] = 0
  messages = {}

Step 1: Interaction occurs
  Event: Node 50 interacts with Node 60 at time 100
  Edge: (50, 60, time=100, edge_idx=0)

Step 2: Create raw message (in TGN.get_raw_messages)
  For node 50 (source):
    source_memory = memory[50]  # [0, 0, ..., 0] (172 dims)
    destination_memory = memory[60]  # [0, 0, ..., 0] (172 dims)
    edge_features = [0.5, 0.3, ...]  # Edge feature dims
    time_delta = 100 - last_update[50] = 100
    time_encoding = time_encoder(100)  # [0.88, 0.99, ...] (100 dims)
    
    message_50 = concat([source_memory, destination_memory, edge_features, time_encoding])
    # Result: ~454-dimensional vector
  
  For node 60 (destination):
    source_memory = memory[60]  # [0, 0, ..., 0] (172 dims)
    destination_memory = memory[50]  # [0, 0, ..., 0] (172 dims)
    edge_features = [0.5, 0.3, ...]  # Same edge feature dims
    time_delta = 100 - last_update[60] = 100
    time_encoding = time_encoder(100)  # [0.88, 0.99, ...] (100 dims)
    
    message_60 = concat([source_memory, destination_memory, edge_features, time_encoding])
    # Result: ~454-dimensional vector

Step 3: Store raw message (store_raw_messages)
  memory.messages[50].append((message_50, timestamp=100))
  memory.messages[60].append((message_60, timestamp=100))
  
  State:
    messages = {
      50: [(message_50, 100)],
      60: [(message_60, 100)]
    }

Step 4: More interactions accumulate (before updating memory)
  - Node 50 interacts with Node 70 at time 150
  - Node 50 interacts with Node 60 again at time 200
  
  State:
    messages = {
      50: [
        (message_vector_50_1, 100),  # From time 100
        (message_vector_50_2, 150),  # From time 150
        (message_vector_50_3, 200)   # From time 200
      ],
      60: [(message_vector_60_1, 100), (message_vector_60_2, 200)],
      70: [(message_vector_70_1, 150)]
    }

Step 5: Aggregate messages (message_aggregator.aggregate)
  For node 50: Combine 3 messages
  - "last" aggregator: Takes most recent → message_vector_50_3
  - "mean" aggregator: Averages all 3 messages
  
  Result: unique_messages[50] = aggregated_message (single vector)

Step 6: Process message (message_function.compute_message)
  Apply MLP or identity function to transform aggregated message
  processed_message[50] = MLP(aggregated_message)  # e.g., 454 dims → 100 dims

Step 7: Update memory (memory_updater.update_memory)
  Use GRU/RNN to update memory based on processed message:
    new_memory[50] = GRU(old_memory[50], processed_message[50])
    memory[50] = new_memory[50]  # Updated! [0.2, 0.5, -0.1, ..., 0.3]
    last_update[50] = 200  # Updated timestamp

Step 8: Clear messages (clear_messages)
  After memory is updated, remove processed messages:
    memory.messages[50] = []  # Cleared!

Final State:
  memory[50] = [0.2, 0.5, -0.1, ..., 0.3]  # Updated (172-dim vector)
  last_update[50] = 200
  messages[50] = []  # Empty

Key Points:
-----------
1. Memory is a SINGLE vector per node (not multiple stored updates)
2. Messages are TEMPORARY buffers (stored before aggregation, cleared after update)
3. Multiple messages can accumulate before memory is updated (batch processing)
4. Memory is updated incrementally (not replaced, but evolved via GRU/RNN)
5. Memory persists across interactions (represents node's evolving state)

Flow Summary:
-------------
Interaction → Create Message → Store Message → (More Interactions) →
Aggregate Messages → Process Message → Update Memory → Clear Messages
"""

import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes).to(self.device),
                                    requires_grad=False)

    self.messages = defaultdict(list)

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend(node_id_to_messages[node])

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) for x in v]

    return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_memory(self, memory_backup):
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) for x in v]

  def detach_memory(self):
    self.memory.detach_()

    # Detach all stored messages
    for k, v in self.messages.items():
      new_node_messages = []
      for message in v:
        new_node_messages.append((message[0].detach(), message[1]))

      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []
