"""
Message Function Module for TGN
================================

Purpose: Transforms raw messages (created from interactions) into processed messages
that will be used to update node memories. Acts as a compression/transformation step
between message creation and memory updating.

Role in TGN Flow:
-----------------
1. Edge interaction occurs → Raw message created (in TGN.get_raw_messages)
2. Raw messages stored → Temporarily buffered in memory
3. Messages aggregated → Multiple messages per node combined (MessageAggregator)
4. Messages processed → MessageFunction transforms raw → processed ← THIS MODULE
5. Memory updated → Processed messages used by MemoryUpdater (GRU/RNN)

What are Raw Messages?
----------------------
Raw messages are concatenations of:
- source_memory: [memory_dimension] - Source node's current memory state
- destination_memory: [memory_dimension] - Destination node's current memory state
- edge_features: [n_edge_features] - Features of the edge/interaction
- time_encoding: [time_encoder.dimension] - Encoded time delta since last update

raw_message_dimension = 2 * memory_dimension + n_edge_features + time_encoder.dimension

Example: If memory_dim=500, edge_feat=10, time_enc=100:
  raw_message_dimension = 2*500 + 10 + 100 = 1110 dimensions

Implementations:
----------------
1. MLPMessageFunction: Compresses raw messages via 2-layer MLP
   - Layer 1: Linear(raw_dim → raw_dim//2) + ReLU
   - Layer 2: Linear(raw_dim//2 → message_dim)
   - Learnable parameters: 2 weight matrices + 2 bias vectors
   - Formula: output = Linear₂(ReLU(Linear₁(raw_message)))

2. IdentityMessageFunction: Pass-through (no transformation)
   - Returns raw_messages unchanged
   - Used when message_function="identity"

Running Example:
----------------
Setup: raw_message_dimension=1110, message_dimension=100, batch_size=32 edges

Step 1: Raw messages created from batch interactions
  - Each edge creates 2 messages (one for source, one for destination)
  - Total: 2 * batch_size = 64 raw messages
  - Example batch: [(50, 60), (70, 80), (50, 90), ...] (32 edges)
  - Source messages: 32 messages for nodes [50, 70, 50, ...]
  - Destination messages: 32 messages for nodes [60, 80, 90, ...]

Step 2: Messages aggregated by MessageAggregator (BEFORE message function)
  - Source messages aggregated: unique_sources = [50, 70, ...] (e.g., 20 unique nodes)
    * Node 50 appears twice → aggregated into 1 message
    * unique_messages shape: [n_unique_sources, 1110] = [20, 1110]
  - Destination messages aggregated: unique_destinations = [60, 80, 90, ...] (e.g., 25 unique nodes)
    * unique_messages shape: [n_unique_destinations, 1110] = [25, 1110]

Step 3: Message function called TWICE per batch (once for sources, once for destinations)
  
  Call 1: Process source messages
    Input: unique_messages = [20, 1110] (after aggregation)
    MLP transformation (if using MLPMessageFunction):
      Layer 1: [20, 1110] @ W₁[1110, 555] + b₁[555] → [20, 555]
      ReLU: max(0, [20, 555]) → [20, 555]
      Layer 2: [20, 555] @ W₂[555, 100] + b₂[100] → [20, 100]
    Output: processed_source_messages = [20, 100]
  
  Call 2: Process destination messages
    Input: unique_messages = [25, 1110] (after aggregation)
    MLP transformation:
      Layer 1: [25, 1110] @ W₁[1110, 555] + b₁[555] → [25, 555]
      ReLU: max(0, [25, 555]) → [25, 555]
      Layer 2: [25, 555] @ W₂[555, 100] + b₂[100] → [25, 100]
    Output: processed_destination_messages = [25, 100]
  
  Parameters: W₁(1110×555=616,050) + b₁(555) + W₂(555×100=55,500) + b₂(100) = 672,205 total

Step 4: Processed messages passed to MemoryUpdater
  - processed_source_messages: [20, 100] → updates memory for 20 unique source nodes
  - processed_destination_messages: [25, 100] → updates memory for 25 unique destination nodes
  - Total: 45 memory updates (n_unique_sources + n_unique_destinations)

If using IdentityMessageFunction:
  processed_source_messages = unique_source_messages  # [20, 1110] (no compression)
  processed_destination_messages = unique_destination_messages  # [25, 1110] (no compression)

Key Points:
-----------
- Message function is applied AFTER aggregation, BEFORE memory update
- Input shape is [n_unique_nodes, raw_message_dim] where n_unique_nodes <= batch_size
  (after MessageAggregator combines multiple messages per node)
- Function is called TWICE per batch: once for source nodes, once for destination nodes
- Total messages processed per batch: n_unique_sources + n_unique_destinations
  (typically between batch_size and 2*batch_size, depending on node overlap)
- MLP compresses high-dimensional raw messages to fixed message_dimension
- Compression enables efficient memory updates and learned representations
- Identity function preserves full raw message (no compression)
"""

from torch import nn


class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  """

  def compute_message(self, raw_messages):
    return None


class MLPMessageFunction(MessageFunction):
  def __init__(self, raw_message_dimension, message_dimension):
    super(MLPMessageFunction, self).__init__()

    self.mlp = self.layers = nn.Sequential(
      nn.Linear(raw_message_dimension, raw_message_dimension // 2),
      nn.ReLU(),
      nn.Linear(raw_message_dimension // 2, message_dimension),
    )

  def compute_message(self, raw_messages):
    messages = self.mlp(raw_messages)

    return messages


class IdentityMessageFunction(MessageFunction):

  def compute_message(self, raw_messages):

    return raw_messages


def get_message_function(module_type, raw_message_dimension, message_dimension):
  if module_type == "mlp":
    return MLPMessageFunction(raw_message_dimension, message_dimension)
  elif module_type == "identity":
    return IdentityMessageFunction()
