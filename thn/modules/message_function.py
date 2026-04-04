"""
Message Function Module for THGN
=================================

Purpose: Transforms raw messages (created from hyperedge interactions) into processed messages
that will be used to update node memories. Acts as a compression/transformation step
between message creation and memory updating.

Role in THGN Flow:
------------------
1. Hyperedge interaction occurs → Raw message created (in THGN.get_raw_messages)
   (Each node in a hyperedge receives a message about the interaction)
2. Raw messages stored → Temporarily buffered in memory
3. Messages aggregated → Multiple messages per node combined (MessageAggregator)
4. Messages processed → MessageFunction transforms raw → processed ← THIS MODULE
5. Memory updated → Processed messages used by MemoryUpdater (GRU/RNN)

What are Raw Messages?
----------------------
Raw messages are concatenations of information about a hyperedge interaction from a node's perspective.
The exact composition depends on how THGN.get_raw_messages() aggregates information from other nodes
in the hyperedge, but typically includes:

- node_memory: [memory_dimension] - Current node's memory state
- aggregated_other_nodes_memory: [memory_dimension] - Aggregated (mean/sum) memory of other nodes in hyperedge
- hyperedge_features: [n_hyperedge_features] - Features of the hyperedge/interaction
- time_encoding: [time_encoder.dimension] - Encoded time delta since last update

Note on Raw Message Dimension:
-------------------------------
The raw message dimension formula for THGN differs from TGN:

TGN (binary edges):
  raw_message_dimension = 2 * memory_dimension + n_edge_features + time_encoder.dimension
  Example: If memory_dim=500, edge_feat=10, time_enc=100:
    raw_message_dimension = 2*500 + 10 + 100 = 1110 dimensions

THGN (hyperedges):
  raw_message_dimension = memory_dimension + aggregated_other_nodes_dim + n_hyperedge_features + time_encoder.dimension
  
  The exact formula depends on how other nodes in the hyperedge are aggregated in get_raw_messages():
  - If using mean pooling: aggregated_other_nodes_dim = memory_dimension
  - If using sum: aggregated_other_nodes_dim = memory_dimension
  - If using concatenation: aggregated_other_nodes_dim = (k-1) * memory_dimension (where k = hyperedge size)
  
  Example (with mean pooling): If memory_dim=500, hyperedge_feat=10, time_enc=100:
    raw_message_dimension = 500 + 500 + 10 + 100 = 1110 dimensions
    (Same dimension as TGN in this case, but composition is different)

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
Setup: raw_message_dimension=1110, message_dimension=100, batch_size=32 hyperedges

Step 1: Raw messages created from batch hyperedge interactions
  - Each hyperedge creates messages for all nodes in that hyperedge
  - Example batch: [[50, 60, 70], [80, 90], [50, 100], ...] (32 hyperedges)
  - Total messages: Sum of hyperedge sizes (e.g., 3 + 2 + 2 + ... = ~70 messages)
  - Messages for nodes: [50, 60, 70, 80, 90, 50, 100, ...]

Step 2: Messages aggregated by MessageAggregator (BEFORE message function)
  - Messages aggregated: unique_nodes = [50, 60, 70, 80, 90, 100, ...] (e.g., 45 unique nodes)
    * Node 50 appears in 2 hyperedges → aggregated into 1 message
    * unique_messages shape: [n_unique_nodes, 1110] = [45, 1110]

Step 3: Message function called ONCE per batch (for all unique nodes)
  
  Call: Process all unique node messages
    Input: unique_messages = [45, 1110] (after aggregation)
    MLP transformation (if using MLPMessageFunction):
      Layer 1: [45, 1110] @ W₁[1110, 555] + b₁[555] → [45, 555]
      ReLU: max(0, [45, 555]) → [45, 555]
      Layer 2: [45, 555] @ W₂[555, 100] + b₂[100] → [45, 100]
    Output: processed_messages = [45, 100]
  
  Parameters: W₁(1110×555=616,050) + b₁(555) + W₂(555×100=55,500) + b₂(100) = 672,205 total

Step 4: Processed messages passed to MemoryUpdater
  - processed_messages: [45, 100] → updates memory for 45 unique nodes
  - Total: 45 memory updates (one per unique node that participated in batch hyperedges)

If using IdentityMessageFunction:
  processed_messages = unique_messages  # [45, 1110] (no compression)

Key Points:
-----------
- Message function is applied AFTER aggregation, BEFORE memory update
- Input shape is [n_unique_nodes, raw_message_dim] where n_unique_nodes <= total_nodes_in_batch
  (after MessageAggregator combines multiple messages per node)
- Function is called ONCE per batch (for all unique nodes, not separately for sources/destinations)
- Total messages processed per batch: n_unique_nodes
  (typically between batch_size and sum_of_hyperedge_sizes, depending on node overlap)
- MLP compresses high-dimensional raw messages to fixed message_dimension
- Compression enables efficient memory updates and learned representations
- Identity function preserves full raw message (no compression)
- The message function is hyperedge-agnostic: it operates on the raw message tensor structure,
  which is the same whether messages come from binary edges (TGN) or hyperedges (THGN)
"""

from torch import nn


class MessageFunction(nn.Module):
  """
  Module which computes the message for a given interaction.
  
  In THGN, messages come from hyperedge interactions (each node in a hyperedge receives
  a message), but the transformation logic remains the same as in TGN.
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
  else:
    raise ValueError("Message function {} not implemented".format(module_type))
