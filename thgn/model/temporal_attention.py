"""
Temporal Attention Layer for THGN
==================================

Purpose: Computes node embeddings by attending over temporal neighbors. Uses multi-head attention
to weight neighbors based on node features, hyperedge features, and time encodings, producing
context-aware embeddings for hyperedge prediction.

Role in THGN Flow:
-----------------
1. Memory updated → Node memories reflect recent interactions (via message/memory flow)
2. Temporal neighbors found → For each node, find neighbors before query time (via clique structure)
3. Temporal attention applied → This module attends over neighbors ← THIS MODULE
4. Embeddings computed → Final node embeddings used for hyperedge prediction

Connection to Memory/Message Flow:
----------------------------------
- Memory (updated via messages) becomes part of node features: features = memory + raw_features
- Temporal attention uses these memory-enhanced features as queries
- Memory stores "what happened"; attention uses it to create task-specific embeddings

What is Query Time?
-------------------
Query time = timestamp of the hyperedge being predicted. During training:
- Each hyperedge in batch has its own query time (its timestamp)
- Only neighbors/interactions BEFORE query time are considered
- Prevents data leakage: future interactions don't influence past predictions
- Example: Hyperedge [50, 60, 70] at t=500 → query time = 500, only use neighbors before 500

Key Difference from Regular GAT:
---------------------------------
- Regular GAT: Attention over all spatial neighbors (no time consideration)
- Temporal Attention: Attention over TEMPORAL neighbors (only before query time)
- Includes time encodings in query and key
- Recursive: multi-layer (each layer attends over neighbors from previous layer)

Running Example:
---------------
Setup: Node 50 at query time 500, memory_dim=172, time_dim=100, n_neighbors=20

Step 1: Get node features (includes memory)
  src_features = memory[50] + raw_features = [0.2, 0.5, ..., 0.3] (172 dims)
  src_time_encoding = time_encoder(0) = [1.0, 0.99, ...] (100 dims)  # Query node at query time
  query = [src_features(172) + time(100)] = [272 dims]

Step 2: Get temporal neighbors (before time 500)
  Sample 20 most recent hyperedges: [H1(t=100), H2(t=200), ..., H20(t=1000)]
  For each hyperedge, extract nodes (up to 20 nodes per hyperedge, randomly sampled if larger)
  neighbors = [60, 70, 80, ..., 90, 100, ...] (up to 20² = 400 neighbors total)
  neighbor_features = [memory[60] + raw[60], memory[70] + raw[70], ...] (up to 400 × 172 dims)
    * Each neighbor's features = memory + raw features (element-wise addition)
    * Then processed through n_layers-1 temporal attention layers (recursive)
  hyperedge_features = [hyperedge_feat_1, hyperedge_feat_1, ..., hyperedge_feat_2, ...] (up to 400 × 10 dims)
    * Repeated for nodes from same hyperedge
  neighbor_time_encodings = [time_enc(500-100), time_enc(500-100), ..., time_enc(500-200), ...] (up to 400 × 100 dims)
    * Same timestamp for nodes from same hyperedge
  key = [neighbor_features(172) + hyperedge(10) + time(100)] = [up to 400 × 282 dims]

Step 3: Multi-head attention
  Attention(Q, K, V) = softmax(QK^T / √d_k) V
  
  Formula Breakdown:
  - Q (Query): Source node representation [1, batch_size, query_dim]
    * What we're looking for: src_features + src_time_encoding
    * Shape: [1, 32, 272] (32 nodes, 272-dim query)
  
  - K (Key): Neighbor representations [n_neighbors², batch_size, key_dim]
    * What each neighbor offers: neighbor_features + hyperedge_features + neighbor_time_encoding
    * Shape: [400, 32, 282] (up to 400 neighbors from 20 hyperedges, 32 nodes, 282-dim key)
  
  - V (Value): Same as K in this implementation
    * The actual information from each neighbor
    * Shape: [400, 32, 282]
  
  - QK^T: Matrix multiplication computes similarity between query and each key
    * QK^T = [1, 32, 272] @ [282, 32, 400] = [1, 32, 400]
    * Each entry [0, i, j] = similarity between source node i and neighbor j
  
  - √d_k: Scaling factor (square root of key dimension)
    * Prevents large dot products from dominating softmax
    * d_k = 282, so √d_k ≈ 16.79
  
  - softmax(...): Normalizes similarities to probabilities (attention weights)
    * Applied along neighbor dimension (masked for padding)
    * Each row sums to 1 (over valid neighbors only): weights[i, :] = [0.1, 0.3, 0.05, ..., 0.02, 0, 0, ...]
  
  - Multiply by V: Weighted sum of neighbor values
    * softmax(...) @ V = [1, 32, 400] @ [400, 32, 282] = [1, 32, 272]
    * Note: Output dimension matches query_dim (272), not value_dim (282)
    * Final output: attended representation for each source node
  
  Result:
  - Attention weights: [32, 1, 400] (how much each neighbor matters, with padding masked)
  - Attended output: [32, 272] (weighted combination of neighbors, matches query_dim)

Step 4: Merge with source (skip connection)
  - Concatenate: [attn_output(272) ; src_features(172)] = [32, 444]
  - MLP Layer 1: [444] → [172] (hidden dimension)
  - MLP Layer 2: [172] → [172] (output_dimension)
  - Final embedding: [32, 172]

Input/Output:
------------
Input:
  - src_node_features: [batch, 172] - Source node features (includes memory)
  - src_time_features: [batch, 1, 100] - Time encoding for source (query time)
  - neighbors_features: [batch, n_neighbors², 172] - Neighbor features (includes their memories)
    * Shape is n_neighbors² because we sample n_neighbors hyperedges, each with up to n_neighbors nodes
  - neighbors_time_features: [batch, n_neighbors², 100] - Time encodings for neighbors
  - edge_features: [batch, n_neighbors², 10] - Hyperedge features (features of the interaction/hyperedge)
  - neighbors_padding_mask: [batch, n_neighbors²] - Mask for padding (True = invalid)

Output:
  - attn_output: [batch, 172] - Final embedding (memory + attended neighbors)
  - attn_output_weights: [batch, n_neighbors²] - Attention weights (which neighbors matter)

What are Q, K, V in the Code?
------------------------------
- Q (Query): Line 192 - Concatenation of source node features + source time encoding
  * source_node_features: [batch, 172] - Memory + raw features (element-wise addition)
  * src_time_features: [batch, 1, 100] - Time encoding for query time (always time_encoder(0))
  * query = [src_features(172) + time(100)] = [batch, 1, 272]

- K (Key): Line 193 - Concatenation of neighbor features + hyperedge features + neighbor time encodings
  * neighbors_features: [batch, n_neighbors², 172] - Neighbor memory + raw features (processed through n_layers-1)
  * edge_features: [batch, n_neighbors², 10] - Hyperedge features (repeated for nodes from same hyperedge)
  * neighbors_time_features: [batch, n_neighbors², 100] - Time encoding of (query_time - interaction_time)
  * key = [neighbor_features(172) + hyperedge(10) + time(100)] = [batch, n_neighbors², 282]

- V (Value): Line 210 - Same as K (value=key)
  * Represents the information content from each neighbor
  * After attention weights are computed, V is weighted and summed

Key Points:
-----------
- Query time is per-hyperedge (not per-batch) - each hyperedge uses its own timestamp
- Negative hyperedges use same query time as corresponding positive hyperedge
- Memory-enhanced features are used as queries (memory from message/memory flow)
- Memory + raw features uses element-wise addition (not concatenation) - keeps dimension constant
- Neighbor features also include memory + raw (processed through recursive temporal attention)
- Neighbor time encoding = time_encoder(query_time - interaction_time) - "how long ago"
- Temporal neighbors are filtered by query time (only before query time)
- Recursive: used in multi-layer fashion (neighbors computed from previous layer)
- Neighbors come from clique structure: hyperedges converted to pairwise connections via get_neighbor_finder
"""

import torch
from torch import nn

from utils.utils import ConcatMergeLayer


class TemporalAttentionLayer(torch.nn.Module):
  """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the hyperedge/interaction timestamps.
  """

  def __init__(self, n_node_features, n_neighbors_features, n_edge_features, time_dim,
               output_dimension, n_head=2,
               dropout=0.1):
    super(TemporalAttentionLayer, self).__init__()

    self.n_head = n_head

    self.feat_dim = n_node_features
    self.time_dim = time_dim

    self.query_dim = n_node_features + time_dim
    self.key_dim = n_neighbors_features + time_dim + n_edge_features

    self.merger = ConcatMergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=n_head,
                                                   dropout=dropout)

  def forward(self, src_node_features, src_time_features, neighbors_features,
              neighbors_time_features, edge_features, neighbors_padding_mask):
    """
    Temporal attention model for hypergraphs.
    
    :param src_node_features: float Tensor of shape [batch_size, n_node_features]
      Source node features (includes memory + raw features)
    :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
      Time encoding for source node at query time
    :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
      Neighbor node features (includes their memories, processed through recursive layers)
    :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors, time_dim]
      Time encodings for neighbors (query_time - interaction_time)
    :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
      Hyperedge features (features of the interaction/hyperedge between source and each neighbor)
      Note: In THGN, these are hyperedge features, but parameter name kept as edge_features
      for compatibility with calling code.
    :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
      Mask indicating invalid neighbors (True = padding/invalid, False = valid)
    :return:
    attn_output: float Tensor of shape [batch_size, output_dimension]
      Final node embedding after temporal attention
    attn_output_weights: float Tensor of shape [batch_size, n_neighbors]
      Attention weights showing which neighbors matter most
    """

    src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)

    query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
    key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)

    # Reshape tensors to expected shape by multi head attention
    query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
    key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

    # Compute mask of which source nodes have no valid neighbors
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
    # force the attention to just 'attend' on this neighbor (which has the same features as all
    # the others since they are fake neighbors) and will produce an equivalent result to the
    # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                              key_padding_mask=neighbors_padding_mask)

    attn_output = attn_output.squeeze()
    attn_output_weights = attn_output_weights.squeeze()

    # Source nodes with no neighbors have an all zero attention output. The attention output is
    # then added or concatenated to the original source node features and then fed into an MLP.
    # This means that an all zero vector is not used.
    attn_output = attn_output.masked_fill(invalid_neighborhood_mask, 0)
    attn_output_weights = attn_output_weights.masked_fill(invalid_neighborhood_mask, 0)

    # Skip connection with temporal attention over neighborhood and the features of the node itself
    attn_output = self.merger(attn_output, src_node_features)

    return attn_output, attn_output_weights
