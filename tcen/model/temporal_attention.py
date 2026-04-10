"""
Temporal Attention Layer for TGN
=================================

Purpose: Computes node embeddings by attending over temporal neighbors. Uses multi-head attention
to weight neighbors based on node features, edge features, and time encodings, producing
context-aware embeddings for link prediction.

Role in TGN Flow:
-----------------
1. Memory updated → Node memories reflect recent interactions (via message/memory flow)
2. Temporal neighbors found → For each node, find neighbors before query time
3. Temporal attention applied → This module attends over neighbors ← THIS MODULE
4. Embeddings computed → Final node embeddings used for link prediction

Connection to Memory/Message Flow:
----------------------------------
- Memory (updated via messages) becomes part of node features: features = memory + raw_features
- Temporal attention uses these memory-enhanced features as queries
- Memory stores "what happened"; attention uses it to create task-specific embeddings

What is Query Time?
-------------------
Query time = timestamp of the edge being predicted. During training:
- Each edge in batch has its own query time (its timestamp)
- Only neighbors/interactions BEFORE query time are considered
- Prevents data leakage: future interactions don't influence past predictions
- Example: Edge (50, 60, t=500) → query time = 500, only use neighbors before 500

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
  neighbors = [60, 70, 80, ...] (20 neighbors)
  neighbor_features = [memory[60] + raw[60], memory[70] + raw[70], ...] (20 × 172 dims)
    * Each neighbor's features = memory + raw features (element-wise addition)
    * Then processed through n_layers-1 temporal attention layers (recursive)
  edge_features = [edge_feat_1, edge_feat_2, ...] (20 × 10 dims)
  neighbor_time_encodings = [time_enc(500-100), time_enc(500-200), ...] (20 × 100 dims)
  key = [neighbor_features(172) + edge(10) + time(100)] = [20 × 282 dims]

Step 3: Multi-head attention
  Attention(Q, K, V) = softmax(QK^T / √d_k) V
  
  Formula Breakdown:
  - Q (Query): Source node representation [1, batch_size, query_dim]
    * What we're looking for: src_features + src_time_encoding
    * Shape: [1, 32, 272] (32 nodes, 272-dim query)
  
  - K (Key): Neighbor representations [n_neighbors, batch_size, key_dim]
    * What each neighbor offers: neighbor_features + edge_features + neighbor_time_encoding
    * Shape: [20, 32, 282] (20 neighbors, 32 nodes, 282-dim key)
  
  - V (Value): Same as K in this implementation
    * The actual information from each neighbor
    * Shape: [20, 32, 282]
  
  - QK^T: Matrix multiplication computes similarity between query and each key
    * QK^T = [1, 32, 272] @ [282, 32, 20] = [1, 32, 20]
    * Each entry [0, i, j] = similarity between source node i and neighbor j
  
  - √d_k: Scaling factor (square root of key dimension)
    * Prevents large dot products from dominating softmax
    * d_k = 282, so √d_k ≈ 16.79
  
  - softmax(...): Normalizes similarities to probabilities (attention weights)
    * Applied along neighbor dimension
    * Each row sums to 1: weights[i, :] = [0.1, 0.3, 0.05, ..., 0.02]
  
  - Multiply by V: Weighted sum of neighbor values
    * softmax(...) @ V = [1, 32, 20] @ [20, 32, 282] = [1, 32, 272]
    * Note: Output dimension matches query_dim (272), not value_dim (282)
    * Final output: attended representation for each source node
  
  Result:
  - Attention weights: [32, 1, 20] (how much each neighbor matters)
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
  - neighbors_features: [batch, 20, 172] - Neighbor features (includes their memories)
  - neighbors_time_features: [batch, 20, 100] - Time encodings for neighbors
  - edge_features: [batch, 20, 10] - Edge features
  - neighbors_padding_mask: [batch, 20] - Mask for padding (True = invalid)

Output:
  - attn_output: [batch, 172] - Final embedding (memory + attended neighbors)
  - attn_output_weights: [batch, 20] - Attention weights (which neighbors matter)

What are Q, K, V in the Code?
------------------------------
- Q (Query): Line 135 - Concatenation of source node features + source time encoding
  * source_node_features: [batch, 172] - Memory + raw features (element-wise addition)
  * src_time_features: [batch, 1, 100] - Time encoding for query time (always time_encoder(0))
  * query = [src_features(172) + time(100)] = [batch, 1, 272]

- K (Key): Line 136 - Concatenation of neighbor features + edge features + neighbor time encodings
  * neighbors_features: [batch, 20, 172] - Neighbor memory + raw features (processed through n_layers-1)
  * edge_features: [batch, 20, 10] - Edge features between source and each neighbor
  * neighbors_time_features: [batch, 20, 100] - Time encoding of (query_time - interaction_time)
  * key = [neighbor_features(172) + edge(10) + time(100)] = [batch, 20, 282]

- V (Value): Line 153 - Same as K (value=key)
  * Represents the information content from each neighbor
  * After attention weights are computed, V is weighted and summed

Key Points:
-----------
- Query time is per-edge (not per-batch) - each edge uses its own timestamp
- Negative edges use same query time as corresponding positive edge
- Memory-enhanced features are used as queries (memory from message/memory flow)
- Memory + raw features uses element-wise addition (not concatenation) - keeps dimension constant
- Neighbor features also include memory + raw (processed through recursive temporal attention)
- Neighbor time encoding = time_encoder(query_time - interaction_time) - "how long ago"
- Temporal neighbors are filtered by query time (only before query time)
- Recursive: used in multi-layer fashion (neighbors computed from previous layer)
"""

import torch
from torch import nn

from utils.utils import MergeLayer


class TemporalAttentionLayer(torch.nn.Module):
  """
  Temporal attention layer. Return the temporal embedding of a node given the node itself,
   its neighbors and the edge timestamps.
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

    self.merger = MergeLayer(self.query_dim, n_node_features, n_node_features, output_dimension)

    self.multi_head_target = nn.MultiheadAttention(embed_dim=self.query_dim,
                                                   kdim=self.key_dim,
                                                   vdim=self.key_dim,
                                                   num_heads=n_head,
                                                   dropout=dropout)

  def forward(self, src_node_features, src_time_features, neighbors_features,
              neighbors_time_features, edge_features, neighbors_padding_mask):
    """
    "Temporal attention model
    :param src_node_features: float Tensor of shape [batch_size, n_node_features]
    :param src_time_features: float Tensor of shape [batch_size, 1, time_dim]
    :param neighbors_features: float Tensor of shape [batch_size, n_neighbors, n_node_features]
    :param neighbors_time_features: float Tensor of shape [batch_size, n_neighbors,
    time_dim]
    :param edge_features: float Tensor of shape [batch_size, n_neighbors, n_edge_features]
    :param neighbors_padding_mask: float Tensor of shape [batch_size, n_neighbors]
    :return:
    attn_output: float Tensor of shape [1, batch_size, n_node_features]
    attn_output_weights: [batch_size, 1, n_neighbors]
    """

    src_node_features_unrolled = torch.unsqueeze(src_node_features, dim=1)

    query = torch.cat([src_node_features_unrolled, src_time_features], dim=2)
    key = torch.cat([neighbors_features, edge_features, neighbors_time_features], dim=2)

    # print(neighbors_features.shape, edge_features.shape, neighbors_time_features.shape)
    # Reshape tensors so to expected shape by multi head attention
    query = query.permute([1, 0, 2])  # [1, batch_size, num_of_features]
    key = key.permute([1, 0, 2])  # [n_neighbors, batch_size, num_of_features]

    # Compute mask of which source nodes have no valid neighbors
    invalid_neighborhood_mask = neighbors_padding_mask.all(dim=1, keepdim=True)
    # If a source node has no valid neighbor, set it's first neighbor to be valid. This will
    # force the attention to just 'attend' on this neighbor (which has the same features as all
    # the others since they are fake neighbors) and will produce an equivalent result to the
    # original tgat paper which was forcing fake neighbors to all have same attention of 1e-10
    neighbors_padding_mask[invalid_neighborhood_mask.squeeze(), 0] = False

    # print(query.shape, key.shape)

    attn_output, attn_output_weights = self.multi_head_target(query=query, key=key, value=key,
                                                              key_padding_mask=neighbors_padding_mask)

    # mask = torch.unsqueeze(neighbors_padding_mask, dim=2)  # mask [B, N, 1]
    # mask = mask.permute([0, 2, 1])
    # attn_output, attn_output_weights = self.multi_head_target(q=query, k=key, v=key,
    #                                                           mask=mask)

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
