"""
Embedding Module for TGN
========================

Purpose: Computes final node embeddings for link prediction using recursive temporal graph
convolution. Builds multi-hop temporal neighborhoods by recursively attending over neighbors
at different layers. Orchestrates the recursive structure and delegates attention computation
to TemporalAttentionLayer.

Role in TGN Flow:
-----------------
1. Memory updated → Node memories reflect recent interactions (via message/memory flow)
2. Temporal neighbors found → For each node, find neighbors before query time
3. Recursive embedding computation → This module recursively computes embeddings ← THIS MODULE
4. Temporal attention applied → TemporalAttentionLayer attends over neighbors (called by this module)
5. Final embeddings → Used for link prediction

Key Concept - Recursive Structure:
----------------------------------
To compute embedding at layer N:
  - Recursively compute embedding at layer N-1 for source node
  - Find temporal neighbors (before query time)
  - Recursively compute embeddings at layer N-1 for each neighbor
  - Apply temporal attention at layer N over neighbors (which already have layer N-1 embeddings)

Base case (n_layers=0): Returns memory + raw_features (no neighbors, no attention)

This builds multi-hop temporal neighborhoods: each layer attends over neighbors that already
aggregated their own neighborhoods from previous layers.

Connection to TemporalAttentionLayer:
-------------------------------------
- This module orchestrates the recursive structure and data preparation
- TemporalAttentionLayer performs the actual attention computation
- GraphAttentionEmbedding creates TemporalAttentionLayer instances (one per layer)
- aggregate() method calls TemporalAttentionLayer.forward() with prepared data

Implementations:
----------------
1. GraphAttentionEmbedding: Uses TemporalAttentionLayer (learned attention weights)
2. GraphSumEmbedding: Simple sum aggregation (no attention)
3. IdentityEmbedding: Returns memory only (no graph structure)
4. TimeEmbedding: Time-scaled memory (memory × (1 + MLP(time_delta)))

2-Layer Running Example (n_layers=2):
-------------------------------------
Setup: Node 50 at query time 500, memory_dim=172, time_dim=100, n_neighbors=3

LAYER 2 (Top Level):
  Step 1: Recursively compute source node 50 at layer 1
    → Calls compute_embedding(50, t=500, n_layers=1) [goes to LAYER 1]
  
  Step 2: Find temporal neighbors of node 50 (before time 500)
    → neighbors = [60, 70, 80] (3 neighbors)
    → edge_times = [300, 400, 450] (when interactions occurred)
    → edge_deltas = [500-300, 500-400, 500-450] = [200, 100, 50]
  
  Step 3: Recursively compute neighbor embeddings at layer 1
    → For neighbor 60: compute_embedding(60, t=500, n_layers=1) [goes to LAYER 1]
    → For neighbor 70: compute_embedding(70, t=500, n_layers=1) [goes to LAYER 1]
    → For neighbor 80: compute_embedding(80, t=500, n_layers=1) [goes to LAYER 1]
  
  Step 4: Apply temporal attention at layer 2
    → TemporalAttentionLayer.forward() called with:
      - src_features = layer1_embedding[50] (from Step 1)
      - neighbors = [layer1_embedding[60], layer1_embedding[70], layer1_embedding[80]] (from Step 3)
      - edge_features, time_encodings, etc.
    → Output: layer2_embedding[50] = [172 dims]

LAYER 1 (Recursive):
  For source node 50:
    Step 1: Base case (n_layers=0)
      → source_features = memory[50] + raw[50] = [172 dims]
      → Return (this is layer1_embedding[50])
  
  For neighbor 60:
    Step 1: Find neighbors of 60 (before time 500)
      → neighbors = [90, 100] (2-hop neighbors of node 50)
      → edge_times = [200, 250]
      → edge_deltas = [500-200, 500-250] = [300, 250]
    
    Step 2: Compute base embeddings for neighbors 90, 100
      → base[90] = memory[90] + raw[90] = [172 dims]
      → base[100] = memory[100] + raw[100] = [172 dims]
    
    Step 3: Apply temporal attention at layer 1 for neighbor 60
      → TemporalAttentionLayer.forward() called with:
        - src_features = memory[60] + raw[60]
        - neighbors = [base[90], base[100]]
        - edge_features, time_encodings, etc.
      → Output: layer1_embedding[60] = [172 dims]
  
  Similar process for neighbors 70 and 80...

Final Result:
  - layer2_embedding[50] incorporates:
    - Direct neighbors: 60, 70, 80 (1-hop)
    - Neighbors of neighbors: 90, 100, ... (2-hop)
    - All weighted by temporal attention at each layer

Key Points:
-----------
- Recursion builds multi-hop temporal neighborhoods (n_layers = max hop distance)
- Each layer uses neighbors that already have embeddings from previous layers
- Temporal neighbors are from entire training graph (not just current batch)
- Graph can be multigraph (multiple edges between same nodes at different times)
- Query time is per-edge (timestamp of edge being predicted)
- Memory + raw features uses element-wise addition (not concatenation)
"""

import torch
from torch import nn
import numpy as np
import math

from model.temporal_attention import TemporalAttentionLayer


class EmbeddingModule(nn.Module):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               dropout):
    super(EmbeddingModule, self).__init__()
    self.node_features = node_features
    self.edge_features = edge_features
    # self.memory = memory
    self.neighbor_finder = neighbor_finder
    self.time_encoder = time_encoder
    self.n_layers = n_layers
    self.n_node_features = n_node_features
    self.n_edge_features = n_edge_features
    self.n_time_features = n_time_features
    self.dropout = dropout
    self.embedding_dimension = embedding_dimension
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return NotImplemented


class IdentityEmbedding(EmbeddingModule):
  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    return memory[source_nodes, :]


class TimeEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True, n_neighbors=1):
    super(TimeEmbedding, self).__init__(node_features, edge_features, memory,
                                        neighbor_finder, time_encoder, n_layers,
                                        n_node_features, n_edge_features, n_time_features,
                                        embedding_dimension, device, dropout)

    class NormalLinear(nn.Linear):
      # From Jodie code
      def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.normal_(0, stdv)
        if self.bias is not None:
          self.bias.data.normal_(0, stdv)

    self.embedding_layer = NormalLinear(1, self.n_node_features)

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    source_embeddings = memory[source_nodes, :] * (1 + self.embedding_layer(time_diffs.unsqueeze(1)))

    return source_embeddings


class GraphEmbedding(EmbeddingModule):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphEmbedding, self).__init__(node_features, edge_features, memory,
                                         neighbor_finder, time_encoder, n_layers,
                                         n_node_features, n_edge_features, n_time_features,
                                         embedding_dimension, device, dropout)

    self.use_memory = use_memory
    self.device = device

  def compute_embedding(self, memory, source_nodes, timestamps, n_layers, n_neighbors=20, time_diffs=None,
                        use_time_proj=True):
    """Recursive implementation of curr_layers temporal graph attention layers.

    src_idx_l [batch_size]: users / items input ids.
    cut_time_l [batch_size]: scalar representing the instant of the time where we want to extract the user / item representation.
    curr_layers [scalar]: number of temporal convolutional layers to stack.
    num_neighbors [scalar]: number of temporal neighbor to consider in each convolutional layer.
    """

    assert (n_layers >= 0)

    source_nodes_torch = torch.from_numpy(source_nodes).long().to(self.device)
    timestamps_torch = torch.unsqueeze(torch.from_numpy(timestamps).float().to(self.device), dim=1)

    # query node always has the start time -> time span == 0
    source_nodes_time_embedding = self.time_encoder(torch.zeros_like(
      timestamps_torch))

    source_node_features = self.node_features[source_nodes_torch, :]

    if self.use_memory:
      source_node_features = memory[source_nodes, :] + source_node_features

    if n_layers == 0:
      return source_node_features
    else:

      source_node_conv_embeddings = self.compute_embedding(memory,
                                                           source_nodes,
                                                           timestamps,
                                                           n_layers=n_layers - 1,
                                                           n_neighbors=n_neighbors)

      neighbors, edge_idxs, edge_times = self.neighbor_finder.get_temporal_neighbor(
        source_nodes,
        timestamps,
        n_neighbors=n_neighbors)

      neighbors_torch = torch.from_numpy(neighbors).long().to(self.device)

      edge_idxs = torch.from_numpy(edge_idxs).long().to(self.device)

      edge_deltas = timestamps[:, np.newaxis] - edge_times

      edge_deltas_torch = torch.from_numpy(edge_deltas).float().to(self.device)

      neighbors = neighbors.flatten()
      neighbor_embeddings = self.compute_embedding(memory,
                                                   neighbors,
                                                   np.repeat(timestamps, n_neighbors),
                                                   n_layers=n_layers - 1,
                                                   n_neighbors=n_neighbors)

      effective_n_neighbors = n_neighbors if n_neighbors > 0 else 1
      neighbor_embeddings = neighbor_embeddings.view(len(source_nodes), effective_n_neighbors, -1)
      edge_time_embeddings = self.time_encoder(edge_deltas_torch)

      edge_features = self.edge_features[edge_idxs, :]

      mask = neighbors_torch == 0

      source_embedding = self.aggregate(n_layers, source_node_conv_embeddings,
                                        source_nodes_time_embedding,
                                        neighbor_embeddings,
                                        edge_time_embeddings,
                                        edge_features,
                                        mask)

      return source_embedding

  def aggregate(self, n_layers, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    return NotImplemented


class GraphSumEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphSumEmbedding, self).__init__(node_features=node_features,
                                            edge_features=edge_features,
                                            memory=memory,
                                            neighbor_finder=neighbor_finder,
                                            time_encoder=time_encoder, n_layers=n_layers,
                                            n_node_features=n_node_features,
                                            n_edge_features=n_edge_features,
                                            n_time_features=n_time_features,
                                            embedding_dimension=embedding_dimension,
                                            device=device,
                                            n_heads=n_heads, dropout=dropout,
                                            use_memory=use_memory)
    self.linear_1 = torch.nn.ModuleList([torch.nn.Linear(embedding_dimension + n_time_features +
                                                         n_edge_features, embedding_dimension)
                                         for _ in range(n_layers)])
    self.linear_2 = torch.nn.ModuleList(
      [torch.nn.Linear(embedding_dimension + n_node_features + n_time_features,
                       embedding_dimension) for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    neighbors_features = torch.cat([neighbor_embeddings, edge_time_embeddings, edge_features],
                                   dim=2)
    neighbor_embeddings = self.linear_1[n_layer - 1](neighbors_features)
    neighbors_sum = torch.nn.functional.relu(torch.sum(neighbor_embeddings, dim=1))

    source_features = torch.cat([source_node_features,
                                 source_nodes_time_embedding.squeeze()], dim=1)
    source_embedding = torch.cat([neighbors_sum, source_features], dim=1)
    source_embedding = self.linear_2[n_layer - 1](source_embedding)

    return source_embedding


class GraphAttentionEmbedding(GraphEmbedding):
  def __init__(self, node_features, edge_features, memory, neighbor_finder, time_encoder, n_layers,
               n_node_features, n_edge_features, n_time_features, embedding_dimension, device,
               n_heads=2, dropout=0.1, use_memory=True):
    super(GraphAttentionEmbedding, self).__init__(node_features, edge_features, memory,
                                                  neighbor_finder, time_encoder, n_layers,
                                                  n_node_features, n_edge_features,
                                                  n_time_features,
                                                  embedding_dimension, device,
                                                  n_heads, dropout,
                                                  use_memory)

    self.attention_models = torch.nn.ModuleList([TemporalAttentionLayer(
      n_node_features=n_node_features,
      n_neighbors_features=n_node_features,
      n_edge_features=n_edge_features,
      time_dim=n_time_features,
      n_head=n_heads,
      dropout=dropout,
      output_dimension=n_node_features)
      for _ in range(n_layers)])

  def aggregate(self, n_layer, source_node_features, source_nodes_time_embedding,
                neighbor_embeddings,
                edge_time_embeddings, edge_features, mask):
    attention_model = self.attention_models[n_layer - 1]

    source_embedding, _ = attention_model(source_node_features,
                                          source_nodes_time_embedding,
                                          neighbor_embeddings,
                                          edge_time_embeddings,
                                          edge_features,
                                          mask)

    return source_embedding


def get_embedding_module(module_type, node_features, edge_features, memory, neighbor_finder,
                         time_encoder, n_layers, n_node_features, n_edge_features, n_time_features,
                         embedding_dimension, device,
                         n_heads=2, dropout=0.1, n_neighbors=None,
                         use_memory=True):
  if module_type == "graph_attention":
    return GraphAttentionEmbedding(node_features=node_features,
                                    edge_features=edge_features,
                                    memory=memory,
                                    neighbor_finder=neighbor_finder,
                                    time_encoder=time_encoder,
                                    n_layers=n_layers,
                                    n_node_features=n_node_features,
                                    n_edge_features=n_edge_features,
                                    n_time_features=n_time_features,
                                    embedding_dimension=embedding_dimension,
                                    device=device,
                                    n_heads=n_heads, dropout=dropout, use_memory=use_memory)
  elif module_type == "graph_sum":
    return GraphSumEmbedding(node_features=node_features,
                              edge_features=edge_features,
                              memory=memory,
                              neighbor_finder=neighbor_finder,
                              time_encoder=time_encoder,
                              n_layers=n_layers,
                              n_node_features=n_node_features,
                              n_edge_features=n_edge_features,
                              n_time_features=n_time_features,
                              embedding_dimension=embedding_dimension,
                              device=device,
                              n_heads=n_heads, dropout=dropout, use_memory=use_memory)

  elif module_type == "identity":
    return IdentityEmbedding(node_features=node_features,
                             edge_features=edge_features,
                             memory=memory,
                             neighbor_finder=neighbor_finder,
                             time_encoder=time_encoder,
                             n_layers=n_layers,
                             n_node_features=n_node_features,
                             n_edge_features=n_edge_features,
                             n_time_features=n_time_features,
                             embedding_dimension=embedding_dimension,
                             device=device,
                             dropout=dropout)
  elif module_type == "time":
    return TimeEmbedding(node_features=node_features,
                         edge_features=edge_features,
                         memory=memory,
                         neighbor_finder=neighbor_finder,
                         time_encoder=time_encoder,
                         n_layers=n_layers,
                         n_node_features=n_node_features,
                         n_edge_features=n_edge_features,
                         n_time_features=n_time_features,
                         embedding_dimension=embedding_dimension,
                         device=device,
                         dropout=dropout,
                         n_neighbors=n_neighbors)
  else:
    raise ValueError("Embedding Module {} not supported".format(module_type))


