"""
THGN (Temporal Hypergraph Network) Main Model
==============================================

Purpose: Main orchestrator class that combines all THGN components to perform temporal hyperedge
prediction. Manages memory updates, embedding computation, and hyperedge probability prediction.
This is the top-level class that coordinates Memory, MessageAggregator, MessageFunction,
MemoryUpdater, and EmbeddingModule.

Role in THGN Flow:
------------------
1. Initialization → Sets up all components (memory, message functions, embedding module, etc.)
2. Training/Inference → Receives batch of hyperedges with timestamps
3. Memory update → Updates node memories based on hyperedge interactions (message flow)
4. Embedding computation → Computes temporal embeddings using recursive attention
5. Hyperedge prediction → Scores hyperedge probabilities using embeddings

Key Components:
----------------
- Memory: Stores evolving per-node state vectors
- MessageAggregator: Combines multiple messages per node (last/mean)
- MessageFunction: Transforms raw messages (MLP/identity)
- MemoryUpdater: Updates memory using GRU/RNN
- EmbeddingModule: Computes final embeddings via recursive temporal attention
- MergeLayer: MLP decoder for hyperedge prediction (mean pooling + MLP)

Complete Training Flow with Running Example:
---------------------------------------------
Batch: [([50, 60, 70], t=500, idx=0), ([80, 90], t=600, idx=1)]
Negative samples: [[23, 45], [67, 89]] (for hyperedges 0 and 1), same time step

Step 1: Memory Update (if memory_update_at_start=True)
  - Update memory from previous batches' stored messages
  - Memory now reflects all interactions before time 500

Step 2: Compute Embeddings
  - All hyperedges use SAME memory state (from before batch)
  - Hyperedge ([50, 60, 70], t=500): query_time = 500, uses memory before 500
  - Hyperedge ([80, 90], t=600): query_time = 600, uses memory before 600
  - Note: Hyperedge at 600 doesn't see hyperedge at 500 from same batch (temporal consistency)
  - Computes embeddings for: [50, 60, 70, 80, 90, 23, 45, 67, 89] (all nodes in hyperedges and negatives)
  - Returns: embeddings for all unique nodes

Step 3: Hyperedge Prediction
  - Positive hyperedges: score([50, 60, 70]), score([80, 90])
  - Negative hyperedges: score([23, 45]), score([67, 89])
  - Uses MergeLayer: mean_pool([emb_50, emb_60, emb_70]) → [172] → MLP → [1] (score)
  - Applies sigmoid → probabilities

Step 4: Memory Update (after embeddings)
  - Create raw messages for current batch:
    * For node 50: message = [mem[50], mean([mem[60], mem[70]]), hyperedge_feat, time_enc(500-last_update[50])]
    * For node 60: message = [mem[60], mean([mem[50], mem[70]]), hyperedge_feat, time_enc(500-last_update[60])]
    * For node 70: message = [mem[70], mean([mem[50], mem[60]]), hyperedge_feat, time_enc(500-last_update[70])]
    * Similar for nodes 80, 90
  - Store messages (if memory_update_at_start=True) or update immediately
  - Memory now includes interactions from this batch (for next batch)

Key Methods:
------------
- compute_temporal_embeddings(): Computes embeddings for all nodes in hyperedges and negatives
- compute_hyperedge_probabilities(): Main entry point - predicts hyperedge probabilities
- update_memory(): Updates memory via message flow (aggregate → process → update)
- get_raw_messages(): Creates raw messages from hyperedge interactions
- get_updated_memory(): Non-destructive memory update (returns copy)

Important Points:
-----------------
- Within-batch hyperedges don't influence each other: All hyperedges in batch use same memory state
  (from before batch). This ensures temporal consistency but means earlier hyperedges in batch
  don't affect later hyperedges' embeddings.
- Query time is per-hyperedge: Each hyperedge uses its own timestamp as query time
- Memory update timing: If memory_update_at_start=True, memory is updated from previous
  batches BEFORE computing embeddings, then current batch messages are stored for next batch
- Negative hyperedges use same query time as corresponding positive hyperedge
- Embeddings are computed on-demand (not stored) - computed fresh for each prediction
- The model can handle multigraphs (multiple hyperedges between same nodes at different times)
- Hyperedge sizes are variable: Each hyperedge can have any number of nodes (≥ 1)
- Raw message dimension: memory_dim + aggregated_other_nodes_dim + hyperedge_feat_dim + time_dim
  where aggregated_other_nodes_dim = memory_dim (if using mean pooling for other nodes)
"""

import logging
import numpy as np
import torch
from collections import defaultdict

from utils.utils import MergeLayer
from modules.memory import Memory
from modules.message_aggregator import get_message_aggregator
from modules.message_function import get_message_function
from modules.memory_updater import get_memory_updater
from modules.embedding_module import get_embedding_module
from model.time_encoding import TimeEncode


class THGN(torch.nn.Module):
  def __init__(self, neighbor_finder, node_features, edge_features, device, n_layers=2,
               n_heads=2, dropout=0.1, use_memory=False,
               memory_update_at_start=True, message_dimension=100,
               memory_dimension=500, embedding_module_type="graph_attention",
               message_function="mlp",
               mean_time_shift=0, std_time_shift=1, n_neighbors=None, aggregator_type="last",
               memory_updater_type="gru",
               use_node_embedding_in_message=False,
               dyrep=False):
    """
    Initialize THGN model.
    
    Parameters:
    -----------
    neighbor_finder: NeighborFinder
      Finds temporal neighbors for nodes (hyperedge-level sampling)
    node_features: np.ndarray [n_nodes, n_node_features]
      Raw node features
    edge_features: np.ndarray [n_hyperedges, n_edge_features]
      Raw hyperedge features (one per hyperedge)
    device: torch.device
      Device to run model on
    n_layers: int
      Number of recursive temporal attention layers
    n_heads: int
      Number of attention heads
    dropout: float
      Dropout probability
    use_memory: bool
      Whether to use memory module
    memory_update_at_start: bool
      If True, update memory from previous batches before computing embeddings
    message_dimension: int
      Dimension of processed messages (after MessageFunction)
    memory_dimension: int
      Dimension of node memory vectors
    embedding_module_type: str
      Type of embedding module ("graph_attention", "graph_sum", "identity", "time")
    message_function: str
      Type of message function ("mlp", "identity")
    mean_time_shift: float
      Mean time shift for normalization (unified, no source/destination distinction)
    std_time_shift: float
      Standard deviation of time shift for normalization
    n_neighbors: int
      Number of hyperedges to sample per node (results in up to n_neighbors² total neighbor nodes)
    aggregator_type: str
      Message aggregation strategy ("last", "mean")
    memory_updater_type: str
      Memory updater type ("gru", "rnn")
    use_node_embedding_in_message: bool
      If True, use computed embeddings in messages instead of memory
    dyrep: bool
      If True, use memory directly as embeddings (DyRep-style)
    """
    super(THGN, self).__init__()

    self.n_layers = n_layers
    self.neighbor_finder = neighbor_finder
    self.device = device
    self.logger = logging.getLogger(__name__)

    self.node_raw_features = torch.from_numpy(node_features.astype(np.float32)).to(device)
    self.edge_raw_features = torch.from_numpy(edge_features.astype(np.float32)).to(device)

    self.n_node_features = self.node_raw_features.shape[1]
    self.n_nodes = self.node_raw_features.shape[0]
    self.n_edge_features = self.edge_raw_features.shape[1]
    self.embedding_dimension = self.n_node_features
    self.n_neighbors = n_neighbors
    self.embedding_module_type = embedding_module_type
    self.use_node_embedding_in_message = use_node_embedding_in_message
    self.dyrep = dyrep

    self.use_memory = use_memory
    self.time_encoder = TimeEncode(dimension=self.n_node_features)
    self.memory = None

    # Unified time shift parameters (no source/destination distinction for hypergraphs)
    self.mean_time_shift = mean_time_shift
    self.std_time_shift = std_time_shift

    if self.use_memory:
      self.memory_dimension = memory_dimension
      self.memory_update_at_start = memory_update_at_start
      
      # Raw message dimension for THGN:
      # - node_memory: memory_dimension
      # - aggregated_other_nodes_memory: memory_dimension (if using mean pooling)
      # - hyperedge_features: n_edge_features
      # - time_encoding: time_encoder.dimension
      # Total: 2 * memory_dimension + n_edge_features + time_encoder.dimension
      # (Same formula as TGN, but interpretation is different: aggregated_other_nodes instead of single destination)
      raw_message_dimension = 2 * self.memory_dimension + self.n_edge_features + \
                              self.time_encoder.dimension
      message_dimension = message_dimension if message_function != "identity" else raw_message_dimension
      
      self.memory = Memory(n_nodes=self.n_nodes,
                           memory_dimension=self.memory_dimension,
                           input_dimension=message_dimension,
                           message_dimension=message_dimension,
                           device=device)
      self.message_aggregator = get_message_aggregator(aggregator_type=aggregator_type,
                                                       device=device)
      self.message_function = get_message_function(module_type=message_function,
                                                   raw_message_dimension=raw_message_dimension,
                                                   message_dimension=message_dimension)
      self.memory_updater = get_memory_updater(module_type=memory_updater_type,
                                               memory=self.memory,
                                               message_dimension=message_dimension,
                                               memory_dimension=self.memory_dimension,
                                               device=device)

    self.embedding_module_type = embedding_module_type

    self.embedding_module = get_embedding_module(module_type=embedding_module_type,
                                                 node_features=self.node_raw_features,
                                                 edge_features=self.edge_raw_features,
                                                 memory=self.memory,
                                                 neighbor_finder=self.neighbor_finder,
                                                 time_encoder=self.time_encoder,
                                                 n_layers=self.n_layers,
                                                 n_node_features=self.n_node_features,
                                                 n_edge_features=self.n_edge_features,
                                                 n_time_features=self.n_node_features,
                                                 embedding_dimension=self.embedding_dimension,
                                                 device=self.device,
                                                 n_heads=n_heads, dropout=dropout,
                                                 use_memory=use_memory,
                                                 n_neighbors=self.n_neighbors)

    # MergeLayer decoder for hyperedge prediction (mean pooling + MLP)
    # Takes variable-size sets of node embeddings and outputs scalar scores
    self.affinity_score = MergeLayer(self.n_node_features, self.n_node_features, 1)

  def compute_temporal_embeddings(self, hyperedges, negative_hyperedges, edge_times, edge_idxs, n_neighbors=20):
    """
    Compute temporal embeddings for all nodes in hyperedges and negative hyperedges.
    
    This method computes embeddings for each node occurrence (one per hyperedge), even if the same
    node appears in multiple hyperedges. Each occurrence uses its hyperedge's timestamp as the
    query time, resulting in potentially different embeddings for the same node at different times.
    
    Key Changes from TGN:
    ---------------------
    1. **Input format**: Takes hyperedges (list of lists) and negative_hyperedges (list of lists)
       instead of separate source_nodes, destination_nodes, negative_nodes arrays.
    
    2. **Node extraction**: Extracts all nodes from hyperedges and negatives, computing embeddings
       per occurrence (like TGN). If node 50 appears in 2 hyperedges with different timestamps,
       it gets 2 different embeddings.
    
    3. **Unified time shift**: Uses single mean_time_shift and std_time_shift (no source/destination
       distinction) since hypergraphs don't have directional edges.
    
    4. **Memory update**: Calls get_raw_messages() k times per hyperedge (once for each of the k nodes),
       similar to how TGN calls it twice per edge (once for source, once for destination). Each node
       in the hyperedge gets its own message that aggregates information from the other nodes in the
       hyperedge.
    
    5. **Return format**: Returns tensor of all embeddings + mappings to group by hyperedge, instead
       of three separate tensors (source, destination, negative).
    
    6. **Negative sampling**: In THGN, negative hyperedges are completely random (all nodes randomly
       sampled), unlike TGN where the source node is kept fixed and only the destination is randomly
       sampled. This is simpler and doesn't require designating an "anchor" node in hyperedges.
    
    Parameters:
    -----------
    hyperedges: list of lists
      Positive hyperedges in the batch. Each hyperedge is a list of node IDs.
      Example: [[50, 60, 70], [80, 90]] for batch_size=2
    negative_hyperedges: list of lists
      Negative hyperedges in the batch. Must have same sizes as corresponding positive hyperedges.
      Example: [[23, 45, 67], [12, 34]] for batch_size=2 (sizes [3, 2] match)
    edge_times: np.ndarray [batch_size]
      Timestamp of each hyperedge interaction
    edge_idxs: np.ndarray [batch_size]
      Index of each hyperedge (for accessing hyperedge features)
    n_neighbors: int
      Number of hyperedges to sample per node (results in up to n_neighbors² total neighbor nodes)
    
    Returns:
    --------
    node_embeddings: torch.Tensor [total_node_occurrences, embed_dim]
      Embeddings for all node occurrences. Each node in each hyperedge gets one embedding.
      If node 50 appears in 2 hyperedges, it appears twice in this tensor with potentially
      different embeddings (due to different query times).
    
    hyperedge_to_node_indices: dict {int: list of int}
      Maps each positive hyperedge index to the indices in node_embeddings tensor.
      Example: {0: [0, 1, 2], 1: [3, 4]} means hyperedge 0 uses embeddings at indices 0,1,2
    
    negative_hyperedge_to_node_indices: dict {int: list of int}
      Maps each negative hyperedge index to the indices in node_embeddings tensor.
      Example: {0: [5, 6, 7], 1: [8, 9]} means negative hyperedge 0 uses embeddings at indices 5,6,7
    
    Example:
    --------
    Input:
      hyperedges = [[50, 60, 70], [50, 80]]
      negative_hyperedges = [[23, 45, 67], [12, 34]]
      edge_times = [500, 600]
      edge_idxs = [0, 1]
    
    Processing:
      - Extract nodes: [50, 60, 70, 50, 80, 23, 45, 67, 12, 34] (10 occurrences)
      - Timestamps: [500, 500, 500, 600, 600, 500, 500, 500, 600, 600]
      - Compute embeddings for all 10 occurrences
    
    Return:
      node_embeddings: [10, embed_dim]
        - indices 0-2: nodes [50, 60, 70] from hyperedge 0 at t=500
        - indices 3-4: nodes [50, 80] from hyperedge 1 at t=600 (node 50 has different embedding!)
        - indices 5-7: nodes [23, 45, 67] from negative hyperedge 0 at t=500
        - indices 8-9: nodes [12, 34] from negative hyperedge 1 at t=600
      
      hyperedge_to_node_indices = {0: [0, 1, 2], 1: [3, 4]}
      negative_hyperedge_to_node_indices = {0: [5, 6, 7], 1: [8, 9]}
    """
    n_hyperedges = len(hyperedges)
    
    # Extract all nodes from hyperedges and negatives, keeping track of which hyperedge each belongs to
    all_nodes = []
    all_timestamps = []
    hyperedge_to_node_indices = {}
    negative_hyperedge_to_node_indices = {}
    
    # Process positive hyperedges
    current_idx = 0
    for i, hyperedge in enumerate(hyperedges):
      hyperedge_indices = []
      for node in hyperedge:
        all_nodes.append(node)
        all_timestamps.append(edge_times[i])
        hyperedge_indices.append(current_idx)
        current_idx += 1
      hyperedge_to_node_indices[i] = hyperedge_indices
    
    # Process negative hyperedges
    for i, negative_hyperedge in enumerate(negative_hyperedges):
      negative_indices = []
      for node in negative_hyperedge:
        all_nodes.append(node)
        all_timestamps.append(edge_times[i])  # Use same timestamp as corresponding positive
        negative_indices.append(current_idx)
        current_idx += 1
      negative_hyperedge_to_node_indices[i] = negative_indices
    
    all_nodes = np.array(all_nodes)
    all_timestamps = np.array(all_timestamps)
    
    # Get all unique nodes that appear in positive hyperedges (for memory update)
    all_positive_nodes = []
    for hyperedge in hyperedges:
      all_positive_nodes.extend(hyperedge)
    all_positive_nodes = np.array(all_positive_nodes)
    
    memory = None
    time_diffs = None
    if self.use_memory:
      if self.memory_update_at_start:
        # Update memory for all nodes with messages stored in previous batches
        # DEBUG: Check which nodes have messages
        # nodes_with_messages = [node_id for node_id in range(self.n_nodes) 
        #                        if node_id in self.memory.messages and len(self.memory.messages[node_id]) > 0]
        # if len(nodes_with_messages) > 0:
        #   # DEBUG: Print message info for nodes with messages
        #   print(f"\n[DEBUG Batch 26+] Nodes with messages: {len(nodes_with_messages)}")
        #   for node_id in nodes_with_messages[:10]:  # Print first 10
        #     msg_timestamps = [msg[1].item() if isinstance(msg[1], torch.Tensor) else msg[1] 
        #                      for msg in self.memory.messages[node_id]]
        #     last_update = self.memory.last_update[node_id].item()
        #     print(f"  Node {node_id}: {len(self.memory.messages[node_id])} messages, "
        #           f"timestamps: {msg_timestamps}, last_update: {last_update}")
        
        memory, last_update = self.get_updated_memory(list(range(self.n_nodes)),
                                                      self.memory.messages)
      else:
        memory = self.memory.get_memory(list(range(self.n_nodes)))
        last_update = self.memory.last_update

      # Compute time differences for all node occurrences
      # Unified time shift normalization (no source/destination distinction)
      time_diffs = torch.LongTensor(all_timestamps).to(self.device) - last_update[all_nodes].long()
      time_diffs = (time_diffs - self.mean_time_shift) / self.std_time_shift

    # Compute the embeddings using the embedding module
    node_embeddings = self.embedding_module.compute_embedding(memory=memory,
                                                              source_nodes=all_nodes,
                                                              timestamps=all_timestamps,
                                                              n_layers=self.n_layers,
                                                              n_neighbors=n_neighbors,
                                                              time_diffs=time_diffs)

    if self.use_memory:
      if self.memory_update_at_start:
        # Persist the updates to the memory for all nodes in positive hyperedges
        # DEBUG: Track memory update for Node 37
        # if 37 in all_positive_nodes:
        #   before_mem_37 = self.memory.last_update[37].item()
        #   msgs_before = len(self.memory.messages.get(37, []))
        #   print(f"[DEBUG Batch 27] BEFORE update_memory: Node 37 last_update={before_mem_37}, messages={msgs_before}")
        
        self.update_memory(all_positive_nodes, self.memory.messages)
        
        # DEBUG: Track after memory update
        # if 37 in all_positive_nodes:
        #   after_mem_37 = self.memory.last_update[37].item()
        #   msgs_after = len(self.memory.messages.get(37, []))
        #   print(f"[DEBUG Batch 27] AFTER update_memory: Node 37 last_update={after_mem_37}, messages={msgs_after}")

        assert torch.allclose(memory[all_positive_nodes], self.memory.get_memory(all_positive_nodes), atol=1e-5), \
          "Something wrong in how the memory was updated"

        # Remove messages for the positives since we have already updated the memory using them
        # DEBUG: Track message clearing
        # if 37 in all_positive_nodes:
        #   msgs_before_clear = len(self.memory.messages.get(37, []))
        #   print(f"[DEBUG Batch 27] BEFORE clear_messages: Node 37 messages={msgs_before_clear}")
        
        self.memory.clear_messages(all_positive_nodes)
        
        # DEBUG: Track after clearing
        # if 37 in all_positive_nodes:
        #   msgs_after_clear = len(self.memory.messages.get(37, []))
        #   print(f"[DEBUG Batch 27] AFTER clear_messages: Node 37 messages={msgs_after_clear}")

      # Create raw messages for each node in each hyperedge
      # Similar to TGN calling get_raw_messages twice per edge (once for source, once for dest),
      # THGN calls it k times per hyperedge (once for each of the k nodes)
      for i, (hyperedge, edge_time, edge_idx) in enumerate(zip(hyperedges, edge_times, edge_idxs)):
        # Get embeddings for nodes in this hyperedge
        hyperedge_node_indices = hyperedge_to_node_indices[i]
        hyperedge_embeddings = node_embeddings[hyperedge_node_indices]
        
        # For each node in the hyperedge, create a message
        for node_idx, node_id in enumerate(hyperedge):
          # DEBUG: Track message creation for Node 37
          # if node_id == 37:
          #   print(f"[DEBUG Batch 27] Creating message for Node 37: hyperedge={hyperedge}, edge_time={edge_time}, "
          #         f"current_last_update={self.memory.last_update[37].item()}")
          
          # Get this node's embedding
          node_embedding = hyperedge_embeddings[node_idx:node_idx+1]  # Keep batch dimension
          
          # Get other nodes in the hyperedge (for aggregation)
          other_node_ids = [hyperedge[j] for j in range(len(hyperedge)) if j != node_idx]
          other_embeddings = hyperedge_embeddings[[j for j in range(len(hyperedge)) if j != node_idx]]
          
          # Create raw message for this node
          # get_raw_messages will aggregate other nodes' memories/embeddings internally
          unique_nodes, messages = self.get_raw_messages([node_id],
                                                         node_embedding,
                                                         other_node_ids,
                                                         other_embeddings,
                                                         np.array([edge_time]),
                                                         np.array([edge_idx]))
          
          # DEBUG: Track message storage for Node 37
          # if node_id == 37:
          #   msg_ts = list(messages[37])[0][1].item() if 37 in messages and len(messages[37]) > 0 else None
          #   print(f"[DEBUG Batch 27] Storing message for Node 37: timestamp={msg_ts}, "
          #         f"messages_before_storage={len(self.memory.messages.get(37, []))}")
          
          if self.memory_update_at_start:
            self.memory.store_raw_messages(unique_nodes, messages)
            
            # DEBUG: Track after storage
            # if node_id == 37:
            #   msgs_after = len(self.memory.messages.get(37, []))
            #   msg_timestamps = [msg[1].item() if isinstance(msg[1], torch.Tensor) else msg[1] 
            #                    for msg in self.memory.messages.get(37, [])]
            #   print(f"[DEBUG Batch 27] AFTER store_raw_messages: Node 37 messages={msgs_after}, timestamps={msg_timestamps}")
          else:
            self.update_memory(unique_nodes, messages)

      if self.dyrep:
        # For dyrep, use memory directly as embeddings
        # Need to map back to original node occurrences
        node_embeddings = memory[all_nodes]

    return node_embeddings, hyperedge_to_node_indices, negative_hyperedge_to_node_indices

  def compute_hyperedge_probabilities(self, hyperedges, negative_hyperedges, edge_times, edge_idxs, n_neighbors=20):
    """
    Compute probabilities for hyperedges by first computing temporal embeddings using the THGN encoder
    and then feeding them into the MergeLayer decoder (mean pooling + MLP).
    
    This method scores variable-size hyperedges by:
    1. Computing embeddings for all nodes in hyperedges and negatives
    2. Extracting embeddings for each hyperedge using the mappings
    3. Padding variable-size hyperedges to max_size for batching
    4. Using MergeLayer (mean pooling + MLP) to score each hyperedge
    5. Returning probabilities for positive and negative hyperedges
    
    Key Changes from TGN:
    ---------------------
    1. **Input format**: Takes hyperedges (list of lists) and negative_hyperedges (list of lists)
       instead of separate source_nodes, destination_nodes, negative_nodes arrays.
    
    2. **Embedding extraction**: Uses mappings from compute_temporal_embeddings to extract embeddings
       for each hyperedge, rather than having separate tensors for sources/destinations/negatives.
    
    3. **Variable-size handling**: Pads hyperedges to max_size and uses masks to indicate valid vs
       padded nodes, since hyperedges can have different sizes (unlike TGN's fixed-size edges of 2).
    
    4. **Scoring method**: Uses MergeLayer with mean pooling (set aggregation) instead of TGN's
       concatenation approach (pair aggregation). Mean pooling is permutation-invariant, treating
       hyperedges as sets rather than ordered pairs.
    
    5. **Decoder input**: MergeLayer expects [batch_size, max_hyperedge_size, embed_dim] with optional
       mask, while TGN's decoder expects [batch_size, embed_dim] pairs.
    
    Parameters:
    -----------
    hyperedges: list of lists
      Positive hyperedges in the batch. Each hyperedge is a list of node IDs.
      Example: [[50, 60, 70], [80, 90]] for batch_size=2
    negative_hyperedges: list of lists
      Negative hyperedges in the batch. Must have same sizes as corresponding positive hyperedges.
      Example: [[23, 45, 67], [12, 34]] for batch_size=2 (sizes [3, 2] match)
    edge_times: np.ndarray [batch_size]
      Timestamp of each hyperedge interaction
    edge_idxs: np.ndarray [batch_size]
      Index of each hyperedge (for accessing hyperedge features)
    n_neighbors: int
      Number of hyperedges to sample per node (results in up to n_neighbors² total neighbor nodes)
    
    Returns:
    --------
    pos_prob: torch.Tensor [batch_size]
      Probabilities for positive hyperedges (sigmoid applied)
    neg_prob: torch.Tensor [batch_size]
      Probabilities for negative hyperedges (sigmoid applied)
    
    Example:
    --------
    Input:
      hyperedges = [[50, 60, 70], [80, 90]]
      negative_hyperedges = [[23, 45, 67], [12, 34]]
      edge_times = [500, 600]
      edge_idxs = [0, 1]
    
    Processing:
      1. Compute embeddings: node_embeddings, mappings = compute_temporal_embeddings(...)
      2. Extract embeddings for each hyperedge:
         - hyperedge 0: [emb_50, emb_60, emb_70] (size 3)
         - hyperedge 1: [emb_80, emb_90] (size 2)
         - negative 0: [emb_23, emb_45, emb_67] (size 3)
         - negative 1: [emb_12, emb_34] (size 2)
      3. Pad to max_size=3:
         - hyperedge 0: [emb_50, emb_60, emb_70] (no padding)
         - hyperedge 1: [emb_80, emb_90, 0] (padded)
         - mask: [[1,1,1], [1,1,0], [1,1,1], [1,1,0]]
      4. Score all hyperedges: scores = MergeLayer(padded_embs, mask)
      5. Split: pos_prob = scores[:2], neg_prob = scores[2:]
    
    Return:
      pos_prob: [0.85, 0.72]  # High probabilities for positive hyperedges
      neg_prob: [0.15, 0.23]  # Low probabilities for negative hyperedges
    """
    n_hyperedges = len(hyperedges)
    
    # Compute temporal embeddings for all nodes
    node_embeddings, hyperedge_to_node_indices, negative_hyperedge_to_node_indices = \
        self.compute_temporal_embeddings(hyperedges, negative_hyperedges, edge_times, edge_idxs, n_neighbors)
    
    # Extract embeddings for each hyperedge
    all_hyperedge_embeddings = []
    all_hyperedge_sizes = []
    
    # Process positive hyperedges
    for i, hyperedge in enumerate(hyperedges):
      node_indices = hyperedge_to_node_indices[i]
      hyperedge_embs = node_embeddings[node_indices]  # [k, embed_dim]
      all_hyperedge_embeddings.append(hyperedge_embs)
      all_hyperedge_sizes.append(len(hyperedge))
    
    # Process negative hyperedges
    for i, negative_hyperedge in enumerate(negative_hyperedges):
      node_indices = negative_hyperedge_to_node_indices[i]
      hyperedge_embs = node_embeddings[node_indices]  # [k, embed_dim]
      all_hyperedge_embeddings.append(hyperedge_embs)
      all_hyperedge_sizes.append(len(negative_hyperedge))
    
    # Find max hyperedge size for padding
    max_size = max(all_hyperedge_sizes) if all_hyperedge_sizes else 1
    embed_dim = node_embeddings.shape[1]
    total_hyperedges = len(all_hyperedge_embeddings)
    
    # Pad all hyperedges to max_size and create mask
    padded_embeddings = torch.zeros(total_hyperedges, max_size, embed_dim, device=node_embeddings.device)
    mask = torch.zeros(total_hyperedges, max_size, device=node_embeddings.device, dtype=torch.float32)
    
    for i, (hyperedge_embs, size) in enumerate(zip(all_hyperedge_embeddings, all_hyperedge_sizes)):
      padded_embeddings[i, :size] = hyperedge_embs  # [max_size, embed_dim]
      mask[i, :size] = 1.0  # Mark valid nodes
    
    # Score all hyperedges using MergeLayer (mean pooling + MLP)
    scores = self.affinity_score(padded_embeddings, mask)  # [total_hyperedges, 1]
    scores = scores.squeeze(dim=1)  # [total_hyperedges]
    
    # Split into positive and negative scores
    pos_score = scores[:n_hyperedges]
    neg_score = scores[n_hyperedges:]
    
    return pos_score.sigmoid(), neg_score.sigmoid()

  def update_memory(self, nodes, messages):
    """
    Update node memories by aggregating, processing, and applying messages from hyperedge interactions.
    
    This method is structure-agnostic and works identically to TGN's update_memory. The only difference
    is that messages come from hyperedge interactions instead of binary edge interactions, but the
    message dictionary structure and processing pipeline are identical.
    
    Steps:
    1. Aggregate: Combine multiple messages per node (MessageAggregator)
    2. Process: Transform raw messages to processed messages (MessageFunction)
    3. Update: Update memory using GRU/RNN (MemoryUpdater)
    
    Parameters:
    -----------
    nodes: list of int
      Node IDs that have messages to process
    messages: dict
      Dictionary mapping node_id to list of (message_tensor, timestamp) tuples
      Format: {node_id: [(message_tensor, timestamp), ...]}
      Messages come from hyperedge interactions (created by get_raw_messages)
    
    Returns:
    --------
    None (updates memory in-place)
    """
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    # Update the memory with the aggregated messages
    self.memory_updater.update_memory(unique_nodes, unique_messages,
                                      timestamps=unique_timestamps)

  def get_updated_memory(self, nodes, messages):
    """
    Get updated memory without modifying the actual memory (non-destructive).
    
    Similar to update_memory, but returns copies of updated memory and last_update timestamps
    without modifying the actual memory state. Used when we need to compute embeddings with
    updated memory but don't want to persist the updates yet.
    
    Parameters:
    -----------
    nodes: list of int
      Node IDs that have messages to process
    messages: dict
      Dictionary mapping node_id to list of (message_tensor, timestamp) tuples
      Format: {node_id: [(message_tensor, timestamp), ...]}
      Messages come from hyperedge interactions (created by get_raw_messages)
    
    Returns:
    --------
    updated_memory: torch.Tensor [n_nodes, memory_dimension]
      Copy of memory after applying updates (does not modify actual memory)
    updated_last_update: torch.Tensor [n_nodes]
      Copy of last_update timestamps after applying updates
    """
    # Aggregate messages for the same nodes
    unique_nodes, unique_messages, unique_timestamps = \
      self.message_aggregator.aggregate(
        nodes,
        messages)

    if len(unique_nodes) > 0:
      unique_messages = self.message_function.compute_message(unique_messages)

    updated_memory, updated_last_update = self.memory_updater.get_updated_memory(unique_nodes,
                                                                                 unique_messages,
                                                                                 timestamps=unique_timestamps)

    return updated_memory, updated_last_update

  def get_raw_messages(self, node_id, node_embedding, other_node_ids, other_embeddings, edge_time, edge_idx):
    """
    Create raw messages for a single node from a hyperedge interaction.
    
    For each node in a hyperedge, creates a message that includes:
    - The node's own memory (or embedding if use_node_embedding_in_message=True)
    - Aggregated memories of other nodes in the hyperedge (mean pooling)
    - Hyperedge features
    - Time encoding (time delta since last update)
    
    Key Changes from TGN:
    ---------------------
    1. **Input format**: Takes single node_id and other_node_ids (list) instead of batch of
       source_nodes and destination_nodes. Called once per node in hyperedge (k times for
       hyperedge of size k).
    
    2. **Aggregation**: Aggregates other nodes' memories using mean pooling instead of using
       a single destination memory. This handles variable-size hyperedges while keeping
       message dimension fixed.
    
    3. **Message structure**: [node_memory, aggregated_other_nodes_memory, hyperedge_features, time_encoding]
       Same dimension as TGN: 2 * memory_dim + hyperedge_feat_dim + time_dim
    
    Parameters:
    -----------
    node_id: list of int (length 1)
      The node ID to create message for (passed as list for consistency with TGN's batch format)
    node_embedding: torch.Tensor [1, embed_dim]
      Embedding of the node (computed from compute_temporal_embeddings)
    other_node_ids: list of int
      Other node IDs in the hyperedge (excluding node_id)
    other_embeddings: torch.Tensor [k-1, embed_dim]
      Embeddings of other nodes in the hyperedge
    edge_time: np.ndarray (length 1)
      Timestamp of the hyperedge interaction
    edge_idx: np.ndarray (length 1)
      Index of the hyperedge (for accessing hyperedge features)
    
    Returns:
    --------
    unique_nodes: np.ndarray
      Unique node IDs (just [node_id] in this case, since we process one node at a time)
    messages: dict
      Dictionary mapping node_id to list of (message_tensor, timestamp) tuples
      Format: {node_id: [(message_tensor, timestamp)]}
    
    Example:
    --------
    Hyperedge [50, 60, 70] at time 500, edge_idx=0
    
    Call 1 (for node 50):
      node_id = [50]
      node_embedding = [emb_50]  # [1, embed_dim]
      other_node_ids = [60, 70]
      other_embeddings = [emb_60, emb_70]  # [2, embed_dim]
      
      node_memory = mem[50] or emb_50  # [memory_dim]
      aggregated_other = mean([mem[60], mem[70]])  # [memory_dim]
      hyperedge_feat = edge_raw_features[0]  # [hyperedge_feat_dim]
      time_enc = time_encoder(500 - last_update[50])  # [time_dim]
      
      message = [node_memory, aggregated_other, hyperedge_feat, time_enc]  # [raw_message_dim]
      
      Returns: ([50], {50: [(message_tensor, 500)]})
    
    Similar calls for nodes 60 and 70...
    """
    # Convert to tensors and handle single values
    node_id = node_id[0] if isinstance(node_id, (list, np.ndarray)) else node_id
    edge_time = edge_time[0] if isinstance(edge_time, np.ndarray) else edge_time
    edge_idx = edge_idx[0] if isinstance(edge_idx, np.ndarray) else edge_idx
    
    edge_time_tensor = torch.tensor([edge_time], dtype=torch.float32).to(self.device)
    edge_features = self.edge_raw_features[edge_idx:edge_idx+1]  # [1, hyperedge_feat_dim]
    
    # Get node's memory (or use embedding if flag is set)
    node_memory = self.memory.get_memory([node_id]) if not \
      self.use_node_embedding_in_message else node_embedding.squeeze(0)
    
    # If we got memory, it's [1, memory_dim], squeeze to [memory_dim]
    if node_memory.dim() > 1:
      node_memory = node_memory.squeeze(0)
    
    # Get other nodes' memories (or use embeddings if flag is set)
    if len(other_node_ids) > 0:
      if not self.use_node_embedding_in_message:
        other_memories = self.memory.get_memory(other_node_ids)  # [k-1, memory_dim]
      else:
        other_memories = other_embeddings  # [k-1, embed_dim]
      
      # Aggregate other nodes' memories using mean pooling
      aggregated_other_memory = torch.mean(other_memories, dim=0)  # [memory_dim]
    else:
      # Edge case: hyperedge with only one node (shouldn't happen, but handle gracefully)
      aggregated_other_memory = torch.zeros_like(node_memory)
    
    # Compute time delta encoding
    time_delta = edge_time_tensor - self.memory.last_update[node_id]
    time_delta_encoding = self.time_encoder(time_delta.unsqueeze(dim=1)).view(-1)  # [time_dim]
    
    # Concatenate to form raw message: [node_mem, aggregated_other_mem, hyperedge_feat, time_enc]
    raw_message = torch.cat([node_memory, aggregated_other_memory, edge_features.squeeze(0),
                            time_delta_encoding], dim=0)  # [raw_message_dim]
    
    # Create messages dictionary
    messages = defaultdict(list)
    # Store timestamp as tensor (not float) so it can be stacked by message aggregator
    messages[node_id].append((raw_message, edge_time_tensor.squeeze()))
    
    return np.array([node_id]), messages

  def set_neighbor_finder(self, neighbor_finder):
    """
    Update the neighbor finder for both the model and embedding module.
    
    This method is used during training/evaluation to switch between different neighbor finders
    (e.g., training neighbor finder vs full graph neighbor finder for validation/test).
    
    In THGN, the neighbor finder performs hyperedge-level sampling (samples n_neighbors hyperedges,
    then extracts nodes from each hyperedge), resulting in up to n_neighbors² total neighbor nodes.
    
    Parameters:
    -----------
    neighbor_finder: NeighborFinder
      The new neighbor finder to use for temporal neighbor queries
    """
    self.neighbor_finder = neighbor_finder
    self.embedding_module.neighbor_finder = neighbor_finder
