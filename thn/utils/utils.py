import numpy as np
import torch


class ConcatMergeLayer(torch.nn.Module):
  """
  Two-layer MLP that merges two embeddings by concatenation (TGN-style).
  
  Used in temporal attention to merge attention output with source node features.
  Formula: output = W2 * ReLU(W1 * [x1; x2] + b1) + b2
  where [x1; x2] is concatenation of two embeddings.
  
  This is different from MergeLayer (used for hyperedge prediction), which performs
  mean pooling over a set of node embeddings.
  
  Parameters:
  -----------
  dim1: int
    Dimension of first embedding (e.g., query_dim from attention output)
  dim2: int
    Dimension of second embedding (e.g., n_node_features from source node)
  dim3: int
    Hidden layer dimension
  dim4: int
    Output dimension (typically n_node_features)
  """
  def __init__(self, dim1, dim2, dim3, dim4):
    super().__init__()
    self.fc1 = torch.nn.Linear(dim1 + dim2, dim3)
    self.fc2 = torch.nn.Linear(dim3, dim4)
    self.act = torch.nn.ReLU()

    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)

  def forward(self, x1, x2):
    """
    Forward pass: Concatenate two embeddings and pass through MLP.
    
    Parameters:
    -----------
    x1: torch.Tensor [batch_size, dim1]
      First embedding (e.g., attention output)
    x2: torch.Tensor [batch_size, dim2]
      Second embedding (e.g., source node features)
    
    Returns:
    --------
    output: torch.Tensor [batch_size, dim4]
      Merged embedding after MLP transformation
    """
    x = torch.cat([x1, x2], dim=1)
    h = self.act(self.fc1(x))
    return self.fc2(h)


class MergeLayer(torch.nn.Module):
  """
  Two-layer MLP decoder for hyperedge prediction using mean pooling aggregation.
  
  Purpose: Takes a variable-size set of node embeddings (representing a hyperedge) and
  outputs a scalar score indicating the probability that this set of nodes forms a hyperedge
  at the given timestamp.
  
  Architecture:
  ------------
  1. Mean Pooling: Aggregates variable-size set of node embeddings into fixed-size vector
     hyperedge_emb = mean([node_emb_1, node_emb_2, ..., node_emb_k])  # [embed_dim]
  
  2. Two-Layer MLP: Maps pooled hyperedge embedding to scalar score
     score = W2 * ReLU(W1 * hyperedge_emb + b1) + b2
  
  Complete Formula:
  ----------------
  Given hyperedge S = {v₁, v₂, ..., vₖ} with node embeddings {z₁, z₂, ..., zₖ}:
  
    pooled_emb = (1/k) * Σᵢ zᵢ                    # Mean pooling [embed_dim]
    h = ReLU(W1 * pooled_emb + b1)                # Hidden layer [hidden_dim]
    score = W2 * h + b2                            # Output layer [1]
  
  Where:
    - k = |S| (hyperedge size, variable)
    - zᵢ ∈ ℝ^embed_dim (node embedding)
    - W1 ∈ ℝ^(hidden_dim × embed_dim), b1 ∈ ℝ^hidden_dim
    - W2 ∈ ℝ^(1 × hidden_dim), b2 ∈ ℝ
  
  Permutation Invariance:
  ---------------------
  Mean pooling is permutation-invariant: mean([z₁, z₂]) = mean([z₂, z₁])
  This ensures the decoder treats hyperedges as sets (order doesn't matter).
  
  Alternative Approaches (not implemented):
  ----------------------------------------
  1. DeepSets: More expressive permutation-invariant aggregation
     - Transform each node embedding: z'ᵢ = MLP(zᵢ)
     - Aggregate: pooled_emb = Σᵢ z'ᵢ
     - Then MLP: score = MLP(pooled_emb)
     - Pros: Learns member importance, more expressive
     - Cons: More complex, higher compute cost
  
  2. Attention Over Members: Most expressive approach
     - Compute attention weights: αᵢ = softmax(Q @ Kᵢ^T)
     - Weighted aggregation: pooled_emb = Σᵢ αᵢ * Vᵢ
     - Then MLP: score = MLP(pooled_emb)
     - Pros: Learns which members matter most, interpretable, best performance
     - Cons: Most expensive (O(n²)), more complex, risk of overfitting
  
  Usage in THGN:
  ------------
  During training/inference:
    - Positive hyperedges: score(hyperedge_nodes) → high probability
    - Negative hyperedges: score(random_nodes) → low probability
    - Loss: Binary cross-entropy between predicted and true labels
  
  Example:
  -------
  Input: Hyperedge with 3 nodes
    node_embeddings = [[0.2, 0.5, ...], [0.3, 0.1, ...], [0.4, 0.2, ...]]  # [3, 172]
    mask = [1, 1, 1]  # All nodes valid
  
  Process:
    pooled = mean([0.2,0.5,...], [0.3,0.1,...], [0.4,0.2,...]) = [0.3, 0.27, ...]  # [172]
    h = ReLU(W1 * pooled + b1)  # [hidden_dim]
    score = W2 * h + b2  # [1]
  
  Output: score = 0.85 (high probability this is a valid hyperedge)
  
  Parameters:
  ----------
  embed_dim: int
    Dimension of node embeddings (input to mean pooling)
  hidden_dim: int
    Dimension of hidden layer in MLP
  output_dim: int
    Dimension of output (typically 1 for binary classification)
  """
  def __init__(self, embed_dim, hidden_dim, output_dim=1):
    super().__init__()
    self.embed_dim = embed_dim
    self.hidden_dim = hidden_dim
    self.output_dim = output_dim
    
    # Two-layer MLP: embed_dim → hidden_dim → output_dim
    self.fc1 = torch.nn.Linear(embed_dim, hidden_dim)
    self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
    self.act = torch.nn.ReLU()
    
    # Xavier initialization (same as TGN)
    torch.nn.init.xavier_normal_(self.fc1.weight)
    torch.nn.init.xavier_normal_(self.fc2.weight)
  
  def forward(self, node_embeddings, mask=None):
    """
    Forward pass: Mean pool node embeddings, then apply MLP.
    
    Parameters:
    ----------
    node_embeddings: torch.Tensor [batch_size, max_hyperedge_size, embed_dim]
      Node embeddings for each hyperedge in the batch.
      Variable-size hyperedges are padded to max_hyperedge_size.
    mask: torch.Tensor [batch_size, max_hyperedge_size], optional
      Binary mask indicating valid nodes (1) vs padding (0).
      If None, assumes all nodes are valid.
    
    Returns:
    -------
    torch.Tensor [batch_size, output_dim]
      Scalar scores for each hyperedge in the batch.
    """
    batch_size, max_size, embed_dim = node_embeddings.shape
    
    # Handle mask: if provided, use it; otherwise assume all valid
    if mask is None:
      mask = torch.ones(batch_size, max_size, device=node_embeddings.device, dtype=torch.float32)
    else:
      mask = mask.float()
    
    # Mean pooling over valid nodes only
    # Sum embeddings, weighted by mask
    masked_embeddings = node_embeddings * mask.unsqueeze(-1)  # [batch, max_size, embed_dim]
    summed = masked_embeddings.sum(dim=1)  # [batch, embed_dim]
    
    # Count valid nodes per hyperedge (avoid division by zero)
    valid_counts = mask.sum(dim=1, keepdim=True)  # [batch, 1]
    valid_counts = torch.clamp(valid_counts, min=1.0)  # Ensure at least 1 to avoid division by zero
    
    # Mean: divide by number of valid nodes
    pooled_emb = summed / valid_counts  # [batch, embed_dim]
    
    # Two-layer MLP
    h = self.act(self.fc1(pooled_emb))  # [batch, hidden_dim]
    score = self.fc2(h)  # [batch, output_dim]
    
    return score


class MLP(torch.nn.Module):
  """
  Three-layer MLP for node classification: dim → 80 → 10 → 1
  
  Architecture: x → Linear(dim, 80) → ReLU → Dropout → 
                Linear(80, 10) → ReLU → Dropout → Linear(10, 1)
  
  Used for supervised node classification task in THGN. Takes node embeddings
  (computed from hypergraph structure) and outputs binary classification probability.
  
  Note: The hyperedge structure affects how node embeddings are computed (via
  temporal hypergraph attention), but once embeddings are obtained, this MLP
  operates identically to the TGN version - it's just a classifier on node features.
  
  Example:
    node_emb [batch, 100] → MLP(100) → output [batch] (classification prob)
  """
  def __init__(self, dim, drop=0.3):
    super().__init__()
    self.fc_1 = torch.nn.Linear(dim, 80)
    self.fc_2 = torch.nn.Linear(80, 10)
    self.fc_3 = torch.nn.Linear(10, 1)
    self.act = torch.nn.ReLU()
    self.dropout = torch.nn.Dropout(p=drop, inplace=False)

  def forward(self, x):
    x = self.act(self.fc_1(x))
    x = self.dropout(x)
    x = self.act(self.fc_2(x))
    x = self.dropout(x)
    return self.fc_3(x).squeeze(dim=1)


class EarlyStopMonitor(object):
  """
  Implements early stopping based on validation metric improvement.
  
  Logic:
    - If (curr_val - best_val) / |best_val| > tolerance: reset counter, update best
    - Otherwise: increment counter
    - Stop if counter >= max_round (no improvement for max_round epochs)
  
  Formula: relative_improvement = (curr - best) / |best|
  
  Example:
    max_round=5, tolerance=1e-10, best_val=0.85
    - curr_val=0.86 → improvement > tolerance → reset counter
    - curr_val=0.8501 → improvement < tolerance → counter++
    - After 5 consecutive non-improvements → return True (stop)
  
  Note: This utility is independent of graph structure (works identically for
  graphs and hypergraphs) - it's just a training helper that monitors metrics.
  """
  def __init__(self, max_round=3, higher_better=True, tolerance=1e-10):
    self.max_round = max_round
    self.num_round = 0

    self.epoch_count = 0
    self.best_epoch = 0

    self.last_best = None
    self.higher_better = higher_better
    self.tolerance = tolerance

  def early_stop_check(self, curr_val):
    if not self.higher_better:
      curr_val *= -1
    if self.last_best is None:
      self.last_best = curr_val
    elif (curr_val - self.last_best) / np.abs(self.last_best) > self.tolerance:
      self.last_best = curr_val
      self.num_round = 0
      self.best_epoch = self.epoch_count
    else:
      self.num_round += 1

    self.epoch_count += 1

    return self.num_round >= self.max_round


class RandHyperedgeSampler(object):
  """
  Samples random negative hyperedges for hyperedge prediction (negative sampling).
  
  For each positive hyperedge of size k, samples a random set of k nodes that likely
  doesn't form a real hyperedge, creating negative examples for binary classification.
  
  Sampling: For each positive hyperedge of size k, randomly selects k unique nodes
  from the node list. Ensures no duplicate nodes within each negative hyperedge.
  With seed: ensures reproducible negative samples for evaluation.
  
  Key Difference from TGN's RandEdgeSampler:
  -----------------------------------------
  - TGN: Samples pairs (src, dst) - fixed size 2
  - THGN: Samples sets of nodes - variable size k (matches positive hyperedge size)
  
  Example:
    Positive hyperedges: [[50, 60, 70], [80, 90], [100, 110, 120, 130]]
    Sizes: [3, 2, 4]
    Sample negatives: [[23, 45, 89], [12, 67], [34, 56, 78, 90]]
    Model learns: P([50,60,70])=1, P([23,45,89])=0, etc.
  
  Note: Does not check if sampled set exactly matches any positive hyperedge.
  This is acceptable because: (1) probability is very low for large node sets,
  (2) even if it matches, it's still a valid negative example (just happens to be
  a real hyperedge at a different time or in a different context).
  """
  def __init__(self, node_list, seed=None):
    """
    Initialize negative hyperedge sampler.
    
    Parameters:
    ----------
    node_list: array-like
      List of all unique nodes in the dataset (can contain duplicates, will be uniquified)
    seed: int, optional
      Random seed for reproducible sampling (used for validation/test sets)
    """
    self.seed = None
    self.node_list = np.unique(node_list)  # All unique nodes in dataset
    
    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)
  
  def sample(self, hyperedge_sizes):
    """
    Sample negative hyperedges matching the sizes of positive hyperedges.
    
    For each positive hyperedge of size k, samples k random unique nodes to form
    a negative hyperedge. Ensures no duplicate nodes within each hyperedge.
    
    Parameters:
    ----------
    hyperedge_sizes: list of int
      Size of each positive hyperedge in the batch.
      Example: [3, 2, 4] means first positive has 3 nodes, second has 2, third has 4.
      Length of this list determines how many negative hyperedges to sample.
    
    Returns:
    -------
    list of lists
      Each element is a list of node IDs forming a negative hyperedge.
      Length matches len(hyperedge_sizes).
      Example: [[23, 45, 89], [12, 67], [34, 56, 78, 90]] for sizes [3, 2, 4]
    """
    negative_hyperedges = []
    
    for size in hyperedge_sizes:
      # Sample k unique nodes randomly (without replacement)
      if self.seed is None:
        sampled_nodes = np.random.choice(self.node_list, size=size, replace=False)
      else:
        sampled_nodes = self.random_state.choice(self.node_list, size=size, replace=False)
      
      # Convert to list (order doesn't matter for hyperedges, but we return as list)
      negative_hyperedges.append(sampled_nodes.tolist())
    
    return negative_hyperedges

  def sample_with_overlap(self, positive_hyperedges, overlap_ratio=0.99):
    """
    Sample negative hyperedges with controlled overlap to positives.

    For each positive hyperedge of size k:
      - keep floor(overlap_ratio * k) nodes from the positive hyperedge
      - replace the rest with random nodes sampled from the global node pool
      - keep size fixed at k and avoid duplicates

    Parameters:
    ----------
    positive_hyperedges: list of lists
      Positive hyperedges for the current batch.
    overlap_ratio: float
      Desired overlap ratio in [0, 1].

    Returns:
    -------
    list of lists
      Negative hyperedges with same sizes as positives.
    """
    if overlap_ratio < 0 or overlap_ratio > 1:
      raise ValueError(f"overlap_ratio must be in [0, 1], got {overlap_ratio}")

    negative_hyperedges = []

    for positive_hyperedge in positive_hyperedges:
      pos_nodes = list(dict.fromkeys(positive_hyperedge))
      k = len(pos_nodes)
      if k == 0:
        negative_hyperedges.append([])
        continue

      keep_n = int(np.floor(overlap_ratio * k))
      keep_n = max(0, min(keep_n, k))
      replace_n = k - keep_n

      if keep_n > 0:
        if self.seed is None:
          kept_nodes = np.random.choice(pos_nodes, size=keep_n, replace=False).tolist()
        else:
          kept_nodes = self.random_state.choice(pos_nodes, size=keep_n, replace=False).tolist()
      else:
        kept_nodes = []

      neg_nodes = list(kept_nodes)
      neg_set = set(neg_nodes)
      pos_set = set(pos_nodes)

      if replace_n > 0:
        # Prefer nodes outside the positive set so negatives are harder but not identical.
        candidate_pool = [n for n in self.node_list if n not in neg_set and n not in pos_set]
        if len(candidate_pool) < replace_n:
          candidate_pool = [n for n in self.node_list if n not in neg_set]

        if self.seed is None:
          sampled_replacements = np.random.choice(candidate_pool, size=replace_n, replace=False).tolist()
        else:
          sampled_replacements = self.random_state.choice(candidate_pool, size=replace_n, replace=False).tolist()
        neg_nodes.extend(sampled_replacements)

      # If replacements were requested, avoid exact equality with positive whenever possible.
      if replace_n > 0 and set(neg_nodes) == pos_set:
        available = [n for n in self.node_list if n not in set(neg_nodes)]
        if available:
          if self.seed is None:
            replacement = np.random.choice(available)
          else:
            replacement = self.random_state.choice(available)
          neg_nodes[-1] = replacement

      negative_hyperedges.append([int(x) for x in neg_nodes])

    return negative_hyperedges
  
  def reset_random_state(self):
    """
    Reset random state to initial seed (for reproducible evaluation).
    
    Used during validation/testing to ensure same negative samples across epochs.
    """
    if self.seed is not None:
      self.random_state = np.random.RandomState(self.seed)


def get_neighbor_finder(data, uniform, max_node_idx=None):
  """
  Builds temporal adjacency structure for hypergraphs at the hyperedge level.
  
  For each hyperedge H = [v₁, v₂, ..., vₖ], stores the hyperedge itself (not pairwise connections):
  - For each node vᵢ in H, adds the entire hyperedge to vᵢ's hyperedge list
  - This preserves hyperedge structure and allows sampling at the interaction level
  
  Key Insight:
  -----------
  Sampling at hyperedge level (not node level) preserves hypergraph semantics:
  - Sample n_neighbors most recent hyperedges (actual interactions)
  - For each hyperedge, include all nodes (or sample if hyperedge is too large)
  - This ensures we get n_neighbors interactions, not just n_neighbors individual nodes
  
  Example:
  -------
  Hyperedge: [50, 60, 70] with edge_idx=0, timestamp=100
  Result:
    adj_list[50] = [(0, 100, [50, 60, 70])]  # (edge_idx, timestamp, node_list)
    adj_list[60] = [(0, 100, [50, 60, 70])]
    adj_list[70] = [(0, 100, [50, 60, 70])]
  
  Parameters:
  ----------
  data: Data object
    Must have: hyperedges (list of lists), edge_idxs, timestamps
  uniform: bool
    Whether to use uniform sampling (passed to NeighborFinder)
  max_node_idx: int, optional
    Maximum node ID. If None, computed from data.
  
  Returns:
  -------
  NeighborFinder
    Configured for temporal neighbor queries on hypergraph data with hyperedge-level sampling
  """
  # Compute max node index if not provided
  if max_node_idx is None:
    # Find maximum node ID across all hyperedges
    max_node_idx = 0
    for hyperedge in data.hyperedges:
      if len(hyperedge) > 0:
        max_node_idx = max(max_node_idx, max(hyperedge))
  
  # Initialize adjacency list: one list per node
  # Each entry will be (edge_idx, timestamp, list_of_nodes_in_hyperedge)
  adj_list = [[] for _ in range(max_node_idx + 1)]
  
  # Process each hyperedge: store hyperedge-level information
  for hyperedge, edge_idx, timestamp in zip(data.hyperedges, data.edge_idxs, data.timestamps):
    # Skip empty hyperedges (shouldn't happen, but safety check)
    if len(hyperedge) == 0:
      continue
    
    # For each node in the hyperedge, add the entire hyperedge to its list
    # Store as (edge_idx, timestamp, list_of_all_nodes_in_hyperedge)
    hyperedge_nodes = list(hyperedge)  # Make a copy of the node list
    for node in hyperedge:
      adj_list[node].append((edge_idx, timestamp, hyperedge_nodes))
  
  return NeighborFinder(adj_list, uniform=uniform)


class NeighborFinder:
  """
  Finds temporal neighbors for nodes at specific timestamps (core of THGN).
  
  Maintains sorted hyperedge lists (by timestamp) for efficient temporal queries.
  Uses hyperedge-level sampling: samples n_neighbors most recent hyperedges, then extracts
  nodes from each hyperedge (with random sampling if hyperedge size > n_neighbors).
  
  Two sampling strategies:
    1. Most Recent (default): takes n_neighbors most recent hyperedges before cut_time
    2. Uniform: randomly samples n_neighbors hyperedges, then re-sorts by time
  
  Key methods:
    - find_before(node, time): Returns all hyperedges before time (O(log n) via binary search)
    - get_temporal_neighbor(nodes, times, n_neighbors): Samples temporal neighborhoods
  
  Hyperedge-Level Sampling:
  ------------------------
  - Samples n_neighbors hyperedges (actual interactions)
  - For each hyperedge: includes all nodes if size ≤ n_neighbors, else randomly samples n_neighbors nodes
  - Maximum total neighbors = n_neighbors²
  - Preserves hyperedge structure better than node-level sampling
  
  Flattening and Implicit Grouping:
  ---------------------------------
  Nodes from sampled hyperedges are flattened into a single list for attention computation.
  This approach has both advantages and limitations:
  
  Advantages:
    - Correct sampling: We sample n_neighbors actual hyperedges (interactions), not just
      n_neighbors individual nodes, ensuring we capture the right temporal context
    - Implicit grouping preserved: Nodes from the same hyperedge share:
      * Same edge_idx (repeated for all nodes in the hyperedge)
      * Same edge_time (repeated for all nodes in the hyperedge)
      * Same edge_features (will be repeated when passed to attention)
    - Attention can learn relationships: The shared metadata (edge_idx, edge_time, edge_features)
      allows the attention mechanism to implicitly learn that nodes came from the same interaction
  
  Limitations:
    - Explicit grouping lost: The attention mechanism treats all neighbors independently
      (nodes from the same hyperedge are not explicitly grouped during attention computation)
    - However, this is still better than pairwise sampling because:
      * We sample hyperedges (not individual nodes), preserving interaction-level context
      * Shared metadata provides implicit signals for the model to learn hyperedge relationships
  
  Example:
    Node 50's stored hyperedges: [H1(t=100, nodes=[50,60,70]), H2(t=200, nodes=[50,80]), H3(t=300, nodes=[50,90,100,110])]
      Note: Hyperedges are stored with ALL nodes including the source node itself
    
    get_temporal_neighbor([50], [1000], n_neighbors=2) → most recent 2 hyperedges: H2, H3
      - H2: Filter out node 50 → neighbors=[80]
      - H3: Filter out node 50 → neighbors=[90, 100, 110]
    
    Returns: neighbors=[[80, 90, 100, 110]], edge_times=[[200, 300, 300, 300]], edge_idxs=[[1, 2, 2, 2]]
      Note: Source node 50 is filtered out from each hyperedge (no self-loops)
      Note: Actual output shape is [batch, n_neighbors²] with padding
  """
  def __init__(self, adj_list, uniform=False, seed=None):
    # Store hyperedge-level information: (edge_idx, timestamp, node_list)
    self.node_to_hyperedges = []  # List of hyperedges per node
    self.node_to_edge_idxs = []   # Edge indices for each hyperedge
    self.node_to_edge_timestamps = []  # Timestamps for each hyperedge
    
    for hyperedges in adj_list:
      # Hyperedges is a list of tuples (edge_idx, timestamp, node_list)
      # Sort by timestamp
      sorted_hyperedges = sorted(hyperedges, key=lambda x: x[1])
      self.node_to_hyperedges.append([x[2] for x in sorted_hyperedges])  # Store node lists
      self.node_to_edge_idxs.append(np.array([x[0] for x in sorted_hyperedges]))
      self.node_to_edge_timestamps.append(np.array([x[1] for x in sorted_hyperedges]))
    
    self.uniform = uniform
    
    if seed is not None:
      self.seed = seed
      self.random_state = np.random.RandomState(self.seed)
    else:
      self.seed = None
      self.random_state = None
  
  def find_before(self, src_idx, cut_time):
    """
    Extracts all hyperedges happening before cut_time for node src_idx.
    The returned hyperedges are sorted by time.
    
    Parameters:
    ----------
    src_idx: int
      Node ID to query
    cut_time: float
      Timestamp cutoff (only hyperedges before this time are returned)
    
    Returns:
    -------
    hyperedges: list of lists
      List of hyperedges (each is a list of node IDs) before cut_time
    edge_idxs: np.array
      Hyperedge indices corresponding to each hyperedge
    timestamps: np.array
      Timestamps of each hyperedge
    """
    # Check if node exists in the neighbor finder (bounds check)
    # This handles cases where nodes from negative samples or validation/test data
    # might not be present in the training neighbor finder
    if src_idx >= len(self.node_to_edge_timestamps) or src_idx < 0:
      # Node not found - return empty results (same as node with no neighbors)
      return [], np.array([]), np.array([])
    
    i = np.searchsorted(self.node_to_edge_timestamps[src_idx], cut_time)
    
    return self.node_to_hyperedges[src_idx][:i], self.node_to_edge_idxs[src_idx][:i], self.node_to_edge_timestamps[src_idx][:i]
  
  def get_temporal_neighbor(self, source_nodes, timestamps, n_neighbors=20):
    """
    Given a list of node ids and relative cut times, extracts a sampled temporal 
    neighborhood of each node using hyperedge-level sampling.
    
    Sampling Strategy:
    -----------------
    1. Sample n_neighbors most recent hyperedges (or uniform if flag set)
    2. For each hyperedge:
       - If size ≤ n_neighbors: include all nodes
       - If size > n_neighbors: randomly sample n_neighbors nodes
    3. Flatten to n_neighbors² total neighbors (with padding)
    
    Parameters:
    ----------
    source_nodes: array-like [batch_size]
      Node IDs to query
    timestamps: array-like [batch_size]
      Cutoff timestamps for each node (only hyperedges before this time)
    n_neighbors: int
      Number of hyperedges to sample (default: 20)
      Maximum total neighbors = n_neighbors²
    
    Returns:
    -------
    neighbors: np.array [batch_size, n_neighbors²]
      Flattened list of neighbor node IDs from sampled hyperedges
      Padded with 0s if fewer than n_neighbors² available
    edge_idxs: np.array [batch_size, n_neighbors²]
      Hyperedge indices for each neighbor (repeated for nodes from same hyperedge)
      Padded with 0s if fewer than n_neighbors² available
    edge_times: np.array [batch_size, n_neighbors²]
      Timestamps for each neighbor (same for nodes from same hyperedge)
      Padded with 0s if fewer than n_neighbors² available
    """
    assert (len(source_nodes) == len(timestamps))
    
    max_neighbors = n_neighbors * n_neighbors  # Maximum total neighbors
    tmp_max_neighbors = max_neighbors if max_neighbors > 0 else 1
    
    neighbors = np.zeros((len(source_nodes), tmp_max_neighbors)).astype(np.int32)
    edge_times = np.zeros((len(source_nodes), tmp_max_neighbors)).astype(np.float32)
    edge_idxs = np.zeros((len(source_nodes), tmp_max_neighbors)).astype(np.int32)
    
    for i, (source_node, timestamp) in enumerate(zip(source_nodes, timestamps)):
      # Get all hyperedges before cut_time
      source_hyperedges, source_edge_idxs, source_edge_times = self.find_before(source_node, timestamp)
      
      if len(source_hyperedges) > 0 and n_neighbors > 0:
        # Step 1: Sample n_neighbors hyperedges
        if self.uniform:
          # Uniform sampling: randomly sample n_neighbors hyperedges
          if len(source_hyperedges) > n_neighbors:
            if self.random_state is not None:
              sampled_hyperedge_idx = self.random_state.randint(0, len(source_hyperedges), n_neighbors)
            else:
              sampled_hyperedge_idx = np.random.randint(0, len(source_hyperedges), n_neighbors)
            sampled_hyperedges = [source_hyperedges[j] for j in sampled_hyperedge_idx]
            sampled_edge_idxs = source_edge_idxs[sampled_hyperedge_idx]
            sampled_edge_times = source_edge_times[sampled_hyperedge_idx]
            
            # Re-sort by time
            sort_order = sampled_edge_times.argsort()
            sampled_hyperedges = [sampled_hyperedges[j] for j in sort_order]
            sampled_edge_idxs = sampled_edge_idxs[sort_order]
            sampled_edge_times = sampled_edge_times[sort_order]
          else:
            sampled_hyperedges = source_hyperedges
            sampled_edge_idxs = source_edge_idxs
            sampled_edge_times = source_edge_times
        else:
          # Most recent: take last n_neighbors hyperedges
          num_to_take = min(n_neighbors, len(source_hyperedges))
          sampled_hyperedges = source_hyperedges[-num_to_take:]
          sampled_edge_idxs = source_edge_idxs[-num_to_take:]
          sampled_edge_times = source_edge_times[-num_to_take:]
        
        # Step 2: Extract nodes from each hyperedge (with random sampling if needed)
        all_neighbors = []
        all_edge_idxs = []
        all_edge_times = []
        
        for hyperedge_nodes, edge_idx, edge_time in zip(sampled_hyperedges, sampled_edge_idxs, sampled_edge_times):
          # Remove source node from hyperedge (don't include self)
          hyperedge_nodes_filtered = [n for n in hyperedge_nodes if n != source_node]
          
          if len(hyperedge_nodes_filtered) == 0:
            continue  # Skip if only source node in hyperedge
          
          # Sample nodes from hyperedge
          if len(hyperedge_nodes_filtered) <= n_neighbors:
            # Take all nodes
            sampled_nodes = hyperedge_nodes_filtered
          else:
            # Randomly sample n_neighbors nodes
            if self.random_state is not None:
              sampled_nodes = self.random_state.choice(hyperedge_nodes_filtered, size=n_neighbors, replace=False).tolist()
            else:
              sampled_nodes = np.random.choice(hyperedge_nodes_filtered, size=n_neighbors, replace=False).tolist()
          
          # Add to flattened list
          all_neighbors.extend(sampled_nodes)
          all_edge_idxs.extend([edge_idx] * len(sampled_nodes))
          all_edge_times.extend([edge_time] * len(sampled_nodes))
        
        # Step 3: Pad to max_neighbors and store
        num_neighbors = len(all_neighbors)
        if num_neighbors > 0:
          # Truncate if exceeds max (shouldn't happen, but safety check)
          if num_neighbors > max_neighbors:
            all_neighbors = all_neighbors[:max_neighbors]
            all_edge_idxs = all_edge_idxs[:max_neighbors]
            all_edge_times = all_edge_times[:max_neighbors]
            num_neighbors = max_neighbors
          
          # Store (right-aligned with padding on left)
          neighbors[i, max_neighbors - num_neighbors:] = all_neighbors
          edge_idxs[i, max_neighbors - num_neighbors:] = all_edge_idxs
          edge_times[i, max_neighbors - num_neighbors:] = all_edge_times
    
    return neighbors, edge_idxs, edge_times
