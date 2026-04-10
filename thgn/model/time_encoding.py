"""
Time Encoding Module (TGAT-based) for THGN
============================================

Purpose: Converts time differences (scalars) into dense vector representations that
neural networks can process. Neural networks work with vectors, not single numbers.

What is Time Delta?
-------------------
Time delta is the difference between two timestamps. 
Example: If node 50 last interacted at time 100, and current time is 150, 
         then time_delta = 150 - 100 = 50 (time units since last interaction).

Key Concept:
------------
Time delta (number) → Vector representation
Example: time_delta = 50 → [0.88, 0.999, 1.0, ...] (100-dimensional vector)

Why Vector Encoding Instead of Raw Number?
------------------------------------------
1. Non-linearity: Cosine creates non-linear patterns (raw number is linear)
2. Multi-scale: Captures both short-term and long-term temporal patterns simultaneously
3. Periodicity: Naturally handles periodic patterns (daily, weekly cycles)
4. Learnable: Model learns which time scales matter for different relationships

Mathematical Formulation:
--------------------------
Weight Initialization:
  weight[i] = 1 / (10 ^ (9 * i / (dimension - 1)))
  Creates frequencies from high (1.0) to low (1e-9)

Forward Pass:
  output[i] = cos(weight[i] * t)
  For each dimension i, multiply time delta t by weight, then apply cosine

Complete Formula:
  TimeEncode(t) = [cos(w₀·t), cos(w₁·t), ..., cos(wₙ₋₁·t)]
  where wᵢ = 1 / (10 ^ (9·i/(n-1)))

Example:
--------
Input: time_delta = 50.0 (normalized)
Weights (dimension=4): [1.0, 0.001, 0.000001, 0.000000001]
Output: [cos(50), cos(0.05), cos(0.00005), cos(0.00000005)]
      ≈ [-0.26, 0.999, 1.0, 1.0]

Different time scales affect different dimensions:
- Small time deltas: mostly first dimension changes
- Large time deltas: more dimensions change
- Model learns which dimensions matter for different relationships

Learnable Parameters:
---------------------
- Weights are initialized with frequency pattern but are LEARNABLE
- During training, weights update via backpropagation
- Model learns optimal time encoding for the specific task
- Same time delta produces different encoding as training progresses

Usage in THGN:
--------------
1. Compute time delta: time_delta = current_time - last_update_time
2. Normalize: normalized = (time_delta - mean) / std
3. Encode: time_encoding = time_encoder(normalized)
4. Use in messages/embeddings: [memory, neighbor_memory, edge_feat, time_encoding]

Note: For hypergraphs, time encoding works the same as for graphs - it encodes
the time since a node's last interaction, regardless of whether that interaction
was in an edge (2 nodes) or a hyperedge (multiple nodes).

Reference: Temporal Graph Attention Network (TGAT)
"""

import torch
import numpy as np


class TimeEncode(torch.nn.Module):
  """
  Time Encoding module that converts time deltas into dense vector representations.
  
  Based on TGAT (Temporal Graph Attention Network) time encoding approach.
  Uses cosine of scaled time deltas with learnable frequency weights.
  
  Works identically for graphs and hypergraphs - encodes time since last interaction
  for any node, regardless of the size of the interaction (edge or hyperedge).
  """
  def __init__(self, dimension):
    super(TimeEncode, self).__init__()

    self.dimension = dimension
    self.w = torch.nn.Linear(1, dimension)

    self.w.weight = torch.nn.Parameter((torch.from_numpy(1 / 10 ** np.linspace(0, 9, dimension)))
                                       .float().reshape(dimension, -1))
    self.w.bias = torch.nn.Parameter(torch.zeros(dimension).float())

  def forward(self, t):
    # t has shape [batch_size, seq_len]
    # Add dimension at the end to apply linear layer --> [batch_size, seq_len, 1]
    t = t.unsqueeze(dim=2)

    # output has shape [batch_size, seq_len, dimension]
    output = torch.cos(self.w(t))

    return output
