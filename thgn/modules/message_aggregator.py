"""
Message Aggregator Module for THGN
===================================

Purpose: Combines multiple pending messages per node into a single aggregated message.
When a node has multiple hyperedge interactions before memory is updated, it accumulates multiple
messages. The aggregator reduces these to one message per node for memory updates.

Role in THGN Flow:
------------------
1. Hyperedge interactions occur → Raw messages created and stored in memory.messages
   (Each node in a hyperedge receives a message about the interaction)
2. Messages accumulate → Multiple messages per node: {node_id: [(message, timestamp), ...]}
3. Messages aggregated → MessageAggregator combines multiple messages per node ← THIS MODULE
4. Messages processed → MessageFunction transforms aggregated messages
5. Memory updated → Processed messages used by MemoryUpdater to update node memories

Input Format:
-------------
messages: Dictionary mapping node_id to list of (message_vector, timestamp) tuples
  Format: {node_id: [(message_tensor, timestamp), ...]}
  
Example:
  messages = {
    50: [(msg1_tensor, 100), (msg2_tensor, 150), (msg3_tensor, 200)],
    60: [(msg4_tensor, 100)]
  }
  node_ids = [50, 60]
  
  Note: In THGN, a single hyperedge interaction (e.g., [50, 60, 70] at time 100) creates
  messages for all nodes in that hyperedge (50, 60, 70), but the aggregator still processes
  them the same way as in TGN.

Output Format:
--------------
Returns tuple: (to_update_node_ids, unique_messages, unique_timestamps)
  - to_update_node_ids: List of unique node IDs that have messages
  - unique_messages: Tensor [n_unique_nodes, message_dimension] with aggregated messages
  - unique_timestamps: Tensor [n_unique_nodes] with timestamps (last message's timestamp)

Implementations:
----------------
1. LastMessageAggregator: Keeps only the most recent message per node
   - Strategy: messages[node_id][-1] (last element in list)
   - Use case: Only most recent hyperedge interaction matters
   - Formula: aggregated_message = messages[node_id][-1][0]

2. MeanMessageAggregator: Averages all messages per node
   - Strategy: Element-wise mean of all message vectors
   - Use case: All hyperedge interactions contribute equally
   - Formula: aggregated_message = mean([m[0] for m in messages[node_id]], dim=0)

Running Example:
----------------
Setup: Node 50 has 3 messages (from 3 different hyperedge interactions), Node 60 has 1 message
  Hyperedge interactions:
    - H1: [50, 60, 70] at time 100 → messages for nodes 50, 60, 70
    - H2: [50, 80] at time 150 → message for node 50
    - H3: [50, 90, 100] at time 200 → messages for nodes 50, 90, 100
  
  Accumulated messages:
    messages = {
      50: [
        (torch.tensor([0.1, 0.2, ...]), 100),  # msg1 from H1: [message_dim dims]
        (torch.tensor([0.3, 0.4, ...]), 150),  # msg2 from H2: [message_dim dims]
        (torch.tensor([0.5, 0.6, ...]), 200)   # msg3 from H3: [message_dim dims]
      ],
      60: [
        (torch.tensor([0.7, 0.8, ...]), 100)   # msg4 from H1: [message_dim dims]
      ]
    }

LastMessageAggregator:
  Node 50: aggregated = msg3 (most recent at time 200, from H3)
  Node 60: aggregated = msg4 (from H1)
  Output: ([50, 60], tensor([msg3, msg4]), tensor([200, 100]))

MeanMessageAggregator:
  Node 50: aggregated = mean([msg1, msg2, msg3], dim=0)
           = [(0.1+0.3+0.5)/3, (0.2+0.4+0.6)/3, ...]  # Element-wise average
  Node 60: aggregated = msg4
  Output: ([50, 60], tensor([mean_msg, msg4]), tensor([200, 100]))

Key Points:
-----------
- Aggregation happens BEFORE message processing (MessageFunction)
- Multiple messages can accumulate during batch processing (from multiple hyperedge interactions)
- Timestamp returned is always from the last message (most recent)
- Empty message lists are skipped (nodes with no messages excluded)
- Aggregation reduces variable number of messages per node to exactly 1 per node
- The aggregator is hyperedge-agnostic: it operates on the message dictionary structure,
  which is the same whether messages come from binary edges (TGN) or hyperedges (THGN)
"""

from collections import defaultdict
import torch
import numpy as np


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
  
  In THGN, messages come from hyperedge interactions (each node in a hyperedge receives
  a message), but the aggregation logic remains the same as in TGN.
  """
  def __init__(self, device):
    super(MessageAggregator, self).__init__()
    self.device = device

  def aggregate(self, node_ids, messages):
    """
    Given a list of node ids, and a list of messages of the same length, aggregate different
    messages for the same id using one of the possible strategies.
    :param node_ids: A list of node ids of length batch_size
    :param messages: A tensor of shape [batch_size, message_length]
    :param timestamps A tensor of shape [batch_size]
    :return: A tensor of shape [n_unique_node_ids, message_length] with the aggregated messages
    """

  def group_by_id(self, node_ids, messages, timestamps):
    node_id_to_messages = defaultdict(list)

    for i, node_id in enumerate(node_ids):
      node_id_to_messages[node_id].append((messages[i], timestamps[i]))

    return node_id_to_messages


class LastMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(LastMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Only keep the last message for each node (most recent hyperedge interaction)"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            # Sort messages by timestamp to ensure chronological order
            # Extract timestamps and sort
            sorted_messages = sorted(messages[node_id], 
                                   key=lambda x: x[1].item() if isinstance(x[1], torch.Tensor) else x[1])
            
            # Take the most recent message (last after sorting)
            to_update_node_ids.append(node_id)
            unique_messages.append(sorted_messages[-1][0])
            unique_timestamps.append(sorted_messages[-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    # DEBUG: Print aggregated info
    # if len(to_update_node_ids) > 0:
    #   print(f"[DEBUG Aggregator] Aggregated {len(to_update_node_ids)} nodes with messages")
    #   for i, node_id in enumerate(to_update_node_ids[:10]):  # First 10
    #     ts = unique_timestamps[i].item() if isinstance(unique_timestamps[i], torch.Tensor) else unique_timestamps[i]
    #     print(f"  Node {node_id}: timestamp={ts}")

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Average all messages for each node (all hyperedge interactions contribute equally)"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        # Sort messages by timestamp to ensure chronological order
        sorted_messages = sorted(messages[node_id],
                               key=lambda x: x[1].item() if isinstance(x[1], torch.Tensor) else x[1])
        
        n_messages += len(sorted_messages)
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in sorted_messages]), dim=0))
        # Use timestamp from most recent message
        unique_timestamps.append(sorted_messages[-1][1])

    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


def get_message_aggregator(aggregator_type, device):
  if aggregator_type == "last":
    return LastMessageAggregator(device=device)
  elif aggregator_type == "mean":
    return MeanMessageAggregator(device=device)
  else:
    raise ValueError("Message aggregator {} not implemented".format(aggregator_type))
