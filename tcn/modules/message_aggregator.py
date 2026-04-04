"""
Message Aggregator Module for TGN
=================================

Purpose: Combines multiple pending messages per node into a single aggregated message.
When a node has multiple interactions before memory is updated, it accumulates multiple
messages. The aggregator reduces these to one message per node for memory updates.

Role in TGN Flow:
-----------------
1. Edge interactions occur → Raw messages created and stored in memory.messages
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

Output Format:
-------------
Returns tuple: (to_update_node_ids, unique_messages, unique_timestamps)
  - to_update_node_ids: List of unique node IDs that have messages
  - unique_messages: Tensor [n_unique_nodes, message_dimension] with aggregated messages
  - unique_timestamps: Tensor [n_unique_nodes] with timestamps (last message's timestamp)

Implementations:
----------------
1. LastMessageAggregator: Keeps only the most recent message per node
   - Strategy: messages[node_id][-1] (last element in list)
   - Use case: Only most recent interaction matters
   - Formula: aggregated_message = messages[node_id][-1][0]

2. MeanMessageAggregator: Averages all messages per node
   - Strategy: Element-wise mean of all message vectors
   - Use case: All interactions contribute equally
   - Formula: aggregated_message = mean([m[0] for m in messages[node_id]], dim=0)

Running Example:
----------------
Setup: Node 50 has 3 messages, Node 60 has 1 message
  messages = {
    50: [
      (torch.tensor([0.1, 0.2, ...]), 100),  # msg1: [1110 dims]
      (torch.tensor([0.3, 0.4, ...]), 150),  # msg2: [1110 dims]
      (torch.tensor([0.5, 0.6, ...]), 200)   # msg3: [1110 dims]
    ],
    60: [
      (torch.tensor([0.7, 0.8, ...]), 100)   # msg4: [1110 dims]
    ]
  }

LastMessageAggregator:
  Node 50: aggregated = msg3 (most recent at time 200)
  Node 60: aggregated = msg4
  Output: ([50, 60], tensor([msg3, msg4]), tensor([200, 100]))

MeanMessageAggregator:
  Node 50: aggregated = mean([msg1, msg2, msg3], dim=0)
           = [(0.1+0.3+0.5)/3, (0.2+0.4+0.6)/3, ...]  # Element-wise average
  Node 60: aggregated = msg4
  Output: ([50, 60], tensor([mean_msg, msg4]), tensor([200, 100]))

Key Points:
-----------
- Aggregation happens BEFORE message processing (MessageFunction)
- Multiple messages can accumulate during batch processing
- Timestamp returned is always from the last message (most recent)
- Empty message lists are skipped (nodes with no messages excluded)
- Aggregation reduces variable number of messages per node to exactly 1 per node
"""

from collections import defaultdict
import torch
import numpy as np


class MessageAggregator(torch.nn.Module):
  """
  Abstract class for the message aggregator module, which given a batch of node ids and
  corresponding messages, aggregates messages with the same node id.
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
    """Only keep the last message for each node"""    
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []
    
    to_update_node_ids = []
    
    for node_id in unique_node_ids:
        if len(messages[node_id]) > 0:
            to_update_node_ids.append(node_id)
            unique_messages.append(messages[node_id][-1][0])
            unique_timestamps.append(messages[node_id][-1][1])
    
    unique_messages = torch.stack(unique_messages) if len(to_update_node_ids) > 0 else []
    unique_timestamps = torch.stack(unique_timestamps) if len(to_update_node_ids) > 0 else []

    return to_update_node_ids, unique_messages, unique_timestamps


class MeanMessageAggregator(MessageAggregator):
  def __init__(self, device):
    super(MeanMessageAggregator, self).__init__(device)

  def aggregate(self, node_ids, messages):
    """Average all messages for each node"""
    unique_node_ids = np.unique(node_ids)
    unique_messages = []
    unique_timestamps = []

    to_update_node_ids = []
    n_messages = 0

    for node_id in unique_node_ids:
      if len(messages[node_id]) > 0:
        n_messages += len(messages[node_id])
        to_update_node_ids.append(node_id)
        unique_messages.append(torch.mean(torch.stack([m[0] for m in messages[node_id]]), dim=0))
        unique_timestamps.append(messages[node_id][-1][1])

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
