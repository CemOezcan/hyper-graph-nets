
import torch
import torch_scatter
import torch.nn.functional as F

from torch import nn

from src.migration.attention import AttentionModel
from src.util import device, MultiGraph


class GraphNet(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, message_passing_aggregator, attention=False):
        super().__init__()
        self.mesh_edge_model = model_fn(output_size)
        self.world_edge_model = model_fn(output_size)
        self.node_model = model_fn(output_size)
        self.attention = attention
        if attention:
            self.attention_model = AttentionModel()
        self.message_passing_aggregator = message_passing_aggregator

        self.linear_layer = nn.LazyLinear(1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        senders = edge_set.senders.to(device)
        receivers = edge_set.receivers.to(device)
        sender_features = torch.index_select(
            input=node_features, dim=0, index=senders)
        receiver_features = torch.index_select(
            input=node_features, dim=0, index=receivers)
        features = [sender_features, receiver_features, edge_set.features]
        features = torch.cat(features, dim=-1)
        if edge_set.name == "mesh_edges":
            return self.mesh_edge_model(features)
        else:
            return self.world_edge_model(features)

    # TODO refactor
    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]
                   ), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        data = data.to(device)
        segment_ids = segment_ids.to(device)
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(
                segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape).to(device)
        if operation == 'sum':
            result = torch_scatter.scatter_add(
                data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(
                data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(
                data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(
                data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'std':
            result = torch_scatter.scatter_std(
                data.float(), segment_ids, out=result, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        num_nodes = node_features.shape[0]
        features = [node_features]
        for edge_set in edge_sets:
            if self.attention and self.message_passing_aggregator == 'pna':
                attention_input = self.linear_layer(edge_set.features)
                attention_input = self.leaky_relu(attention_input)
                attention = F.softmax(attention_input, dim=0)
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation='sum'))
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation='mean'))
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation='max'))
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation='min'))
            elif self.attention:
                attention_input = self.linear_layer(edge_set.features)
                attention_input = self.leaky_relu(attention_input)
                attention = F.softmax(attention_input, dim=0)
                features.append(
                    self.unsorted_segment_operation(torch.mul(edge_set.features, attention), edge_set.receivers,
                                                    num_nodes, operation=self.message_passing_aggregator))
            elif self.message_passing_aggregator == 'pna':
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='sum'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='mean'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='max'))
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers,
                                                    num_nodes, operation='min'))
            else:
                features.append(
                    self.unsorted_segment_operation(edge_set.features, edge_set.receivers, num_nodes,
                                                    operation=self.message_passing_aggregator))
        features = torch.cat(features, dim=-1)
        return self.node_model(features)

    def forward(self, graph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(
                graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(
            graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features += graph.node_features
        if mask is not None:
            mask = mask.repeat(new_node_features.shape[-1])
            mask = mask.view(
                new_node_features.shape[0], new_node_features.shape[1])
            new_node_features = torch.where(
                mask, new_node_features, graph.node_features)
        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        return MultiGraph(new_node_features, new_edge_sets)
