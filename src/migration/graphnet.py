from collections import OrderedDict

import torch
import torch_scatter
import torch.nn.functional as F

from torch import nn

from src.migration.attention import AttentionModel
from src.util import device, MultiGraph


class GraphNet(nn.Module):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, message_passing_aggregator, attention=False,
                 hierarchical=True, edge_sets=[]):
        super().__init__()
        self.hierarchical = hierarchical
        self.attention = attention

        self.node_model_cross = model_fn(output_size)
        self.edge_models = nn.ModuleDict({name: model_fn(output_size) for name in edge_sets})

        if hierarchical:
            self.hyper_node_model_up = model_fn(output_size)
            self.hyper_node_model_cross = model_fn(output_size)
            self.node_model_down = model_fn(output_size)

        if attention:
            self.attention_model = AttentionModel()
            self.linear_layer = nn.LazyLinear(1)
            self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.message_passing_aggregator = message_passing_aggregator

    def _update_edge_features(self, node_features, edge_set):
        """Aggregrates node features, and applies edge function."""
        node_features = torch.cat(tuple(node_features), dim=0)
        senders = edge_set.senders.to(device)
        receivers = edge_set.receivers.to(device)

        sender_features = torch.index_select(input=node_features, dim=0, index=senders)
        receiver_features = torch.index_select(input=node_features, dim=0, index=receivers)

        features = [sender_features, receiver_features, edge_set.features]
        features = torch.cat(features, dim=-1)

        return self.edge_models[edge_set.name](features)

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
        hyper_node_offset = len(node_features[0])
        node_features = torch.cat(tuple(node_features), dim=0)
        num_nodes = node_features.shape[0]
        features = [node_features]

        features = self.aggregation(
            list(filter(lambda x: x.name in self.edge_models.keys(), edge_sets)),
            features,
            num_nodes
        )
        updated_nodes_cross = self.node_model_cross(features[:hyper_node_offset])
        node_features_2 = torch.cat((updated_nodes_cross, node_features[hyper_node_offset:]), dim=0)

        if not self.hierarchical:
            return [updated_nodes_cross, node_features[hyper_node_offset:]]

        features = self.aggregation(
            list(filter(lambda x: x.name == 'intra_cluster_to_cluster', edge_sets)),
            [node_features_2],
            num_nodes
        )
        updated_hyper_nodes_up = self.hyper_node_model_up(features[hyper_node_offset:])
        node_features_3 = torch.cat((node_features_2[:hyper_node_offset], updated_hyper_nodes_up), dim=0)

        features = self.aggregation(
            list(filter(lambda x: x.name == 'inter_cluster', edge_sets)),
            [node_features_3],
            num_nodes
        )
        updated_hyper_nodes_cross = self.hyper_node_model_cross(features[hyper_node_offset:])
        node_features_4 = torch.cat((node_features_3[:hyper_node_offset], updated_hyper_nodes_cross), dim=0)

        features = self.aggregation(
            list(filter(lambda x: x.name == 'intra_cluster_to_mesh', edge_sets)),
            [node_features_4],
            num_nodes
        )
        updated_nodes_down = self.node_model_down(features[:hyper_node_offset])

        return [updated_nodes_down, node_features_4[hyper_node_offset:]]

    def aggregation(self, edge_sets, features, num_nodes):
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

        return torch.cat(features, dim=-1)

    def forward(self, graph, mask=None):
        """Applies GraphNetBlock and returns updated MultiGraph."""
        # apply edge functions
        new_edge_sets = []
        for edge_set in graph.edge_sets:
            updated_features = self._update_edge_features(graph.node_features, edge_set)
            new_edge_sets.append(edge_set._replace(features=updated_features))

        # apply node function
        new_node_features = self._update_node_features(graph.node_features, new_edge_sets)

        # add residual connections
        new_node_features = list(map(sum, zip(new_node_features, graph.node_features)))
        if mask is not None:
            new_node_features = self._mask_operation(mask, new_node_features, graph)
        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in zip(new_edge_sets, graph.edge_sets)]
        return MultiGraph(new_node_features, new_edge_sets)

    @staticmethod
    def _mask_operation(mask, new_node_features, graph):
        # TODO: why is this function necessary?
        mask = mask.repeat(new_node_features.shape[-1])
        mask = mask.view(new_node_features.shape[0], new_node_features.shape[1])
        return torch.where(mask, new_node_features, graph.node_features)
