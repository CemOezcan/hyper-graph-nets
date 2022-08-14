"""Model for FlagSimple."""

import torch
import torch_scatter
import torch.nn.functional as F

from torch import nn
import src.rmp.get_rmp as rmp
from src.rmp.remote_message_passing import RemoteMessagePassing
from src.migration.normalizer import Normalizer
from src.migration.meshgraphnet import MeshGraphNet
from src import util
from src.rmp.ricci import Ricci
from src.util import NodeType, EdgeSet, MultiGraph, device, MultiGraphWithPos


class FlagModel(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, params):
        super(FlagModel, self).__init__()
        self._params = params
        self._output_normalizer = Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = Normalizer(size=3 + NodeType.SIZE, name='node_normalizer')
        self._node_dynamic_normalizer = Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = Normalizer(size=7, name='mesh_edge_normalizer')
        self._intra_edge_normalizer = Normalizer(size=4, name='intra_edge_normalizer')
        self._inter_edge_normalizer = Normalizer(size=4, name='intra_edge_normalizer')

        # TODO: Parameterize
        self._model_type = 'flag'

        self.message_passing_steps = params.get('message_passing_steps')
        self.message_passing_aggregator = params.get('aggregation')
        self._attention = params.get('attention') == 'True'
        self._hierarchical = params.get('rmp').get('connector') == 'hierarchical'
        self._ricci = params.get('rmp').get('ricci') == 'True'

        self.learned_model = MeshGraphNet(
            output_size=params.get('size'),
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            attention=self._attention,
            hierarchical=self._hierarchical,
            ricci=self._ricci).to(device)

        # TODO: Parameterize clustering algorithm and node connector
        self._remote_graph = rmp.get_rmp(params)
        self._remote_graph.initialize(self._intra_edge_normalizer, self._inter_edge_normalizer)

    # TODO check if redundant: see graphnet.py_world_edge_normalizer
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
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(
                segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
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
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result

    def build_graph(self, inputs, is_training):
        """Builds input graph."""
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev|world_pos']
        node_type = inputs['node_type']
        velocity = world_pos - prev_world_pos
        one_hot_node_type = F.one_hot(
            node_type[:, 0].to(torch.int64), NodeType.SIZE)

        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        cells = inputs['cells']
        decomposed_cells = util.triangles_to_edges(cells)
        senders, receivers = decomposed_cells['two_way_connectivity']

        mesh_pos = inputs['mesh_pos']
        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=receivers))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)

        mesh_edges = EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)

        # TODO: Add world_edges here? (FlagDynamic instead of FlagSimple)
        # TODO: Change data structure

        num_nodes = node_type.shape[0]
        max_node_dynamic = self.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                           num_nodes,
                                                           operation='max').to(device)
        min_node_dynamic = self.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                           num_nodes,
                                                           operation='min').to(device)
        node_dynamic = self._node_dynamic_normalizer(max_node_dynamic - min_node_dynamic)

        graph = MultiGraphWithPos(node_features=[self._node_normalizer(node_features, is_training)],
                                  edge_sets=[mesh_edges], target_feature=world_pos,
                                  model_type=self._model_type, node_dynamic=node_dynamic)

        # No ripples: graph = MultiGraph(node_features=self._node_normalizer(node_features), edge_sets=[mesh_edges])
        # TODO: Normalize hyper nodes
        if self._params.get('rmp').get('ricci'):
            ricci = Ricci()
            graph = ricci.run(graph, inputs, self._mesh_edge_normalizer, is_training)
        graph = self._remote_graph.create_graph(graph, is_training)
        return graph

    def forward(self, graph):
        # TODO: Get rid of parameter: is_training
        return self.learned_model(graph)

    def update(self, inputs, per_node_network_output):
        """Integrate model outputs."""

        acceleration = self._output_normalizer.inverse(per_node_network_output) # TODO: generalize to multiple node types  [:len(inputs['world_pos'])]

        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev|world_pos']
        position = 2 * cur_position + acceleration - prev_position
        return position

    def get_output_normalizer(self):
        return self._output_normalizer

    def reset_remote_graph(self):
        self._remote_graph.reset_clusters()

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._mesh_edge_normalizer,
                   path + "_mesh_edge_normalizer.pth")
        torch.save(self._world_edge_normalizer,
                   path + "_world_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")
        torch.save(self._node_dynamic_normalizer,
                   path + "_node_dynamic_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth")
        self._output_normalizer = torch.load(path + "_output_normalizer.pth")
        self._mesh_edge_normalizer = torch.load(
            path + "_mesh_edge_normalizer.pth")
        self._world_edge_normalizer = torch.load(
            path + "_world_edge_normalizer.pth")
        self._node_normalizer = torch.load(path + "_node_normalizer.pth")
        self._node_dynamic_normalizer = torch.load(
            path + "_node_dynamic_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()
