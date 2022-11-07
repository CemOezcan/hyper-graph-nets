from typing import Callable, List

import torch
from torch import Tensor

from src.migration.graphnet import GraphNet
from src.util import EdgeSet, MultiGraph


class MultiScaleGraphNet(GraphNet):
    """Multi-Edge and Multi-Node Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str, edge_sets: List[str]):
        super().__init__(model_fn, output_size, message_passing_aggregator, edge_sets)

        self.hyper_node_model_up = model_fn(output_size)
        self.hyper_node_model_cross = model_fn(output_size)
        self.node_model_down = model_fn(output_size)

    def forward(self, graph: MultiGraph, mask=None) -> MultiGraph:
        # TODO: Decide if 5 or 3 mp-steps
        # TODO: Modularize
        new_edge_sets = dict()

        # 1
        # update_edges(mesh, world)
        mesh_edges = list(filter(lambda x: x.name == 'mesh_edges', graph.edge_sets))[0]
        updates_mesh_features = self._update_edge_features(graph.node_features, mesh_edges)
        mesh_edges = mesh_edges._replace(features=updates_mesh_features)
        new_edge_sets['mesh_edges'] = mesh_edges
        # update_nodes(mesh_nodes, mesh, world)
        new_node_features = super()._update_node_features(graph.node_features, [mesh_edges])
        new_node_features = torch.add(new_node_features[0], graph.node_features[0])
        graph.node_features[0] = new_node_features

        # update_edges(up)
        up_edges = list(filter(lambda x: x.name == 'intra_cluster_to_cluster', graph.edge_sets))[0]
        updates_up_features = self._update_edge_features(graph.node_features, up_edges)
        up_edges = up_edges._replace(features=updates_up_features)
        new_edge_sets['intra_cluster_to_cluster'] = up_edges
        # update_nodes(hyper, up)
        new_hyper_node_features = self._update_hyper_node_features(graph.node_features, [up_edges], self.hyper_node_model_up)
        new_hyper_node_features = torch.add(new_hyper_node_features[1], graph.node_features[1])
        graph.node_features[1] = new_hyper_node_features

        # 2
        # update_edges(inter)
        # TODO: Overwrite
        inter_edges = list(filter(lambda x: x.name == 'inter_cluster', graph.edge_sets))[0]
        updates_inter_features = self._update_edge_features(graph.node_features, inter_edges)
        inter_edges = inter_edges._replace(features=updates_inter_features)
        # update_nodes(hyper, inter) TODO: up? (not in MSMGN-Paper)
        new_hyper_node_features = self._update_hyper_node_features(graph.node_features, [inter_edges], self.hyper_node_model_cross)
        new_hyper_node_features = torch.add(new_hyper_node_features[1], graph.node_features[1])
        graph.node_features[1] = new_hyper_node_features

        # 3
        # update_edges(inter)
        updates_inter_features = self._update_edge_features(graph.node_features, inter_edges)
        inter_edges = inter_edges._replace(features=updates_inter_features)
        # update_nodes(hyper, inter)
        new_hyper_node_features = self._update_hyper_node_features(graph.node_features, [inter_edges], self.hyper_node_model_cross)
        new_hyper_node_features = torch.add(new_hyper_node_features[1], graph.node_features[1])
        graph.node_features[1] = new_hyper_node_features

        # 4
        # update_edges(inter)
        updates_inter_features = self._update_edge_features(graph.node_features, inter_edges)
        inter_edges = inter_edges._replace(features=updates_inter_features)
        new_edge_sets['inter_cluster'] = inter_edges
        # update_nodes(hyper, inter)
        new_hyper_node_features = self._update_hyper_node_features(graph.node_features, [inter_edges], self.hyper_node_model_cross)
        new_hyper_node_features = torch.add(new_hyper_node_features[1], graph.node_features[1])
        graph.node_features[1] = new_hyper_node_features

        # update_edges(down)
        # TODO: Only geometric features for hypernodes?
        down_edges = list(filter(lambda x: x.name == 'intra_cluster_to_mesh', graph.edge_sets))[0]
        updates_down_features = self._update_edge_features(graph.node_features, down_edges)
        down_edges = down_edges._replace(features=updates_down_features)
        new_edge_sets['intra_cluster_to_mesh'] = down_edges
        # update_nodes(mesh, down)
        new_node_features = self.update(graph.node_features, [down_edges])
        new_node_features = torch.add(new_node_features[0], graph.node_features[0])
        graph.node_features[0] = new_node_features

        # 5
        # update_edges(mesh, world)
        updates_mesh_features = self._update_edge_features(graph.node_features, mesh_edges)
        mesh_edges = mesh_edges._replace(features=updates_mesh_features)
        new_edge_sets['mesh_edges'] = mesh_edges
        # update_nodes(mesh, mesh, world)
        new_node_features = super()._update_node_features(graph.node_features, [mesh_edges])
        new_node_features = torch.add(new_node_features[0], graph.node_features[0])
        graph.node_features[0] = new_node_features

        edge_set_tuples = list()

        for es in graph.edge_sets:
            edge_set_tuples.append((new_edge_sets[es.name], es))

        new_edge_sets = [es._replace(features=es.features + old_es.features)
                         for es, old_es in edge_set_tuples]

        return MultiGraph(graph.node_features, new_edge_sets)


    def _update_hyper_node_features(self, node_features: List[Tensor], edge_sets: List[EdgeSet], model) -> List[Tensor]:
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
        updated_nodes = model(features[hyper_node_offset:])

        return [node_features[:hyper_node_offset], updated_nodes]

    def update(self, node_features: List[Tensor], edge_sets: List[EdgeSet]) -> List[Tensor]:
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
        updated_nodes_cross = self.node_model_down(features[:hyper_node_offset])

        return [updated_nodes_cross, node_features[hyper_node_offset:]]

