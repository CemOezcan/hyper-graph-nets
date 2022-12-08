from typing import Callable, List

import torch
from torch import Tensor, nn

from src.migration.graphnet import GraphNet
from src.util import EdgeSet, MultiGraph


class MultiScaleGraphNet(GraphNet):
    """Multi-Edge and Multi-Node Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str, edge_sets: List[str]):
        super().__init__(model_fn, output_size, message_passing_aggregator, edge_sets)

        self.hyper_node_model_up = model_fn(output_size)
        self.hyper_node_models_cross = nn.ModuleList([model_fn(output_size) for _ in range(3)])
        self.node_model_down = model_fn(output_size)

    def forward(self, graph: MultiGraph, mask=None) -> MultiGraph:
        # TODO: Decide if 5 or 3 mp-steps
        # TODO: Modularize
        # TODO: world_edges
        new_edge_sets = dict()

        # 1
        # update_edges(mesh, world)
        self.perform_edge_updates(graph, 'mesh_edges', new_edge_sets)
        self.perform_edge_updates(graph, 'world_edges', new_edge_sets)
        temp_edge_sets = {'mesh_edges', 'world_edges'}.intersection(self.edge_models.keys())
        # update_nodes(mesh_nodes, mesh, world)
        self._update_node_features(graph, [new_edge_sets[x] for x in temp_edge_sets])

        # update_edges(up)
        self.perform_edge_updates(graph, 'intra_cluster_to_cluster', new_edge_sets)
        # update_nodes(hyper_nodes, up)
        self._update_hyper_node_features(graph, [new_edge_sets['intra_cluster_to_cluster']], self.hyper_node_model_up)

        # 2, 3, 4
        for i in range(3):
            # update_edges(hyper)
            self.perform_edge_updates(graph, 'inter_cluster', new_edge_sets)
            self.perform_edge_updates(graph, 'inter_cluster_world', new_edge_sets)
            temp_edge_sets = {'inter_cluster', 'inter_cluster_world'}.intersection(self.edge_models.keys())
            # update_nodes(hyper_nodes, hyper)
            self._update_hyper_node_features(graph, [new_edge_sets[x] for x in temp_edge_sets], self.hyper_node_models_cross[i])

        # update_edges(down)
        # TODO: Only geometric features for hypernodes?
        self.perform_edge_updates(graph, 'intra_cluster_to_mesh', new_edge_sets)
        # update_nodes(mesh, down)
        self._update_down(graph, [new_edge_sets['intra_cluster_to_mesh']])

        # 5
        # update_edges(mesh, world)
        self.perform_edge_updates(graph, 'mesh_edges', new_edge_sets)
        self.perform_edge_updates(graph, 'world_edges', new_edge_sets)
        temp_edge_sets = {'mesh_edges', 'world_edges'}.intersection(self.edge_models.keys())

        # update_nodes(mesh, mesh, world)
        self._update_node_features(graph, [new_edge_sets[x] for x in temp_edge_sets])

        return MultiGraph(graph.node_features, new_edge_sets.values())
