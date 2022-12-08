from typing import Callable, List

import torch
from torch import Tensor

from src.migration.graphnet import GraphNet
from src.util import EdgeSet, MultiGraph


class HeteroGraphNet(GraphNet):
    """Multi-Edge and Multi-Node Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str, edge_sets: List[str]):
        super().__init__(model_fn, output_size, message_passing_aggregator, edge_sets)
        self.hyper_node_model_cross = model_fn(output_size)

    def _update_node_features(self, graph: MultiGraph, edge_sets: List[EdgeSet]):
        """Aggregrates edge features, and applies node function."""
        node_features = graph.node_features
        hyper_node_offset = len(node_features[0])
        node_features = torch.cat(tuple(node_features), dim=0)
        num_nodes = node_features.shape[0]
        features = [node_features]

        features = self.aggregation(
            list(filter(lambda x: x.name in self.edge_models.keys(), edge_sets)),
            features,
            num_nodes
        )
        updated_nodes = self.node_model_cross(features[:hyper_node_offset])
        updated_hyper_nodes = self.hyper_node_model_cross(features[hyper_node_offset:])
        graph.node_features[0] = torch.add(updated_nodes, graph.node_features[0])
        graph.node_features[1] = torch.add(updated_hyper_nodes, graph.node_features[1])


