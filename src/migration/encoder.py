from typing import Callable, List

import torch
from torch import nn

from src.util import MultiGraph


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp: Callable, latent_size: int, edge_sets: List[str], hierarchical=True):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size

        self.node_model = self._make_mlp(latent_size)
        self.edge_models = nn.ModuleDict({name: self._make_mlp(latent_size) for name in edge_sets})
        self.hierarchical = hierarchical

        if hierarchical:
            self.hyper_node_model = self._make_mlp(latent_size)

    def forward(self, graph: MultiGraph) -> MultiGraph:
        node_latents = [self.node_model(graph.node_features[0])]
        if self.hierarchical:
            try:
                node_latents.append(self.hyper_node_model(graph.node_features[1]))
            except (IndexError, AttributeError):
                pass
        else:
            try:
                node_latents.append(self.node_model(graph.node_features[1]))
            except (IndexError, AttributeError):
                pass


        new_edges_sets = []
        for edge_set in graph.edge_sets:
            feature = edge_set.features
            try:
                latent = self.edge_models[edge_set.name](feature)
                new_edges_sets.append(edge_set._replace(features=latent))
            except KeyError:
                continue

        return MultiGraph(node_latents, new_edges_sets)
