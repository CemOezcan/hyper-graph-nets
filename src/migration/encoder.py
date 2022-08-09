import torch
from torch import nn

from src.util import MultiGraph


class Encoder(nn.Module):
    """Encodes node and edge features into latent features."""

    def __init__(self, make_mlp, latent_size):
        super().__init__()
        self._make_mlp = make_mlp
        self._latent_size = latent_size

        self.node_model = self._make_mlp(latent_size)
        self.hyper_node_model = self._make_mlp(latent_size)

        self.mesh_edge_model = self._make_mlp(latent_size)
        self.inter_cluster_model = self._make_mlp(latent_size)
        self.intra_cluster_model = self._make_mlp(latent_size)

    def forward(self, graph):
        node_latents = [self.node_model(graph.node_features[0])]
        try:
            node_latents.append(self.hyper_node_model(graph.node_features[1]))
        except IndexError:
            pass

        new_edges_sets = []

        for index, edge_set in enumerate(graph.edge_sets):
            feature = edge_set.features
            if edge_set.name == "mesh_edges":
                latent = self.mesh_edge_model(feature)
            elif edge_set.name == "inter_cluster":
                latent = self.inter_cluster_model(feature)
            elif edge_set.name == "intra_cluster":
                latent = self.intra_cluster_model(feature)
            else:
                raise IndexError('Edge type {} unknown.'.format(edge_set.name))

            new_edges_sets.append(edge_set._replace(features=latent))

        return MultiGraph(node_latents, new_edges_sets)
