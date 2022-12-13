from typing import List

import torch
from torch import Tensor
import torch.nn.functional as F
from src.rmp.abstract_connector import AbstractConnector
from src.rmp.hierarchical_connector import HierarchicalConnector
from src.util import device, EdgeSet, MultiGraphWithPos, MultiGraph


class MultigraphConnector(HierarchicalConnector):
    """
    Naive remote message passing with fully connected clusters.
    """

    def __init__(self, fully_connect):
        super().__init__(fully_connect)

    def initialize(self, intra, inter, hyper, noise_scale=0):
        self.noise_scale = 0.005
        super().initialize(intra, inter, hyper)
        return []

    def run(self, graph: MultiGraphWithPos, clusters: List[Tensor], neighbors: List[Tensor], is_training: bool) -> MultiGraph:
        graph = super().run(graph, clusters, neighbors, is_training)
        nf, hnf = graph.node_features[0], graph.node_features[1]
        new_nf = torch.cat((nf, torch.tensor([[1, 0]] * len(nf))), dim=1)
        new_hnf = torch.cat((hnf, torch.tensor([[0, 1]] * len(hnf))), dim=1)

        mesh_edges = self.get_edges(graph, 'mesh_edges')
        world_edges = self.get_edges(graph, 'world_edges')
        inter_cluster = self.get_edges(graph, 'inter_cluster')
        intra_cluster_to_cluster = self.get_edges(graph, 'intra_cluster_to_cluster')
        intra_cluster_to_mesh = self.get_edges(graph, 'intra_cluster_to_mesh')

        new_me = torch.cat(
            (mesh_edges.features, torch.tensor([[1, 0, 0, 0]] * len(mesh_edges.features))),
            dim=1)
        mesh_edges = mesh_edges._replace(features=new_me, name='mesh_edges')

        new_ic = torch.cat(
            (inter_cluster.features, torch.tensor([[0, 1, 0, 0]] * len(inter_cluster.features))),
            dim=1)
        inter_cluster = inter_cluster._replace(features=new_ic, name='mesh_edges')

        new_icc = torch.cat(
            (intra_cluster_to_cluster.features,
             torch.tensor([[0, 0, 1, 0]] * len(intra_cluster_to_cluster.features))),
            dim=1)
        intra_cluster_to_cluster = intra_cluster_to_cluster._replace(features=new_icc, name='mesh_edges')

        new_icm = torch.cat(
            (intra_cluster_to_mesh.features,
             torch.tensor([[0, 0, 0, 1]] * len(intra_cluster_to_mesh.features))),
            dim=1)
        intra_cluster_to_mesh = intra_cluster_to_mesh._replace(features=new_icm, name='mesh_edges')


        mesh_edges = mesh_edges._replace(
            senders=torch.cat(
                (mesh_edges.senders, inter_cluster.senders, intra_cluster_to_cluster.senders, intra_cluster_to_mesh.senders),
                dim=0
            ),
            receivers=torch.cat(
                (mesh_edges.receivers, inter_cluster.receivers, intra_cluster_to_cluster.receivers, intra_cluster_to_mesh.receivers),
                dim=0
            ),
            features=torch.cat(
                (mesh_edges.features, inter_cluster.features, intra_cluster_to_cluster.features, intra_cluster_to_mesh.features),
                dim=0
            )
        )


        graph = graph._replace(edge_sets=[mesh_edges, world_edges])
        graph = graph._replace(node_features=[new_nf, new_hnf])

        return graph

    def get_edges(self, graph, edge_set_name) -> EdgeSet:
        return list(filter(lambda x: x.name == edge_set_name, graph.edge_sets))[0]
