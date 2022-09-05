from typing import List, Tuple

import numpy as np
import scipy.spatial as ss
import torch
from torch import Tensor

from src import util
from src.migration.normalizer import Normalizer
from src.rmp.abstract_connector import AbstractConnector
from src.util import MultiGraphWithPos, EdgeSet, device, MultiGraph


class HierarchicalConnector(AbstractConnector):
    """
    Implementation of a hierarchical remote message passing strategy for hierarchical graph neural networks.
    """
    def __init__(self, fully_connect):
        super().__init__()
        self._fully_connect = fully_connect

    def initialize(self, intra, inter):
        super().initialize(intra, inter)
        return ['intra_cluster_to_mesh', 'intra_cluster_to_cluster', 'inter_cluster']

    def run(self, graph: MultiGraphWithPos, clusters: List[Tensor], is_training: bool) -> MultiGraph:
        device_0 = 'cpu'
        clustering_features = torch.cat((graph.target_feature, graph.mesh_features), dim=1).to(device_0)
        node_feature = graph.node_features.to(device_0)
        model_type = graph.model_type
        num_nodes = len(graph.node_features)

        hyper_edges = list()

        # Intra cluster communication
        # TODO: Decouple computation of senders and receivers from the computation of edge features
        hyper_nodes = torch.arange(num_nodes, len(clusters) + num_nodes).to(device_0)
        clustering_means = list()
        node_feature_means = list()
        for cluster in clusters:
            clustering_means.append(
                list(torch.mean(torch.index_select(input=clustering_features, dim=0, index=cluster), dim=0)))
            node_feature_means.append(
                list(torch.mean(torch.index_select(input=node_feature, dim=0, index=cluster), dim=0)))

        clustering_means = torch.tensor(clustering_means).to(device_0)
        node_feature_means = torch.tensor(node_feature_means).to(device_0)

        graph = graph._replace(target_feature=[clustering_features, clustering_means])
        graph = graph._replace(node_features=[node_feature, node_feature_means])

        clustering_features = graph.target_feature

        snd_to_cluster = list()
        snd_to_mesh = list()
        rcv_to_cluster = list()
        rcv_to_mesh = list()
        edges_to_cluster = list()
        edges_to_mesh = list()
        for hyper_node, cluster in zip(hyper_nodes, clusters):
            hyper_node = torch.tensor([hyper_node] * len(cluster)).to(device_0)
            senders, receivers, edge_features = self._get_subgraph(model_type, clustering_features, hyper_node, cluster)

            senders_to_mesh, senders_to_cluster = torch.split(senders, int(len(senders) / 2))
            receivers_to_mesh, receivers_to_cluster = torch.split(receivers, int(len(receivers) / 2))
            e_m, e_c = torch.split(edge_features, int(len(edge_features) / 2))

            snd_to_mesh.append(senders_to_mesh)
            snd_to_cluster.append(senders_to_cluster)
            rcv_to_mesh.append(receivers_to_mesh)
            rcv_to_cluster.append(receivers_to_cluster)
            edges_to_cluster.append(e_c)
            edges_to_mesh.append(e_m)

        snd_to_cluster = torch.cat(snd_to_cluster, dim=0)
        rcv_to_cluster = torch.cat(rcv_to_cluster, dim=0)
        edges_to_cluster = self._intra_normalizer(torch.cat(edges_to_cluster, dim=0).to(device), is_training)

        world_edges_to_cluster = EdgeSet(
            name='intra_cluster_to_cluster',
            features=edges_to_cluster,
            senders=snd_to_cluster,
            receivers=rcv_to_cluster)

        hyper_edges.append(world_edges_to_cluster)

        snd_to_mesh = torch.cat(snd_to_mesh, dim=0)
        rcv_to_mesh = torch.cat(rcv_to_mesh, dim=0)
        edges_to_mesh = self._intra_normalizer(torch.cat(edges_to_mesh, dim=0).to(device), is_training)

        world_edges_to_mesh = EdgeSet(
            name='intra_cluster_to_mesh',
            features=edges_to_mesh,
            senders=snd_to_mesh,
            receivers=rcv_to_mesh)

        hyper_edges.append(world_edges_to_mesh)

        # Inter cluster communication
        if self._fully_connect or clustering_means.shape[0] < 4:
            senders, receivers, edge_features = self._fully_connected(clustering_features, torch.tensor([hyper_nodes[0]]), model_type)
        else:
            senders, receivers, edge_features = self._delaunay(clustering_features, num_nodes, model_type)

        world_edges = EdgeSet(
            name='inter_cluster',
            features=self._inter_normalizer(edge_features.to(device), is_training),
            senders=senders,
            receivers=receivers)

        hyper_edges.append(world_edges)

        # Expansion
        edge_sets = graph.edge_sets
        edge_sets.extend(hyper_edges)

        return MultiGraph(node_features=graph.node_features, edge_sets=edge_sets)

    def _delaunay(self, clustering_features, num_nodes, model_type):
        _, points = torch.split(clustering_features[1], 3, dim=1)
        tri = ss.Delaunay(points)
        simplices = torch.tensor([list(map(lambda x: x + num_nodes, simplex)) for simplex in tri.simplices]).to('cpu')
        a = util.triangles_to_edges(simplices)
        return self._get_subgraph(model_type, clustering_features, a['senders'], a['receivers'])

    def _fully_connected(self, clustering_features, hyper_nodes, model_type):
        edges = torch.combinations(hyper_nodes, with_replacement=True).to('cpu')
        senders, receivers = torch.unbind(edges, dim=-1)
        mask = torch.not_equal(senders, receivers).to('cpu')
        edges = edges[mask]
        senders, receivers = torch.unbind(edges, dim=-1)
        return self._get_subgraph(model_type, clustering_features, senders, receivers)
