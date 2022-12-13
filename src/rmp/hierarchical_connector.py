import math
from typing import List

import numpy as np
import scipy.spatial as ss
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from src import util
from src.rmp.abstract_connector import AbstractConnector
from src.util import MultiGraphWithPos, EdgeSet, device, MultiGraph


class HierarchicalConnector(AbstractConnector):
    """
    Implementation of a hierarchical remote message passing strategy for hierarchical graph neural networks.
    """
    def __init__(self, fully_connect, noise_scale):
        super().__init__(fully_connect, noise_scale)

    def initialize(self, intra, inter, hyper):
        super().initialize(intra, inter, hyper)
        # TODO: fix
        return ['intra_cluster_to_mesh', 'intra_cluster_to_cluster', 'inter_cluster']#, 'inter_cluster_world']

    def run(self, graph: MultiGraphWithPos, clusters: List[Tensor], neighbors: List[Tensor], is_training: bool) -> MultiGraph:
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
        if is_training:
            zero_size = torch.zeros(clustering_means.size(), dtype=torch.float32).to(device_0)
            noise = torch.normal(zero_size, std=self._noise_scale).to(device_0)
            clustering_means += noise

        node_feature_means = torch.tensor(node_feature_means).to(device_0)
        spread_mesh = list()
        spread_world = list()
        for i in range(len(clustering_means)):
            means_mesh = torch.tensor(clustering_means[i][-3:]).to(device_0).repeat(len(clusters[i]), 1)
            points_mesh = torch.index_select(clustering_features[:, -3:], 0, clusters[i])

            means_world = torch.tensor(clustering_means[i][:3]).to(device_0).repeat(len(clusters[i]), 1)
            points_world = torch.index_select(clustering_features[:, :3], 0, clusters[i])

            spread_mesh.append(max([torch.dist(m, p) for m, p in zip(means_mesh, points_mesh)]))
            spread_world.append(max([torch.dist(m, p) for m, p in zip(means_world, points_world)]))

        spread_mesh, spread_world = torch.tensor(spread_mesh).to(device_0), torch.tensor(spread_world).to(device_0)
        cluster_sizes = torch.tensor([len(x) for x in clusters]).to(device_0)
        feature_augmentation = torch.stack([cluster_sizes, spread_mesh, spread_world], dim=-1).to(device)
        feature_augmentation = self._hyper_normalizer(feature_augmentation, is_training)
        node_feature_means = torch.cat([node_feature_means.to(device), feature_augmentation.to(device)], dim=-1).to(device)


        graph = graph._replace(target_feature=[clustering_features, clustering_means])
        graph = graph._replace(node_features=[node_feature.to(device), node_feature_means])

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
            senders, receivers, edge_features = self._fully_connected(clustering_features, hyper_nodes, model_type)
        else:
            # senders, receivers, edge_features = self._delaunay(clustering_features, num_nodes, model_type)
            senders, receivers, edge_features = self._downscale_triangulation(clustering_features, neighbors, model_type)

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

    def world_hyer_edges(self, graph: MultiGraphWithPos, clusters, clustering_means, hyper_nodes, num_nodes,
                         model_type, clustering_features, is_training):
        # TODO: use and reimplement
        index = 0
        device_0 = 'cpu'
        # Add inter cluster edges only if world edges to the receiving cluster exist
        world_edge_rec = list(filter(lambda x: x.name == 'world_edges', graph.edge_sets))[0].receivers
        colliding_hyper_nodes = list()
        for h, c in zip(hyper_nodes, clusters):
            if set(c.tolist()).intersection(set(world_edge_rec.tolist())):
                colliding_hyper_nodes.append(h)

        if not colliding_hyper_nodes:
            dist = list()
            for i in range(len(clustering_means)):
                if i != index - num_nodes:
                    dist.append(torch.dist(clustering_means[i][:3], clustering_means[index - num_nodes][:3]))
                else:
                    dist.append(torch.inf)
            colliding_hyper_nodes.append(num_nodes + np.argmin(dist))

        senders, receivers, edge_features = self._get_subgraph(
            model_type,
            clustering_features,
            torch.tensor([index] * len(colliding_hyper_nodes)).to(device_0),
            torch.tensor(colliding_hyper_nodes)

        )

        # Remove the obstacle cluster as the receiver of inter cluster edges
        indices_mask = (senders == index).nonzero()
        senders = senders[indices_mask].squeeze()
        receivers = receivers[indices_mask].squeeze()
        edge_features = edge_features[indices_mask].squeeze()

        world_edges = EdgeSet(
            name='inter_cluster_world',
            features=self._inter_normalizer(edge_features.to(device), is_training)[:, :4],
            senders=senders,
            receivers=receivers
        )

        return world_edges


    def _delaunay(self, clustering_features, num_nodes, model_type):
        _, points = torch.split(clustering_features[1], 3, dim=1)
        # TODO: Nearest neighbor triangulation
        points = points[:, :2]

        tri = ss.Delaunay(points)
        simplices = torch.tensor([list(map(lambda x: x + num_nodes, simplex)) for simplex in tri.simplices]).to('cpu')
        a = util.triangles_to_edges(simplices)
        return self._get_subgraph(model_type, clustering_features, a['senders'], a['receivers'])

    def _downscale_triangulation(self, clustering_features, neighbors, model_type):
        num_nodes = len(clustering_features[0])
        edges = torch.stack(neighbors) + torch.tensor(num_nodes).repeat(len(neighbors), 2)
        senders, receivers = torch.split(edges, 1, dim=1)
        return self._get_subgraph(model_type, clustering_features, senders.squeeze(), receivers.squeeze())

    def _fully_connected(self, clustering_features, hyper_nodes, model_type):
        edges = torch.combinations(hyper_nodes, with_replacement=True).to('cpu')
        senders, receivers = torch.unbind(edges, dim=-1)
        mask = torch.not_equal(senders, receivers).to('cpu')
        edges = edges[mask]
        senders, receivers = torch.unbind(edges, dim=-1)
        return self._get_subgraph(model_type, clustering_features, senders, receivers)
