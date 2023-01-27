from typing import List

import torch

from src.migration.normalizer import Normalizer
from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.rmp.abstract_connector import AbstractConnector
from src.util import MultiGraphWithPos, EdgeSet, MultiGraph, device
from src.rmp.coarser_mesh import CoarserClustering
from profilehooks import profile
import src.util

class RemoteMessagePassing:
    """
    Remote message passing for graph neural networks.
    """

    def __init__(self, clustering_algorithm: AbstractClusteringAlgorithm, connector: AbstractConnector):
        """
        Initialize the remote message passing strategy.

        Parameters
        ----------
            clustering_algorithm : AbstractClusteringAlgorithm
                 The clustering algorithm

            connector : AbstractConnector
                The connector

        """
        self._clustering_algorithm = clustering_algorithm
        self._node_connector = connector
        self._clusters = None
        self._connected_graphs = []

    def initialize(self, intra: Normalizer, inter: Normalizer, hyper: Normalizer) -> List:
        """
        Initialize normalizers after fetching the subclass according to the given configuration file

        Parameters
        ----------
        hyper : Normalizer
            Normalizer for hyper nodes

        intra : Normalizer
            Normalizer for intra cluster edges

        inter : Normalizer
            Normalizer for inter cluster edges

        Returns
        -------
            List
                An empty list

        """
        return self._node_connector.initialize(intra, inter, hyper)

    def create_graph(self, graph: MultiGraphWithPos, trajectory_index: int, step: int, is_training: bool) -> MultiGraph:
        """
        Template method: Identify clusters and connect them using remote edges.

        Parameters
        ----------
            graph : Input graph
            is_training : Whether the input is a training instance or not

        Returns
        -------
            MultiGraph
                The input graph with additional edges for remote message passing

        """
        print("trajectory index: {}, step: {}".format(trajectory_index, step))
        graph = graph._replace(node_features=graph.node_features[0])

        if type(self._clustering_algorithm) == CoarserClustering:
            if is_training and len(self._connected_graphs) > trajectory_index and len(self._connected_graphs[trajectory_index]) > step:
                new_graph_cpu = self._connected_graphs[trajectory_index][step]
                edge_sets = []
                for i in range(len(new_graph_cpu.edge_sets)):
                    edge_set = new_graph_cpu.edge_sets[i]
                    edge_sets.append(EdgeSet(edge_set.name, edge_set.features.to(src.util.device), edge_set.senders.to(src.util.device), edge_set.receivers.to(src.util.device)))
                node_features = []
                for i in range(len(new_graph_cpu.node_features)):
                    node_features.append(new_graph_cpu.node_features[i].to(src.util.device))
                new_graph = MultiGraph(node_features=node_features, edge_sets=edge_sets)
            else:
                if self._clusters is None:
                    self._clusters = self._clustering_algorithm.run(graph)
                self._neighbors = [
                    self._clustering_algorithm.represented_nodes,
                    self._clustering_algorithm.representing_nodes,
                    self._clustering_algorithm.mesh_edge_senders,
                    self._clustering_algorithm.mesh_edge_receivers,
                    self._clustering_algorithm.world_edge_senders,
                    self._clustering_algorithm.world_edge_receivers
                ]
                new_graph = self._node_connector.run(graph, self._clusters, self._neighbors, is_training)
                if len(self._connected_graphs) <= trajectory_index:
                    self._connected_graphs.append([])
                edge_sets = []
                for i in range(len(new_graph.edge_sets)):
                    edge_set = new_graph.edge_sets[i]
                    edge_sets.append(EdgeSet(edge_set.name, edge_set.features.to('cpu'), edge_set.senders.to('cpu'), edge_set.receivers.to('cpu')))
                node_features = []
                for i in range(len(new_graph.node_features)):
                    node_features.append(new_graph.node_features[i].to('cpu'))
                new_graph_cpu = MultiGraph(node_features=node_features, edge_sets=edge_sets)
                self._connected_graphs[trajectory_index].append(new_graph_cpu)
            torch.cuda.empty_cache()
        else:
            if self._clusters is None:
                print(type(self._clustering_algorithm))
                if graph.obstacle_nodes is not None:
                    self.remove_obstacles(graph)
                else:
                    self._clusters = self._clustering_algorithm.run(graph)
            self._neighbors = self._clustering_algorithm.neigboring_clusters
            new_graph = self._node_connector.run(graph, self._clusters, self._neighbors, is_training)
        return new_graph

    def remove_obstacles(self, graph):
        indices = graph.obstacle_nodes.nonzero().squeeze()
        fst = indices[0]
        lst = indices[-1]

        if fst == 0:
            # include [lst + 1, -1]
            new_nodes = graph.node_features[lst + 1:]
            new_target = graph.target_feature[lst + 1:]
            new_mesh = graph.mesh_features[lst + 1:]
            edges = graph.unnormalized_edges
            s, r, f = edges.senders, edges.receivers, edges.features
            s_mask = torch.gt(s, lst)
            r_mask = torch.gt(r, lst)
            mask = torch.logical_and(s_mask, r_mask)

            new_s = torch.tensor([sender - lst - 1 for sender in s[mask]])
            new_r = torch.tensor([receiver - lst - 1 for receiver in r[mask]])

            new_edges = EdgeSet(name='mesh_edges',
                                features=f[mask],
                                receivers=new_r,
                                senders=new_s)

            new_graph = graph._replace(node_features=new_nodes, target_feature=new_target, mesh_features=new_mesh, unnormalized_edges=new_edges)
            offset = lst + 1
            b4 = True

        else:
            # include [0, fst - 1]
            new_nodes = graph.node_features[:fst]
            new_target = graph.target_feature[:fst]
            new_mesh = graph.mesh_features[:fst]
            edges = graph.unnormalized_edges
            s, r, f = edges.senders, edges.receivers, edges.features
            s_mask = torch.lt(s, fst)
            r_mask = torch.lt(r, fst)
            mask = torch.logical_and(s_mask, r_mask)

            new_edges = EdgeSet(name='mesh_edges',
                                features=f[mask],
                                receivers=r[mask],
                                senders=s[mask])

            new_graph = graph._replace(node_features=new_nodes, target_feature=new_target, mesh_features=new_mesh, unnormalized_edges=new_edges)

            offset = lst - fst + 1
            b4 = False

        self._clusters = self._clustering_algorithm.run(new_graph, offset,
                                                        b4) if self._clusters is None else self._clusters

        if fst == 0:
            for i in range(len(self._clusters)):
                tensor = torch.tensor([lst + 1] * len(self._clusters[i])).to('cpu')
                self._clusters[i] = torch.add(tensor, self._clusters[i])

    def reset_clusters(self):
        """
        Reset the current clustering structure and therefore force its recomputation
        on the next call of :func:`RemoteMessagePassing.create_graph`
        """
        self._clusters = None

    def visualize_cluster(self, graph):
        """
        Visualize the clusters of the input graph.
        """
        self._clustering_algorithm.visualize_cluster(graph)
