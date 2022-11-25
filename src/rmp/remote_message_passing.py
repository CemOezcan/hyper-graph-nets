from typing import List

import torch

from src.migration.normalizer import Normalizer
from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.rmp.abstract_connector import AbstractConnector
from src.util import MultiGraphWithPos, EdgeSet, MultiGraph


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

    def initialize(self, intra: Normalizer, inter: Normalizer) -> List:
        """
        Initialize normalizers after fetching the subclass according to the given configuration file

        Parameters
        ----------
        intra : Normalizer
            Normalizer for intra cluster edges

        inter : Normalizer
            Normalizer for inter cluster edges

        Returns
        -------
            List
                An empty list

        """
        return self._node_connector.initialize(intra, inter)

    def create_graph(self, graph: MultiGraphWithPos, is_training: bool) -> MultiGraph:
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
        graph = graph._replace(node_features=graph.node_features[0])

        if self._clusters is None:
            if graph.node_dynamic is not None:
                self.remove_obstacles(graph)
            else:
                self._clusters = self._clustering_algorithm.run(graph)

        new_graph = self._node_connector.run(graph, self._clusters, is_training)
        return new_graph

    def remove_obstacles(self, graph):
        indices = graph.node_dynamic.nonzero().squeeze()
        fst = indices[0]
        lst = indices[-1]

        if fst == 0:
            # include [lst + 1, -1]
            new_nodes = graph.node_features[lst + 1:]
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

            new_graph = graph._replace(node_features=new_nodes, unnormalized_edges=new_edges)
            offset = lst + 1
            b4 = True

        else:
            # include [0, fst - 1]
            new_nodes = graph.node_features[:fst]
            edges = graph.unnormalized_edges
            s, r, f = edges.senders, edges.receivers, edges.features
            s_mask = torch.lt(s, fst)
            r_mask = torch.lt(r, fst)
            mask = torch.logical_and(s_mask, r_mask)

            new_edges = EdgeSet(name='mesh_edges',
                                features=f[mask],
                                receivers=r[mask],
                                senders=s[mask])

            new_graph = graph._replace(node_features=new_nodes, unnormalized_edges=new_edges)

            offset = lst - fst + 1
            b4 = False

        self._clusters = self._clustering_algorithm.run(new_graph, offset,
                                                        b4) if self._clusters is None else self._clusters

        if fst == 0:
            for i in range(len(self._clusters)):
                for j in range(len(self._clusters[i])):
                    self._clusters[i][j] += lst + 1

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

    @staticmethod
    def _graph_to_device(graph: MultiGraphWithPos, dev):
        """
        Send a graph to the given device

        Parameters
        ----------
            graph : MultiGraphWithPos
                The graph
            dev :
                The device

        Returns
        -------
            MultiGraphWithPos
                The graph on device dev

        """
        return MultiGraphWithPos(node_features=graph.node_features.to(dev),
                                 edge_sets=[
                                     EdgeSet(
                                         name=e.name,
                                         features=e.features.to(dev),
                                         senders=e.senders.to(dev),
                                         receivers=e.receivers.to(dev)
                                     ) for e in graph.edge_sets
                                 ],
                                 target_feature=graph.target_feature.to(dev),
                                 model_type=graph.model_type,
                                 node_dynamic=graph.node_dynamic.to(dev)
                                 )
