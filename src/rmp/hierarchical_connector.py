from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor

from src.migration.normalizer import Normalizer
from src.rmp.abstract_connector import AbstractConnector
from src.util import MultiGraphWithPos, EdgeSet, device


class HierarchicalConnector(AbstractConnector):
    """
    Implementation of a hierarchical remote message passing strategy for hierarchical graph neural networks.
    """
    def __init__(self, normalizer: Normalizer):
        super().__init__(normalizer)

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos, clusters: List[Tensor], is_training: bool) -> MultiGraphWithPos:
        target_feature = graph.target_feature
        node_feature = graph.node_features
        model_type = graph.model_type
        num_nodes = len(graph.node_features)

        hyper_edges = list()

        # Intra cluster communication
        # TODO: Parameter: num. representatives
        # TODO: Maybe change the current convention of appending hypernodes to normal nodes --> Use separate MLPs for different node types
        hyper_nodes = torch.tensor(range(num_nodes, len(clusters) + num_nodes)).to(device)
        target_feature_means = list()
        node_feature_means = list()
        for cluster in clusters:
            # TODO: tensor operation?
            target_feature_means.append(
                list(torch.mean(torch.index_select(input=target_feature, dim=0, index=cluster), dim=0).to(device)))
            node_feature_means.append(
                list(torch.mean(torch.index_select(input=node_feature, dim=0, index=cluster), dim=0).to(device)))

        target_feature_means = torch.tensor(target_feature_means).to(device)
        node_feature_means = torch.tensor(node_feature_means).to(device)

        graph = graph._replace(target_feature=[target_feature, target_feature_means])
        graph = graph._replace(node_features=[node_feature, node_feature_means])
        # TODO: node_dynamic

        target_feature = graph.target_feature

        edges = list()
        snd = list()
        rcv = list()
        for hyper_node, cluster in zip(hyper_nodes, clusters):
            hyper_node = torch.tensor([hyper_node] * len(cluster)).to(device)

            senders, receivers, edge_features = self._get_subgraph(model_type, target_feature, hyper_node, cluster)
            snd.append(senders.to(device))
            rcv.append(receivers.to(device))
            edges.append(edge_features.to(device))

        edges = self._normalizer(torch.cat(edges, dim=0).to(device)).to(device)
        snd = torch.cat(snd, dim=0).to(device)
        rcv = torch.cat(rcv, dim=0).to(device)
        world_edges = EdgeSet(
            name='intra_cluster',
            features=self._normalizer(edges, None, is_training).to(device),
            receivers=rcv,
            senders=snd)

        hyper_edges.append(world_edges)

        # Inter cluster communication
        senders, receivers, edge_features = self._get_subgraph(model_type, target_feature, hyper_nodes, hyper_nodes)

        edge_features = self._normalizer(edge_features.to(device)).to(device)
        world_edges = EdgeSet(
            name='inter_cluster',
            features=self._normalizer(edge_features, None, is_training).to(device),
            receivers=receivers.to(device),
            senders=senders.to(device))

        hyper_edges.append(world_edges)

        # Expansion
        edge_sets = graph.edge_sets
        edge_sets.extend(hyper_edges)

        return MultiGraphWithPos(node_features=graph.node_features,
                                 edge_sets=edge_sets, target_feature=graph.target_feature,
                                 model_type=graph.model_type, node_dynamic=graph.node_dynamic)
