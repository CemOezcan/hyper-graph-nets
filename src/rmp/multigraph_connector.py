from typing import List

import torch

from src.migration.normalizer import Normalizer
from src.rmp.abstract_connector import AbstractConnector
from src.util import device, EdgeSet, MultiGraphWithPos


class MultigraphConnector(AbstractConnector):
    """
    Naive remote message passing with fully connected clusters.
    """

    def __init__(self, normalizer: Normalizer):
        super().__init__(normalizer)

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos, clusters: List[List], is_training: bool) -> MultiGraphWithPos:
        target_feature = graph.target_feature
        model_type = graph.model_type
        remote_edges = list()

        # Intra cluster communication
        # TODO: Parameter: num. representatives
        core_size = 5
        core_nodes = torch.tensor(self._get_representatives(clusters, core_size))

        edges = list()
        snd = list()
        rcv = list()
        for cluster in core_nodes:
            senders, receivers, edge_features = self._get_subgraph(model_type, target_feature, cluster, cluster)
            snd.append(senders)
            rcv.append(receivers)
            edges.append(edge_features)

        edges = self._normalizer(torch.cat(edges, dim=0))
        snd = torch.cat(snd, dim=0)
        rcv = torch.cat(rcv, dim=0)
        world_edges = EdgeSet(
            name='remote_edges',
            features=self._normalizer(edges, None, is_training),
            receivers=rcv,
            senders=snd)

        remote_edges.append(world_edges)

        # Inter cluster communication
        core_size = 1
        core_nodes = torch.tensor(sum(self._get_representatives(core_nodes, core_size), list()))
        senders, receivers, edge_features = self._get_subgraph(model_type, target_feature, core_nodes, core_nodes)

        edge_features = self._normalizer(edge_features)
        world_edges = EdgeSet(
            name='remote_edges',
            features=self._normalizer(edge_features, None, is_training),
            receivers=receivers,
            senders=senders)

        remote_edges.append(world_edges)

        # Expansion
        edge_sets = graph.edge_sets
        edge_sets.extend(remote_edges)

        return MultiGraphWithPos(node_features=graph.node_features,
                                 edge_sets=edge_sets, target_feature=graph.target_feature,
                                 model_type=graph.model_type, node_dynamic=graph.node_dynamic)

    def _get_representatives(self, clusters, core_size):
        representatives = list()
        for cluster in clusters:
            cluster_size = len(cluster)
            # TODO: pick based on max node_dynamics or distance from central node (representative)
            random_mask = torch.randperm(n=cluster_size)[0:core_size]
            representatives.append(random_mask)

        representatives = [x.tolist() for x in representatives]

        core_nodes = list()
        for indices, cluster in zip(representatives, clusters):
            tuples = filter(lambda x: x[0] in indices, enumerate(cluster))
            core_nodes.append(list(map(lambda x: x[1], tuples)))

        return core_nodes

    def _get_subgraph(self, model_type, target_feature, senders_list, receivers_list):
        senders = torch.cat(
            (senders_list.clone().detach(), receivers_list.clone().detach()), dim=0)
        receivers = torch.cat(
            (receivers_list.clone().detach(), senders_list.clone().detach()), dim=0)

        # TODO: Make model independent
        if model_type == 'flag' or model_type == 'deform_model':
            relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                       torch.index_select(input=target_feature, dim=0, index=receivers))
            edge_features = torch.cat(
                (relative_target_feature, torch.norm(relative_target_feature, dim=-1, keepdim=True)), dim=-1)
        else:
            raise Exception("Model type is not specified in RippleNodeConnector.")

        return senders, receivers, edge_features

