from typing import List

import torch

from src.rmp.abstract_connector import AbstractConnector
from src.util import device, EdgeSet, MultiGraphWithPos


class MultigraphConnector(AbstractConnector):
    """
    Naive remote message passing with fully connected clusters.
    """

    def __init__(self, fully_connect):
        super().__init__()
        self._fully_connect = fully_connect

    def initialize(self, intra, inter):
        super().initialize(intra, inter)
        return ['inter_cluster', 'intra_cluster']

    def run(self, graph: MultiGraphWithPos, clusters: List[List], is_training: bool) -> MultiGraphWithPos:
        device_0 = 'cpu'
        target_feature = graph.target_feature.to(device_0)
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
            senders, receivers, edge_features = self._get_subgraph(model_type, [target_feature], cluster, cluster)
            snd.append(senders)
            rcv.append(receivers)
            edges.append(edge_features)

        edges = torch.cat(edges, dim=0).to(device)
        snd = torch.cat(snd, dim=0)
        rcv = torch.cat(rcv, dim=0)
        world_edges = EdgeSet(
            name='intra_cluster',
            features=self._intra_normalizer(edges, is_training),
            receivers=rcv,
            senders=snd
        )

        remote_edges.append(world_edges)

        # Inter cluster communication
        core_size = 1
        core_nodes = torch.tensor(sum(self._get_representatives(core_nodes, core_size), list())).to(device_0)
        senders, receivers, edge_features = self._get_subgraph(model_type, [target_feature], core_nodes, core_nodes)

        edge_features = edge_features.to(device)
        world_edges = EdgeSet(
            name='inter_cluster',
            features=self._inter_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders
        )

        remote_edges.append(world_edges)

        # Expansion
        edge_sets = graph.edge_sets
        edge_sets.extend(remote_edges)

        return MultiGraphWithPos(node_features=[graph.node_features],
                                 edge_sets=edge_sets, target_feature=[graph.target_feature],
                                 model_type=graph.model_type, node_dynamic=graph.node_dynamic)

    @staticmethod
    def _get_representatives(clusters: List[List], core_size: int) -> List[List]:
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
