import numpy as np
from src.graph_balancer.abstract_graph_balancer import AbstractGraphBalancer
from src.util import MultiGraphWithPos


class RandomGraphBalancer(AbstractGraphBalancer):
    def __init__(self, params):
        super().__init__()
        self._edge_amount = params.get('graph_balancer').get(
            'random').get('edge_amount')

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        added_edges = {'senders': [], 'receivers': []}
        vertices_amt = graph.node_features[0].size(dim=0)
        random_edge_pairs = np.random.choice(
            vertices_amt, size=(self._edge_amount, 2), replace=False)
        for e in random_edge_pairs:
            added_edges['senders'].append(e[0])
            added_edges['receivers'].append(e[1])
        graph = self.add_graph_balance_edges(
            graph, added_edges, mesh_edge_normalizer, is_training)
        self._wandb.log({'random added edges': len(added_edges['senders'])})
        return graph
