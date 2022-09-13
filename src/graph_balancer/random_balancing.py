import numpy as np
import wandb

from src.graph_balancer.abstract_graph_balancer import AbstractGraphBalancer
from src.util import MultiGraphWithPos


class RandomGraphBalancer(AbstractGraphBalancer):
    def __init__(self, params):
        super().__init__()
        self._edge_amount = params.get('graph_balancer').get(
            'random').get('edge_amount')
        self._remove_edges = params.get('graph_balancer').get('remove_edges')

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos):
        added_edges = {'senders': [], 'receivers': []}
        removed_edges = {'senders': [], 'receivers': []}
        vertices_amt = graph.node_features[0].size(dim=0)
        random_edge_pairs = np.random.choice(
            vertices_amt, size=(self._edge_amount, 2), replace=False)
        for e in random_edge_pairs:
            added_edges['senders'].append(e[0])
            added_edges['receivers'].append(e[1])
        wandb.log({'random added edges': len(added_edges['senders'])})
        if self._remove_edges:
            random_edge_removal = np.random.choice(
                vertices_amt, size=(self._edge_amount, 2), replace=False)
            for e in random_edge_removal:
                removed_edges['senders'].append(e[0])
                removed_edges['receivers'].append(e[1])
            wandb.log({'random removed edges': len(removed_edges['senders'])})
            return added_edges, removed_edges
        return added_edges, None
