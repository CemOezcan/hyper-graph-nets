from typing import Dict, Tuple

from src.graph_balancer.abstract_graph_balancer import AbstractGraphBalancer
from src.util import MultiGraphWithPos


class GraphBalancer:
    """
    Remote message passing for graph neural networks.
    """

    def __init__(self, balancer: AbstractGraphBalancer):
        """
        Initialize the graph balancer strategy.
        """
        self._balancer = balancer

    def initialize(self):
        pass

    def get_balanced_graph(self, graph: MultiGraphWithPos, mesh_edge_normalizer, is_training: bool) -> Tuple[MultiGraphWithPos, Dict]:
        return self._balancer.run(graph, mesh_edge_normalizer, is_training)

    def create_graph(self, graph: MultiGraphWithPos, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        return self._balancer.create_graph(graph, mesh_edge_normalizer, is_training)

    def reset_balancer(self):
        self._balancer.reset_edges()
    
    def add_graph_balance_edges(self, graph: MultiGraphWithPos, added_edges: Dict, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        return self._balancer.add_graph_balance_edges(graph, added_edges, mesh_edge_normalizer, is_training)
