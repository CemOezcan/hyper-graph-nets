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

    def create_graph(self, graph: MultiGraphWithPos, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        return self._balancer.create_graph(graph, mesh_edge_normalizer, is_training)

    def reset_balancer(self):
        self._balancer.reset_edges()
