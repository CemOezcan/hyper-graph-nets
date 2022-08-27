from src.util import MultiGraphWithPos


class GraphBalancer:
    """
    Remote message passing for graph neural networks.
    """

    def __init__(self, balancer):
        """
        Initialize the graph balancer strategy.
        """
        self._balancer = balancer

    def initialize(self):
        pass

    def get_balanced_graph(self, graph: MultiGraphWithPos, inputs, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        return self._balancer.run(graph, inputs, mesh_edge_normalizer, is_training)
