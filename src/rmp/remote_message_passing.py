from src.migration.normalizer import Normalizer
from src.rmp.hierarchical_connector import HierarchicalConnector
from src.rmp.random_clustering import RandomClustering
from src.rmp.hdbscan import HDBSCAN
from src.rmp.multigraph_connector import MultigraphConnector
from src.util import MultiGraphWithPos


class RemoteMessagePassing:
    """
    Remote message passing for graph neural networks.
    """
    def __init__(self, normalizer: Normalizer):
        """
        Initialize the remote message passing strategy.

        Parameters
        ----------
        normalizer : Normalizer for remote edges
        """
        # TODO: Parameterize
        self._clustering_algorithm = HDBSCAN()
        self._node_connector = MultigraphConnector(normalizer)

    def create_graph(self, graph: MultiGraphWithPos, is_training: bool) -> MultiGraphWithPos:
        """
        Template method: Identify clusters and connect them using remote edges.

        Parameters
        ----------
        graph : Input graph
        is_training : Whether the input is a training instance or not

        Returns the input graph with additional edges for remote message passing.
        -------

        """
        # TODO: Replace lists with tensors
        graph = graph._replace(node_features=graph.node_features[0])
        clusters = self._clustering_algorithm.run(graph)
        new_graph = self._node_connector.run(graph, clusters, is_training)

        return new_graph
