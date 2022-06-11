
from src.rmp.random_clustering import RandomClustering
from src.rmp.full_connector import FullConnector


class RemoteMessagePassing():

    def __init__(self, normalizer):
        self._clustering_algorithm = RandomClustering()
        self._node_connector = FullConnector(normalizer)

    def create_graph(self, graph, is_training):
        clusters = self._clustering_algorithm.run(graph)
        new_graph = self._node_connector.run(graph, clusters, is_training)

        return new_graph
