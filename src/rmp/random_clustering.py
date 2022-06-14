import torch

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm


class RandomClustering(AbstractClusteringAlgorithm):
    """
    Naive clustering strategy (Baseline). Pick clusters at random.
    """
    def __init__(self):
        super().__init__()

    def _initialize(self):
        self._num_clusters = 5

    def run(self, graph):
        # TODO: return Clusters and Representatives (Core and Border)

        # Cluster
        indices = []

        target_feature_matrix = graph.target_feature
        num_nodes = target_feature_matrix.shape[0]
        cluster_size = num_nodes // self._num_clusters
        cluster_size_rest = num_nodes % self._num_clusters

        for i in range(self._num_clusters - 1):
            start_index = i * cluster_size
            end_index = (i + 1) * cluster_size
            indices.append(range(start_index, end_index))
        indices.append(range((self._num_clusters - 1) * cluster_size,
                             self._num_clusters * cluster_size + cluster_size_rest))

        indices = [list(x) for x in indices]

        # TODO: Differentiate between core and border
        return indices
