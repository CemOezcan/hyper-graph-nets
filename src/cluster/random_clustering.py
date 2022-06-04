import torch

from src.cluster.abstract_clustering_algorithm import AbstractClusteringAlgorithm


class RandomClustering(AbstractClusteringAlgorithm):

    def __init__(self):
        super().__init__()

    def _initialize(self):
        pass

    def run(self, graph):
        # TODO: return Clusters and Representatives (Core and Border)

        # Cluster
        indices = []
        # TODO: Parameter: num. clusters
        num_clusters = 5

        target_feature_matrix = graph.target_feature
        num_nodes = target_feature_matrix.shape[0]
        cluster_size = num_nodes // num_clusters
        cluster_size_rest = num_nodes % num_clusters

        for i in range(num_clusters - 1):
            start_index = i * cluster_size
            end_index = (i + 1) * cluster_size
            indices.append((start_index, end_index))
        indices.append(((num_clusters - 1) * cluster_size, num_clusters * cluster_size + cluster_size_rest))

        # Reprs.
        selected_nodes = []
        for ripple in indices:
            cluster_size = ripple[1] - ripple[0]
            # TODO: Parameter: num. representatives
            core_size = min(5, cluster_size)
            random_mask = torch.randperm(n=cluster_size)[0:core_size]
            selected_nodes.append(random_mask)

        # TODO: Differentiate between core and border
        return indices, selected_nodes
