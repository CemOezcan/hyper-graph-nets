import hdbscan
from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm


class HDBSCAN(AbstractClusteringAlgorithm):
    """
    Hierarchical Density Based Clustering for Applications with Noise.
    """
    def __init__(self):
        super().__init__()

    def _initialize(self):
        pass

    def run(self, graph):
        # TODO: More features
        X = graph.target_feature
        clustering = hdbscan.HDBSCAN().fit(X)
        labels = clustering.labels_ + 1

        enum = list(zip(labels, range(len(X))))
        clusters = [list(map(lambda x: x[1], filter(lambda x: x[0] == label, enum))) for label in set(labels)]
        # TODO: Special case for clusters[0] (noise)

        return clusters
