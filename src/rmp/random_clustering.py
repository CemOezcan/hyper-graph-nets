from typing import List

import numpy as np

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos


class RandomClustering(AbstractClusteringAlgorithm):
    """
    Naive clustering strategy (Baseline). Pick clusters at random.
    """
    def __init__(self, sampling, num_clusters, alpha):
        super().__init__()
        self._sampling = sampling
        self._num_clusters = num_clusters
        self._alpha = alpha

    def _initialize(self):
        pass

    def _cluster(self, graph: MultiGraphWithPos) -> List[int]:
        return list(map(int, np.multiply(np.random.rand(graph.target_feature.shape[0]), self._num_clusters)))
