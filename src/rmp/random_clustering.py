from typing import List

import numpy as np
import torch
from torch import Tensor

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos


class RandomClustering(AbstractClusteringAlgorithm):
    """
    Naive clustering strategy (Baseline). Pick clusters at random.
    """
    def __init__(self, num_clusters, sampling, alpha, threshold):
        super().__init__()
        self._sampling = sampling
        self._num_clusters = num_clusters
        self._alpha = alpha
        self._threshold = threshold

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        labels = self._empty_cluster_handling(list(self._cluster(graph)))
        self._labels = labels
        indices = self._labels_to_indices(labels)

        if self._sampling:
            for i, cluster in enumerate(indices):
                perm = torch.randperm(cluster.size(0))
                idx = perm[:int(len(cluster) * self._alpha) + 1]
                indices[i] = cluster[idx]

        return indices

    def _cluster(self, graph: MultiGraphWithPos) -> List[int]:
        return list(map(int, np.multiply(np.random.rand(graph.target_feature.shape[0]), self._num_clusters)))
