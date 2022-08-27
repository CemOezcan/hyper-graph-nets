from typing import List

import numpy as np
import torch
from torch import Tensor

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos, device


class RandomClustering(AbstractClusteringAlgorithm):
    """
    Naive clustering strategy (Baseline). Pick clusters at random.
    """
    def __init__(self):
        super().__init__()

    def _initialize(self):
        self._num_clusters = 10

    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        num_nodes = graph.target_feature.shape[0]

        labels = list(map(int, np.multiply(np.random.rand(num_nodes), self._num_clusters)))
        indices = self._labels_to_indices(labels)

        for i, cluster in enumerate(indices):
            perm = torch.randperm(cluster.size(0))
            idx = perm[:len(cluster) // 2]
            indices[i] = cluster[idx]

        return indices
