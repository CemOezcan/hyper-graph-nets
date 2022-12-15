import math
from typing import List

import numpy as np
import sklearn
from sklearn.preprocessing import StandardScaler

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos


class KMeansClustering(AbstractClusteringAlgorithm):
    """
    K-Means Clustering
    """

    def __init__(self, num_clusters, sampling, alpha, threshold):
        super().__init__()
        self._sampling = sampling
        self._num_clusters = num_clusters
        self._alpha = alpha
        self._threshold = threshold

    def _initialize(self):
        pass

    def _cluster(self, graph: MultiGraphWithPos) -> List[int]:
        sc = StandardScaler()
        X = graph.mesh_features.to('cpu')[:, :2]
        X = sc.fit_transform(X)
        sc =sklearn.cluster.KMeans(n_clusters=self._num_clusters, random_state=0)

        return sc.fit(X).labels_
