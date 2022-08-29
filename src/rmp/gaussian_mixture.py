from typing import List

import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from sklearn.mixture import GaussianMixture

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos


class GaussianMixtureClustering(AbstractClusteringAlgorithm):
    """
    Gaussian Mixture Clustering
    """

    def __init__(self, num_clusters, sampling, spotter_threshold, alpha):
        super().__init__()
        self._sampling = sampling
        self._num_clusters = num_clusters
        self._spotter_threshold = spotter_threshold
        self._alpha = alpha

    def _initialize(self):
        pass

    def _cluster(self, graph: MultiGraphWithPos) -> List[int]:
        sc = StandardScaler()
        X = torch.cat((graph.target_feature, graph.mesh_features), dim=1).to('cpu')
        X = sc.fit_transform(X)
        # TODO: Change parameter to 'kmeans++'
        clustering = GaussianMixture(n_components=self._num_clusters, random_state=0, init_params='kmeans').fit(X)
        return clustering.predict(X)

    


