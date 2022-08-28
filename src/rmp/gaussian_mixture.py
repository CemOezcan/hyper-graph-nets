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

    def __init__(self, num_clusters, sampling, spotter_threshold):
        super().__init__()
        self._sampling = sampling
        self._num_clusters = num_clusters
        self._spotter_threshold = spotter_threshold

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        sc = StandardScaler()
        X = torch.cat((graph.target_feature, graph.mesh_features), dim=1).to('cpu')
        X = sc.fit_transform(X)
        clustering = GaussianMixture(n_components=self._num_clusters, random_state=0, init_params='kmeans').fit(X)
        labels = clustering.predict(X)

        if self._sampling:
            return self._labels_to_indices(labels)
        else:
            spotter = self.spotter(graph, labels, self._spotter_threshold)

        return self._labels_to_indices(labels)

