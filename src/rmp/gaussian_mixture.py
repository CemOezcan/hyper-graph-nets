from typing import List

from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos


class GaussianMixtureClustering(AbstractClusteringAlgorithm):
    """
    Gaussian Mixture Clustering
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
        clustering = GaussianMixture(n_components=self._num_clusters, random_state=0, init_params='k-means++').fit(X)

        return clustering.predict(X)
