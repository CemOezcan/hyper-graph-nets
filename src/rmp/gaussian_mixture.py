from typing import List

import torch
from torch import Tensor
from sklearn.mixture import GaussianMixture
import wandb

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos, device


class GaussianMixtureClustering(AbstractClusteringAlgorithm):
    """
    Gaussian Mixture Clustering
    """

    def __init__(self, num_clusters):
        super().__init__()
        self._num_clusters = num_clusters
        self._wandb = wandb.init(reinit=False)

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        X = torch.cat((graph.target_feature, graph.mesh_features), dim=1)
        clustering = GaussianMixture(
            n_components=self._num_clusters, random_state=0, init_params='k-means').fit(X.to('cpu'))
        labels = clustering.predict(X)

        return self._labels_to_indices(labels)

