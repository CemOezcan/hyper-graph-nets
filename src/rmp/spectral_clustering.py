import math
from typing import List

import numpy as np
import sklearn
import torch
from torch import Tensor
import wandb
from sklearn.preprocessing import MinMaxScaler

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos


class SpectralClustering(AbstractClusteringAlgorithm):
    """
    Spectral Clustering
    """

    def __init__(self, num_clusters):
        super().__init__()
        self._num_clusters = num_clusters
        self._wandb = wandb.init(reinit=False)

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        X = self._compute_affinity_matrix(graph)
        sc = sklearn.cluster.SpectralClustering(n_clusters=self._num_clusters, random_state=0, affinity='precomputed', assign_labels='cluster_qr')
        labels = sc.fit(X).labels_

        return self._labels_to_indices(labels)

    @staticmethod
    def _compute_affinity_matrix(graph: MultiGraphWithPos):
        # TODO: Inverse normalization of edge features?
        num_nodes = graph.node_features[0].shape[0]
        affinity_matrix = np.zeros((num_nodes, num_nodes), float)

        mesh_edges = graph.unnormalized_edges

        edges = list(
            zip(
                mesh_edges.senders.numpy(),
                mesh_edges.receivers.numpy(),
                [1 / math.sqrt(f[3] ** 2 + f[6] ** 2) for f in mesh_edges.features]
            )
        )

        maximum = 0
        indices = list()
        for edge in edges:
            if math.isinf(edge[2]):
                indices.append((edge[0], edge[1]))
                continue
            maximum = max(maximum, edge[2])
            affinity_matrix[edge[0], edge[1]] = edge[2]

        for i in indices:
            affinity_matrix[i] = maximum + 1

        return affinity_matrix

