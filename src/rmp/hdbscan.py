from typing import List

import hdbscan
import torch
from torch import Tensor

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos, device
from sklearn.cluster import KMeans

class HDBSCAN(AbstractClusteringAlgorithm):
    """
    Hierarchical Density Based Clustering for Applications with Noise.
    """
    def __init__(self):
        super().__init__()

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        # TODO: Currently, all clusterings of the initial state of a trajectory return the same result, hence ...
        # TODO: More features !!! (or don't run clustering algorithm more than once for efficiency)
        X = graph.target_feature
        clustering = hdbscan.HDBSCAN(core_dist_n_jobs=-1).fit(X.to('cpu'))
        labels = clustering.labels_ + 1

        enum = list(zip(labels, range(len(X))))
        clusters = [list(map(lambda x: x[1], filter(lambda x: x[0] == label, enum))) for label in set(labels)]
        # TODO: Special case for clusters[0] (noise)

        indices = [torch.tensor(cluster) for cluster in clusters[1:]]

        return indices
