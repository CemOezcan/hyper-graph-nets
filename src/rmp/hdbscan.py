from typing import List

import hdbscan
import numpy as np
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
        # TODO: Add velocity as fourth dimension, but only for later instances in a trajectory
        # TODO: Experimental parameter: Many clusters vs few clusters (min_pts=None vs. min_pts=10)
        min_cluster_size = 10
        min_samples = 5
        X = torch.cat((graph.target_feature, graph.mesh_features), dim=1)
        clustering = hdbscan.HDBSCAN(core_dist_n_jobs=-1, min_cluster_size=min_cluster_size, min_samples=min_samples).fit(X.to('cpu'))
        labels = clustering.labels_ + 1

        enum = list(zip(labels, range(len(X))))
        clusters = [list(map(lambda x: x[1], filter(lambda x: x[0] == label, enum))) for label in set(labels)]
        #indices = [torch.tensor(cluster) for cluster in clusters[1:]]
        # TODO: Special case for clusters[0] (noise)
        indices = self.exemplars(X, clustering.exemplars_)

        return indices

    def exemplars(self, X, exemplars):
        indices = list()
        for i in range(len(exemplars)):
            indices.append(list())
            for ex in exemplars[i]:
                mask = torch.eq(X, torch.tensor(
                    ex, device=device).repeat(X.shape[0], 1))
                for m in range(len(mask)):
                    value = True
                    for bool in mask[m]:
                        value = value and bool
                    if value:
                        indices[i].append(m)
        return [torch.tensor(x) for x in indices]

    def highest_dynamics(self, graph, clusters, min_cluster_size):
        dyn = [abs(x) for x in graph.node_dynamic.tolist()]
        new = list()

        for x in clusters:
            new.append(list())
            for i in x:
                new[-1].append(dyn[i])

        indices = list()
        for a in range(len(new)):
            n = new[a]
            idx = np.argsort([-number for number in n])[:min_cluster_size]
            indices.append([clusters[a][i] for i in idx])

        return [torch.tensor(x) for x in indices]
