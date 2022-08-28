from typing import List

import hdbscan
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import Tensor
import wandb

from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos, device
from sklearn.cluster import KMeans, DBSCAN


class HDBSCAN(AbstractClusteringAlgorithm):
    """
    Hierarchical Density Based Clustering for Applications with Noise.
    """

    def __init__(self, sampling, threshold):
        super().__init__()
        self._sampling = sampling
        self._threshold = threshold
        self._wandb = wandb.init(reinit=False)

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        # TODO: Currently, all clusterings of the initial state of a trajectory return the same result, hence ...
        # TODO: More features !!! (or don't run clustering algorithm more than once for efficiency)
        # TODO: Add velocity as fourth dimension, but only for later instances in a trajectory
        # TODO: Experimental parameter: Many clusters vs few clusters (min_pts=None vs. min_pts=10)
        min_cluster_size = 10
        min_samples = 2
        # TODO: Normalize
        sc = StandardScaler()
        X = graph.target_feature.to('cpu')
        X = sc.fit_transform(X)
        clustering = hdbscan.HDBSCAN(core_dist_n_jobs=-1, max_cluster_size=50, prediction_data=True).fit(X)
        labels = clustering.labels_
        self._wandb.log({'hdbscan cluster': labels.max(
        ), 'hdbscan noise': len([x for x in labels if x < 0])})

        if not self._sampling:
            indices = self._labels_to_indices(labels)
        else:
        # TODO: Special case for clusters[0] (noise)
            exemplars = self.exemplars(clustering)
            spotter = self.spotter(clustering, self._threshold)

            indices = [torch.tensor(list(set(e + s)))
                       for e, s in zip(exemplars, spotter)]

        return indices

    def exemplars(self, clustering):
        selected_clusters = clustering.condensed_tree_._select_clusters()
        raw_condensed_tree = clustering.condensed_tree_._raw_tree

        exemplars = []
        for cluster in selected_clusters:
            cluster_exemplars = np.array([], dtype=np.int64)
            for leaf in clustering._prediction_data._recurse_leaf_dfs(cluster):
                leaf_max_lambda = raw_condensed_tree['lambda_val'][
                    raw_condensed_tree['parent'] == leaf].max()
                points = raw_condensed_tree['child'][
                    (raw_condensed_tree['parent'] == leaf) &
                    (raw_condensed_tree['lambda_val'] == leaf_max_lambda)]
                cluster_exemplars = np.hstack([cluster_exemplars, points])
            exemplars.append(list(cluster_exemplars))

        return exemplars

    def spotter(self, clustering, threshold):
        soft_clusters = hdbscan.all_points_membership_vectors(clustering)
        spotter_candidates = [
            self.top_two_probs_diff(x) for x in soft_clusters]
        prob_diff = np.array([x[0] for x in spotter_candidates])
        prob_sum = np.sum(np.sort(soft_clusters, )[:, -2:], axis=1)
        metric = 1 - prob_diff[:] / prob_sum[:]
        spotter = np.where(metric > threshold)[0]
        indices = [[] for i in range(clustering.labels_.max() + 1)]
        [indices[spotter_candidates[x][1]].append(x) for x in spotter]
        return indices

    @staticmethod
    def top_two_probs_diff(probs):
        cluster = np.argsort(probs)
        return [probs[cluster[-1]] - probs[cluster[-2]], cluster[-1]]

    @staticmethod
    def assign_noise_to_cluster(clustering):
        soft_clusters = hdbscan.all_points_membership_vectors(
            clustering)
        return [np.argsort(x)[-1] for x in soft_clusters]

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
