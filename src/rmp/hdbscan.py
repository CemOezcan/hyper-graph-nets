import random
from typing import List

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos
from torch import Tensor

import hdbscan


class HDBSCAN(AbstractClusteringAlgorithm):
    """
    Hierarchical Density Based Clustering for Applications with Noise.
    """

    def __init__(self, sampling, max_cluster_size, min_cluster_size, min_samples, spotter_threshold):
        super().__init__()
        self._sampling = sampling
        self._max_cluster_size = max_cluster_size
        self._min_cluster_size = min_cluster_size
        self._spotter_threshold = spotter_threshold
        self._min_samples = min_samples

    def _initialize(self):
        pass
    
    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        """
        Run clustering algorithm given a multigraph or point cloud.

        Parameters
        ----------
        graph :  Input data for the algorithm, represented by a multigraph or a point cloud.

        Returns clustering as a list.
        -------

        """
        clustering = self._cluster(graph)
        labels = clustering.labels_
        self._num_clusters = len(self._labels_to_indices(labels))

        if not self._sampling:
            return self._labels_to_indices(labels)
        
        spotter = self.spotter(clustering, self._spotter_threshold)
        exemplars = self.exemplars(clustering)
        top_k = self.highest_dynamics(graph, labels, self._alpha)
        return self._combine_samples(spotter, exemplars, top_k)

    def _cluster(self, graph: MultiGraphWithPos) -> List[int]:
        # TODO: Currently, all clusterings of the initial state of a trajectory return the same result, hence ...
        # TODO: More features !!! (or don't run clustering algorithm more than once for efficiency)
        # TODO: Add velocity as fourth dimension, but only for later instances in a trajectory
        # TODO: Experimental parameter: Many clusters vs few clusters (min_pts=None vs. min_pts=10)
        # TODO: Normalize
        sc = StandardScaler()
        X = graph.target_feature.to('cpu')
        X = sc.fit_transform(X)
        clustering = hdbscan.HDBSCAN(core_dist_n_jobs=-1, max_cluster_size=self._max_cluster_size, min_samples=self._min_samples,
                                     min_cluster_size=self._min_cluster_size, prediction_data=True).fit(X)
        labels = clustering.labels_
        self._wandb.log({'hdbscan cluster': labels.max(
        ), 'hdbscan noise': len([x for x in labels if x < 0])})
        return clustering

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
