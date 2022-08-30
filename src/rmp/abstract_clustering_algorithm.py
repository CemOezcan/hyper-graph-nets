from abc import ABC, abstractmethod
import random
from typing import List

import numpy as np
import torch
import wandb
from src.util import MultiGraphWithPos
from torch import Tensor


class AbstractClusteringAlgorithm(ABC):
    """
    Abstract superclass for clustering algorithms.
    """

    def __init__(self):
        """
        Initializes the clustering algorithm
        """
        self._num_clusters = 10
        self._treshold = 0
        self._alpha = 0.5
        self._sampling = False
        self._spotter_threshold = 0
        self._wandb = wandb.init(reinit=False)
        self._initialize()
        # TODO: Change graph input to initialize in order to preprocess the graph

    @abstractmethod
    def _initialize(self):
        """
        Special initialization function if parameterized preprocessing is necessary.

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def _cluster(self, graph: MultiGraphWithPos) -> List[int]:
        '''
        Run clustering algorithm given a multigraph or point cloud.
        '''
        raise NotImplementedError

    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        """
        Run clustering algorithm given a multigraph or point cloud.

        Parameters
        ----------
        graph :  Input data for the algorithm, represented by a multigraph or a point cloud.

        Returns clustering as a list.
        -------

        """
        labels = list(self._cluster(graph))

        if not self._sampling:
            return self._labels_to_indices(labels)
        
        spotter = self.spotter(graph, labels, self._spotter_threshold)
        exemplars = self.exemplars(labels, spotter, self._alpha)
        return self._combine_samples(spotter, exemplars)


    def _labels_to_indices(self, labels: List[int]) -> List[Tensor]:
        """
        Groups data points into clusters, given their class labels.

        Parameters
        ----------
        labels : Class labels of data points, where labels[i] represents the class label of data point i

        Returns clustering as a list.
        -------

        """
        indices = [list() for _ in range(max(labels) + 1)]

        for i in range(len(labels)):
            if labels[i] >= 0:
                indices[labels[i]].append(i)

        return [torch.tensor(x) for x in indices]

    def spotter(self, graph: MultiGraphWithPos, labels: List[int], threshold: int) -> List[int]:
        '''Given a graph with edges, traverse all edges and find vertices that are connected to each other, but belong to different clusters'''
        edge_set = [x for x in graph.edge_sets if x.name == 'mesh_edges']
        # for the sender and receiver of the edge set, find the corresponding label
        sender_labels = [labels[x] for x in edge_set[0].senders]
        receiver_labels = [labels[x] for x in edge_set[0].receivers]
        # combine senders, receivers, sender_labels, receiver_labels into a tensor of tuples
        # (sender, receiver, sender_label, receiver_label)
        edge_set_tensor = torch.stack((torch.tensor(edge_set[0].senders).to('cpu'), torch.tensor(edge_set[0].receivers).to('cpu'), torch.tensor(sender_labels), torch.tensor(receiver_labels)), dim=1)
        # find all elements of edge_set_tensor that have a different sender_label and receiver_label
        edges_different_clusters = torch.nonzero(torch.abs(edge_set_tensor[:, 2] - edge_set_tensor[:, 3]) > 0).squeeze()
        # for each element in edges_different_clusters, put them in a list of length labels and put the senders and receivers to their corresponding labels
        indices = [list() for _ in range(self._num_clusters)]
        for i in edges_different_clusters:
            indices[edge_set_tensor[i, 2].item()].append(edge_set_tensor[i, 0].item())
            indices[edge_set_tensor[i, 3].item()].append(edge_set_tensor[i, 1].item())
        result = [list() for _ in range(self._num_clusters)]
        for k, i in enumerate(indices):
            for e in set(i):
                if i.count(e) >= threshold:
                    result[k].append(e)
        self._wandb.log({f'spotter added': sum([len(x) for x in result])})
        return result

    def exemplars(self, labels: List[List[int]], spotter: List[List[int]], alpha: float) -> List[List[int]]:
        # for each list in labels, remove the elements of the list if the elements are also in spotter
        result = [list() for _ in range(self._num_clusters)]
        for i, e in enumerate(labels):
            if i not in spotter[e]:
                result[e].append(i)
        #randomly sample from the remaining elements of each list in result according to the ratio alpha
        for i in range(self._num_clusters):
            random.shuffle(result[i])
            result[i] = result[i][:int(len(result[i]) * alpha)]
        self._wandb.log({f'exemplars added': sum([len(x) for x in result])})
        return result

    def _combine_samples(self, spotter: List[List[int]], exemplars: List[List[int]]) -> List[List[int]]:
        # combine the lists of spotter and exemplars
        result = [list() for _ in range(self._num_clusters)]
        for i in range(self._num_clusters):
            result[i] = torch.tensor(spotter[i] + exemplars[i])
        return result

    def highest_dynamics(self, graph: MultiGraphWithPos, clusters: List[List[int]], top_k: int) -> List[List[int]]:
        dyn = [abs(x) for x in graph.node_dynamic.tolist()]
        cluster_dyn = [[dyn[index] for index in cluster] for cluster in clusters]

        indices = list()
        for a in range(len(cluster_dyn)):
            cluster = cluster_dyn[a]
            k = min(len(cluster), top_k)
            idx = np.argsort([-dynamics for dynamics in cluster])[:k]
            indices.append([clusters[a][i] for i in idx])

        return indices
