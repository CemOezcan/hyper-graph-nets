from abc import ABC, abstractmethod
import random
from typing import List

import torch
import wandb
from src.util import MultiGraphWithPos
from torch import Tensor
import seaborn as sns
import colorcet as cc
import numpy as np

class AbstractClusteringAlgorithm(ABC):
    """
    Abstract superclass for clustering algorithms.
    """

    def __init__(self):
        """
        Initializes the clustering algorithm
        """
        self._num_clusters = 10
        self._threshold = 0
        self._alpha = 0.5
        self._sampling = False
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
        labels = self._empty_cluster_handling(list(self._cluster(graph)))
        self._labels = labels

        if not self._sampling:
            indices = self._labels_to_indices(labels)
            if self.__class__.__name__ == 'RandomClustering':
                for i, cluster in enumerate(indices):
                    perm = torch.randperm(cluster.size(0))
                    idx = perm[:len(cluster) // 2]
                    indices[i] = cluster[idx]
            return indices
        
        spotter = self.spotter(graph, labels, self._alpha, self._threshold)
        exemplars = self.exemplars(labels, spotter, self._alpha)
        top_k = self.highest_dynamics(graph, labels, self._alpha)
        return self._combine_samples(spotter, exemplars, top_k)

    def visualize_cluster(self, coordinates):
        palette = [[e * 255 for e in x] for x in sns.color_palette(cc.glasbey, len(set(self._labels)))]
        coordinates = [np.concatenate([c, palette[l]]) for c, l in zip(coordinates, self._labels)]
        self._wandb.log({'cluster': [wandb.Object3D(np.vstack(coordinates))]})

    def _empty_cluster_handling(self, labels: List[int]):
        '''If a cluster is empty, add a random element from another cluster'''
        result = [list() for _ in range(self._num_clusters)]
        for i in range(len(labels)):
            result[labels[i]].append(i)
        for i in range(self._num_clusters):
            if len(result[i]) == 0:
                labels[random.choice(result[random.randint(0, self._num_clusters - 1)])] = i
        return labels

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

    def spotter(self, graph: MultiGraphWithPos, labels: List[int], alpha: float, threshold: int) -> List[List[int]]:
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
        result = [list() for _ in range(self._num_clusters)]
        for i in edges_different_clusters:
            result[edge_set_tensor[i, 2].item()].append(edge_set_tensor[i, 0].item())
            result[edge_set_tensor[i, 3].item()].append(edge_set_tensor[i, 1].item())
        for i in range(self._num_clusters):
            result[i] = [x for x in set(result[i]) if result[i].count(x) >= threshold]
        result = self._reduce_samples(result, alpha, True)
        self._wandb.log({f'spotter added': sum([len(x) for x in result])})
        return result

    def exemplars(self, labels: List[List[int]], spotter: List[List[int]], alpha: float) -> List[List[int]]:
        # for each list in labels, remove the elements of the list if the elements are also in spotter
        result = [list() for _ in range(self._num_clusters)]
        for i, e in enumerate(labels):
            if i not in spotter[e]:
                result[e].append(i)
        #randomly sample from the remaining elements of each list in result according to the ratio alpha
        result = self._reduce_samples(result, alpha, True)
        self._wandb.log({f'exemplars added': sum([len(x) for x in result])})
        return result

    def _combine_samples(self, spotter: List[List[int]], exemplars: List[List[int]], top_k: List[List[int]]) -> List[List[int]]:
        # combine the lists of spotter and exemplars
        result = [list() for _ in range(self._num_clusters)]
        for i in range(self._num_clusters):
            result[i] = torch.tensor(list(set(spotter[i] + exemplars[i] + top_k[i])))
        return result

    def highest_dynamics(self, graph: MultiGraphWithPos, clusters: List[int], alpha: float) -> List[List[int]]:
        # for each index in clusters, put the index in the corresponding list in result
        result = [list() for _ in range(self._num_clusters)]
        for i in range(len(clusters)):
            result[clusters[i]].append(i)
        # for each list in result, sort the indices in descending order according to the graph's dynamics
        for i in range(self._num_clusters):
            result[i] = sorted(result[i], key=lambda x: graph.node_dynamic[x], reverse=True)
        # for each list in result, take the alpha percentage indices
        result = self._reduce_samples(result, alpha, False)
        self._wandb.log({f'highest dynamics added': sum([len(x) for x in result])})
        return result

    def _reduce_samples(self, result: List[List[int]], alpha: float, shuffle: bool) -> List[List[int]]:
        for i in range(self._num_clusters):
            if shuffle:
                random.shuffle(result[i])
            threshold = max(int(alpha * 100), int(len(result[i]) * alpha))
            threshold = min(len(result[i]), threshold)
            result[i] = result[i][:threshold]
        return result
