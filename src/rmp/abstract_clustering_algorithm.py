from abc import ABC, abstractmethod
from typing import List

import torch
from torch import Tensor

from src.util import MultiGraphWithPos


class AbstractClusteringAlgorithm(ABC):
    """
    Abstract superclass for clustering algorithms.
    """

    def __init__(self):
        """
        Initializes the clustering algorithm
        """
        self._num_clusters = 10
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
    def run(self, graph: MultiGraphWithPos) -> List[Tensor]:
        """
        Run clustering algorithm given a multigraph or point cloud.

        Parameters
        ----------
        graph :  Input data for the algorithm, represented by a multigraph or a point cloud.

        Returns clustering as a list.
        -------

        """
        raise NotImplementedError

    def _labels_to_indices(self, labels: List[int]) -> List[Tensor]:
        """
        Groups data points into clusters, given their class labels.

        Parameters
        ----------
        labels : Class labels of data points, where labels[i] represents the class label of data point i

        Returns clustering as a list.
        -------

        """
        indices = [list() for _ in range(self._num_clusters)]

        for i in range(len(labels)):
            indices[labels[i]].append(i)

        return [torch.tensor(x) for x in indices]

    def spotter(self, graph: MultiGraphWithPos, labels: List[int]) -> List[int]:
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
        # return tensor of indices
        return [torch.tensor(x) for x in indices]

    def spotter_with_probability(self, graph: MultiGraphWithPos, labels: List[int], probabilities, threshold) -> List[int]:
        '''For an array of probabilities with size (labels, n), where n is the number of vertices'''
        edge_set = [x for x in graph.edge_sets if x.name == 'mesh_edges']
        # for the sender and receiver of the edge set, find the corresponding label
        sender_labels = [labels[x] for x in edge_set[0].senders]
        receiver_labels = [labels[x] for x in edge_set[0].receivers]     
        # combine senders, receivers, sender_labels, receiver_labels into a tensor of tuples
        # (sender, receiver, sender_label, receiver_label)
        edge_set_tensor = torch.stack((torch.tensor(edge_set[0].senders).to('cpu'), torch.tensor(edge_set[0].receivers).to('cpu'), torch.tensor(sender_labels), torch.tensor(receiver_labels)), dim=1)
        # for each element in edge_set_tensor, put the sender in a different tensor with both labels and probabilities
        sender_tensor = torch.stack((edge_set_tensor[:, 0], edge_set_tensor[:, 2], edge_set_tensor[:, 3], probabilities[edge_set_tensor[:, 2], edge_set_tensor[:, 0]], probabilities[edge_set_tensor[:, 3], edge_set_tensor[:, 0]]), dim=1)
        receiver_tensor = torch.stack((edge_set_tensor[:, 1], edge_set_tensor[:, 3], edge_set_tensor[:, 2], probabilities[edge_set_tensor[:, 2], edge_set_tensor[:, 1]], probabilities[edge_set_tensor[:, 3], edge_set_tensor[:, 1]]), dim=1)
        # compute the difference and sum of probabilities and put them in the tensor
        sender_tensor[:, 5] = sender_tensor[:, 4] - sender_tensor[:, 3]
        receiver_tensor[:, 5] = receiver_tensor[:, 3] - receiver_tensor[:, 4]
        sender_tensor[:, 6] = sender_tensor[:, 4] + sender_tensor[:, 3]
        receiver_tensor[:, 6] = receiver_tensor[:, 3] + receiver_tensor[:, 4]
        # divide the difference by the sum of probabilities and check if they are above the threshold
        sender_tensor[:, 7] = sender_tensor[:, 5] / sender_tensor[:, 6]
        receiver_tensor[:, 7] = receiver_tensor[:, 5] / receiver_tensor[:, 6]
        # find all elements of sender_tensor that have a different sender_label and receiver_label and above the threshold
        sender_edges_different_clusters = torch.nonzero(torch.abs(sender_tensor[:, 7] > threshold) > 0).squeeze()
        receiver_edges_different_clusters = torch.nonzero(torch.abs(receiver_tensor[:, 7] > threshold) > 0).squeeze()
        # for each element in sender_edges_different_clusters, put them in a list of length labels and put the senders and receivers to their corresponding labels
        indices = [list() for _ in range(self._num_clusters)]
        for i in sender_edges_different_clusters:
            indices[sender_tensor[i, 1].item()].append(sender_tensor[i, 0].item())
        for i in receiver_edges_different_clusters:
            indices[receiver_tensor[i, 1].item()].append(receiver_tensor[i, 0].item())
        # return tensor of indices after checking for duplicates in labels list
        return [torch.tensor(list(set(x))) for x in indices]