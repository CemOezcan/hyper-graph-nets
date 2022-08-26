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
