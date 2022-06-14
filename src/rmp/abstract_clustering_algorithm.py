from abc import ABC, abstractmethod
from typing import List

from src.util import MultiGraph


class AbstractClusteringAlgorithm(ABC):
    """
    Abstract superclass for clustering algorithms.
    """

    def __init__(self):
        """
        Initializes the clustering algorithm
        """
        self._num_clusters = 0
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
    def run(self, graph: MultiGraph) -> List[List]:
        """
        Run clustering algorithm given a multigraph or point cloud.

        Parameters
        ----------
        graph :  Input data for the algorithm, represented by a multigraph or a point cloud.

        Returns clustering as a list. The first cluster contains noisy nodes.
        -------

        """
        raise NotImplementedError
