from abc import ABC, abstractmethod
from typing import List

from src.util import MultiGraphWithPos


class AbstractGraphProcessor(ABC):
    """
    Abstract superclass for processing graphs.
    """

    def __init__(self):
        """
        Initializes the graph processing algorithm
        """
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """
        Special initialization function if parameterized preprocessing is necessary.

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def run(self, graph: MultiGraphWithPos) -> MultiGraphWithPos:
        """
        Run processing algorithm given a multigraph.

        Parameters
        ----------
        graph :  Input data for the algorithm, represented by a multigraph or a point cloud.

        Returns a processed MultiGraphWithPos
        -------

        """
        raise NotImplementedError
