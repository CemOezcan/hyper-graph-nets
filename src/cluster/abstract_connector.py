from abc import ABC, abstractmethod


class AbstractConnector(ABC):

    def __init__(self, normalizer):
        self._normalizer = normalizer
        self._initialize()

    @abstractmethod
    def _initialize(self):
        raise NotImplementedError

    @abstractmethod
    def run(self, graph, clusters, representatives, is_training):
        raise NotImplementedError
