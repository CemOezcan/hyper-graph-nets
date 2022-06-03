from abc import abstractmethod


class Clustering():

    def __init__(self, graph):
        self._initialize()
        return

    @abstractmethod
    def _initialize(self):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError
