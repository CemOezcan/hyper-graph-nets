
from src.rmp.abstract_connector import AbstractConnector


class HierarchicalConnector(AbstractConnector):

    def __init__(self, normalizer):
        super().__init__(normalizer)

    def _initialize(self):
        pass

    def run(self, graph, clusters, representatives, is_training):
        pass
