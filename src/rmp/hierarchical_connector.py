
from src.rmp.abstract_connector import AbstractConnector


class HierarchicalConnector(AbstractConnector):
    """
    Implementation of a hierarchical remote message passing strategy for hierarchical graph neural networks.
    """
    def __init__(self, normalizer):
        super().__init__(normalizer)

    def _initialize(self):
        pass

    def run(self, graph, clusters, is_training):
        pass
