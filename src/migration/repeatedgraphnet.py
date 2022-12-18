import time
from typing import Callable, List

import torch
from torch import Tensor

from src.migration.graphnet import GraphNet
from src.util import EdgeSet, MultiGraph


class RepeatedGraphNet(GraphNet):
    """Multi-Edge and Multi-Node Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str, edge_sets: List[str], repetitions=2):
        super().__init__(model_fn, output_size, message_passing_aggregator, edge_sets)
        self.repetitions = repetitions

    def forward(self, graph: MultiGraph, mask=None) -> MultiGraph:
        for _ in range(self.repetitions):
            graph = super().forward(graph)

        return graph
