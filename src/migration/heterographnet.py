from typing import Callable, List

import torch
from torch import Tensor

from src.migration.graphnet import GraphNet
from src.util import EdgeSet, MultiGraph


class HeteroGraphNet(GraphNet):
    """Multi-Edge and Multi-Node Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str, edge_sets: List[str]):
        super().__init__(model_fn, output_size, message_passing_aggregator, edge_sets)

    def forward(self, graph: MultiGraph, mask=None) -> MultiGraph:
        # update_edges(mesh, world, inter, intra_up, intra_down)
        # update_nodes(mesh, hyper)

        return
