from typing import Callable, List

import torch
from torch import Tensor

from src.migration.graphnet import GraphNet
from src.util import EdgeSet, MultiGraph


class MultiScaleGraphNet(GraphNet):
    """Multi-Edge and Multi-Node Interaction Network with residual connections."""

    def __init__(self, model_fn: Callable, output_size: int, message_passing_aggregator: str, edge_sets: List[str]):
        super().__init__(model_fn, output_size, message_passing_aggregator, edge_sets)

    def forward(self, graph: MultiGraph, mask=None) -> MultiGraph:
        # TODO: Decide if 5 or 3 mp-steps

        # 1
        # update_edges(mesh, world)
        # update_nodes(mesh_nodes, mesh, world)

        # update_edges(up)
        # update_nodes(hyper, up)

        # 2
        # update_edges(inter)
        # update_nodes(hyper, inter) TODO: up? (not in MSMGN-Paper)
        # 3
        # update_edges(inter)
        # update_nodes(hyper, inter)
        # 4
        # update_edges(inter)
        # update_nodes(hyper, inter)

        # update_edges(down)
        # update_nodes(mesh, down)

        # 5
        # update_edges(mesh, world)
        # update_nodes(mesh, mesh, world)

        return
