from typing import Callable

from torch import nn, Tensor

from src.util import MultiGraph


class Decoder(nn.Module):
    """Decodes node features from graph."""

    def __init__(self, make_mlp: Callable, output_size: int):
        super().__init__()
        self.model = make_mlp(output_size)

    def forward(self, graph: MultiGraph) -> Tensor:
        return self.model(graph.node_features)
