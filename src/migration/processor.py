from typing import Callable, List, Type

from torch import nn

from src.migration.graphnet import GraphNet
from src.migration.hypergraphnet import HyperGraphNet
from src.util import MultiGraph


class Processor(nn.Module):
    """
    The Graph Neural Network that transforms the input graph.
    """

    def __init__(self, make_mlp: Callable, output_size: int, message_passing_steps: int,
                 message_passing_aggregator: str, edge_sets: List[str], graphnet_block: Type[GraphNet]):
        super().__init__()
        blocks = []
        for _ in range(message_passing_steps):
            blocks.append(
                graphnet_block(model_fn=make_mlp, output_size=output_size,
                               message_passing_aggregator=message_passing_aggregator,  edge_sets=edge_sets
                               )
            )
        self.graphnet_blocks = nn.Sequential(*blocks)

    def forward(self, latent_graph: MultiGraph) -> MultiGraph:
        return self.graphnet_blocks(latent_graph)
