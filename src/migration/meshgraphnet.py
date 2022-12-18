
import functools

from collections import OrderedDict
from typing import List, Type, Tuple

from torch import nn, Tensor

from src.migration.encoder import Encoder
from src.migration.graphnet import GraphNet
from src.migration.heterographnet import HeteroGraphNet
from src.migration.hypergraphnet import HyperGraphNet
from src.migration.multigraphnet import MultiGraphNet
from src.migration.multiscalegraphnet import MultiScaleGraphNet
from src.migration.processor import Processor
from src.migration.decoder import Decoder
from src.migration.repeatedgraphnet import RepeatedGraphNet
from src.util import device, MultiGraph


class MeshGraphNet(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self, output_size: int, latent_size: int, num_layers: int, message_passing_aggregator: str,
                 message_passing_steps: int, architecture: str, edge_sets: List[str]):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator
        graphnet_block, hierarchical = self.get_architecture(architecture)

        self.encoder = Encoder(make_mlp=self._make_mlp,
                               latent_size=self._latent_size,
                               hierarchical=hierarchical,
                               edge_sets=edge_sets)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator,
                                   edge_sets=edge_sets,
                                   graphnet_block=graphnet_block)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)

    def forward(self, graph: MultiGraph) -> Tensor:
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph = self.encoder(graph)
        latent_graph = self.processor(latent_graph)
        latent_graph = latent_graph._replace(node_features=latent_graph.node_features[0])
        return self.decoder(latent_graph)

    def _make_mlp(self, output_size: int, layer_norm=True) -> nn.Module:
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(
                network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network

    @staticmethod
    def get_architecture(architecture: str) -> Tuple[Type[GraphNet], bool]:
        """
        Returns the specified GNN architecture.

        Parameters
        ----------
            architecture : str
                The name of the architecture

        Returns
        -------
            Tuple[Type[GraphNet], bool]
                The GraphNet block and whether it is hierarchical or not

        """
        if architecture == 'hyper':
            return HyperGraphNet, True
        elif architecture == 'multiscale':
            return MultiScaleGraphNet, True
        elif architecture == 'hetero':
            return HeteroGraphNet, True
        elif architecture == 'multi':
            return MultiGraphNet, False
        elif architecture == 'repeated':
            return RepeatedGraphNet, False
        else:
            return GraphNet, False


# TODO refactor into new file
class LazyMLP(nn.Module):
    def __init__(self, output_sizes: List[int]):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" +
                                      str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input: Tensor) -> Tensor:
        input = input.to(device)
        y = self.layers(input)
        return y
