
import functools

from collections import OrderedDict
from torch import nn

from src.migration.encoder import Encoder
from src.migration.processor import Processor
from src.migration.decoder import Decoder
from src.util import device


class MeshGraphNet(nn.Module):
    """Encode-Process-Decode GraphNet model."""

    def __init__(self, output_size, latent_size, num_layers, message_passing_aggregator,
                 message_passing_steps, hierarchical, edge_sets):
        super().__init__()
        self._latent_size = latent_size
        self._output_size = output_size
        self._num_layers = num_layers
        self._message_passing_steps = message_passing_steps
        self._message_passing_aggregator = message_passing_aggregator

        self.encoder = Encoder(make_mlp=self._make_mlp,
                               latent_size=self._latent_size,
                               hierarchical=hierarchical,
                               edge_sets=edge_sets)
        self.processor = Processor(make_mlp=self._make_mlp, output_size=self._latent_size,
                                   message_passing_steps=self._message_passing_steps,
                                   message_passing_aggregator=self._message_passing_aggregator,
                                   edge_sets=edge_sets,
                                   hierarchical=hierarchical)
        self.decoder = Decoder(make_mlp=functools.partial(self._make_mlp, layer_norm=False),
                               output_size=self._output_size)

    def forward(self, graph):
        """Encodes and processes a multigraph, and returns node features."""
        latent_graph = self.encoder(graph)
        latent_graph = self.processor(latent_graph)
        latent_graph = latent_graph._replace(node_features=latent_graph.node_features[0])
        return self.decoder(latent_graph)

    def _make_mlp(self, output_size, layer_norm=True):
        """Builds an MLP."""
        widths = [self._latent_size] * self._num_layers + [output_size]
        network = LazyMLP(widths)
        if layer_norm:
            network = nn.Sequential(
                network, nn.LayerNorm(normalized_shape=widths[-1]))
        return network


# TODO refactor into new file
class LazyMLP(nn.Module):
    def __init__(self, output_sizes):
        super().__init__()
        num_layers = len(output_sizes)
        self._layers_ordered_dict = OrderedDict()
        for index, output_size in enumerate(output_sizes):
            self._layers_ordered_dict["linear_" +
                                      str(index)] = nn.LazyLinear(output_size)
            if index < (num_layers - 1):
                self._layers_ordered_dict["relu_" + str(index)] = nn.ReLU()
        self.layers = nn.Sequential(self._layers_ordered_dict)

    def forward(self, input):
        input = input.to(device)
        y = self.layers(input)
        return y
