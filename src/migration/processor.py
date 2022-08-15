
from torch import nn

from src.migration.graphnet import GraphNet
from src.migration.hypergraphnet import HyperGraphNet


class Processor(nn.Module):
    """
    This class takes the nodes with the most influential feature (sum of square)
    The the chosen numbers of nodes in each ripple will establish connection(features and distances) with the most influential nodes and this connection will be learned
    Then the result is add to output latent graph of encoder and the modified latent graph will be feed into original processor

    Option: choose whether to normalize the high rank node connection
    """

    def __init__(self, make_mlp, output_size, message_passing_steps, message_passing_aggregator, edge_sets, hierarchical=True):
        super().__init__()
        graphnet_block = HyperGraphNet if hierarchical else GraphNet
        self.graphnet_blocks = nn.Sequential()
        for index in range(message_passing_steps):
            self.graphnet_blocks.append(
                graphnet_block(model_fn=make_mlp, output_size=output_size,
                               message_passing_aggregator=message_passing_aggregator,  edge_sets=edge_sets
                               )
            )

    def forward(self, latent_graph):
        return self.graphnet_blocks(latent_graph)
