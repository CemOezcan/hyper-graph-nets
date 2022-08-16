
import torch
from src.migration.graphnet import GraphNet


class HyperGraphNet(GraphNet):
    """Multi-Edge Interaction Network with residual connections."""

    def __init__(self, model_fn, output_size, message_passing_aggregator, edge_sets):
        super().__init__(model_fn, output_size, message_passing_aggregator, edge_sets)

        self.hyper_node_model_up = model_fn(output_size)
        self.hyper_node_model_cross = model_fn(output_size)
        self.node_model_down = model_fn(output_size)

    def _update_node_features(self, node_features, edge_sets):
        """Aggregrates edge features, and applies node function."""
        hyper_node_offset = len(node_features[0])
        num_nodes = torch.cat(tuple(node_features), dim=0).shape[0]

        nf = super()._update_node_features(node_features, edge_sets)
        updated_nodes_cross, hyper_node_features = nf[0], nf[1]
        node_features_2 = torch.cat((updated_nodes_cross, hyper_node_features), dim=0)

        features = self.aggregation(
            list(filter(lambda x: x.name == 'intra_cluster_to_cluster', edge_sets)),
            [node_features_2],
            num_nodes
        )
        updated_hyper_nodes_up = self.hyper_node_model_up(features[hyper_node_offset:])
        node_features_3 = torch.cat((node_features_2[:hyper_node_offset], updated_hyper_nodes_up), dim=0)

        features = self.aggregation(
            list(filter(lambda x: x.name == 'inter_cluster', edge_sets)),
            [node_features_3],
            num_nodes
        )
        updated_hyper_nodes_cross = self.hyper_node_model_cross(features[hyper_node_offset:])
        node_features_4 = torch.cat((node_features_3[:hyper_node_offset], updated_hyper_nodes_cross), dim=0)

        features = self.aggregation(
            list(filter(lambda x: x.name == 'intra_cluster_to_mesh', edge_sets)),
            [node_features_4],
            num_nodes
        )
        updated_nodes_down = self.node_model_down(features[:hyper_node_offset])

        return [updated_nodes_down, node_features_4[hyper_node_offset:]]