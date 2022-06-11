import torch

from src.rmp.abstract_connector import AbstractConnector
from src.util import device, EdgeSet, MultiGraphWithPos


class FullConnector(AbstractConnector):

    def __init__(self, normalizer):
        super().__init__(normalizer)

    def _initialize(self):
        pass

    def run(self, graph, clusters, is_training):
        # Reprs.
        representatives = []
        for ripple in clusters:
            cluster_size = ripple[1] - ripple[0]
            # TODO: Parameter: num. representatives
            core_size = min(5, cluster_size)
            random_mask = torch.randperm(n=cluster_size)[0:core_size]
            representatives.append(random_mask)

        # Fully Connected
        model_type = graph.model_type
        node_dynamic = graph.node_dynamic
        _, sort_indices = torch.sort(node_dynamic, dim=0, descending=True)

        core_nodes = []
        for (start_index, end_index), node_mask in zip(clusters, representatives):
            if end_index > start_index:
                cluster = sort_indices[start_index:end_index]
                core_nodes.append(cluster[node_mask])

        target_feature = graph.target_feature
        remote_edges = []
        for core_node in core_nodes:
            receivers_list = core_node
            senders_list = core_node
            senders = torch.cat(
                (torch.tensor(senders_list, device=device), torch.tensor(receivers_list, device=device)), dim=0)
            receivers = torch.cat(
                (torch.tensor(receivers_list, device=device), torch.tensor(senders_list, device=device)), dim=0)

            # TODO: Make model independent
            if model_type == 'flag' or model_type == 'deform_model':
                relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                           torch.index_select(input=target_feature, dim=0, index=receivers))
                edge_features = torch.cat(
                    (relative_target_feature, torch.norm(relative_target_feature, dim=-1, keepdim=True)), dim=-1)
            else:
                raise Exception("Model type is not specified in RippleNodeConnector.")

            edge_features = self._normalizer(edge_features)
            world_edges = EdgeSet(
                name='remote_edges',
                features=self._normalizer(edge_features, None, is_training),
                receivers=receivers,
                senders=senders)

            remote_edges.append(world_edges)

        edge_sets = graph.edge_sets
        edge_sets.extend(remote_edges)

        return MultiGraphWithPos(node_features=graph.node_features,
                                 edge_sets=edge_sets, target_feature=graph.target_feature,
                                 model_type=graph.model_type, node_dynamic=graph.node_dynamic)
