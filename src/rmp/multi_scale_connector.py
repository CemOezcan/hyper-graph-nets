import math
from typing import List

import numpy as np
import scipy.spatial as ss
import torch
from matplotlib import pyplot as plt
from torch import Tensor
from torch import linalg as la

import src.util
from src import util
from src.rmp.abstract_connector import AbstractConnector
from src.rmp.coarser_mesh import CoarserClustering
from src.util import MultiGraphWithPos, EdgeSet, device, MultiGraph

class MultiScaleConnector(AbstractConnector):
    """
    Implementation of a hierarchical remote message passing strategy for hierarchical graph neural networks.
    """
    def __init__(self, fully_connect, noise_scale, hyper_node_features):
        super().__init__(fully_connect, noise_scale, hyper_node_features)

    def initialize(self, intra, inter, hyper):
        super().initialize(intra, inter, hyper)
        # TODO: fix
        return ['intra_cluster_to_mesh', 'intra_cluster_to_cluster', 'inter_cluster', 'inter_cluster_world']

    def run(self, graph: MultiGraphWithPos, clusters: int, neighbors: List[Tensor], is_training: bool) -> MultiGraph:
        device = 'cpu'

        represented_nodes = neighbors[0]
        representing_nodes = neighbors[1]

        nodes_features_list = []
        mesh_features_list = []
        target_features_list = []
        for i in range(clusters):
            indexes = (representing_nodes.to(device) == i).nonzero(as_tuple=False)
            indexes = represented_nodes[indexes]
            new_node_features = torch.mean(graph.node_features.to(device)[indexes], 0)
            new_mesh_features = torch.mean(graph.mesh_features.to(device)[indexes], 0)
            new_target_feature = torch.mean(graph.target_feature.to(device)[indexes], 0)

            nodes_features_list.append(new_node_features)
            mesh_features_list.append(new_mesh_features)
            target_features_list.append(new_target_feature)

        nodes_features = torch.cat(nodes_features_list)
        mesh_features = torch.cat(mesh_features_list)
        target_features = torch.cat(target_features_list)

        low_mesh_senders = neighbors[2]
        low_mesh_receivers = neighbors[3]
        mesh_u = mesh_features[low_mesh_receivers] - mesh_features[low_mesh_senders]
        mesh_u_norm = la.norm(mesh_u, dim=1).reshape((mesh_u.shape[0], 1))
        mesh_x = target_features[low_mesh_receivers] - target_features[low_mesh_senders]
        mesh_x_norm = la.norm(mesh_x, dim=1).reshape((mesh_x.shape[0], 1))
        mesh_edge_features = torch.cat((mesh_u, mesh_u_norm, mesh_x, mesh_x_norm), 1)

        # add world edges
        high_world_edge_senders = graph.edge_sets[1].senders.to(device)
        high_world_edge_receivers = graph.edge_sets[1].receivers.to(device)

        world_edge_senders_list = []
        world_edge_receivers_list = []
        for node_index in range(high_world_edge_senders.shape[0]):
            senders_indices = (represented_nodes == high_world_edge_senders[node_index]).nonzero(
                as_tuple=False).squeeze()
            senders = torch.unique(representing_nodes[senders_indices].squeeze())
            receivers_indices = (represented_nodes == high_world_edge_receivers[node_index]).nonzero(
                as_tuple=False).squeeze()
            receivers = torch.unique(representing_nodes[receivers_indices].squeeze())
            for sender_index in range(senders.shape[0]):
                for receiver_index in range(receivers.shape[0]):
                    sender = senders[sender_index].item()
                    receiver = receivers[receiver_index].item()
                    world_edge_senders_list.append(sender)
                    world_edge_receivers_list.append(receiver)

        low_world_senders = torch.tensor([])
        low_world_receivers = torch.tensor([])

        if len(world_edge_senders_list) != 0:
            world_merged = torch.unique(
                torch.stack((
                    torch.tensor(world_edge_senders_list, dtype=torch.int64),
                    torch.tensor(world_edge_receivers_list, dtype=torch.int64)), 1), dim=0)
            low_world_senders = world_merged[:, 0]
            low_world_receivers = world_merged[:, 1]

        world_x = target_features[low_world_receivers] - target_features[low_world_senders]
        world_x_norm = la.norm(world_x, dim=1).reshape((world_x.shape[0], 1))
        world_edge_features = torch.cat((world_x, world_x_norm), 1)

        merged_representing_nodes = representing_nodes + graph.node_features.shape[0]
        merged_low_mesh_senders = low_mesh_senders + graph.node_features.shape[0]
        merged_low_mesh_receivers = low_mesh_receivers + graph.node_features.shape[0]
        merged_low_world_senders = low_world_senders + graph.node_features.shape[0]
        merged_low_world_receivers = low_world_receivers + graph.node_features.shape[0]

        edge_sets = graph.edge_sets
        edge_sets.extend([
            EdgeSet("inter_cluster", mesh_edge_features.to(src.util.device), merged_low_mesh_senders.to(src.util.device), merged_low_mesh_receivers.to(src.util.device)),
            EdgeSet("inter_cluster_world", world_edge_features.to(src.util.device), merged_low_world_senders.to(src.util.device), merged_low_world_receivers.to(src.util.device))
        ])

        if is_training and self._noise_scale is not None:
            zero_size = torch.zeros(nodes_features.shape, dtype=torch.float32)
            noise = torch.normal(zero_size, std=self._noise_scale)
            nodes_features += noise

        # Expansion
        node_features = [graph.node_features.to(src.util.device), nodes_features.to(src.util.device)]

        mesh_features = torch.cat((graph.mesh_features.to(device), mesh_features.to(device)), 0)
        target_feature = torch.cat((graph.target_feature.to(device), target_features.to(device)), 0)

        u_down_sampling = mesh_features[merged_representing_nodes] - mesh_features[represented_nodes]
        u_down_sampling_norm = la.norm(u_down_sampling, dim=1).reshape(merged_representing_nodes.shape[0], 1)
        x_down_sampling = target_feature[merged_representing_nodes] - target_feature[represented_nodes]
        x_down_sampling_norm = la.norm(x_down_sampling, dim=1).reshape(merged_representing_nodes.shape[0], 1)
        down_sampling_features = torch.cat((u_down_sampling, u_down_sampling_norm, x_down_sampling, x_down_sampling_norm), 1).to(src.util.device)

        u_up_sampling = mesh_features[represented_nodes] - mesh_features[merged_representing_nodes]
        u_up_sampling_norm = la.norm(u_up_sampling, dim=1).reshape(merged_representing_nodes.shape[0], 1)
        x_up_sampling = target_feature[represented_nodes] - target_feature[merged_representing_nodes]
        x_up_sampling_norm = la.norm(x_up_sampling, dim=1).reshape(merged_representing_nodes.shape[0], 1)
        up_sampling_features = torch.cat((u_up_sampling, u_up_sampling_norm, x_up_sampling, x_up_sampling_norm), 1).to(src.util.device)

        edge_sets.extend([
            EdgeSet("intra_cluster_to_cluster", down_sampling_features, represented_nodes.to(src.util.device), merged_representing_nodes.to(src.util.device)),
            EdgeSet("intra_cluster_to_mesh", up_sampling_features, merged_representing_nodes.to(src.util.device), represented_nodes.to(src.util.device))
        ])

        return MultiGraph(node_features=node_features, edge_sets=edge_sets)
