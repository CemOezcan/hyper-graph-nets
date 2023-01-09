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

    def run(self, graph: MultiGraphWithPos, clusters: MultiGraphWithPos, neighbors: List[Tensor], is_training: bool) -> MultiGraph:
        device = 'cpu'

        down_sampled_grapgh = clusters
        represented_nodes = neighbors[0]
        representing_nodes = neighbors[1]
        representing_nodes += graph.node_features.shape[0]

        low_mesh_senders = down_sampled_grapgh.edge_sets[0].senders + graph.node_features.shape[0]
        low_mesh_receivers = down_sampled_grapgh.edge_sets[0].receivers + graph.node_features.shape[0]
        low_world_senders = down_sampled_grapgh.edge_sets[1].senders + graph.node_features.shape[0]
        low_world_receivers = down_sampled_grapgh.edge_sets[1].receivers + graph.node_features.shape[0]

        edge_sets = graph.edge_sets
        edge_sets.extend([
            EdgeSet("inter_cluster", down_sampled_grapgh.edge_sets[0].features.to(src.util.device), low_mesh_senders.to(src.util.device), low_mesh_receivers.to(src.util.device)),
            EdgeSet("inter_cluster_world", down_sampled_grapgh.edge_sets[1].features.to(src.util.device), low_world_senders.to(src.util.device), low_world_receivers.to(src.util.device))
        ])

        down_sampled_node_features = down_sampled_grapgh.node_features
        if is_training and self._noise_scale is not None:
            zero_size = torch.zeros(down_sampled_node_features.shape, dtype=torch.float32)
            noise = torch.normal(zero_size, std=self._noise_scale)
            down_sampled_node_features += noise

        # Expansion
        node_features = [graph.node_features.to(src.util.device), down_sampled_node_features.to(src.util.device)]

        mesh_features = torch.cat((graph.mesh_features.to(device), down_sampled_grapgh.mesh_features.to(device)), 0)
        target_feature = torch.cat((graph.target_feature.to(device), down_sampled_grapgh.target_feature.to(device)), 0)

        u_down_sampling = mesh_features[representing_nodes] - mesh_features[represented_nodes]
        u_down_sampling_norm = la.norm(u_down_sampling, dim=1).reshape(representing_nodes.shape[0], 1)
        x_down_sampling = target_feature[representing_nodes] - target_feature[represented_nodes]
        x_down_sampling_norm = la.norm(x_down_sampling, dim=1).reshape(representing_nodes.shape[0], 1)
        down_sampling_features = torch.cat((u_down_sampling, u_down_sampling_norm, x_down_sampling, x_down_sampling_norm), 1).to(src.util.device)

        u_up_sampling = mesh_features[represented_nodes] - mesh_features[representing_nodes]
        u_up_sampling_norm = la.norm(u_up_sampling, dim=1).reshape(representing_nodes.shape[0], 1)
        x_up_sampling = target_feature[represented_nodes] - target_feature[representing_nodes]
        x_up_sampling_norm = la.norm(x_up_sampling, dim=1).reshape(representing_nodes.shape[0], 1)
        up_sampling_features = torch.cat((u_up_sampling, u_up_sampling_norm, x_up_sampling, x_up_sampling_norm), 1).to(src.util.device)

        edge_sets.extend([
            EdgeSet("intra_cluster_to_cluster", down_sampling_features, represented_nodes.to(src.util.device), representing_nodes.to(src.util.device)),
            EdgeSet("intra_cluster_to_mesh", up_sampling_features, representing_nodes.to(src.util.device), represented_nodes.to(src.util.device))
        ])

        return MultiGraph(node_features=node_features, edge_sets=edge_sets)
