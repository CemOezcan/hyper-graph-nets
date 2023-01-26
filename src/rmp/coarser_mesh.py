import math
from typing import List

import numpy as np
import sklearn
from src.rmp.abstract_clustering_algorithm import AbstractClusteringAlgorithm
from src.util import MultiGraphWithPos
from src.util import EdgeSet
import torch
from torch import linalg as la


class CoarserClustering(AbstractClusteringAlgorithm):

    def __init__(self, ratio):
        super().__init__()
        self._ratio = ratio
        self.nodes_count = 0
        self.mesh_edge_senders_list = []
        self.mesh_edge_receivers_list = []
        self.world_edge_senders_list = []
        self.world_edge_receivers_list = []
        self.mesh_edge_senders = torch.empty(0, dtype=torch.int64)
        self.mesh_edge_receivers = torch.empty(0, dtype=torch.int64)
        self.world_edge_senders = torch.empty(0, dtype=torch.int64)
        self.world_edge_receivers = torch.empty(0, dtype=torch.int64)

        self.represented_nodes = torch.empty(0, dtype=torch.int64)
        self.representing_nodes = torch.empty(0, dtype=torch.int64)
        self.traversed_nodes = torch.empty(0, dtype=torch.int64)
        self.traversed_nodes_indexes = torch.empty(0, dtype=torch.int64)

    def _initialize(self):
        pass

    def _cluster(self, graph: MultiGraphWithPos) -> List[int]:
        pass

    def run(self, graph: MultiGraphWithPos, depth=0, b4=True) -> MultiGraphWithPos:
        device = 'cpu'
        self.nodes_count = 0
        self.mesh_edge_senders_list = []
        self.mesh_edge_receivers_list = []
        self.world_edge_senders_list = []
        self.world_edge_receivers_list = []
        self.mesh_edge_senders = torch.empty(0, dtype=torch.int64)
        self.mesh_edge_receivers = torch.empty(0, dtype=torch.int64)
        self.world_edge_senders = torch.empty(0, dtype=torch.int64)
        self.world_edge_receivers = torch.empty(0, dtype=torch.int64)
        self.represented_nodes = torch.empty(0, dtype=torch.int64)
        self.representing_nodes = torch.empty(0, dtype=torch.int64)
        self.traversed_nodes = torch.empty(0, dtype=torch.int64)
        self.traversed_nodes_indexes = torch.empty(0, dtype=torch.int64)

        # down sample the mesh
        first_mesh_node_index = (graph.obstacle_nodes == 0).nonzero(as_tuple=True)[0][0].item()
        mesh = self._cluster_subgraph(graph, torch.tensor([first_mesh_node_index], dtype=torch.int64), depth)

        # down sample the obstacle
        first_obstacle_node_index = (graph.obstacle_nodes == 1).nonzero(as_tuple=True)[0][0].item()
        obstacle = self._cluster_subgraph(graph, torch.tensor([first_obstacle_node_index], dtype=torch.int64), depth)

        # add world edges


        represented_nodes = self.represented_nodes.to(device)
        high_world_edge_senders = graph.edge_sets[1].senders.to(device)
        high_world_edge_receivers = graph.edge_sets[1].receivers.to(device)

        for node_index in range(high_world_edge_senders.shape[0]):
            senders_indices = (represented_nodes == high_world_edge_senders[node_index]).nonzero(as_tuple=False).squeeze()
            senders = torch.unique(self.representing_nodes[senders_indices].squeeze())
            receivers_indices = (represented_nodes == high_world_edge_receivers[node_index]).nonzero(as_tuple=False).squeeze()
            receivers = torch.unique(self.representing_nodes[receivers_indices].squeeze())

            for sender_index in range(senders.shape[0]):
                for receiver_index in range(receivers.shape[0]):
                    sender = senders[sender_index].item()
                    receiver = receivers[receiver_index].item()
                    self.world_edge_senders_list.append(sender)
                    self.world_edge_receivers_list.append(receiver)

        mesh_merged = torch.unique(
            torch.stack((
                torch.tensor(self.mesh_edge_senders_list, dtype = torch.int64),
                torch.tensor(self.mesh_edge_receivers_list, dtype = torch.int64)), 1), dim=0)
        self.mesh_edge_senders = mesh_merged[:, 0]
        self.mesh_edge_receivers = mesh_merged[:, 1]
        
        print(len(self.world_edge_senders_list))
        world_merged = torch.unique(
            torch.stack((
                torch.tensor(self.world_edge_senders_list, dtype=torch.int64),
                torch.tensor(self.world_edge_receivers_list, dtype=torch.int64)), 1), dim=0)
        print(world_merged.shape)
        self.world_edge_senders = world_merged[:, 0]
        self.world_edge_receivers = world_merged[:, 1]

        return self.nodes_count


    def _cluster_subgraph(self, graph: MultiGraphWithPos, node_index=0, depth=0):
        device = 'cpu'

        # get neighbours
        neighbours_indexes = self.get_neigbors_indexes(graph, node_index)

        # set the first order neighbours as traversed
        x = torch.cat((neighbours_indexes[0].to(device), torch.tensor([node_index], dtype=torch.int64).to(device)), 0)
        self.traversed_nodes = torch.cat((self.traversed_nodes, x), 0)

        xx = torch.full(x.shape, -1)
        self.traversed_nodes_indexes = torch.cat((self.traversed_nodes_indexes, xx), 0)

        neighbours = []

        # repeat for neighbour nodes
        for neighbours_index in neighbours_indexes[1]:
            traversed = (self.traversed_nodes.to(device) == neighbours_index.to(device)).nonzero(as_tuple=False)
            if traversed.shape[0] == 0:
                neighbour = self._cluster_subgraph(graph, neighbours_index, depth=depth + 1)
                neighbours.append(neighbour)
            elif self.traversed_nodes_indexes[traversed[0][0]] >= 0:
                neighbours.append(self.traversed_nodes_indexes[traversed[0][0]])

        # create new node
        indexes = torch.cat((neighbours_indexes[0].to(device), neighbours_indexes[1].to(device), torch.tensor([node_index], dtype=torch.int64).to(device)), 0)

        new_node_index = self.nodes_count
        self.nodes_count += 1

        traversed = (self.traversed_nodes.to(device) == node_index.to(device)).nonzero(as_tuple=False)
        self.traversed_nodes_indexes[traversed] = new_node_index

        # add to represented nodes
        self.represented_nodes = torch.cat((self.represented_nodes.to(device), indexes.to(device)), 0)
        representing_nodes = torch.full((indexes.shape[0], 1), new_node_index).squeeze()
        self.representing_nodes = torch.cat((self.representing_nodes, representing_nodes), 0)

        # add edges to neighbours
        for neighbours_index in neighbours:
            self.mesh_edge_senders_list += [new_node_index, neighbours_index]
            self.mesh_edge_receivers_list += [neighbours_index, new_node_index]

        return new_node_index

    def get_neigbors_indexes(self, graph, node_index, order=2):
        device = 'cpu'
        edge_set = [edge for edge in graph.edge_sets if edge.name == 'mesh_edges'][0]
        node_indexes = torch.tensor([node_index], dtype=torch.int64)
        neighbours = []
        for i in range(order):
            edges_indexes = torch.empty(0, dtype=torch.int64)
            for node_index in node_indexes:
                edges_indexes = torch.cat((edges_indexes.to(device), (edge_set.senders == node_index).nonzero(as_tuple=False).to(device)), 0)
            edges_indexes = torch.reshape(edges_indexes, (edges_indexes.shape[0], 1))
            edges_indexes = torch.unique(edges_indexes)
            neighbours_indexes = edge_set.receivers[edges_indexes]
            neighbours_indexes = torch.unique(neighbours_indexes)
            neighbours.append(neighbours_indexes)
            node_indexes = neighbours_indexes
        return neighbours

