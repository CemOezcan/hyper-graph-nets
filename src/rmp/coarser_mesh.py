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
        self.represented_nodes = torch.empty(0, dtype=torch.int64)
        self.representing_nodes = torch.empty(0, dtype=torch.int64)
        self.traversed_nodes = torch.empty(0, dtype=torch.int64)
        self.traversed_nodes_indexes = torch.empty(0, dtype=torch.int64)
        self.new_graph = MultiGraphWithPos(node_features=torch.empty(0),
                                           edge_sets=[
                                                        EdgeSet("inter_cluster", torch.empty(0), torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)),
                                                        EdgeSet("inter_cluster_world", torch.empty(0), torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64))
                                                     ],
                                           mesh_features=torch.empty(0),
                                           target_feature=torch.empty(0),
                                           model_type='',
                                           unnormalized_edges=EdgeSet("mesh_edges", torch.empty(0), torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64)),
                                           node_dynamic=None,
                                           obstacle_nodes=0)

    def _initialize(self):
        pass

    def _cluster(self, graph: MultiGraphWithPos) -> List[int]:
        pass

    def run(self, graph: MultiGraphWithPos, depth=0, b4=True) -> MultiGraphWithPos:
        device = 'cpu'

        self.represented_nodes = torch.empty(0, dtype=torch.int64)
        self.representing_nodes = torch.empty(0, dtype=torch.int64)
        self.traversed_nodes = torch.empty(0, dtype=torch.int64)
        self.traversed_nodes_indexes = torch.empty(0, dtype=torch.int64)
        self.new_graph = MultiGraphWithPos(node_features=torch.empty(0),
                                           edge_sets=[
                                               EdgeSet("inter_cluster", torch.empty(0),
                                                       torch.empty(0, dtype=torch.int64),
                                                       torch.empty(0, dtype=torch.int64)),
                                               EdgeSet("inter_cluster_world", torch.empty(0),
                                                       torch.empty(0, dtype=torch.int64),
                                                       torch.empty(0, dtype=torch.int64))
                                           ],
                                           mesh_features=torch.empty(0),
                                           target_feature=torch.empty(0),
                                           model_type='',
                                           unnormalized_edges=EdgeSet("mesh_edges", torch.empty(0),
                                                                      torch.empty(0, dtype=torch.int64),
                                                                      torch.empty(0, dtype=torch.int64)),
                                           node_dynamic=None,
                                           obstacle_nodes=0)

        # down sample the mesh
        first_mesh_node_index = (graph.obstacle_nodes == 0).nonzero(as_tuple=True)[0][0].item()
        mesh = self._cluster_subgraph(graph, torch.tensor([first_mesh_node_index], dtype=torch.int64), depth)

        # down sample the obstacle
        first_obstacle_node_index = (graph.obstacle_nodes == 1).nonzero(as_tuple=True)[0][0].item()
        obstacle = self._cluster_subgraph(graph, torch.tensor([first_obstacle_node_index], dtype=torch.int64), depth)

        # add world edges

        world_edge_senders = torch.empty(0, dtype=torch.int64)
        world_edge_receivers = torch.empty(0, dtype=torch.int64)
        world_edge_features = torch.empty(0, dtype=torch.int64)

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
                    if senders[sender_index] == 496:
                        print("Sdsdsds")
                    if receivers[receiver_index] == 496:
                        print("Sdsdsds")
                    sender = senders[sender_index].reshape(1)
                    receiver = receivers[receiver_index].reshape(1)
                    world_edge_senders = torch.cat((world_edge_senders, sender), 0)
                    world_edge_receivers = torch.cat((world_edge_receivers, receiver), 0)

        merged = torch.unique(torch.stack((world_edge_senders, world_edge_receivers), 1), dim=0)
        world_edge_senders = merged[:, 0]
        world_edge_receivers = merged[:, 1]
        for index in range(world_edge_senders.shape[0]):
            if world_edge_receivers[index] == 950:
                print("Sfsfsd")
            x_ij = self.new_graph.target_feature[world_edge_receivers[index]] - self.new_graph.target_feature[world_edge_senders[index]]
            x_ij_norm = la.norm(x_ij).reshape(1)
            f1 = torch.cat((x_ij, x_ij_norm), 0).reshape(1, 4)
            world_edge_features = torch.cat((world_edge_features, f1), 0)
        self.new_graph.edge_sets[1] = EdgeSet("inter_cluster_world", world_edge_features, world_edge_senders, world_edge_receivers)
        return self.new_graph


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
                #neighbours.append(self.traversed_nodes[traversed[0][0]])

        # create new node
        indexes = torch.cat((neighbours_indexes[0], neighbours_indexes[1]), 0)
        indexes = torch.cat((indexes.to(device), torch.tensor([node_index], dtype=torch.int64).to(device)), 0)

        new_node_features = torch.mean(graph.node_features.to(device)[indexes], 0)
        new_node_features = torch.reshape(new_node_features, (1, new_node_features.shape[0]))

        new_mesh_features = torch.mean(graph.mesh_features[indexes], 0)
        new_mesh_features = torch.reshape(new_mesh_features, (1, new_mesh_features.shape[0]))

        x = graph.target_feature[indexes]
        new_target_feature = torch.mean(graph.target_feature[indexes], 0)
        new_target_feature = torch.reshape(new_target_feature, (1, new_target_feature.shape[0]))

        node_features = torch.cat((self.new_graph.node_features.to(device), new_node_features.to(device)), 0) ##
        mesh_features = torch.cat((self.new_graph.mesh_features.to(device), new_mesh_features.to(device)), 0)
        target_feature = torch.cat((self.new_graph.target_feature.to(device), new_target_feature.to(device)), 0)
        new_node_index = target_feature.shape[0] - 1

        traversed = (self.traversed_nodes.to(device) == node_index.to(device)).nonzero(as_tuple=False)
        self.traversed_nodes_indexes[traversed] = new_node_index

        # add to represented nodes
        represented_nodes = torch.cat((neighbours_indexes[0], neighbours_indexes[1]), 0)
        self.represented_nodes = torch.cat((self.represented_nodes.to(device), represented_nodes.to(device)), 0)
        representing_nodes = torch.full((represented_nodes.shape[0], 1), new_node_index).squeeze()
        self.representing_nodes = torch.cat((self.representing_nodes, representing_nodes), 0)
        if 950 in self.representing_nodes:
            print("SDfsdfsdf")

        # add edges
        mesh_edge_senders = self.new_graph.edge_sets[0].senders
        mesh_edge_receivers = self.new_graph.edge_sets[0].receivers
        mesh_edge_features = self.new_graph.edge_sets[0].features

        world_senders = self.new_graph.edge_sets[1].senders
        world_receivers = self.new_graph.edge_sets[1].receivers

        # add edges to neighbours
        for neighbours_index in neighbours:
            mesh_edge_senders = torch.cat((mesh_edge_senders, torch.tensor([new_node_index, neighbours_index], dtype=torch.int64)), 0)
            mesh_edge_receivers = torch.cat((mesh_edge_receivers, torch.tensor([neighbours_index, new_node_index], dtype=torch.int64)), 0)

            u_ij = mesh_features[neighbours_index] - mesh_features[new_node_index]
            u_ij_norm = la.norm(u_ij).reshape(1)
            x_ij = target_feature[neighbours_index] - target_feature[new_node_index]
            x_ij_norm = la.norm(x_ij).reshape(1)
            f1 = torch.cat((u_ij, u_ij_norm, x_ij, x_ij_norm), 0).reshape(1, 8)

            u_ji = mesh_features[new_node_index] - mesh_features[neighbours_index]
            u_ji_norm = la.norm(u_ji).reshape(1)
            x_ji = target_feature[new_node_index] - target_feature[neighbours_index]
            x_ji_norm = la.norm(x_ji).reshape(1)
            f2 = torch.cat((u_ji, u_ji_norm, x_ji, x_ji_norm), 0).reshape(1, 8)
            mesh_edge_features = torch.cat((mesh_edge_features, f1, f2), 0)

            #self.new_graph.edge_sets[0].features = torch.cat((self.new_graph.edge_sets[0].features, node_features), 0)

            # world_senders = torch.cat((world_senders, torch.tensor([new_node_index, neighbours_index], dtype=torch.int64)), 0)
            # world_receivers = torch.cat((world_receivers, torch.tensor([neighbours_index, new_node_index], dtype=torch.int64)), 0)
            #self.new_graph.edge_sets[1].features = torch.cat((self.new_graph.edge_sets[1].features, node_features), 0)

        # for neighbour in neighbours:
        #     for next_neighbour in neighbours:
        #         if neighbour == next_neighbour:
        #             continue
        #         mesh_senders = torch.cat((mesh_senders, torch.tensor([neighbour], dtype=torch.int64)), 0)
        #         mesh_receivers = torch.cat((mesh_receivers, torch.tensor([next_neighbour], dtype=torch.int64)), 0)
        #         world_senders = torch.cat((world_senders, torch.tensor([neighbour], dtype=torch.int64)), 0)
        #         world_receivers = torch.cat((world_receivers, torch.tensor([next_neighbour], dtype=torch.int64)), 0)

        edge_sets = [
            EdgeSet("inter_cluster", mesh_edge_features, mesh_edge_senders, mesh_edge_receivers),
            EdgeSet("inter_cluster_world", torch.empty(0), world_senders, world_receivers)
        ]

        # compute edge featues

        self.new_graph = MultiGraphWithPos(node_features=node_features,
                          edge_sets=edge_sets,
                          mesh_features=mesh_features,
                          target_feature=target_feature,
                          model_type='plate',
                          unnormalized_edges=0,
                          node_dynamic=graph.node_dynamic,
                          obstacle_nodes=0
                          )
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

