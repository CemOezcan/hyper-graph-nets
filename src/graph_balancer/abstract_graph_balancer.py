from abc import ABC, abstractmethod
import networkx as nx
from typing import Dict, Tuple

import torch
from src.util import EdgeSet, MultiGraphWithPos, device


class AbstractGraphBalancer(ABC):
    """
    Abstract superclass for processing graphs.
    """

    def __init__(self):
        """
        Initializes the graph processing algorithm
        """
        self._added_edges = None
        self._mask = None
        self._initialize()

    @abstractmethod
    def _initialize(self):
        """
        Special initialization function if parameterized preprocessing is necessary.

        Returns
        -------

        """
        raise NotImplementedError

    @abstractmethod
    def run(self, graph: MultiGraphWithPos, mesh_edge_normalizer, is_training: bool) -> Tuple[MultiGraphWithPos, Dict]:
        """
        Run processing algorithm given a multigraph.

        Parameters
        ----------
        graph :  Input data for the algorithm, represented by a multigraph or a point cloud.

        Returns a processed MultiGraphWithPos
        -------

        """
        raise NotImplementedError

    @staticmethod
    def add_graph_balance_edges(graph: MultiGraphWithPos, added_edges: Dict, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        mesh_pos = graph.mesh_features
        world_pos = graph.target_feature
        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=torch.tensor(added_edges['senders'], dtype=torch.long, device=device)) -
                              torch.index_select(input=world_pos, dim=0, index=torch.tensor(added_edges['receivers'], dtype=torch.long, device=device)))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, torch.tensor(added_edges['senders'], dtype=torch.long, device=device)) -
                             torch.index_select(mesh_pos, 0, torch.tensor(added_edges['receivers'], dtype=torch.long, device=device)))
        edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)
        graph.edge_sets.append(EdgeSet(name='balance', features=mesh_edge_normalizer(edge_features, is_training), senders=torch.tensor(
            added_edges['senders'], dtype=torch.long, device=device), receivers=torch.tensor(added_edges['receivers'], dtype=torch.long, device=device)))
        return graph

    def remove_graph_balance_edges(self, graph: MultiGraphWithPos, mask: torch.Tensor, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        graph_edges = graph.edge_sets[0]
        graph_edge_set = EdgeSet(name=graph_edges.name, features=mesh_edge_normalizer(graph_edges.features[self._mask], is_training), senders=graph_edges.senders[self._mask], receivers=graph_edges.receivers[self._mask])
        #create a new graph with the filtered edges
        edge_sets = [graph_edge_set, graph.edge_sets[1]]
        graph = graph._replace(edge_sets=edge_sets)
        return graph
        
    @staticmethod
    def _determine_mask(graph_edges: EdgeSet, removed_edges: Dict) -> torch.Tensor:
        mask = torch.ones(len(graph_edges.senders), dtype=torch.bool, device=device)
        G = nx.Graph()
        G.add_edges_from(zip(removed_edges['senders'], removed_edges['receivers']))
        for i, (sender, receiver) in enumerate(zip(graph_edges.senders, graph_edges.receivers)):
            if G.has_edge(sender.item(), receiver.item()):
                mask[i] = False
        return mask
    
    def create_graph(self, graph: MultiGraphWithPos, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        """
        Create a graph with balance edges.
        """
        if self._added_edges is None:
            graph, added_edges, removed_edges = self.run(graph, mesh_edge_normalizer, is_training)
            self._added_edges = added_edges
            if removed_edges is not None:
                self._mask = self._determine_mask(graph.edge_sets[0], removed_edges)
        else:
            graph = self.add_graph_balance_edges(graph, self._added_edges, mesh_edge_normalizer, is_training)
            if self._mask is not None:
                graph = self.remove_graph_balance_edges(graph, self._mask, mesh_edge_normalizer, is_training)
        return graph

    def reset_edges(self):
        self._added_edges = None

    def reset_mask(self):
        self._mask = None