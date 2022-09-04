from abc import ABC, abstractmethod
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

    def create_graph(self, graph: MultiGraphWithPos, mesh_edge_normalizer, is_training: bool) -> MultiGraphWithPos:
        """
        Create a graph with balance edges.
        """
        if self._added_edges is None:
            graph, added_edges = self.run(graph, mesh_edge_normalizer, is_training)
            self._added_edges = added_edges
        else:
            graph = self.add_graph_balance_edges(graph, self._added_edges, mesh_edge_normalizer, is_training)
        return graph

    def reset_edges(self):
        self._added_edges = None
