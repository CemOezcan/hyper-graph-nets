from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import Tensor
import torch

from src.migration.normalizer import Normalizer
from src.util import MultiGraphWithPos, MultiGraph


class AbstractConnector(ABC):
    """
    Abstract superclass for the expansion of the input graph with remote edges.
    """

    def __init__(self):
        """
        Initializes the remote message passing strategy.

        """
        self._intra_normalizer = None
        self._inter_normalizer = None

    def initialize(self, intra: Normalizer, inter: Normalizer) -> List:
        """
        Initialize normalizers after fetching the subclass according to the given configuration file

        Parameters
        ----------
        intra : Normalizer
            Normalizer for intra cluster edges

        inter : Normalizer
            Normalizer for inter cluster edges

        Returns
        -------
            List
                An empty list

        """
        self._intra_normalizer = intra
        self._inter_normalizer = inter
        return list()

    @abstractmethod
    def run(self, graph: MultiGraph, clusters: List[List[int]], is_training: bool) -> MultiGraphWithPos:
        """
        Adds remote edges to the input graph.

        Parameters
        ----------
            graph : MultiGraph
                Input graph

            clusters : List[List[int]]
                Clustering of the graph

            is_training: bool
                Training or test sample

        Returns
        -------
            MultiGraphWithPos
                The input graph including remote edges.

        """
        raise NotImplementedError

    @staticmethod
    def _get_subgraph(model_type: str, target_feature: List[Tensor], senders_list: Tensor, receivers_list: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        target_feature = torch.cat(tuple(map(lambda x: x.clone().detach(), target_feature)), dim=0)
        senders = torch.cat((senders_list.clone().detach(), receivers_list.clone().detach()), dim=0)
        receivers = torch.cat((receivers_list.clone().detach(), senders_list.clone().detach()), dim=0)

        # TODO: Make model independent
        if model_type == 'flag' or model_type == 'plate':
            relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                       torch.index_select(input=target_feature, dim=0, index=receivers))
            world, mesh = torch.split(relative_target_feature, 3, dim=1)
            edge_features = torch.cat(
                (world, torch.norm(world, dim=-1, keepdim=True), mesh, torch.norm(mesh, dim=-1, keepdim=True)),
                dim=-1)
        else:
            raise Exception("Model type is not specified in RippleNodeConnector.")

        return senders, receivers, edge_features
