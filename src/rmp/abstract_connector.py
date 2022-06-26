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

    def __init__(self, normalizer: Normalizer):
        """
        Initializes the remote message passing strategy.

        Parameters
        ----------
        normalizer :  Normalizer for remote edges
        """
        self._normalizer = normalizer
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
    def run(self, graph: MultiGraph, clusters: List[List], is_training: bool) -> MultiGraphWithPos:
        """
        Adds remote edges to the input graph.

        Parameters
        ----------
        graph : Input graph
        clusters : Clustering of the graph
        is_training : Whether the input is a training instance or not

        Returns the input graph including remote edges.
        -------

        """
        raise NotImplementedError

    @staticmethod
    def _get_subgraph(model_type: str, target_feature: List[Tensor], senders_list: Tensor, receivers_list: Tensor) \
            -> Tuple[Tensor, Tensor, Tensor]:
        target_feature = torch.cat(
            tuple(map(lambda x: x.clone().detach(), target_feature)), dim=0)
        senders = torch.cat(
            (senders_list.clone().detach(), receivers_list.clone().detach()), dim=0)
        receivers = torch.cat(
            (receivers_list.clone().detach(), senders_list.clone().detach()), dim=0)

        # TODO: Make model independent
        if model_type == 'flag' or model_type == 'deform_model':
            relative_target_feature = (torch.index_select(input=target_feature, dim=0, index=senders) -
                                       torch.index_select(input=target_feature, dim=0, index=receivers))
            edge_features = torch.cat(
                (relative_target_feature, torch.norm(relative_target_feature, dim=-1, keepdim=True)), dim=-1)
        else:
            raise Exception("Model type is not specified in RippleNodeConnector.")

        return senders, receivers, edge_features
