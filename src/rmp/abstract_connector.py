from abc import ABC, abstractmethod
from typing import List, Tuple

from torch import Tensor
import torch

from src.migration.normalizer import Normalizer
from src.util import MultiGraphWithPos, MultiGraph
from util.Types import ConfigDict


class AbstractConnector(ABC):
    """
    Abstract superclass for the expansion of the input graph with remote edges.
    """

    def __init__(self, fully_connect, noise_scale, hyper_node_features):
        """
        Initializes the remote message passing strategy.

        """
        self._intra_normalizer = None
        self._inter_normalizer = None
        self._hyper_normalizer = None
        self._fully_connect = fully_connect
        self._noise_scale = noise_scale
        self._hyper_node_features = hyper_node_features

    def initialize(self, intra: Normalizer, inter: Normalizer, hyper: Normalizer) -> List:
        """
        Initialize normalizers after fetching the subclass according to the given configuration file

        Parameters
        ----------
        hyper : Normalizer
            Normalizer for hyper nodes

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
        self._hyper_normalizer = hyper

        return list()

    @abstractmethod
    def run(self, graph: MultiGraph, clusters: List[Tensor], neighbors: List[Tensor], is_training: bool) -> MultiGraphWithPos:
        """
        Adds remote edges to the input graph.

        Parameters
        ----------
            graph : MultiGraph
                Input graph

            clusters : List[Tensor]
                Clustering of the graph

            neighbors : List[Tensor]
                Neighboring clusters

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
