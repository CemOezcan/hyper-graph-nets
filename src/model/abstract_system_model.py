from typing import Optional

from torch import nn

from util.Types import *
from abc import ABC, abstractmethod


class AbstractSystemModel(ABC, nn.Module):

    @abstractmethod
    def training_step(self, *args, **kwargs) -> Optional[ValueDict]:
        """
        Perform a single training step.

        Returns: The training loss.

        """
        raise NotImplementedError

    @abstractmethod
    def validation_step(self, *args, **kwargs) -> ValueDict:
        """
        Evaluate given input data and potentially auxiliary information to create a dictionary of resulting values.
        What kinds of things are scored/evaluated depends on the concrete algorithm.
        Args:
            *args:
            **kwargs:

        Returns: A dictionary with different values that are evaluated from the given input data. May e.g., the
        accuracy of the model.

        """
        raise NotImplementedError

    @abstractmethod
    def update(self, *args, **kwargs) -> Result:
        """
        Makes a prediction for the given input data and uses it to compute the predicted system state.

        Args:
            *args:
            **kwargs:

        Returns: Some representation of the predicted system state

        """
        raise NotImplementedError

    @abstractmethod
    def build_graph(self, *args, **kwargs) -> Result:
        """
        Constructs the input graph given a system state.

        Args:
            *args:
            **kwargs:

        Returns: The system state represented by a (heterogeneous hyper-) graph

        """
        raise NotImplementedError

    @abstractmethod
    def expand_graph(self, *args, **kwargs) -> Result:
        """
        Expands the input graph with remote connections.

        Args:
            *args:
            **kwargs:

        Returns: The system state represented by a (heterogeneous hyper-) graph

        """
        raise NotImplementedError

    @abstractmethod
    def rollout(self, *args, **kwargs) -> Result:
        """
        Predict a sub trajectory for n time steps by making n consecutive one-step predictions recursively.

        Args:
            *args:
            **kwargs:

        Returns: The predicted and the ground truth trajectories as well as the corresponding losses for each time step

        """
        raise NotImplementedError

    @abstractmethod
    def n_step_computation(self, *args, **kwargs) -> Result:
        """
        Predict the system state after n time steps. N step predictions are performed recursively within trajectories.

        Args:
            *args:
            **kwargs:

        Returns: The n-step loss

        """
        raise NotImplementedError

    @abstractmethod
    def evaluate(self) -> None:
        raise NotImplementedError
