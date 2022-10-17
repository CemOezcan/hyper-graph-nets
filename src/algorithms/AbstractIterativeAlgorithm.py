from typing import Optional

from util.Types import *
from abc import ABC, abstractmethod


class AbstractIterativeAlgorithm(ABC):
    def __init__(self, config: ConfigDict) -> None:
        """
        Initializes the iterative algorithm.
        Args:
            config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.
        Returns:

        """
        self._config = config

    @abstractmethod
    def initialize(self, task_information: ConfigDict) -> None:
        """
        Due to the interplay between the algorithm and the task, it sometimes makes sense for the task to provide
        additional initial information to the algorithm. This information may e.g., be the dimensionality of the task,
        the kind of training regime to perform etc.
        Args:
            task_information: A dictionary containing information on how to execute the algorithm on the current task

        Returns:

        """
        raise NotImplementedError


    @abstractmethod
    def fit_iteration(self, *args, **kwargs) -> Optional[ValueDict]:
        """
        Train your algorithm for a single iteration. This can e.g., be a single epoch of neural network training,
        a policy update step, or something more complex. Just see this as the outermost for-loop of your algorithm.

        Returns: May return an optional dictionary of values produced during the fit. These may e.g., be statistics
        of the fit such as a training loss.

        """
        raise NotImplementedError

    @abstractmethod
    def one_step_evaluator(self, *args, **kwargs) -> Result:
        """
        Predict the system state for the next time step and evaluate the predictions over the test data.
        Args:
            *args:
            **kwargs:

        Returns: A single result that scores the input, potentially per sample

        """
        raise NotImplementedError

    @abstractmethod
    def n_step_evaluator(self, *args, **kwargs) -> Result:
        """
        Predict the system state after n time steps. N step predictions are performed recursively within trajectories.
         Evaluate the predictions over the test data.
        Args:
            *args:
            **kwargs:

        Returns: A single result that scores the input, potentially per sample

        """
        raise NotImplementedError

    @abstractmethod
    def rollout_evaluator(self, *args, **kwargs) -> Result:
        # TODO: Rename to rollout loss
        """
        Recursive prediction of the system state at the end of trajectories.
         Evaluate the predictions over the test data.
        Args:
            *args:
            **kwargs:

        Returns: A single result that scores the input, potentially per sample

        """
        raise NotImplementedError

    @property
    def config(self) -> ConfigDict:
        return self._config
