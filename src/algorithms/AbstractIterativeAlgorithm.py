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
    def score(self, *args, **kwargs) -> ValueDict:
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
    def predict(self, *args, **kwargs) -> Result:
        """
        Give some prediction for the given input data. The difference to score() is that the prediction is meant to only
         get a set list of input samples and output the model evaluates for it, while the score() function also has
         access to auxiliary information and is meant to give a dictionary of different interesting values.
        Args:
            *args:
            **kwargs:

        Returns: A single result that scores the input, potentially per sample

        """
        raise NotImplementedError

    @property
    def config(self)-> ConfigDict:
        return self._config
