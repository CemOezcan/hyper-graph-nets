import abc
from abc import ABC
from util.Types import *
import plotly.graph_objects as go
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm


class AbstractTask(ABC):
    def __init__(self, algorithm: AbstractIterativeAlgorithm, config: ConfigDict):
        """
        Initializes the current task. Depending on the application, this can be something like a classification task
        for supervised learning (in which case this method would be used to load in the data and labels), or a gym
        environment for a Reinforcement Learning task.

        Args:
            algorithm: The algorithm to train on this task.
            config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.
        """
        self._algorithm = algorithm
        self._config = config
        self._raw_data = {}

    @abc.abstractmethod
    def run_iteration(self):
        raise NotImplementedError("AbstractTask does not implemented run_iteration()")

    @abc.abstractmethod
    def get_scalars(self) -> ScalarDict:
        raise NotImplementedError("AbstractTask does not implemented get_scalars()")

    @abc.abstractmethod
    def plot(self) -> go.Figure:
        """
        Create a plot for the current state of the task and its algorithm
        Returns:

        """
        raise NotImplementedError("AbstractTask does not implemented plot()")

    @property
    def raw_data(self):
        return self._raw_data
