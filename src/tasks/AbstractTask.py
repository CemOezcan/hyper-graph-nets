import os
import pickle
import re
from abc import ABC, abstractmethod
from typing import Tuple

from matplotlib.animation import PillowWriter, FuncAnimation

from src.algorithms.MeshSimulator import MeshSimulator
from src.algorithms.get_algorithm import get_algorithm
from src.data.data_loader import OUT_DIR
from util.Functions import get_from_nested_dict
from util.Types import *
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm


class AbstractTask(ABC):
    def __init__(self, config: ConfigDict):
        """
        Initializes the current task. Depending on the application, this can be something like a classification task
        for supervised learning (in which case this method would be used to load in the data and labels), or a gym
        environment for a Reinforcement Learning task.

        Parameters
        ----------
            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run
        """
        self._config = config

        retrain = config.get('retrain')
        cluster = get_from_nested_dict(config, ['model', 'rmp', 'clustering'])
        num_clusters = get_from_nested_dict(config, ['model', 'rmp', 'num_clusters'])
        balancer = get_from_nested_dict(config, ['model', 'graph_balancer', 'algorithm'])
        mp = get_from_nested_dict(config, ['model', 'message_passing_steps'])
        model_name = f'model_{num_clusters}_cluster:{cluster}_balancer:{balancer}_mp:{mp}_epoch:'

        epochs = [int(file.split('_epoch:')[1][:-4]) for file in os.listdir(OUT_DIR) if re.match(rf'{model_name}[0-9]+\.pkl', file)]
        epochs = list() if retrain else epochs

        if epochs:
            self._current_epoch = max(epochs)
            model_path = os.path.join(OUT_DIR, f'{model_name}{self._current_epoch}.pkl')
            with open(model_path, 'rb') as file:
                self._algorithm = pickle.load(file)
        else:
            self._algorithm = get_algorithm(config)
            self._current_epoch = 0

        self._config = config
        self._raw_data = {}

    @abstractmethod
    def run_iterations(self) -> None:
        """
        Runs all iteration of its iterative algorithm.
        """
        raise NotImplementedError

    @abstractmethod
    def get_scalars(self) -> None:
        """
        Evaluates the final model after all training epochs.

        Returns
        -------

        """

        raise NotImplementedError

    @abstractmethod
    def plot(self, task_name: str) -> Tuple[FuncAnimation, PillowWriter]:
        """
        Create a plot for the current state of the task and its algorithm

        Parameters
        ----------
            task_name : str
                The name of the task

        Returns
        -------
            Tuple[FuncAnimation, PillowWriter]
                The simulations

        """

        raise NotImplementedError

    @property
    def raw_data(self):
        return self._raw_data
