from typing import Optional

from torch.utils.data import DataLoader

from util.Types import *
from abc import ABC, abstractmethod


class AbstractIterativeAlgorithm(ABC):
    """
    Superclass for iterative algorithms
    """

    def __init__(self, config: ConfigDict) -> None:
        """
        Initializes the iterative algorithm.

        Parameters
        ----------
            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.

        """

        self._config = config

    @abstractmethod
    def initialize(self, task_information: ConfigDict) -> None:
        """
        Due to the interplay between the algorithm and the task, it sometimes makes sense for the task to provide
        additional initial information to the algorithm. This information may e.g., be the dimensionality of the task,
        the kind of training regime to perform etc.

        Parameters
        ----------
            task_information : ConfigDict
                A dictionary containing information on how to execute the algorithm on the current task

        Returns
        -------

        """

        raise NotImplementedError

    @abstractmethod
    def fit_iteration(self, train_dataloader: DataLoader) -> Optional[Dict]:
        """
        Train your algorithm for a single iteration. This can e.g., be a single epoch of neural network training,
        a policy update step, or something more complex. Just see this as the outermost for-loop of your algorithm.

        Parameters
        ----------
            train_dataloader : DataLoader
                A data loader containing the training data

        Returns
        -------
            May return an optional dictionary of values produced during the fit. These may e.g., be statistics
            of the fit such as a training loss.

        """

        raise NotImplementedError

    @abstractmethod
    def one_step_evaluator(self, ds_loader: DataLoader, instances: int, task_name: str, logging: bool) -> Optional[Dict]:
        """
        Predict the system state for the next time step and evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : DataLoader
                A data loader containing test/validation instances

            instances : int
                Number of trajectories used to estimate the one-step loss

            task_name : str
                Name of the task

            logging : bool
                Whether to log the results to wandb

        Returns
        -------
            Optional[Dict]
                A single result that scores the input, potentially per sample

        """

        raise NotImplementedError

    @abstractmethod
    def n_step_evaluator(self, ds_loader: DataLoader, task_name: str, n_step_list: List[int], n_traj: int):
        """
        Predict the system state after n time steps. N step predictions are performed recursively within trajectories.
        Evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : DataLoader
                A data loader containing test/validation instances

            task_name : str
                Name of the task

            n_step_list : List[int]
                Different values for n, with which to estimate the n-step loss

            n_traj : int
                Number of trajectories used to estimate the n-step loss

        Returns
        -------

        """

        raise NotImplementedError

    @abstractmethod
    def rollout_evaluator(self, ds_loader: DataLoader, rollouts: int, task_name: str, logging: bool) -> Optional[Dict]:
        """
        Recursive prediction of the system state at the end of trajectories.
        Evaluate the predictions over the test data.

        Parameters
        ----------
            ds_loader : DataLoader
                A data loader containing test/validation instances

            rollouts : int
                Number of trajectories used to estimate the rollout loss

            task_name : str
                Name of the task

            logging : bool
                Whether to log the results to wandb

        Returns
        -------
            Optional[Dict]
                A single result that scores the input, potentially per sample

        """

        raise NotImplementedError

    @property
    def config(self) -> ConfigDict:
        """

        Returns
        -------
            ConfigDict
                The config file

        """
        return self._config
