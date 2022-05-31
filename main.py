import copy
import os
import pickle

import numpy as np
import torch
from cw2 import cluster_work, cw_error, experiment
from cw2.cw_data import cw_logging

from recording.get_recorder import get_recorder
from recording.Recorder import Recorder
from src.algorithms.get_algorithm import get_algorithm
from src.algorithms.MeshSimulator import MeshSimulator
from src.tasks.AbstractTask import AbstractTask
from src.tasks.get_task import get_task
from src.tasks.MeshTask import MeshTask
from src.util import read_yaml
from util.InitializeConfig import initialize_config
from util.Types import ConfigDict, ScalarDict
from src.data.data_loader import OUT_DIR, CONFIG_NAME


class IterativeExperiment(experiment.AbstractIterativeExperiment):
    def __init__(self):
        super(IterativeExperiment, self).__init__()
        self._task: AbstractTask = None
        self._recorder: Recorder = None
        self._config: ConfigDict = None

    def initialize(self, config: ConfigDict, rep: int, logger: cw_logging.LoggerArray) -> None:
        self._config = initialize_config(
            config=copy.deepcopy(config), repetition=rep)

        # initialize random seeds
        numpy_seed = self._config.get("random_seeds").get("numpy")
        pytorch_seed = self._config.get("random_seeds").get("pytorch")
        if numpy_seed is not None:
            np.random.seed(seed=numpy_seed)
        if pytorch_seed is not None:
            torch.manual_seed(seed=pytorch_seed)

        # start with the actual task and algorithm
        algorithm = get_algorithm(config=self._config)
        self._task = get_task(config=self._config, algorithm=algorithm)
        self._recorder = get_recorder(
            config=self._config, algorithm=algorithm, task=self._task)

    def iterate(self, config: ConfigDict, rep: int, n: int) -> ScalarDict:
        self._task.run_iteration()
        scalars = self._recorder.record_iteration(iteration=n)
        return scalars

    def save_state(self, cw_config: dict, rep: int, n: int) -> None:
        pass  # we already save everything in "iterate" for optunawork compatibility

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        if self._recorder is not None:
            try:
                self._recorder.finalize()
            except Exception as e:
                print("Failed finalizing recorder: {}".format(e))


def main(load_model: bool, compute_rollout: bool):
    params = read_yaml(CONFIG_NAME)['params']
    dataset_name = params['task']['dataset']

    if load_model:
        model_path = os.path.join(OUT_DIR, dataset_name) + '/model.pkl'
        with open(model_path, 'rb') as file:
            algorithm = pickle.load(file)
    else:
        algorithm = MeshSimulator(params)

    task = MeshTask(algorithm, params)

    if not load_model:
        task.run_iteration()
    if compute_rollout:
        task.get_scalars()
    task.plot()


if __name__ == '__main__':
    """from optuna_work.experiment_wrappers import wrap_iterative_experiment

    cw = cluster_work.ClusterWork(wrap_iterative_experiment(IterativeExperiment, display_skip_warning=False))
    cw.run()"""
    main(False, True)
