import pickle
import pprint
import random

from cw2 import cluster_work, experiment, cw_error
from cw2.cw_data import cw_logging
from src.algorithms.FlagSimulator import MeshSimulator
from src.algorithms.get_algorithm import get_algorithm
from src.tasks.AbstractTask import AbstractTask
from recording.get_recorder import get_recorder
from src.tasks.FlagTask import MeshTask
from src.tasks.get_task import get_task
from recording.Recorder import Recorder
from util.Types import *
from util.InitializeConfig import initialize_config
import copy
import numpy as np
import torch
import os
from src.util import device, read_yaml

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUT_DIR = os.path.join(DATA_DIR, 'output')


class IterativeExperiment(experiment.AbstractIterativeExperiment):
    def __init__(self):
        super(IterativeExperiment, self).__init__()
        self._task: AbstractTask = None
        self._recorder: Recorder = None
        self._config: ConfigDict = None

    def initialize(self, config: ConfigDict, rep: int, logger: cw_logging.LoggerArray) -> None:
        self._config = initialize_config(config=copy.deepcopy(config), repetition=rep)

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
        self._recorder = get_recorder(config=self._config, algorithm=algorithm, task=self._task)

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


def main(load):
    params = read_yaml()['params']
    dataset_name = params['task']['dataset']

    dataset_dir = os.path.join(DATA_DIR, dataset_name)
    output_dir = os.path.join(OUT_DIR, dataset_name)

    if load:
        dir = 'output/' + dataset_name + '/model.pkl'
        with open(os.path.join(DATA_DIR, dir), 'rb') as file:
            algo = pickle.load(file)
    else:
        algo = MeshSimulator(params)

    task = MeshTask(algo, params)

    if not load:
        task.run_iteration()

    if not os.path.exists(os.path.join(DATA_DIR, 'output/' + dataset_name + '/rollouts.pkl')):
        task.get_scalars()

    task.plot()

if __name__ == '__main__':
    """from optuna_work.experiment_wrappers import wrap_iterative_experiment

    cw = cluster_work.ClusterWork(wrap_iterative_experiment(IterativeExperiment, display_skip_warning=False))
    cw.run()"""
    main(True)
