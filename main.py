import re

from src.util import device, read_yaml
from src.tasks.MeshTask import MeshTask
from src.data.data_loader import CONFIG_NAME, OUT_DIR
from src.algorithms.MeshSimulator import MeshSimulator
import os
import pickle
import random
import sys
import warnings
from multiprocessing import set_start_method

import numpy as np
import torch

from util.Functions import get_from_nested_dict

warnings.filterwarnings('ignore', category=UserWarning)


def main(preprocess: bool, train: bool, compute_rollout: bool):
    params = read_yaml(CONFIG_NAME)['params']
    print(f'Device used for this run: {device}')
    random_seed = params.get('random_seed')
    random.seed(random_seed)
    np.random.seed(seed=random_seed)
    torch.manual_seed(seed=random_seed)

    if train:
        algorithm = MeshSimulator(params)
        task = MeshTask(algorithm, params)
        if preprocess:
            task.preprocess()
        task.run_iteration()

    cluster = get_from_nested_dict(params, ['model', 'rmp', 'clustering'])
    num_clusters = get_from_nested_dict(params, ['model', 'rmp', 'num_clusters'])
    balancer = get_from_nested_dict(params, ['model', 'graph_balancer', 'algorithm'])
    mp = get_from_nested_dict(params, ['model', 'message_passing_steps'])
    model_name = f'model_{num_clusters}_cluster:{cluster}_balancer:{balancer}_mp:{mp}_epoch:'

    epochs = max([file.split('_epoch:')[1][:-4] for file in os.listdir(OUT_DIR) if re.match(rf'{model_name}[0-9]+\.pkl', file)])
    model_path = os.path.join(OUT_DIR, f'{model_name}{epochs}.pkl')
    with open(model_path, 'rb') as file:
        algorithm = pickle.load(file)
        task = MeshTask(algorithm, params)
        task.get_scalars()

    task.plot()


if __name__ == '__main__':
    """from optuna_work.experiment_wrappers import wrap_iterative_experiment

    cw = cluster_work.ClusterWork(wrap_iterative_experiment(IterativeExperiment, display_skip_warning=False))
    cw.run()"""
    args = [True, True, True]
    set_start_method('spawn')
    try:
        args[0] = sys.argv[1] == 'True'
        args[1] = sys.argv[2] == 'True'
        args[2] = sys.argv[3] == 'True'
    except IndexError:
        pass
    main(*args)
