import re

from src.tasks.get_task import get_task
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

warnings.filterwarnings('ignore')


def main(config_name=CONFIG_NAME):
    config_file = read_yaml(config_name)
    retrain = config_file['compute']['retrain']

    params = config_file['params']
    print(f'Device used for this run: {device}')
    random_seed = params.get('random_seed')
    random.seed(random_seed)
    np.random.seed(seed=random_seed)
    torch.manual_seed(seed=random_seed)

    cluster = get_from_nested_dict(params, ['model', 'rmp', 'clustering'])
    num_clusters = get_from_nested_dict(params, ['model', 'rmp', 'num_clusters'])
    balancer = get_from_nested_dict(params, ['model', 'graph_balancer', 'algorithm'])
    mp = get_from_nested_dict(params, ['model', 'message_passing_steps'])
    model_name = f'model_{num_clusters}_cluster:{cluster}_balancer:{balancer}_mp:{mp}_epoch:'

    epochs = [int(file.split('_epoch:')[1][:-4]) for file in os.listdir(OUT_DIR) if re.match(rf'{model_name}[0-9]+\.pkl', file)]
    epochs = list() if retrain else epochs

    if epochs:
        last_epoch = max(epochs)
        model_path = os.path.join(OUT_DIR, f'{model_name}{last_epoch}.pkl')
        with open(model_path, 'rb') as file:
            algorithm = pickle.load(file)

        task = get_task(params, algorithm)
        task.run_iterations(last_epoch)
    else:
        algorithm = MeshSimulator(params)
        task = get_task(params, algorithm)
        task.run_iterations(0)

    task.get_scalars()
    task.plot(f'{model_name}_final')


if __name__ == '__main__':
    set_start_method('spawn')
    try:
        args = sys.argv[1]
        main(args)
    except IndexError:
        main(CONFIG_NAME)
