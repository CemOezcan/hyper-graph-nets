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

    params = config_file['params']
    print(f'Device used for this run: {device}')
    random_seed = params.get('random_seed')
    random.seed(random_seed)
    np.random.seed(seed=random_seed)
    torch.manual_seed(seed=random_seed)

    task = get_task(params)
    task.run_iterations()

    task.get_scalars()


if __name__ == '__main__':
    set_start_method('spawn')
    try:
        args = sys.argv[1]
        main(args)
    except IndexError:
        main(CONFIG_NAME)
