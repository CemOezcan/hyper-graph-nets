import os

from src.data.graphloader import GraphDataLoader
from src.util import read_yaml

from util.Types import ConfigDict
from util.Functions import get_from_nested_dict
from torch.utils.data import DataLoader
from src.data.FlagDatasetIterative import FlagDatasetIterative
from os.path import dirname as up

ROOT_DIR = up(up(up(os.path.join(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
TASK_DIR = os.path.join(DATA_DIR, read_yaml('flag')[
                        'params']['task']['dataset'])
OUT_DIR = os.path.join(TASK_DIR, 'output')
IN_DIR = os.path.join(TASK_DIR, 'input')
CONFIG_NAME = 'flag'

def get_data(config: ConfigDict, split='train', split_and_preprocess=True, add_targets=True):
    dataset_name = get_from_nested_dict(config, list_of_keys=["task", "dataset"], raise_error=True)
    batch_size = config.get('task').get('batch_size')
    if dataset_name == 'flag_minimal' or dataset_name == 'flag_simple':
        dataset = FlagDatasetIterative(path=IN_DIR, split=split, add_targets=add_targets,
                                       split_and_preprocess=split_and_preprocess, batch_size=batch_size, config=config, in_dir=IN_DIR)
        ################################################################################## TODO
        return GraphDataLoader(dataset, batch_size=batch_size, prefetch_factor=config.get('task').get('prefetch_factor'), shuffle=False, num_workers=12)
    else:
        raise NotImplementedError("Implement your data loading here!")
