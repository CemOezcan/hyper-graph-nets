import os
from src.util import read_yaml

from util.Types import ConfigDict
from util.Functions import get_from_nested_dict
from torch.utils.data import DataLoader
from src.data.flagdata import FlagSimpleDatasetIterative
from os.path import dirname as up

ROOT_DIR = up(up(up(os.path.join(os.path.abspath(__file__)))))
DATA_DIR = os.path.join(ROOT_DIR, 'dataset')
TASK_DIR = os.path.join(DATA_DIR, read_yaml()['params']['task']['dataset'])
OUT_DIR = os.path.join(TASK_DIR, 'output')
IN_DIR = os.path.join(TASK_DIR, 'input')


def get_data(config: ConfigDict, split='train', split_and_preprocess=True, add_targets=True):
    dataset = get_from_nested_dict(
        config, list_of_keys=["task", "dataset"], raise_error=True)
    if dataset == 'flag_minimal':
        return DataLoader(FlagSimpleDatasetIterative(path=IN_DIR, split=split, add_targets=add_targets,
                                                     split_and_preprocess=split_and_preprocess), batch_size=config.get('task').get('batch_size'),
                          prefetch_factor=config.get('task').get('prefetch_factor'), shuffle=False, num_workers=0)
    else:
        raise NotImplementedError("Implement your data loading here!")
