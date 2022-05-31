import os

from util.Types import ConfigDict
from util.Functions import get_from_nested_dict
from torch.utils.data import DataLoader
from src.data.flagdata import FlagSimpleDatasetIterative

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(DATA_DIR, 'output')


def get_data(config: ConfigDict, split='train', split_and_preprocess=True, add_targets=True):
    dataset = get_from_nested_dict(config, list_of_keys=["task", "dataset"], raise_error=True)
    if dataset == 'flag_minimal':
        return DataLoader(FlagSimpleDatasetIterative(path=os.path.join(DATA_DIR, 'flag_minimal'), split=split, add_targets=add_targets,
                                                     split_and_preprocess=split_and_preprocess), batch_size=config.get('task').get('batch_size'),
                          prefetch_factor=config.get('task').get('prefetch_factor'), shuffle=False, num_workers=0)
    else:
        raise NotImplementedError("Implement your data loading here!")
