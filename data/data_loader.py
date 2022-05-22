import os

from util.Types import *
from util.Functions import get_from_nested_dict
from src.data.dataset import load_dataset

DATA_DIR = os.path.dirname(os.path.abspath(__file__))


def get_data(config: ConfigDict):
    dataset = get_from_nested_dict(config, list_of_keys=["task", "dataset"], raise_error=True)
    if dataset == "two_moons":
        from sklearn.datasets import make_moons
        import numpy as np
        X, y = make_moons(n_samples=5000, noise=0.2, random_state=0)
        X = (2 * (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))) - 1  # normalize
        X, y = X.astype(np.float32), y.astype(np.float32)
        return {
            "X": X,
            "y": y,
            "dimension": 2
        }
    elif dataset == 'flag_minimal':
        return load_dataset(os.path.join(DATA_DIR, 'flag_minimal'), 'train', batch_size=config.get('task').get('batch_size'),
                            prefetch_factor=config.get('task').get('prefetch_factor'),
                            add_targets=True, split_and_preprocess=True)

    else:
        raise NotImplementedError("Implement your data loading here!")
