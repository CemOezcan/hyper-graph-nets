from util.Types import *
from util.Functions import get_from_nested_dict


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
    else:
        raise NotImplementedError("Implement your data loading here!")
