import numpy as np
from torch.utils.data import DataLoader


class GraphDataLoader(DataLoader):
    """
    A data loader for mesh data.
    """

    def __init__(self, graphs):
        super().__init__(graphs)

    def __iter__(self):
        np.random.seed(0)
        self._iterator = iter(self.dataset)
        return self

    def __next__(self):
        return next(self._iterator)
