
from torch.utils.data import DataLoader


class GraphDataLoader(DataLoader):

    def __init__(self, graphs, **kwargs):
        super().__init__(graphs, **kwargs)

    def __iter__(self):
        return self.dataset

    def __next__(self):
        return next(self.dataset)
