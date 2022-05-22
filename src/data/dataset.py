"""Utility functions for reading the datasets."""

from src.data.flagdata import FlagSimpleDatasetIterative
from torch.utils.data import DataLoader


# TODO: refactor to data.data_loader
# this function returns a torch dataloader
def load_dataset(path, split, add_targets=False, split_and_preprocess=False, batch_size=1, prefetch_factor=2):
    return DataLoader(FlagSimpleDatasetIterative(path=path, split=split, add_targets=add_targets,
                                                 split_and_preprocess=split_and_preprocess), batch_size=batch_size,
                      prefetch_factor=prefetch_factor, shuffle=False, num_workers=0)  # , collate_fn=collate_fn)
