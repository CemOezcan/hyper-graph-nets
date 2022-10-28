"""
Utility class to select an algorithm based on a given config file
"""
from src.algorithms.MeshSimulator import MeshSimulator
from util.Types import *
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from util.Functions import get_from_nested_dict


def get_algorithm(config: ConfigDict) -> AbstractIterativeAlgorithm:
    task = get_from_nested_dict(config, list_of_keys=["task", "task"], raise_error=True)

    if task == "mesh":
        return MeshSimulator(config=config)
    else:
        raise NotImplementedError("Implement your tasks here!")
