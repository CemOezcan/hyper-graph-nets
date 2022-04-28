"""
Utility class to select an algorithm based on a given config file
"""
from util.Types import *
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from util.Functions import get_from_nested_dict


def get_algorithm(config: ConfigDict) -> AbstractIterativeAlgorithm:
    algorithm_name = get_from_nested_dict(config, list_of_keys=["algorithm", "name"], raise_error=True).lower()
    if algorithm_name == "classifier":
        from src.algorithms.BinaryClassifier import BinaryClassifier
        return BinaryClassifier(config=config)
    else:
        raise NotImplementedError("Implement your algorithms here!")
