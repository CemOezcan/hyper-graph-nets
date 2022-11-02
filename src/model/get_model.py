"""
Utility class to select a system model based on a given config file
"""
from src.model.abstract_system_model import AbstractSystemModel
from src.model.cylinder import CylinderModel
from src.model.flag import FlagModel
from src.model.plate import PlateModel
from util.Types import *
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from util.Functions import get_from_nested_dict


def get_model(config: ConfigDict) -> AbstractSystemModel:
    model_name = get_from_nested_dict(config, list_of_keys=["task", "dataset"], raise_error=True).lower()
    if 'flag' in model_name:
        return FlagModel(config.get('model'))
    elif 'plate' in model_name:
        return PlateModel(config.get('model'))
    elif 'cylinder' in model_name:
        return CylinderModel(config.get('model'))
    else:
        raise NotImplementedError("Implement your algorithms here!")
