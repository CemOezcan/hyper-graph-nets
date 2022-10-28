"""
Utility class to select a task based on a given config file
"""
from src.tasks.MeshTask import MeshTask
from util.Types import *
from util.Functions import get_from_nested_dict
from src.tasks.AbstractTask import AbstractTask
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm


def get_task(config: ConfigDict) -> AbstractTask:
    task = get_from_nested_dict(config, list_of_keys=["task", "task"], raise_error=True)

    if task == "mesh":
        return MeshTask(config=config)
    else:
        raise NotImplementedError("Implement your tasks here!")
