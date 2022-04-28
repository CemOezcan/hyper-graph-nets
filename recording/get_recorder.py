from recording.register_loggers import register_loggers
from util.Types import *
from recording.Recorder import Recorder
from src.tasks.AbstractTask import AbstractTask
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm


def get_recorder(config: ConfigDict, algorithm: AbstractIterativeAlgorithm, task: AbstractTask) -> Recorder:
    """
    Processes the config to determine which parts of the algorithm and task to record and creates a corresponding
    recorder.
    Args:
        config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
            used by cw2 for the current run.
        algorithm: An instance of the algorithm to run.
        task: The task instance to perform the algorithm on. Can e.g., contain training data or a gym environment

    Returns: The recorder to use for this experiment trial. Simply calling recorder.record_iteration() will then record
      whatever is specified by the config, algorithm and task for the current algorithm state

    """
    loggers = register_loggers(config=config, algorithm=algorithm, task=task)
    recorder = Recorder(config=config, loggers=loggers, algorithm=algorithm, task=task)
    return recorder
