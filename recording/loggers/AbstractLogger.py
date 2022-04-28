from abc import ABC, abstractmethod
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.tasks.AbstractTask import AbstractTask
from util.Types import *
from recording.loggers.get_logging_writer import get_logging_writer
from recording.loggers.logger_util import process_logger_name
from util.Functions import get_from_nested_dict


class AbstractLogger(ABC):
    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm, task: AbstractTask):
        self._config = config
        self._algorithm = algorithm
        self._task = task
        self._recording_directory: str = get_from_nested_dict(dictionary=config,
                                                              list_of_keys=["_recording_structure", "_recording_dir"],
                                                              raise_error=True)
        self._writer = get_logging_writer(writer_name=self.processed_name,
                                          recording_directory=self._recording_directory)
        self._plot_frequency: int = get_from_nested_dict(dictionary=config,
                                                         list_of_keys=["recording", "plot_frequency"],
                                                         raise_error=True)

    @abstractmethod
    def log_iteration(self, previous_recorded_values: RecordingDict,
                      iteration: int) -> Optional[RecordingDict]:
        """
        Log the current training iteration of the algorithm instance and its task.
        Args:
            previous_recorded_values: Metrics and other information that was computed by previous loggers
            iteration: The current algorithm iteration. Is provided for internal consistency, since we may not want to
              record every algorithm iteration

        Returns:  Some loggers may return their results as a dictionary

        """
        raise NotImplementedError

    def finalize(self) -> None:
        """
        Finalizes the recording, e.g., by saving certain things to disk or by postpressing the results in one way or
        another.
        Returns:

        """
        raise NotImplementedError

    def remove_writer(self) -> None:
        self._writer.handlers = []
        del self._writer

    @property
    def processed_name(self) -> str:
        return process_logger_name(self.__class__.__name__)
