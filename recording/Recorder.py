from util.Types import *
from recording.loggers.AbstractLogger import AbstractLogger
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.tasks.AbstractTask import AbstractTask
from recording.loggers.get_logging_writer import get_logging_writer


class Recorder:
    """
    Records the algorithm and task whenever called by computing common recording values and then delegating the
    recording itself to different recorders.
    """
    def __init__(self, config: ConfigDict,
                 loggers: List[AbstractLogger],
                 algorithm: AbstractIterativeAlgorithm,
                 task: AbstractTask):
        self._loggers = loggers
        self._algorithm = algorithm
        self._task = task
        self._writer = get_logging_writer(self.__class__.__name__,
                                          recording_directory=config.get("_recording_structure").get("_recording_dir"))

    def record_iteration(self, iteration: int) -> RecordingDict:
        self._writer.info("Recording iteration {}".format(iteration))
        recorded_values = {}
        for logger in self._loggers:
            try:
                logger_values = logger.log_iteration(previous_recorded_values=recorded_values,
                                                     iteration=iteration)
                if logger_values is not None:
                    recorded_values[logger.processed_name] = logger_values
            except Exception as e:
                self._writer.error("Error with logger '{}': {}".format(logger.__class__.__name__, e))
        self._writer.info("Finished recording iteration {}\n".format(iteration))
        scalars = recorded_values.get("scalars", {})
        return scalars

    def finalize(self) -> None:
        """
        Finalizes the recording by asking all loggers to finalize. The loggers can then save things to disk or
        postprocess results.
        Returns:

        """
        self._writer.info("Finished experiment! Finalizing recording")
        for logger in self._loggers:
            logger.finalize()
            logger.remove_writer()

        self._writer.info("Finalized recording.")
        self._writer.handlers = []
        del self._writer
