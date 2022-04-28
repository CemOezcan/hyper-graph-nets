from util.Types import *
from recording.loggers.AbstractLogger import AbstractLogger


class NetworkSummaryLogger(AbstractLogger):
    """
    A very basic logger that prints the config file as an output at the start of the experiment. Also saves the config
    as a .yaml in the experiment's directory.
    """

    def log_iteration(self, previous_recorded_values: RecordingDict,
                      iteration: int) -> None:
        if iteration == 0:
            if hasattr(self._algorithm, "network"):
                self._writer.info(self._algorithm.network)

    def finalize(self) -> None:
        pass
