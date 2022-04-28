from util.Types import *
from recording.loggers.AbstractLogger import AbstractLogger
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.tasks.AbstractTask import AbstractTask
from recording.loggers.logger_util import save_to_yaml
from pprint import pformat


class ConfigLogger(AbstractLogger):
    """
    A very basic logger that prints the config file as an output at the start of the experiment. Also saves the config
    as a .yaml in the experiment's directory.
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm, task: AbstractTask):
        super().__init__(config=config, algorithm=algorithm, task=task)
        save_to_yaml(dictionary=config, save_name=self.processed_name,
                     recording_directory=config.get("_recording_structure").get("_recording_dir"))
        self._writer.info("\n" + pformat(object=config, indent=2))

    def log_iteration(self, previous_recorded_values: RecordingDict,
                      iteration: int) -> Optional[RecordingDict]:
        pass

    def finalize(self) -> None:
        pass
