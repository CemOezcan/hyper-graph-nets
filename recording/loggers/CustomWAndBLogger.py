from util.Types import *
from recording.loggers.AbstractLogger import AbstractLogger
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.tasks.AbstractTask import AbstractTask
import wandb
import os
from util.Functions import get_from_nested_dict


class CustomWAndBLogger(AbstractLogger):
    """
    Logs (some of the) recorded results using wandb.ai.
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm, task: AbstractTask):
        super().__init__(config=config, algorithm=algorithm, task=task)
        wandb_params = get_from_nested_dict(config, list_of_keys=["recording", "wandb_params"], raise_error=True)
        project_name = wandb_params.get("project_name")

        recording_structure = config.get("_recording_structure")
        groupname = recording_structure.get("_groupname")[-127:]
        runname = recording_structure.get("_runname")[-127:]
        recording_dir = recording_structure.get("_recording_dir")
        job_name = recording_structure.get("_job_name")

        tags = []
        if get_from_nested_dict(config, list_of_keys=["algorithm", "name"], default_return=None) is not None:
            tags.append(get_from_nested_dict(config, list_of_keys=["algorithm", "name"]))
        if get_from_nested_dict(config, list_of_keys=["task", "task"], default_return=None) is not None:
            tags.append(get_from_nested_dict(config, list_of_keys=["task", "task"]))

        self.wandb_logger = wandb.init(project=project_name,  # name of the whole project
                                       tags=tags,  # tags to search the runs by. Currently contains task and algorithm
                                       job_type=job_name,  # name of your experiment
                                       group=groupname,  # group of identical hyperparameters for different seeds
                                       name=runname,  # individual repetitions

                                       dir=recording_dir,  # local directory for wandb recording

                                       config=config,  # full file config
                                       reinit=False,
                                       settings=wandb.Settings(start_method="thread"))

    def log_iteration(self, previous_recorded_values: RecordingDict, iteration: int) -> None:
        """
        Parses and logs the given dict of recorder metrics to wandb.
        Args:
            previous_recorded_values: A dictionary of previously recorded things
            iteration: The current iteration of the algorithm
        Returns:

        """
        if "scalars" in previous_recorded_values:
            self.wandb_logger.log(data=previous_recorded_values.get("scalars"), step=iteration)
        # extend me to log other things as well!

    def finalize(self) -> None:
        """
        Properly close the wandb logger
        Returns:

        """
        self.wandb_logger.finish()
