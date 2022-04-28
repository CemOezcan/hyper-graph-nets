from recording.loggers.AbstractLogger import AbstractLogger
from util.Types import *
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.tasks.AbstractTask import AbstractTask
import os


class VisualizationLogger(AbstractLogger):

    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm, task: AbstractTask):
        self._num_iterations = config.get("iterations")
        super().__init__(config=config, algorithm=algorithm, task=task)

    def log_iteration(self, previous_recorded_values: RecordingDict,
                      iteration: int):
        if self._plot_frequency > 0 and iteration % self._plot_frequency == 0:
            self._writer.info(msg="Plotting Visualization(s)")
            self._plot(iteration=iteration)

    def _plot(self, iteration: Optional[int] = None):
        fig = self._task.plot()

        save_directory = os.path.join(self._recording_directory, self.processed_name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        fig.write_image(os.path.join(save_directory, self._format_iteration(iteration)))
        fig.data = []
        del fig

    def _format_iteration(self, iteration: Optional[int]) -> str:
        if isinstance(iteration, int):
            format_string = "{:0" + str(len(str(self._num_iterations))) + "d}.pdf"
            return format_string.format(iteration)
        else:
            return "final.pdf"

    def finalize(self) -> None:
        self._plot()
