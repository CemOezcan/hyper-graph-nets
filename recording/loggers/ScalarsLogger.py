from util.Types import *
from recording.loggers.AbstractLogger import AbstractLogger
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.tasks.AbstractTask import AbstractTask
from timeit import default_timer as timer
from recording.loggers.logger_util import save_to_yaml
import os
import numpy as np


def _get_memory_usage() -> float:
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


class ScalarsLogger(AbstractLogger):
    """
    A basic logger for scalar metrics
    """

    def __init__(self, config: ConfigDict, algorithm: AbstractIterativeAlgorithm, task: AbstractTask):
        super().__init__(config=config, algorithm=algorithm, task=task)
        self._previous_duration = timer()  # start time
        self._cumulative_duration = 0
        self._all_scalars = {}

    def log_iteration(self, previous_recorded_values: RecordingDict,
                      iteration: int) -> Optional[RecordingDict]:
        scalars = self._task.get_scalars()
            
        scalars = scalars | self._get_default_scalars()

        self._write_and_save(scalars)
        if self._plot_frequency > 0 and iteration % self._plot_frequency == 0:
            self._plot()
        return scalars

    def finalize(self) -> None:
        """
        Finalizes the recording, e.g., by saving certain things to disk or by postpressing the results in one way or
        another.
        Returns:

        """
        save_to_yaml(dictionary=self._all_scalars, save_name=self.processed_name,
                     recording_directory=self._recording_directory)
        self._plot()

    def _write_and_save(self, scalars: Dict[str, Any]):
        for key, value in scalars.items():
            if key not in self._all_scalars:
                self._all_scalars[key] = []
            self._all_scalars[key].append(scalars[key])
            if isinstance(value, float):
                self._writer.info(msg=key.title() + ": {:.3f}".format(value))
            else:
                self._writer.info(msg=key.title() + ": " + str(value))
                
    def _plot(self):
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        num_scalars = len(self._all_scalars)

        fig = make_subplots(rows=num_scalars, cols=1, shared_xaxes=True,
                            subplot_titles=[key.title().replace("_", " ")
                                            for key in self._all_scalars.keys()],
                            x_title="Iteration")
        for position, (key, values) in enumerate(self._all_scalars.items()):

            fig.add_trace(
                go.Scatter(x=np.arange(len(values)),
                           y=values,
                           mode="lines",
                           name=key.title().replace("_", " ")),
                row=position+1, col=1
            )

        fig.update_layout(height=150*num_scalars, width=800, title_text="Scalars", showlegend=False)
        fig.write_image(os.path.join(self._recording_directory, self.processed_name+".pdf"))
        # todo we currently freeze on this line when trying to exit via a keyboard interrupt for pycharm on windows.
        #  this may be due to the plotly backend exiting prematurely or something similar.
        fig.data = []
        del fig

    def _get_default_scalars(self):
        duration = self._get_duration()
        self._cumulative_duration += duration
        default_scalars = {"duration": duration,
                           "cumulative_duration": self._cumulative_duration}
        if os.name == "posix":  # record memory usage per iteration. Only works on linux
            default_scalars["memory_usage"] = _get_memory_usage()
        return default_scalars

    def _get_duration(self) -> float:
        current_duration = timer()
        duration = current_duration - self._previous_duration
        self._previous_duration = current_duration
        return duration
