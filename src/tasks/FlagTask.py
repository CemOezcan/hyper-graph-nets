from src.tasks.AbstractTask import AbstractTask
from data.data_loader import get_data
from util.Types import *
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import torch
import plotly.graph_objects as go
from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.algorithms.FlagSimulator import FlagModel
from src.data.dataset import load_dataset
import common

device = torch.device('cuda')


class FlagTask(AbstractTask):
    #TODO comments and discussion about nested functions
    def __init__(self, algorithm: AbstractIterativeAlgorithm, config: ConfigDict):
        """
        Initializes all necessary data for a classification task.

        Args:
            config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.
        """
        super().__init__(algorithm=algorithm, config=config)
        self._raw_data = get_data(config=config)

        self.train_loader = load_dataset(self._raw_data, 'train', batch_size=config.get("task").get("batch_size"),
                                                 prefetch_factor=config.get("task").get("prefetch_factor"),
                                                 add_targets=True, split_and_preprocess=True)

        self._test_loader = None
        self._input_dimension = self._raw_data.get("dimension")

        self.classes = self._raw_data.get("classes", None)
        self.out_features = len(self.classes) if self.classes is not None else 1
        self.mask = None

        self._algorithm.initialize(task_information={"input_dimension": self._input_dimension})

    def run_iteration(self):
        assert isinstance(self._algorithm, FlagModel), "Need a classifier to train on a classification task"
        self._algorithm.fit_iteration(train_dataloader=self.train_loader)

    def get_scalars(self) -> ScalarDict:
        assert isinstance(self._algorithm, FlagModel)
        train_scalars = self._algorithm.score(inputs=self._train_X, labels=self._train_y)
        train_scalars = {"train_" + k: v for k, v in train_scalars.items()}

        test_scalars = self._algorithm.score(inputs=self._test_X, labels=self._test_y)
        test_scalars = {"test_" + k: v for k, v in test_scalars.items()}

        return train_scalars | test_scalars

    def plot(self) -> go.Figure:
        if self._input_dimension == 2:  # 2d classification, allowing for a contour plot
            assert isinstance(self._algorithm, FlagModel)
            points_per_axis = 5
            X = self.raw_data.get("X")
            y = self.raw_data.get("y")
            bottom_left = np.min(X, axis=0)
            top_right = np.max(X, axis=0)
            x_margin = (top_right[0] - bottom_left[0]) / 2
            y_margin = (top_right[1] - bottom_left[1]) / 2
            x_positions = np.linspace(bottom_left[0] - x_margin, top_right[0] + x_margin, num=points_per_axis)
            y_positions = np.linspace(bottom_left[1] - y_margin, top_right[1] + y_margin, num=points_per_axis)
            evaluation_grid = np.transpose([np.tile(x_positions, len(y_positions)),
                                            np.repeat(y_positions, len(x_positions))])

            good_samples = X[y == 1]
            bad_samples = X[y == 0]

            reward_evaluation_grid = self._algorithm.predict(evaluation_grid)
            reward_evaluation_grid = reward_evaluation_grid.reshape((points_per_axis, points_per_axis))
            reward_evaluation_grid = np.clip(a=reward_evaluation_grid, a_min=-3, a_max=3)
            fig = go.Figure(data=[go.Contour(x=x_positions, y=y_positions, z=reward_evaluation_grid,
                                             colorscale="Portland", ),
                                  go.Scatter(x=good_samples[::10, 0], y=good_samples[::10, 1],
                                             mode="markers", fillcolor="green", showlegend=False),
                                  go.Scatter(x=bad_samples[::10, 0], y=bad_samples[::10, 1],
                                             mode="markers", fillcolor="red", showlegend=False)
                                  ])
            return fig
        else:
            raise NotImplementedError("plotting not supported for {}-dimensional features", self._input_dimension)


def _rollout(self, model, initial_state, num_steps):
    """Rolls out a model trajectory."""
    node_type = initial_state['node_type']
    self.mask = torch.eq(node_type[:, 0], torch.tensor(
        [common.NodeType.NORMAL.value], device=device))
    self.mask = torch.stack((self.mask, self.mask, self.mask), dim=1)

    def step_fn(prev_pos, cur_pos, trajectory):
        # memory_prev = torch.cuda.memory_allocated(device) / (1024 * 1024)
        with torch.no_grad():
            prediction = model({**initial_state,
                                'prev|world_pos': prev_pos,
                                'world_pos': cur_pos}, is_training=False)

        next_pos = torch.where(self.mask, torch.squeeze(
            prediction), torch.squeeze(cur_pos))

        trajectory.append(cur_pos)
        return cur_pos, next_pos, trajectory

    prev_pos = torch.squeeze(initial_state['prev|world_pos'], 0)
    cur_pos = torch.squeeze(initial_state['world_pos'], 0)
    trajectory = []
    for _ in range(num_steps):
        prev_pos, cur_pos, trajectory = step_fn(prev_pos, cur_pos, trajectory)
    return torch.stack(trajectory)


def evaluate(self, model, trajectory, num_steps=None):
    """Performs model rollouts and create stats."""
    initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
    if num_steps is None:
        num_steps = trajectory['cells'].shape[0]
    prediction = self._rollout(model, initial_state, num_steps)

    # error = tf.reduce_mean((prediction - trajectory['world_pos'])**2, axis=-1)
    # scalars = {'mse_%d_steps' % horizon: tf.reduce_mean(error[1:horizon+1])
    #            for horizon in [1, 10, 20, 50, 100, 200]}

    scalars = None
    traj_ops = {
        'faces': trajectory['cells'],
        'mesh_pos': trajectory['mesh_pos'],
        'gt_pos': trajectory['world_pos'],
        'pred_pos': prediction
    }
    return scalars, traj_ops
