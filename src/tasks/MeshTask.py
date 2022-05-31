import math
import os
import pickle

import matplotlib.pyplot as plt
import matplotlib.animation as ani
import plotly.graph_objects as go
import torch
from data.data_loader import OUT_DIR, get_data
from src.algorithms.AbstractIterativeAlgorithm import \
    AbstractIterativeAlgorithm
from src.algorithms.MeshSimulator import MeshSimulator
from src.data.dataset import load_dataset
from src.tasks.AbstractTask import AbstractTask
from util.Types import ConfigDict, ScalarDict


class MeshTask(AbstractTask):
    # TODO comments and discussion about nested functions
    def __init__(self, algorithm: AbstractIterativeAlgorithm, config: ConfigDict):
        """
        Initializes all necessary data for a classification task.

        Args:
            config: A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.
        """
        super().__init__(algorithm=algorithm, config=config)
        self._raw_data = get_data(config=config)
        self._rollouts = config.get('task').get('rollouts')
        self.train_loader = get_data(config=config)

        self._test_loader = get_data(
            config=config, split='test', split_and_preprocess=False)

        self.mask = None

        self._algorithm.initialize(task_information=config)
        self._dataset_name = config.get('task').get('dataset')

    def run_iteration(self):
        assert isinstance(
            self._algorithm, MeshSimulator), "Need a classifier to train on a classification task"
        self._algorithm.fit_iteration(train_dataloader=self.train_loader)

    # TODO add trajectories from evaluate method
    def get_scalars(self) -> ScalarDict:
        assert isinstance(self._algorithm, MeshSimulator)
        # TODO: Use n_step_eval
        return self._algorithm.evaluator(self._test_loader, self._rollouts)

    def plot(self) -> go.Figure:
        path = os.path.join(OUT_DIR, self._dataset_name)
        rollouts = os.path.join(path, 'rollouts.pkl')

        with open(rollouts, 'rb') as fp:
            rollout_data = pickle.load(fp)

        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot(111, projection='3d')
        skip = 10
        num_steps = rollout_data[0]['gt_pos'].shape[0]
        num_frames = num_steps

        # compute bounds
        bounds = []
        for trajectory in rollout_data:
            bb_min = torch.squeeze(
                trajectory['gt_pos'], dim=0).cpu().numpy().min(axis=(0, 1))
            bb_max = torch.squeeze(
                trajectory['gt_pos'], dim=0).cpu().numpy().max(axis=(0, 1))
            bounds.append((bb_min, bb_max))

        def animate(num):
            step = (num * skip) % num_steps
            traj = (num * skip) // num_steps

            ax.cla()
            bound = bounds[traj]

            ax.set_xlim([bound[0][0], bound[1][0]])
            ax.set_ylim([bound[0][1], bound[1][1]])
            ax.set_zlim([bound[0][2], bound[1][2]])

            pos = torch.squeeze(rollout_data[traj]['pred_pos'], dim=0)[
                step].to('cpu')
            original_pos = torch.squeeze(rollout_data[traj]['gt_pos'], dim=0)[
                step].to('cpu')
            # print(pos[10])
            faces = torch.squeeze(rollout_data[traj]['faces'], dim=0)[
                step].to('cpu')
            ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
            ax.plot_trisurf(original_pos[:, 0], original_pos[:, 1], faces, original_pos[:, 2], shade=True,
                            alpha=0.3)
            ax.set_title('Trajectory %d Step %d' % (traj, step))
            return fig,

        animation = ani.FuncAnimation(
            fig, animate, frames=math.floor(num_frames * 0.1), interval=100)
        writervideo = ani.FFMpegWriter(fps=30)
        animation.save(os.path.join(path, 'animation.mp4'), writer=writervideo)
        plt.show(block=True)
