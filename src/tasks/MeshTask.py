import math
import os
import pickle
import random
import re

import matplotlib.animation as ani
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
import wandb

from src.algorithms.AbstractIterativeAlgorithm import \
    AbstractIterativeAlgorithm
from src.algorithms.MeshSimulator import MeshSimulator
from src.data.data_loader import IN_DIR, OUT_DIR, get_data
from src.tasks.AbstractTask import AbstractTask
from tqdm import tqdm, trange
from util.Types import ConfigDict, ScalarDict
from util.Functions import get_from_nested_dict


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
        self._config = config
        self._raw_data = get_data(config=config)
        self._epochs = config.get('task').get('epochs')
        self._trajectories = config.get('task').get('trajectories')

        self._num_val_files = self._config.get('task').get('validation').get('files')
        self._num_val_trajectories = config.get('task').get('validation').get('trajectories')
        self._num_val_rollouts = self._config.get('task').get('validation').get('rollouts')

        self._num_test_trajectories = config.get('task').get('test').get('trajectories')
        self._num_test_rollouts = config.get('task').get('test').get('rollouts')
        self._num_n_step_rollouts = config.get('task').get('test').get('n_step_rollouts')
        self._n_steps = config.get('task').get('test').get('n_steps')

        self.train_loader = get_data(config=config)

        self._test_loader = get_data(config=config, split='test', split_and_preprocess=False)
        self._valid_loader = get_data(config=config, split='valid')

        self.mask = None

        cluster = get_from_nested_dict(config, ['model', 'rmp', 'clustering'])
        num_clusters = get_from_nested_dict(config, ['model', 'rmp', 'num_clusters'])
        balancer = get_from_nested_dict(config, ['model', 'graph_balancer', 'algorithm'])
        self._mp = get_from_nested_dict(config, ['model', 'message_passing_steps'])
        self._task_name = f'{num_clusters}_cluster:{cluster}_balancer:{balancer}'
        self._algorithm.initialize(task_information=config)
        self._dataset_name = config.get('task').get('dataset')
        self._wandb = wandb.init(reinit=False)

    def run_iteration(self, current_epoch):
        assert isinstance(self._algorithm, MeshSimulator), 'Need a classifier to train on a classification task'

        train_files = [file for file in os.listdir(IN_DIR) if re.match(rf'train_{self._task_name}_[0-9]+\.pth', file)]
        valid_files = [file for file in os.listdir(IN_DIR) if re.match(rf'valid_{self._task_name}_[0-9]+\.pth', file)]
        assert self._num_val_files <= len(valid_files)
        random.shuffle(valid_files)
        valid_files = valid_files[:self._num_val_files]

        for e in trange(current_epoch, self._epochs, desc='Epochs'):
            for train_file in tqdm(train_files, desc='Train files', leave=False):
                with open(os.path.join(IN_DIR, train_file), 'rb') as f:
                    train_data = torch.load(f)
                self._algorithm.fit_iteration(train_dataloader=train_data)
                del train_data

            task_name = f'{self._task_name}_mp:{self._mp}_epoch:{e + 1}'
            self._algorithm.save(task_name)
            # TODO: Always visualize the second trajectory
            del self._test_loader
            self._test_loader = get_data(config=self._config, split='test', split_and_preprocess=False)
            next(self._test_loader)

            one_step = self._algorithm.one_step_evaluator(valid_files, self._num_val_trajectories, task_name)
            rollout = self._algorithm.evaluator(self._test_loader, self._num_val_rollouts, task_name)

            a, w = self.plot()
            dir = self.save_plot(a, w, task_name)

            animation = {"video": wandb.Video(dir, fps=4, format="gif")}
            self._algorithm.log_epoch([one_step, rollout, animation], e + 1)

            if e >= self._config.get('model').get('scheduler_epoch'):
                self._algorithm.lr_scheduler_step()

    def preprocess(self):
        self._algorithm.preprocess(self.train_loader, 'train', self._task_name)
        self._algorithm.preprocess(self._valid_loader, 'valid', self._task_name)

    # TODO add trajectories from evaluate method
    def get_scalars(self) -> ScalarDict:
        assert isinstance(self._algorithm, MeshSimulator)
        task_name = f'{self._task_name}_mp:{self._mp}_epoch:final'
        valid_files = [file for file in os.listdir(IN_DIR) if re.match(rf'valid_{self._task_name}_[0-9]+\.pth', file)]
        self._algorithm.one_step_evaluator(valid_files, self._num_test_trajectories, task_name, logging=False)

        del self._test_loader
        self._test_loader = get_data(config=self._config, split='test', split_and_preprocess=False)
        self._algorithm.evaluator(self._test_loader, self._num_test_rollouts, task_name, logging=False)

        del self._test_loader
        self._test_loader = get_data(config=self._config, split='test', split_and_preprocess=False)
        # TODO: Different rollouts value for n_step_loss
        self._algorithm.n_step_evaluator(self._test_loader, task_name, n_step_list=[self._n_steps], n_traj=self._num_n_step_rollouts)

    def plot(self) -> go.Figure:
        rollouts = os.path.join(OUT_DIR, 'rollouts.pkl')

        with open(rollouts, 'rb') as fp:
            rollout_data = pickle.load(fp)

        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot(111, projection='3d')
        skip = 10
        num_steps = rollout_data[0]['pred_pos'].shape[0]
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

        return animation, writervideo

    @staticmethod
    def save_plot(animation, writervideo, task_name):
        dir = os.path.join(OUT_DIR, f'{task_name}_animation.mp4')
        animation.save(dir, writer=writervideo)
        plt.show(block=True)
        return dir
