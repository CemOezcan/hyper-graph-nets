
import math
import os
import pickle
from typing import Tuple

from matplotlib.animation import PillowWriter, FuncAnimation
import matplotlib.pyplot as plt
import torch
import wandb

from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.algorithms.MeshSimulator import MeshSimulator
from src.data.data_loader import OUT_DIR, get_data
from src.tasks.AbstractTask import AbstractTask
from tqdm import trange
from util.Types import ConfigDict, ScalarDict
from util.Functions import get_from_nested_dict


class MeshTask(AbstractTask):
    # TODO comments and discussion about nested functions
    def __init__(self, algorithm: AbstractIterativeAlgorithm, config: ConfigDict):
        """
        Initializes all necessary data for a mesh simulation task.

        Parameters
        ----------
            algorithm : AbstractIterativeAlgorithm
                # TODO: Remove and initialize here

            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.
        """

        super().__init__(algorithm=algorithm, config=config)
        self._config = config
        self._epochs = config.get('task').get('epochs')
        self._trajectories = config.get('task').get('trajectories')

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
        wandb.init(reinit=False)

    def run_iterations(self, current_epoch: int) -> None:
        """
        Run all training epochs of the mesh simulator.
        Continues the training after the given epoch, if necessary.

        Parameters
        ----------
            current_epoch : int
                Continues training at this epoch

        Returns
        -------

        """
        assert isinstance(self._algorithm, MeshSimulator), 'Need a classifier to train on a classification task'

        for e in trange(current_epoch, self._epochs, desc='Epochs'):
            task_name = f'{self._task_name}_mp:{self._mp}_epoch:{e + 1}'

            self._algorithm.fit_iteration(train_dataloader=self.train_loader)
            one_step = self._algorithm.one_step_evaluator(self._valid_loader, self._num_val_trajectories, task_name)
            rollout = self._algorithm.rollout_evaluator(self._test_loader, self._num_val_rollouts, task_name)

            a, w = self.plot(task_name)
            dir = self.save_plot(a, w, task_name)

            animation = {"video": wandb.Video(dir, fps=5, format="gif")}
            data = {k: v for dictionary in [one_step, rollout, animation] for k, v in dictionary.items()}
            data['epoch'] = e + 1
            self._algorithm.save(task_name)
            self._algorithm.log_epoch(data)

            if e >= self._config.get('model').get('scheduler_epoch'):
                self._algorithm.lr_scheduler_step()

    def get_scalars(self) -> None:
        """
        Estimate and document the one-step, rollout and n-step losses of the mesh simulator.

        Returns
        -------

        """
        assert isinstance(self._algorithm, MeshSimulator)
        task_name = f'{self._task_name}_mp:{self._mp}_epoch:final'

        self._algorithm.one_step_evaluator(self._valid_loader, self._num_test_trajectories, task_name, logging=False)
        self._algorithm.rollout_evaluator(self._test_loader, self._num_test_rollouts, task_name, logging=False)
        self._algorithm.n_step_evaluator(self._test_loader, task_name, n_step_list=[self._n_steps], n_traj=self._num_n_step_rollouts)

    def plot(self, task_name: str) -> Tuple[FuncAnimation, PillowWriter]:
        """
        Simulates and visualizes predicted trajectories as well as their respective ground truth trajectories.
        The predicted trajectories are produced by the current state of the mesh simulator.

        Parameters
        ----------
            task_name : str
                The name of the task

        Returns
        -------
            Tuple[FuncAnimation, PillowWriter]
                The simulations

        """
        rollouts = os.path.join(OUT_DIR, f'{task_name}_rollouts.pkl')

        with open(rollouts, 'rb') as fp:
            rollout_data = pickle.load(fp)

        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot(111, projection='3d')
        skip = 4
        num_steps = rollout_data[0]['pred_pos'].shape[0]
        num_frames = math.floor(num_steps / skip)

        # compute bounds
        bounds = []
        for trajectory in rollout_data:
            bb_min = torch.squeeze(trajectory['gt_pos'], dim=0).cpu().numpy().min(axis=(0, 1))
            bb_max = torch.squeeze(trajectory['gt_pos'], dim=0).cpu().numpy().max(axis=(0, 1))
            bounds.append((bb_min, bb_max))

        def animate(frame):
            step = (frame * skip) % num_steps
            traj = (frame * skip) // num_steps

            ax.cla()
            bound = bounds[traj]

            ax.set_xlim([bound[0][0], bound[1][0]])
            ax.set_ylim([bound[0][1], bound[1][1]])
            ax.set_zlim([bound[0][2], bound[1][2]])

            pos = torch.squeeze(rollout_data[traj]['pred_pos'], dim=0)[step].to('cpu')
            original_pos = torch.squeeze(rollout_data[traj]['gt_pos'], dim=0)[step].to('cpu')
            faces = torch.squeeze(rollout_data[traj]['faces'], dim=0)[step].to('cpu')
            ax.plot_trisurf(pos[:, 0], pos[:, 1], faces, pos[:, 2], shade=True)
            ax.plot_trisurf(original_pos[:, 0], original_pos[:, 1], faces, original_pos[:, 2], shade=True, alpha=0.3)
            ax.set_title('Trajectory %d Step %d' % (traj, step))
            return fig,

        animation = FuncAnimation(fig, animate, frames=num_frames * len(rollout_data))
        writergif = PillowWriter(fps=10)

        return animation, writergif

    @staticmethod
    def save_plot(animation, writervideo, task_name):
        dir = os.path.join(OUT_DIR, f'{task_name}_animation.gif')
        animation.save(dir, writer=writervideo)
        plt.show(block=True)
        return dir

