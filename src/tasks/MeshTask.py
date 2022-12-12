
import math
import os
import pickle
import re
from typing import Tuple

from matplotlib.animation import PillowWriter, FuncAnimation
import matplotlib.pyplot as plt
import torch
import wandb
from matplotlib import tri

from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from src.algorithms.MeshSimulator import MeshSimulator
from src.algorithms.get_algorithm import get_algorithm
from src.data.data_loader import get_directories, get_data
from src.tasks.AbstractTask import AbstractTask
from tqdm import trange

from src.util import device
from util.Types import ConfigDict, ScalarDict
from util.Functions import get_from_nested_dict


class MeshTask(AbstractTask):
    """
    Training and evaluation loops for mesh simulators.
    """

    def __init__(self, config: ConfigDict):
        """
        Initializes all necessary data for a mesh simulation task.

        Parameters
        ----------
            config : ConfigDict
                A (potentially nested) dictionary containing the "params" section of the section in the .yaml file
                used by cw2 for the current run.

        """
        super().__init__(config=config)
        self._config = config
        self._epochs = config.get('task').get('epochs')
        self._trajectories = config.get('task').get('trajectories')
        self._dataset_name = config.get('task').get('dataset')
        _, self._out_dir = get_directories(self._dataset_name)

        self._num_val_trajectories = config.get('task').get('validation').get('trajectories')
        self._num_val_rollouts = self._config.get('task').get('validation').get('rollouts')
        self._num_val_n_step_rollouts = self._config.get('task').get('validation').get('n_step_rollouts')
        self._val_n_steps = self._config.get('task').get('validation').get('n_steps')

        self._num_test_trajectories = config.get('task').get('test').get('trajectories')
        self._num_test_rollouts = config.get('task').get('test').get('rollouts')
        self._num_n_step_rollouts = config.get('task').get('test').get('n_step_rollouts')
        self._n_steps = config.get('task').get('test').get('n_steps')
        self.n_viz = self._config.get('task').get('validation').get('n_viz')

        self.train_loader = get_data(config=config)
        self._test_loader = get_data(config=config, split='valid', split_and_preprocess=False)
        self._valid_loader = get_data(config=config, split='valid')

        cluster = get_from_nested_dict(config, ['model', 'rmp', 'clustering'])
        connector = get_from_nested_dict(config, ['model', 'rmp', 'connector'])
        num_clusters = get_from_nested_dict(config, ['model', 'rmp', 'num_clusters'])
        balancer = get_from_nested_dict(config, ['model', 'graph_balancer', 'algorithm'])
        self._mp = get_from_nested_dict(config, ['model', 'message_passing_steps'])
        self._task_name = f'{num_clusters}_cluster:{cluster}_connector:{connector}_balancer:{balancer}_mp:{self._mp}_epoch:'

        retrain = config.get('retrain')
        epochs = list() if retrain else [
            int(file.split('_epoch:')[1][:-4])
            for file in os.listdir(self._out_dir)
            if re.match(rf'model_{self._task_name}[0-9]+\.pkl', file)
        ]

        if epochs:
            self._current_epoch = max(epochs)
            model_path = os.path.join(self._out_dir, f'model_{self._task_name}{self._current_epoch}.pkl')
            with open(model_path, 'rb') as file:
                self._algorithm = pickle.load(file)
        else:
            self._algorithm = get_algorithm(config)
            self._current_epoch = 0

        self._algorithm.initialize(task_information=config)
        wandb.init(reinit=False)

    def run_iterations(self) -> None:
        """
        Run all training epochs of the mesh simulator.
        Continues the training after the given epoch, if necessary.
        """
        assert isinstance(self._algorithm, MeshSimulator), 'Need a classifier to train on a classification task'
        start_epoch = self._current_epoch
        for e in trange(start_epoch, self._epochs, desc='Epochs'):
            task_name = f'{self._task_name}{e + 1}'

            self._algorithm.fit_iteration(train_dataloader=self.train_loader)
            one_step = self._algorithm.one_step_evaluator(self._valid_loader, self._num_val_trajectories, task_name)
            rollout = self._algorithm.rollout_evaluator(self._test_loader, self._num_val_rollouts, task_name)
            n_step = self._algorithm.n_step_evaluator(self._test_loader, task_name, n_steps=self._val_n_steps, n_traj=self._num_val_n_step_rollouts)

            dir_dict = self.select_plotting(task_name)

            animation = {f"video_{key}": wandb.Video(value, fps=5, format="gif") for key, value in dir_dict.items()}
            data = {k: v for dictionary in [one_step, rollout, n_step, animation] for k, v in dictionary.items()}
            data['epoch'] = e + 1
            self._algorithm.save(task_name)
            self._algorithm.log_epoch(data)
            self._current_epoch = e + 1

            #if e >= self._config.get('model').get('scheduler_epoch'):
            #    self._algorithm.lr_scheduler_step()

    def get_scalars(self) -> None:
        """
        Estimate and document the one-step, rollout and n-step losses of the mesh simulator.

        Returns
        -------

        """
        assert isinstance(self._algorithm, MeshSimulator)
        task_name = f'{self._task_name}final'

        self._algorithm.one_step_evaluator(self._valid_loader, self._num_test_trajectories, task_name, logging=False)
        self._algorithm.rollout_evaluator(self._test_loader, self._num_test_rollouts, task_name, logging=False)
        self._algorithm.n_step_evaluator(self._test_loader, task_name, n_steps=self._n_steps, n_traj=self._num_n_step_rollouts, logging=False)

        self.select_plotting(task_name)

    def select_plotting(self, task_name):
        if 'flag' in self._dataset_name:
            a, w = self.plot(task_name)
            dir = self._save_plot(a, w, task_name)
            return {'': dir}

        elif 'cylinder' in self._dataset_name:
            a, w = self.plot_3(task_name, 'velocity', dpi=100)
            dir_1 = self._save_plot(a, w, task_name)

            a, w = self.plot_3(task_name, 'pressure', dpi=100)
            dir_2 = self._save_plot(a, w, f'{task_name}_pressure')
            return {'': dir_1, 'pressure': dir_2}

        elif 'plate' in self._dataset_name:
            a, w = self.plot_2(task_name)
            dir_1 = self._save_plot(a, w, task_name)

            return {'': dir_1}

    def plot_3(self, task_name, field_name, figsize=(12 * 2, 8 * 2), dpi=None):
        rollouts = os.path.join(self._out_dir, f'{task_name}_rollouts.pkl')
        # TODO: Vizualize pressure levels similarly to cfd
        # TODO: Mask out obstacle nodes and edges

        with open(rollouts, 'rb') as fp:
            rollout_data = pickle.load(fp)

        rollout_data = rollout_data[:self.n_viz]

        fig = plt.figure(figsize=figsize, dpi=dpi)
        if 'plate' in self._dataset_name:
            ax_origin = fig.add_subplot(121)
            ax_pred = fig.add_subplot(122)
        else:
            ax_origin = fig.add_subplot(211)
            ax_pred = fig.add_subplot(212)

        skip = 10
        num_steps = rollout_data[0][f'pred_{field_name}'].shape[0]
        num_frames = math.floor(num_steps / skip)

        # compute bounds
        bounds = []
        for trajectory in rollout_data:
            bb_min = torch.squeeze(trajectory[f'gt_{field_name}'], dim=0).cpu().numpy().min(axis=(0, 1))
            bb_max = torch.squeeze(trajectory[f'gt_{field_name}'], dim=0).cpu().numpy().max(axis=(0, 1))
            bounds.append((bb_min, bb_max))

        def animate(num):
            step = (num * skip) % num_steps
            traj = (num * skip) // num_steps

            ax_origin.cla()
            ax_origin.set_aspect('equal')
            ax_origin.set_axis_off()
            ax_pred.cla()
            ax_pred.set_aspect('equal')
            ax_pred.set_axis_off()

            vmin, vmax = bounds[traj]
            pos = rollout_data[traj]['mesh_pos'][step]
            faces = rollout_data[traj]['faces'][step]
            pred_velocity = rollout_data[traj][f'pred_{field_name}'][step]
            gt_velocity = rollout_data[traj][f'gt_{field_name}'][step]

            if 'plate' in self._dataset_name:
                mask = rollout_data[traj]['mask']
                gt_velocity, _ = self.obstacle(gt_velocity, faces, mask, True)
                pred_velocity, _ = self.obstacle(pred_velocity, faces, mask, True)
                pos, faces = self.obstacle(pos, faces, mask, True)

            triang = tri.Triangulation(pos[:, 0], pos[:, 1], faces)

            ax_origin.tripcolor(triang, gt_velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
            ax_origin.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax_origin.set_title('ORIGIN trajectory %d Step %d' % (traj, step))

            ax_pred.tripcolor(triang, pred_velocity[:, 0], vmin=vmin[0], vmax=vmax[0])
            ax_pred.triplot(triang, 'ko-', ms=0.5, lw=0.3)
            ax_pred.set_title('PRED trajectory %d Step %d' % (traj, step))

            return fig,

        anima = FuncAnimation(fig, animate, frames=num_frames * len(rollout_data), interval=100)
        writergif = PillowWriter(fps=10)

        return anima, writergif

    def plot_2(self, task_name):
        rollouts = os.path.join(self._out_dir, f'{task_name}_rollouts.pkl')
        # TODO: Vizualize stress levels similarly to cfd

        with open(rollouts, 'rb') as fp:
            rollout_data = pickle.load(fp)

        rollout_data = rollout_data[:self.n_viz]

        fig = plt.figure(figsize=(19.2, 10.8))
        ax_origin = fig.add_subplot(121, projection='3d')
        ax_pred = fig.add_subplot(122, projection='3d')
        skip = 10
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

            ax_origin.cla()
            ax_pred.cla()
            bound = bounds[traj]

            ax_origin.set_xlim([bound[0][0], bound[1][0]])
            ax_origin.set_ylim([bound[0][1], bound[1][1]])
            ax_origin.set_zlim([bound[0][2], bound[1][2]])

            ax_pred.set_xlim([bound[0][0], bound[1][0]])
            ax_pred.set_ylim([bound[0][1], bound[1][1]])
            ax_pred.set_zlim([bound[0][2], bound[1][2]])

            mask = rollout_data[traj]['mask']
            pos = torch.squeeze(rollout_data[traj]['pred_pos'], dim=0)[step].to('cpu')
            original_pos = torch.squeeze(rollout_data[traj]['gt_pos'], dim=0)[step].to('cpu')
            faces = torch.squeeze(rollout_data[traj]['faces'], dim=0)[step].to('cpu')

            og, og_faces = self.obstacle(original_pos, faces, mask, True)
            pred, pred_faces = self.obstacle(pos, faces, mask, True)
            obst, obst_faces = self.obstacle(original_pos, faces, mask, False)

            ax_origin.plot_trisurf(obst[:, 0], obst[:, 1], obst_faces, obst[:, 2], shade=True)
            ax_origin.plot_trisurf(og[:, 0], og[:, 1], og_faces, og[:, 2], shade=True, alpha=0.3)

            ax_pred.plot_trisurf(obst[:, 0], obst[:, 1], obst_faces, obst[:, 2], shade=True)
            ax_pred.plot_trisurf(pred[:, 0], pred[:, 1], pred_faces, pred[:, 2], shade=True, alpha=0.3)

            ax_origin.set_title('ORIGIN Trajectory %d Step %d' % (traj, step))
            ax_pred.set_title('PRED Trajectory %d Step %d' % (traj, step))
            return fig,

        anima = FuncAnimation(fig, animate, frames=num_frames * len(rollout_data), interval=100)
        writergif = PillowWriter(fps=10)
        return anima, writergif

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
        rollouts = os.path.join(self._out_dir, f'{task_name}_rollouts.pkl')

        with open(rollouts, 'rb') as fp:
            rollout_data = pickle.load(fp)

        rollout_data = rollout_data[:self.n_viz]

        fig = plt.figure(figsize=(19.2, 10.8))
        ax = fig.add_subplot(111, projection='3d')
        skip = 10
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

        animation = FuncAnimation(fig, animate, frames=num_frames * len(rollout_data), interval=100)
        writergif = PillowWriter(fps=10)

        return animation, writergif

    def obstacle(self, positions, faces, mask, keep):
        if keep:
            mask_2 = torch.logical_not(mask)
        else:
            mask_2 = mask

        obst = positions[mask_2]

        ctr = 0
        index_map = {}
        for i in range(len(positions)):
            if mask_2[i]:
                index_map[i] = ctr
                ctr += 1

        obst_faces = torch.tensor(
            [
                [
                    index_map[int(x[0])],
                    index_map[int(x[1])],
                    index_map[int(x[2])]
                ]
                for x in faces
                if mask_2[x[0]] and mask_2[x[1]] and mask_2[x[2]]]
        )

        return obst, obst_faces

    def _save_plot(self, animation: FuncAnimation, writervideo: PillowWriter, task_name: str) -> str:
        """
        Saves a simulation as a .gif file.

        Parameters
        ----------
            animation : FuncAnimation
                The animation
            writervideo : PillowWriter
                The writer
            task_name : str
                The task name

        Returns
        -------
            str
                The path to the .gif file

        """
        dir = os.path.join(self._out_dir, f'{task_name}_animation.gif')
        animation.save(dir, writer=writervideo)
        plt.show(block=True)
        return dir

