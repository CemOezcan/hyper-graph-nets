import json
import os
import pickle
import random
import time
from queue import Queue, Empty

import numpy as np
import threading as thread

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb

from src.data.data_loader import OUT_DIR, IN_DIR
from src.algorithms.AbstractIterativeAlgorithm import \
    AbstractIterativeAlgorithm
from src.model.flag import FlagModel
from src.util import NodeType, device, detach
from torch.utils.data import DataLoader
from util.Types import ConfigDict, ScalarDict, Union


class MeshSimulator(AbstractIterativeAlgorithm):
    def __init__(self, config: ConfigDict) -> None:
        super().__init__(config=config)
        self._network_config = config.get("model")
        self._dataset_dir = IN_DIR
        self._trajectories = config.get('task').get('trajectories')
        self._dataset_name = config.get('task').get('dataset')

        self._network = None
        self._optimizer = None
        self._scheduler = None
        self._initialized = False

        self.loss_function = F.mse_loss
        self._learning_rate = self._network_config.get("learning_rate")
        self._scheduler_learning_rate = self._network_config.get("scheduler_learning_rate")

        wandb.init(project='rmp')
        wandb.config = {'learning_rate': self._learning_rate, 'epochs': self._trajectories}

    def initialize(self, task_information: ConfigDict) -> None:  # TODO check usability
        if not self._initialized:
            self._network = FlagModel(self._network_config)
            self._optimizer = optim.Adam(self._network.parameters(), lr=self._learning_rate)
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, self._scheduler_learning_rate, last_epoch=-1)
            self._initialized = True

    def set_network(self, network):
        self._network = network

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> ScalarDict:  # TODO check usability
        with torch.no_grad():
            inputs = torch.Tensor(inputs)
            labels = torch.Tensor(labels)
            self._network.evaluate()
            predictions = self._network(inputs)
            predictions = predictions.squeeze()
            loss = self.loss_function(predictions, labels)
            loss = loss.item()

        return {"loss": loss}

    # TODO check usability
    def predict(self, samples: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(samples, np.ndarray):
            samples = torch.Tensor(samples.astype(np.float32))
        evaluations = self._network(samples)
        return detach(evaluations)

    def fit_iteration(self, train_dataloader: DataLoader) -> None:
        self._network.train()
        i = 0
        queue = Queue()
        self.fetch_data(train_dataloader, queue)
        while i < self._trajectories:
            i += 1
            print('Batch: {}'.format(i))
            thread_1 = thread.Thread(target=self.fetch_data, args=(train_dataloader, queue))
            if i < self._trajectories:
                thread_1.start()
            try:
                graphs, trajectory = queue.get()
                start_trajectory = time.time()
                shuffled_graphs = list(zip(graphs, trajectory))
                random.shuffle(shuffled_graphs)
                for graph, data_frame in shuffled_graphs:
                    start_instance = time.time()
                    loss = self._network.training_step(graph, data_frame)
                    loss.backward()

                    self._optimizer.step()
                    self._optimizer.zero_grad()

                    end_instance = time.time()
                    wandb.log({'loss': loss, 'training time per instance': end_instance - start_instance})
                    # self._run.watch(self._network)

                end_trajectory = time.time()
                wandb.log({'training time per trajectory': end_trajectory - start_trajectory}, commit=False)
            except Empty:
                break
            finally:
                self.save()
                if thread_1.is_alive():
                    thread_1.join()

    def fetch_data(self, loader, queue):
        try:
            graphs = list()
            trajectory = next(loader)
            self._network.reset_remote_graph()
            for data_frame in trajectory:
                graphs.append(self._network.build_graph(data_frame, True))
            queue.put((graphs, trajectory))
        except StopIteration:
            return

    @torch.no_grad()
    def one_step_evaluator(self, ds_loader, instances):
        trajectory_loss = list()
        for i, trajectory in enumerate(ds_loader):
            instance_loss = list()
            self._network.reset_remote_graph()

            if i >= instances:
                break

            for data_frame in trajectory:
                graph = self._network.build_graph(data_frame, False)
                loss, pos_error = self._network.validation_step(graph, data_frame)
                instance_loss.append([loss.to('cpu'), pos_error.to('cpu')])

            trajectory_loss.append(instance_loss)

        mean = np.mean(trajectory_loss, axis=0)
        std = np.std(trajectory_loss, axis=0)
        data_frame = pd.DataFrame.from_dict(
            {'mean_loss': [x[0] for x in mean], 'std_loss': [x[0] for x in std],
             'mean_pos_error': [x[1] for x in mean], 'std_pos_error': [x[1] for x in std]
             }
        )

        data_frame.to_csv(os.path.join(OUT_DIR, 'one_step.csv'))

    def evaluator(self, ds_loader, rollouts):
        """Run a model rollout trajectory."""
        trajectories = []
        mse_losses = []
        l1_losses = []
        num_steps = 100
        # TODO: Compute one step loss
        for trajectory in ds_loader:
            self._network.reset_remote_graph()
            _, prediction_trajectory = self.evaluate(trajectory, num_steps=num_steps)
            trajectories.append(prediction_trajectory)
            mse_loss_fn = torch.nn.MSELoss(reduction='none')
            l1_loss_fn = torch.nn.L1Loss(reduction='none')

            mse_loss = mse_loss_fn(trajectory['world_pos'][:num_steps], prediction_trajectory['pred_pos'])
            l1_loss = l1_loss_fn(trajectory['world_pos'][:num_steps], prediction_trajectory['pred_pos'])
            mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1)
            l1_loss = torch.mean(torch.mean(l1_loss, dim=-1), dim=-1)
            mse_losses.append(mse_loss.cpu())
            l1_losses.append(l1_loss.cpu())

        mse_means = torch.mean(torch.stack(mse_losses), dim=0)
        mse_stds = torch.std(torch.stack(mse_losses), dim=0)
        l1_means = torch.mean(torch.stack(l1_losses), dim=0)
        l1_stds = torch.std(torch.stack(l1_losses), dim=0)

        rollout_losses = {'mse_loss': [mse.item() for mse in mse_means],
                          'mse_std': [mse.item() for mse in mse_stds],
                          'l1_loss': [l1.item() for l1 in l1_means],
                          'l1_std': [l1.item() for l1 in l1_stds]}
        data_frame = pd.DataFrame.from_dict(rollout_losses)
        # TODO: Save losses
        # self.save_losses(wandb.run, mse_losses, l1_losses)
        data_frame.to_csv(os.path.join(OUT_DIR, 'rollout_losses.csv'))
        self.save_rollouts(trajectories)
        return rollout_losses

    def evaluate(self, trajectory, num_steps=20):
        """Performs model rollouts and create stats."""
        initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
        if num_steps is None:
            num_steps = trajectory['cells'].shape[0]

        prediction = self._rollout(initial_state, num_steps)

        scalars = None
        traj_ops = {
            'faces': trajectory['cells'],
            'mesh_pos': trajectory['mesh_pos'],
            'gt_pos': trajectory['world_pos'],
            'pred_pos': prediction
        }
        return scalars, traj_ops

    def _rollout(self, initial_state, num_steps):
        """Rolls out a model trajectory."""
        node_type = initial_state['node_type']
        self.mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device))
        self.mask = torch.stack((self.mask, self.mask, self.mask), dim=1)

        prev_pos = torch.squeeze(initial_state['prev|world_pos'], 0)
        cur_pos = torch.squeeze(initial_state['world_pos'], 0)
        trajectory = list()
        for _ in range(num_steps):
            prev_pos, cur_pos, trajectory = self._step_fn(initial_state, prev_pos, cur_pos, trajectory)
        return torch.stack(trajectory)

    def _step_fn(self, initial_state, prev_pos, cur_pos, trajectory):
        with torch.no_grad():
            input = {**initial_state, 'prev|world_pos': prev_pos, 'world_pos': cur_pos}
            graph = self._network.build_graph(input, is_training=False)
            prediction = self._network.update(input, self._network(graph))
        next_pos = torch.where(self.mask, torch.squeeze(prediction), torch.squeeze(cur_pos))
        trajectory.append(cur_pos)
        return cur_pos, next_pos, trajectory

    def n_step_evaluator(self, ds_loader, n_step_list=[3], n_traj=1):
        n_step_mse_losses = {}
        n_step_l1_losses = {}

        # Take n_traj trajectories from valid set for n_step loss calculation
        for i in range(n_traj):
            for trajectory in ds_loader:
                self._network.reset_remote_graph()
                for n_step in n_step_list:
                    self.n_step_computation(
                        n_step_mse_losses, n_step_l1_losses, trajectory, n_step)
        for (kmse, vmse), (kl1, vl1) in zip(n_step_mse_losses.items(), n_step_l1_losses.items()):
            n_step_mse_losses[kmse] = torch.div(vmse, i + 1)
            n_step_l1_losses[kl1] = torch.div(vl1, i + 1)

        return {'n_step_mse_loss': n_step_mse_losses, 'n_step_l1_loss': n_step_l1_losses}

    # TODO remove hardcoded values
    def n_step_computation(self, n_step_mse_losses, n_step_l1_losses, trajectory, n_step):
        mse_losses = []
        l1_losses = []
        for step in range(len(trajectory['world_pos']) - n_step):
            eval_traj = {}
            for k, v in trajectory.items():
                eval_traj[k] = v[step:step + n_step + 1]
            _, prediction_trajectory = self.evaluate(
                eval_traj, n_step + 1)
            mse_loss_fn = torch.nn.MSELoss()
            l1_loss_fn = torch.nn.L1Loss()
            mse_loss = mse_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0),
                                   prediction_trajectory['pred_pos'])
            l1_loss = l1_loss_fn(torch.squeeze(eval_traj['world_pos'], dim=0),
                                 prediction_trajectory['pred_pos'])

            mse_losses.append(mse_loss.cpu())
            l1_losses.append(l1_loss.cpu())
        self._compute_n_step_losses(
            n_step_mse_losses, n_step_l1_losses, n_step, mse_losses, l1_losses)

    @staticmethod
    def _compute_n_step_losses(n_step_mse_losses, n_step_l1_losses, n_step, mse_losses, l1_losses):
        if n_step not in n_step_mse_losses and n_step not in n_step_l1_losses:
            n_step_mse_losses[n_step] = torch.stack(mse_losses)
            n_step_l1_losses[n_step] = torch.stack(l1_losses)
        elif n_step in n_step_mse_losses and n_step in n_step_l1_losses:
            n_step_mse_losses[n_step] = n_step_mse_losses[n_step] + \
                torch.stack(mse_losses)
            n_step_l1_losses[n_step] = n_step_l1_losses[n_step] + \
                torch.stack(l1_losses)
        else:
            raise Exception('Error when computing n step losses!')

    @property
    def network(self):
        return self._network

    def save(self):
        with open(os.path.join(OUT_DIR, 'model.pkl'), 'wb') as file:
            pickle.dump(self, file)

    def save_rollouts(self, rollouts):
        rollouts = [{key: value.to('cpu') for key, value in x.items()} for x in rollouts]
        with open(os.path.join(OUT_DIR, 'rollouts.pkl'), 'wb') as file:
            pickle.dump(rollouts, file)

    @staticmethod
    def save_losses(run, mse_losses, l1_losses):
        run.summary['mean_1_step_mse_loss'] = torch.mean(torch.stack(mse_losses)).item()
        run.summary['mean_1_step_l1_loss'] = torch.mean(torch.stack(l1_losses)).item()
        run.summary['max_1_step_mse_loss'] = torch.max(torch.stack(mse_losses)).item()
        run.summary['max_1_step_l1_loss'] = torch.max(torch.stack(l1_losses)).item()
        run.summary['min_1_step_mse_loss'] = torch.min(torch.stack(mse_losses)).item()
        run.summary['min_1_step_l1_loss'] = torch.min(torch.stack(l1_losses)).item()
        run.summary.update()

    @staticmethod
    def _squeeze_data_frame(data_frame):
        for k, v in data_frame.items():
            data_frame[k] = torch.squeeze(v, 0)
        return data_frame
