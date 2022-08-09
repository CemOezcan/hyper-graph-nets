import json
import os
import pickle
from queue import Queue, Empty

import numpy as np
import threading as thread
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

        self.loss_function = F.mse_loss
        self._learning_rate = self._network_config.get("learning_rate")
        self._scheduler_learning_rate = self._network_config.get(
            "scheduler_learning_rate")
        wandb.init(project='rmp')
        wandb.config = {'learning_rate': self._learning_rate, 'epochs': self._trajectories}

    def initialize(self, task_information: ConfigDict) -> None:  # TODO check usability
        self._network = FlagModel(self._network_config)

        self._optimizer = optim.Adam(
            self._network.parameters(), lr=self._learning_rate)
        self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self._optimizer, self._scheduler_learning_rate, last_epoch=-1)

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
                for graph, data_frame in zip(graphs, trajectory):
                    network_output = self._network(graph)

                    cur_position = data_frame['world_pos']
                    prev_position = data_frame['prev|world_pos']
                    target_position = data_frame['target|world_pos']
                    # TODO check if applicable for other tasks, refactor to model itself
                    target_acceleration = target_position - 2 * cur_position + prev_position
                    target_normalized = self._network.get_output_normalizer()(
                        target_acceleration).to(device)
                    network_output = network_output  # TODO: generalize to multiple node types [:len(target_normalized)]

                    node_type = data_frame['node_type']
                    loss_mask = torch.eq(node_type[:, 0], torch.tensor(
                        [NodeType.NORMAL.value], device=device).int())
                    error = torch.sum(
                        (target_normalized - network_output) ** 2, dim=1)
                    loss = torch.mean(error[loss_mask])

                    self._optimizer.zero_grad()
                    loss.backward()
                    self._optimizer.step()
                    wandb.log({'loss': loss})
                    wandb.watch(self._network)
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

    def evaluator(self, ds_loader, rollouts):
        """Run a model rollout trajectory."""
        trajectories = []
        mse_losses = []
        l1_losses = []
        for _ in range(rollouts):
            for trajectory in ds_loader:
                self._network.reset_remote_graph()
                _, prediction_trajectory = self.evaluate(trajectory)
                mse_loss_fn = torch.nn.MSELoss()
                l1_loss_fn = torch.nn.L1Loss()

                mse_loss = mse_loss_fn(torch.squeeze(
                    trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])
                l1_loss = l1_loss_fn(torch.squeeze(
                    trajectory['world_pos'], dim=0), prediction_trajectory['pred_pos'])

                mse_losses.append(mse_loss.cpu())
                l1_losses.append(l1_loss.cpu())
                trajectories.append(prediction_trajectory)
        loss_record = self.save_losses(mse_losses, l1_losses)
        self.save_rollouts(trajectories)
        return loss_record

    def evaluate(self, trajectory, num_steps=None):
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
    def save_losses(mse_losses, l1_losses):
        loss_record = {}
        loss_record['eval_total_mse_loss'] = torch.sum(
            torch.stack(mse_losses)).item()
        loss_record['eval_total_l1_loss'] = torch.sum(
            torch.stack(l1_losses)).item()
        loss_record['eval_mean_mse_loss'] = torch.mean(
            torch.stack(mse_losses)).item()
        loss_record['eval_max_mse_loss'] = torch.max(
            torch.stack(mse_losses)).item()
        loss_record['eval_min_mse_loss'] = torch.min(
            torch.stack(mse_losses)).item()
        loss_record['eval_mean_l1_loss'] = torch.mean(
            torch.stack(l1_losses)).item()
        loss_record['eval_max_l1_loss'] = torch.max(
            torch.stack(l1_losses)).item()
        loss_record['eval_min_l1_loss'] = torch.min(
            torch.stack(l1_losses)).item()
        loss_record['eval_mse_losses'] = mse_losses
        loss_record['eval_l1_losses'] = l1_losses
        return loss_record

    @staticmethod
    def _squeeze_data_frame(data_frame):
        for k, v in data_frame.items():
            data_frame[k] = torch.squeeze(v, 0)
        return data_frame
