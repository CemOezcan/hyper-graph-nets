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
                instance_loss.append([loss, pos_error])

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
        num_steps = 100

        for i, trajectory in enumerate(ds_loader):
            self._network.reset_remote_graph()
            prediction_trajectory, mse_loss = self._network.rollout(trajectory, num_steps=num_steps)
            trajectories.append(prediction_trajectory)
            mse_losses.append(mse_loss.cpu())

        mse_means = torch.mean(torch.stack(mse_losses), dim=0)
        mse_stds = torch.std(torch.stack(mse_losses), dim=0)

        rollout_losses = {'mse_loss': [mse.item() for mse in mse_means], 'mse_std': [mse.item() for mse in mse_stds]}
        data_frame = pd.DataFrame.from_dict(rollout_losses)

        data_frame.to_csv(os.path.join(OUT_DIR, 'rollout_losses.csv'))
        self.save_rollouts(trajectories)

        return rollout_losses

    def n_step_evaluator(self, ds_loader, n_step_list=[397, 396], n_traj=2):
        # Take n_traj trajectories from valid set for n_step loss calculation
        # TODO: Decide on how to summarize
        losses = list()
        for n_steps in n_step_list:
            n_step_losses = list()
            for i, trajectory in enumerate(ds_loader):
                if i >= n_traj:
                    break
                self._network.reset_remote_graph()
                loss = self._network.n_step_computation(trajectory, n_steps)
                n_step_losses.append(loss)

            means = torch.mean(torch.stack(n_step_losses)).item()
            std = torch.std(torch.stack(n_step_losses)).item()
            losses.append((means, std))

        n_step_stats = {'n_step': n_step_list, 'mean': losses[0], 'std': losses[1]}
        data_frame = pd.DataFrame.from_dict(n_step_stats)
        data_frame.to_csv(os.path.join(OUT_DIR, 'n_step_losses.csv'))

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
