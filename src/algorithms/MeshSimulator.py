import functools
import math
import multiprocessing
import os
import pickle
import random
import time

import numpy as np

import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
from matplotlib import pyplot as plt
from tqdm import tqdm, trange

from src.data.data_loader import OUT_DIR, IN_DIR
from src.algorithms.AbstractIterativeAlgorithm import \
    AbstractIterativeAlgorithm
from src.model.flag import FlagModel
from src.util import detach, EdgeSet, MultiGraph
from torch.utils.data import DataLoader
from util.Types import ConfigDict, ScalarDict, Union


class MeshSimulator(AbstractIterativeAlgorithm):
    def __init__(self, config: ConfigDict) -> None:
        super().__init__(config=config)
        self._network_config = config.get('model')
        self._dataset_dir = IN_DIR
        self._validation = config.get('task').get(
            'validation').get('trajectories')
        self._trajectories = config.get('task').get('trajectories')
        self._dataset_name = config.get('task').get('dataset')
        self._prefetch_factor = config.get('task').get('prefetch_factor')
        assert self._validation <= self._trajectories/self._prefetch_factor
        self._wandb_mode = config.get('logging').get('wandb_mode')
        self._balance_frequency = self._network_config.get(
            'graph_balancer').get('frequency')
        self._rmp_frequency = self._network_config.get('rmp').get('frequency')

        self._batch_size = 1
        self._network = None
        self._optimizer = None
        self._scheduler = None
        self._wandb_run = None
        self._initialized = False

        self.loss_function = F.mse_loss
        self._learning_rate = self._network_config.get('learning_rate')
        self._gamma = self._network_config.get('gamma')

    def initialize(self, task_information: ConfigDict) -> None:  # TODO check usability
        self._wandb_run = wandb.init(project='rmp', config=task_information,
                                     mode=self._wandb_mode)
        wandb.define_metric('epoch')
        wandb.define_metric('validation_loss', step_metric='epoch')
        wandb.define_metric('position_loss', step_metric='epoch')
        wandb.define_metric('validation_mean', step_metric='epoch')
        wandb.define_metric('position_mean', step_metric='epoch')
        if not self._initialized:
            self._batch_size = task_information.get('task').get('batch_size')
            self._network = FlagModel(self._network_config)
            self._optimizer = optim.Adam(
                self._network.parameters(), lr=self._learning_rate)
            self._scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self._optimizer, self._gamma, last_epoch=-1)
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

    def preprocess(self, train_dataloader: DataLoader, split):
        assert self._trajectories % self._prefetch_factor == 0, f'{self._trajectories} must be divisible by prefetch factor {self._prefetch_factor}.'
        is_training = split == 'train'
        print(f'Start preprocessing {split} graphs...')
        data = []
        start_preprocessing = time.time()
        for r in trange(0, self._trajectories, self._prefetch_factor, desc='Preprocessing progress'):
            start_preprocessing_batch = time.time()
            try:
                train = [next(train_dataloader)
                         for _ in range(self._prefetch_factor)]
            except StopIteration:
                break
            with multiprocessing.Pool() as pool:
                for i, result in enumerate(
                        pool.imap(functools.partial(self.fetch_data, is_training=is_training), train)):
                    data.append(result)
                    if (i + 1) % self._prefetch_factor == 0 and i != 0:
                        # TODO: last data storage might not be saved
                        with open(os.path.join(IN_DIR, split + '_ricci_hdbscan' + f'_{int(r / self._prefetch_factor)}.pth'), 'wb') as f:
                            torch.save(data, f)
                        del data
                        data = []
            end_preprocessing_batch = time.time()
            wandb.log(
                {'preprocess time per batch': end_preprocessing_batch - start_preprocessing_batch, 'preprocess completed percentage': int((r / self._trajectories) * 100)})
        end_preprocessing = time.time()
        wandb.log(
            {'preprocess time per batch': end_preprocessing - start_preprocessing})
        print(f'Preprocessing {split} graphs done.')

    def fit_iteration(self, train_dataloader: DataLoader) -> None:
        self._network.train()
        random.shuffle(train_dataloader)
        for trajectory in tqdm(train_dataloader, desc='Trajectories in train file', leave=False):
            random.shuffle(trajectory)
            batches = self.get_batched(trajectory, self._batch_size)
            start_trajectory = time.time()
            for graph, data_frame in tqdm(batches, desc='Batches in trajectory', leave=False):
                start_instance = time.time()

                loss = self._network.training_step(graph, data_frame)
                loss.backward()

                self._optimizer.step()
                self._optimizer.zero_grad()

                end_instance = time.time()
                wandb.log(
                    {'loss': loss, 'training time per instance': end_instance - start_instance})

            end_trajectory = time.time()
            wandb.log({'training time per trajectory': end_trajectory -
                       start_trajectory}, commit=False)

    def get_batched(self, data, batch_size):
        graph_amt = len(data)
        assert graph_amt % batch_size == 0, f'Graph amount {graph_amt} must be divisible by batch size {batch_size}.'
        batches = [data[i: i + batch_size]
                   for i in range(0, len(data), batch_size)]
        graph = batches[0][0][0]
        trajectory_attributes = batches[0][0][1].keys()

        edge_names = [e.name for e in graph.edge_sets]

        batched_data = list()
        for batch in batches:
            edge_dict = {name: {'snd': list(), 'rcv': list(), 'features': list()}
                         for name in edge_names}
            trajectory_dict = {key: list() for key in trajectory_attributes}

            node_features = list()
            for i, (graph, traj) in enumerate(batch):
                # This fixes instance wise clustering
                num_nodes = tuple(
                    map(lambda x: x.shape[0], graph.node_features))
                num_nodes, num_hyper_nodes = num_nodes if len(
                    num_nodes) > 1 else (num_nodes[0], 0)
                hyper_node_offset = batch_size * num_nodes

                node_features.append(graph.node_features)
                for key, value in traj.items():
                    trajectory_dict[key].append(value)

                for e in graph.edge_sets:
                    edge_dict[e.name]['features'].append(e.features)

                    senders = torch.tensor(
                        [x + i * num_nodes if x < hyper_node_offset else x + (batch_size - 1) * num_nodes + i * num_hyper_nodes
                         for x in e.senders.tolist()]
                    )
                    edge_dict[e.name]['snd'].append(senders)

                    receivers = torch.tensor(
                        [x + i * num_nodes if x < hyper_node_offset else x + (batch_size - 1) * num_nodes + i * num_hyper_nodes
                         for x in e.receivers.tolist()]
                    )
                    edge_dict[e.name]['rcv'].append(receivers)

            new_traj = {key: torch.cat(value, dim=0)
                        for key, value in trajectory_dict.items()}

            all_nodes = list(
                map(lambda x: torch.cat(x, dim=0), zip(*node_features)))
            new_graph = MultiGraph(
                node_features=all_nodes,
                edge_sets=[
                    EdgeSet(name=n,
                            features=torch.cat(
                                edge_dict[n]['features'], dim=0),
                            senders=torch.cat(edge_dict[n]['snd'], dim=0),
                            receivers=torch.cat(edge_dict[n]['rcv'], dim=0))
                    for n in edge_dict.keys()
                ]
            )

            batched_data.append((new_graph, new_traj))

        return batched_data

    def fetch_data(self, trajectory, is_training):
        graphs = []
        graph_amt = len(trajectory)
        balanced_edge_set = None
        rmp_clusters = None
        for i, data_frame in enumerate(trajectory):
            graph = self._network.build_graph(data_frame, is_training)

            if i % math.ceil(graph_amt / self._balance_frequency) == 0:
                graph = self._network.run_balancer(
                    graph, data_frame, is_training)
                balanced_edge_set = self._network.get_balanced_edges(graph)
            elif balanced_edge_set:
                [graph.edge_sets.append(e) for e in balanced_edge_set]

            if i % math.ceil(graph_amt / self._rmp_frequency) == 0:
                rmp_clusters = self._network.get_rmp_clusters(graph)
            graph = self._network.connect_rmp_cluster(
                graph, rmp_clusters, is_training)
            graphs.append(graph)

        if is_training:
            targets = [self._network.get_target_unnormalized(
                x) for x in trajectory]
            keys = [key for key in trajectory[0].keys() if key != 'node_type']

            for i in range(len(trajectory)):
                trajectory[i]['target'] = targets[i]
                for key in keys:
                    trajectory[i].pop(key, None)

        data = list(zip(graphs, trajectory))
        batches = self.get_batched(data, 1)
        return batches

    @torch.no_grad()
    def one_step_evaluator(self, ds_loader, instances, epoch):
        trajectory_loss = list()
        for valid_file in ds_loader:
            with open(os.path.join(IN_DIR, valid_file), 'rb') as f:
                valid_data = torch.load(f)
            random.shuffle(valid_data)
            valid_data = [valid_data[:self._validation]]
            for i, trajectory in enumerate(valid_data):
                random.shuffle(trajectory)
                instance_loss = list()
                if i >= instances:
                    break

                for graph, data_frame in trajectory:
                    loss, pos_error = self._network.validation_step(
                        graph, data_frame)
                    instance_loss.append([loss, pos_error])

                trajectory_loss.append(instance_loss)

            del valid_data

        mean = np.mean(trajectory_loss, axis=0)
        std = np.std(trajectory_loss, axis=0)

        data_frame = pd.DataFrame.from_dict(
            {'mean_loss': [x[0] for x in mean], 'std_loss': [x[0] for x in std],
             'mean_pos_error': [x[1] for x in mean], 'std_pos_error': [x[1] for x in std]
             }
        )

        val_loss, pos_loss = zip(*mean)

        wandb.log({
            'validation_loss': wandb.Histogram(
                [x for x in val_loss if np.quantile(val_loss, 0.95) > x > np.quantile(val_loss, 0.01)], num_bins=256),
            'position_loss': wandb.Histogram(
                [x for x in pos_loss if np.quantile(pos_loss, 0.95) > x > np.quantile(pos_loss, 0.01)], num_bins=256),
            'validation_mean': np.mean(val_loss), 'position_mean': np.mean(pos_loss),
            'epoch': epoch}
        )
        data_frame.to_csv(os.path.join(OUT_DIR, 'one_step.csv'))

    def evaluator(self, ds_loader, rollouts):
        """Run a model rollout trajectory."""
        trajectories = []
        mse_losses = []
        num_steps = 100

        for i, trajectory in enumerate(ds_loader):
            if i >= rollouts:
                break
            self._network.reset_remote_graph()
            prediction_trajectory, mse_loss = self._network.rollout(
                trajectory, num_steps=num_steps)
            trajectories.append(prediction_trajectory)
            mse_losses.append(mse_loss.cpu())

        mse_means = torch.mean(torch.stack(mse_losses), dim=0)
        mse_stds = torch.std(torch.stack(mse_losses), dim=0)

        rollout_losses = {'mse_loss': [mse.item() for mse in mse_means], 'mse_std': [
            mse.item() for mse in mse_stds]}
        data_frame = pd.DataFrame.from_dict(rollout_losses)

        # TODO: How are rollouts saved?
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

        n_step_stats = {'n_step': n_step_list,
                        'mean': losses[0], 'std': losses[1]}
        data_frame = pd.DataFrame.from_dict(n_step_stats)
        data_frame.to_csv(os.path.join(OUT_DIR, 'n_step_losses.csv'))

    @property
    def network(self):
        return self._network

    def save(self, epoch):
        with open(os.path.join(OUT_DIR, f'model_{epoch}.pkl'), 'wb') as file:
            pickle.dump(self, file)

    def save_rollouts(self, rollouts):
        rollouts = [{key: value.to('cpu')
                     for key, value in x.items()} for x in rollouts]
        with open(os.path.join(OUT_DIR, 'rollouts.pkl'), 'wb') as file:
            pickle.dump(rollouts, file)

    @staticmethod
    def save_losses(run, mse_losses, l1_losses):
        run.summary['mean_1_step_mse_loss'] = torch.mean(
            torch.stack(mse_losses)).item()
        run.summary['mean_1_step_l1_loss'] = torch.mean(
            torch.stack(l1_losses)).item()
        run.summary['max_1_step_mse_loss'] = torch.max(
            torch.stack(mse_losses)).item()
        run.summary['max_1_step_l1_loss'] = torch.max(
            torch.stack(l1_losses)).item()
        run.summary['min_1_step_mse_loss'] = torch.min(
            torch.stack(mse_losses)).item()
        run.summary['min_1_step_l1_loss'] = torch.min(
            torch.stack(l1_losses)).item()
        run.summary.update()

    @staticmethod
    def _squeeze_data_frame(data_frame):
        for k, v in data_frame.items():
            data_frame[k] = torch.squeeze(v, 0)
        return data_frame

    def lr_scheduler_step(self):
        self._scheduler.step()
