import json
import os
import pickle
from typing import Union
import torch
import numpy as np

from util.Types import ConfigDict, ScalarDict
import torch.nn.functional as F
import torch.optim as optim
from data.data_loader import DATA_DIR
from src.algorithms.AbstractIterativeAlgorithm import \
    AbstractIterativeAlgorithm
from src.model.flag import FlagModel
from src.util import NodeType, device
from torch.utils.data import DataLoader
from util.pytorch.TorchUtil import detach
from util.Types import ConfigDict, ScalarDict


# TODO check if only applicable for flag
class MeshSimulator(AbstractIterativeAlgorithm):
    def __init__(self, config: ConfigDict) -> None:
        super().__init__(config=config)
        # TODO: Config file for flag model
        self._network_config = config.get("model")
        self._dataset_dir = os.path.join(
            DATA_DIR, config.get('task').get('dataset'))
        self._dataset_name = config.get('task').get('dataset')

        self._network = None
        self._optimizer = None
        # TODO: Add scheduler
        # self._scheduler = None

        self.loss_function = F.mse_loss
        self._learning_rate = self._network_config.get("learning_rate")
        # self._scheduler_learning_rate = self._network_config.get("scheduler_learning_rate")

    def initialize(self, task_information: ConfigDict) -> None:
        self._network = FlagModel(self._network_config)

        self._optimizer = optim.Adam(
            self._network.parameters(), lr=self._learning_rate)
        # self._scheduler = torch.optim.lr_scheduler.ExponentialLR(self._optimizer, self._scheduler_learning_rate, last_epoch=-1)

    def score(self, inputs: np.ndarray, labels: np.ndarray) -> ScalarDict:
        with torch.no_grad():
            inputs = torch.Tensor(inputs)
            labels = torch.Tensor(labels)
            self._network.evaluate()
            predictions = self._network(inputs)
            predictions = predictions.squeeze()
            loss = self.loss_function(predictions, labels)
            loss = loss.item()

        return {"loss": loss}

    def predict(self, samples: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(samples, np.ndarray):
            samples = torch.Tensor(samples.astype(np.float32))
        evaluations = self._network(samples)
        return detach(evaluations)

    def fit_iteration(self, train_dataloader: DataLoader) -> None:
        self._network.train()

        for i, data in enumerate(train_dataloader):  # for each batch
            trajectory = self._process_trajectory(
                data, self._network_config, self._dataset_dir, True, True)
            print(i)

            for data_frame in trajectory:
                data_frame = self._squeeze_data_frame(data_frame)
                network_output = self._network(data_frame, is_training=True)

                cur_position = data_frame['world_pos']
                prev_position = data_frame['prev|world_pos']
                target_position = data_frame['target|world_pos']
                # TODO check if applicable for other tasks, refactor to model itself
                target_acceleration = target_position - 2 * cur_position + prev_position
                target_normalized = self._network.get_output_normalizer()(
                    target_acceleration).to(device)

                node_type = data_frame['node_type']
                loss_mask = torch.eq(node_type[:, 0], torch.tensor(
                    [NodeType.NORMAL.value], device=device).int())
                error = torch.sum(
                    (target_normalized - network_output) ** 2, dim=1)
                loss = torch.mean(error[loss_mask])

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            self.save()

    def evaluator(self, ds_loader, rollouts):
        """Run a model rollout trajectory."""
        trajectories = []

        mse_losses = []
        l1_losses = []

        for index in range(rollouts):
            for trajectory in ds_loader:
                trajectory = self._process_trajectory(
                    trajectory, self._network_config, self._dataset_dir, True)

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
            # scalars.append(scalar_data)
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
        self.save_rollouts(trajectories)
        return loss_record

    def evaluate(self, trajectory, num_steps=None):
        """Performs model rollouts and create stats."""
        initial_state = {k: torch.squeeze(
            v, 0)[0] for k, v in trajectory.items()}
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
        self.mask = torch.eq(node_type[:, 0], torch.tensor(
            [NodeType.NORMAL.value], device=device))
        self.mask = torch.stack((self.mask, self.mask, self.mask), dim=1)

        def step_fn(prev_pos, cur_pos, trajectory):
            # memory_prev = torch.cuda.memory_allocated(device) / (1024 * 1024)
            with torch.no_grad():
                prediction = self._network({**initial_state,
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
            prev_pos, cur_pos, trajectory = step_fn(
                prev_pos, cur_pos, trajectory)
        return torch.stack(trajectory)

    @staticmethod
    def _squeeze_data_frame(data_frame):
        for k, v in data_frame.items():
            data_frame[k] = torch.squeeze(v, 0)
        return data_frame

    def _process_trajectory(self, trajectory_data, params, dataset_dir, add_targets_bool=False,
                            split_and_preprocess_bool=False):
        loaded_meta = False
        shapes = {}
        dtypes = {}
        types = {}
        steps = None

        if not loaded_meta:
            try:
                with open(os.path.join(dataset_dir, 'meta.json'), 'r') as fp:
                    meta = json.loads(fp.read())
                shapes = {}
                dtypes = {}
                types = {}
                steps = meta['trajectory_length'] - 2
                for key, field in meta['features'].items():
                    shapes[key] = field['shape']
                    dtypes[key] = field['dtype']
                    types[key] = field['type']
            except FileNotFoundError as e:
                print(e)
                quit()
        trajectory = {}
        # decode bytes into corresponding dtypes
        for key, value in trajectory_data.items():
            raw_data = value.numpy().tobytes()
            mature_data = np.frombuffer(
                raw_data, dtype=getattr(np, dtypes[key]))
            mature_data = torch.from_numpy(mature_data).to(device)
            reshaped_data = torch.reshape(mature_data, shapes[key])
            if types[key] == 'static':
                reshaped_data = torch.tile(
                    reshaped_data, (meta['trajectory_length'], 1, 1))
            elif types[key] == 'dynamic_varlen':
                pass
            elif types[key] != 'dynamic':
                raise ValueError('invalid data format')
            trajectory[key] = reshaped_data

        if add_targets_bool:
            trajectory = self._add_targets(params, steps)(trajectory)
        if split_and_preprocess_bool:
            trajectory = self._split_and_preprocess(params, steps)(trajectory)
        return trajectory

    @staticmethod
    def _add_targets(params, steps):
        # TODO: redundant (see flagdata.py)
        fields = params['field']
        add_history = params['history']

        def fn(trajectory):
            out = {}
            for key, val in trajectory.items():
                out[key] = val[1:-1]
                if key in fields:
                    if add_history:
                        out['prev|' + key] = val[0:-2]
                    out['target|' + key] = val[2:]
            return out

        return fn

    @staticmethod
    def _split_and_preprocess(params, steps):
        # TODO: redundant (see flagdata.py)

        noise_field = params['field']
        noise_scale = params['noise']
        noise_gamma = params['gamma']

        def add_noise(frame):
            zero_size = torch.zeros(
                frame[noise_field].size(), dtype=torch.float32).to(device)
            noise = torch.normal(zero_size, std=noise_scale).to(device)
            other = torch.Tensor([NodeType.NORMAL.value]).to(device)
            mask = torch.eq(frame['node_type'], other.int())[:, 0]
            mask_sequence = []
            for _ in range(noise.shape[1]):
                mask_sequence.append(mask)
            mask = torch.stack(mask_sequence, dim=1)
            noise = torch.where(mask, noise, torch.zeros_like(noise))
            frame[noise_field] += noise
            frame['target|' + noise_field] += (1.0 - noise_gamma) * noise
            return frame

        def element_operation(trajectory):
            trajectory_steps = []
            for i in range(steps):
                trajectory_step = {}
                for key, value in trajectory.items():
                    trajectory_step[key] = value[i]
                noisy_trajectory_step = add_noise(trajectory_step)
                trajectory_steps.append(noisy_trajectory_step)
            return trajectory_steps

        return element_operation

    @property
    def network(self):
        return self._network

    def save(self):
        dir = 'output/' + self._dataset_name + '/model.pkl'
        with open(os.path.join(DATA_DIR, dir), 'wb') as file:
            pickle.dump(self, file)

    def save_rollouts(self, rollouts):
        dir = 'output/' + self._dataset_name + '/rollouts.pkl'
        with open(os.path.join(DATA_DIR, dir), 'wb') as file:
            pickle.dump(rollouts, file)
