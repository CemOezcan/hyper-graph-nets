import json
import os

import torch

import numpy as np
from util.Types import *
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.algorithms.AbstractIterativeAlgorithm import AbstractIterativeAlgorithm
from util.pytorch.TorchUtil import detach
from src.model.flag import FlagModel
from src.util import device, NodeType
from data.data_loader import DATA_DIR

# TODO check if only applicable for flag
class FlagSimulator(AbstractIterativeAlgorithm):
    def __init__(self, config: ConfigDict) -> None:
        super().__init__(config=config)
        # TODO: Config file for flag model
        self._network_config = config.get("model")
        self._dataset_dir = os.path.join(DATA_DIR, config.get('task').get('dataset'))

        self._network = None
        self._optimizer = None
        # TODO: Add scheduler
        # self._scheduler = None

        self.loss_function = F.mse_loss
        self._learning_rate = self._network_config.get("learning_rate")
        # self._scheduler_learning_rate = self._network_config.get("scheduler_learning_rate")

    def initialize(self, task_information: ConfigDict) -> None:
        self._network = FlagModel(self._network_config)

        self._optimizer = optim.Adam(self._network.parameters(), lr=self._learning_rate)
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
        params = self._network_config
        dataset_dir = self._dataset_dir
        is_training = True

        for data in train_dataloader:  # for each batch
            trajectory = self._process_trajectory(data, params, dataset_dir, True, True)

            for data_frame in trajectory:
                data_frame = self._squeeze_data_frame(data_frame)
                network_output = self._network(data_frame, is_training)

                cur_position = data_frame['world_pos']
                prev_position = data_frame['prev|world_pos']
                target_position = data_frame['target|world_pos']
                # TODO check if applicable for other tasks, refactor to model itself
                target_acceleration = target_position - 2 * cur_position + prev_position
                target_normalized = self._network.get_output_normalizer()(target_acceleration).to(device)

                node_type = data_frame['node_type']
                loss_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device).int())
                error = torch.sum((target_normalized - network_output) ** 2, dim=1)
                loss = torch.mean(error[loss_mask])

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

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
            mature_data = np.frombuffer(raw_data, dtype=getattr(np, dtypes[key]))
            mature_data = torch.from_numpy(mature_data).to(device)
            reshaped_data = torch.reshape(mature_data, shapes[key])
            if types[key] == 'static':
                reshaped_data = torch.tile(reshaped_data, (meta['trajectory_length'], 1, 1))
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
            zero_size = torch.zeros(frame[noise_field].size(), dtype=torch.float32).to(device)
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
