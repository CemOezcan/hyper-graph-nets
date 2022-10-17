import json
import os

import numpy as np
import torch
from src.util import device, NodeType
from util.Types import ConfigDict


class Preprocessing():

    def __init__(self, config: ConfigDict, split='train', split_and_preprocess=True, add_targets=True, in_dir=None):
        self._split_and_preprocess_b = split_and_preprocess
        self._add_targets_b = add_targets
        self._add_noise_b = split == 'train'
        self._network_config = config.get("model")
        self._dataset_dir = in_dir

    def preprocess(self, raw_trajectory):
        trajectory = self._process_trajectory(raw_trajectory)
        return trajectory

    def _load_model(self):
        try:
            with open(os.path.join(self._dataset_dir, 'meta.json'), 'r') as fp:
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

        return shapes, dtypes, types, steps, meta

    def _process_trajectory(self, trajectory_data):
        shapes, dtypes, types, steps, meta = self._load_model()
        trajectory = {}

        # decode bytes into corresponding dtypes
        for key, value in trajectory_data.items():
            raw_data = value.tobytes()
            mature_data = np.frombuffer(
                raw_data, dtype=getattr(np, dtypes[key]))
            mature_data = torch.from_numpy(mature_data).to(device)
            reshaped_data = torch.reshape(mature_data, shapes[key])
            if types[key] == 'static':
                reshaped_data = torch.tile(reshaped_data, (meta['trajectory_length'], 1, 1))
            elif types[key] == 'dynamic_varlen':
                pass
            elif types[key] != 'dynamic':
                raise ValueError('invalid data format')
            trajectory[key] = reshaped_data

        if self._add_targets_b:
            trajectory = self._add_targets(steps)(trajectory)
        if self._split_and_preprocess_b:
            trajectory = self._split_and_preprocess(steps)(trajectory)
        return trajectory

    def _split_and_preprocess(self, steps):
        noise_field = self._network_config['field']
        noise_scale = self._network_config['noise']
        noise_gamma = self._network_config['gamma']

        def element_operation(trajectory):
            trajectory_steps = []
            for i in range(steps):
                trajectory_step = {}
                for key, value in trajectory.items():
                    trajectory_step[key] = value[i]
                if self._add_noise_b:
                    trajectory_step = Preprocessing._add_noise(trajectory_step, noise_field, noise_scale, noise_gamma)
                trajectory_steps.append(trajectory_step)
            return trajectory_steps

        return element_operation

    @staticmethod
    def _add_noise(frame, noise_field, noise_scale, noise_gamma):
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

    def _add_targets(self, steps):
        fields = self._network_config['field']
        add_history = self._network_config['history']

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
