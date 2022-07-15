import csv
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch_scatter

from src import util
from src.data import data_loader
from src.data.graphloader import GraphDataLoader
from src.rmp.remote_message_passing import RemoteMessagePassing
from src.util import device, NodeType, EdgeSet, MultiGraphWithPos, MultiGraph
from util.Types import ConfigDict


class Preprocessing():

    def __init__(self, config: ConfigDict, split='train', split_and_preprocess=True, add_targets=True, in_dir=None):
        self._is_train = split == 'train'
        self._model_type = 'flag'
        self._split_and_preprocess_b = split_and_preprocess
        self._add_targets_b = add_targets
        self._network_config = config.get("model")
        self._dataset_dir = in_dir
        self._remote_graph = RemoteMessagePassing()

    def preprocess(self, raw_trajectory):
        graphs = list()
        trajectory = self._process_trajectory(raw_trajectory)
        if self._is_train:
            self._remote_graph.reset_clusters()
            for data_frame in trajectory:
                graphs.append(self._build_graph(data_frame))

        return graphs, trajectory


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
                noisy_trajectory_step = Preprocessing._add_noise(
                    trajectory_step, noise_field, noise_scale, noise_gamma)
                trajectory_steps.append(noisy_trajectory_step)
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

    def _build_graph(self, inputs):
        """Builds input graph."""
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev|world_pos']
        node_type = inputs['node_type']
        velocity = world_pos - prev_world_pos
        one_hot_node_type = F.one_hot(
            node_type[:, 0].to(torch.int64), NodeType.SIZE)

        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        cells = inputs['cells']
        decomposed_cells = util.triangles_to_edges(cells)
        senders, receivers = decomposed_cells['two_way_connectivity']

        mesh_pos = inputs['mesh_pos']
        relative_world_pos = (torch.index_select(input=world_pos, dim=0, index=senders) -
                              torch.index_select(input=world_pos, dim=0, index=receivers))
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        edge_features = torch.cat((
            relative_world_pos,
            torch.norm(relative_world_pos, dim=-1, keepdim=True),
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)), dim=-1)

        mesh_edges = EdgeSet(
            name='mesh_edges',
            features=edge_features,
            receivers=receivers,
            senders=senders)

        # TODO: Change data structure
        num_nodes = node_type.shape[0]
        max_node_dynamic = self.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                           num_nodes,
                                                           operation='max').to(device)
        min_node_dynamic = self.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                           num_nodes,
                                                           operation='min').to(device)
        node_dynamic = max_node_dynamic - min_node_dynamic

        graph = MultiGraphWithPos(node_features=[node_features],
                                  edge_sets=[mesh_edges], target_feature=world_pos,
                                  model_type=self._model_type, node_dynamic=node_dynamic)

        graph = self._remote_graph.create_graph(graph)
        # TODO: Replace
        # graph = MultiGraph(node_features=[node_features], edge_sets=[mesh_edges])

        return graph

    # TODO check if redundant: see graphnet.py_world_edge_normalizer
    def unsorted_segment_operation(self, data, segment_ids, num_segments, operation):
        """
        Computes the sum along segments of a tensor. Analogous to tf.unsorted_segment_sum.

        :param data: A tensor whose segments are to be summed.
        :param segment_ids: The segment indices tensor.
        :param num_segments: The number of segments.
        :return: A tensor of same data type as the data argument.
        """
        assert all([i in data.shape for i in segment_ids.shape]
                   ), "segment_ids.shape should be a prefix of data.shape"

        # segment_ids is a 1-D tensor repeat it to have the same shape as data
        if len(segment_ids.shape) == 1:
            s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
            segment_ids = segment_ids.repeat_interleave(s).view(
                segment_ids.shape[0], *data.shape[1:]).to(device)

        assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

        shape = [num_segments] + list(data.shape[1:])
        result = torch.zeros(*shape)
        if operation == 'sum':
            result = torch_scatter.scatter_add(
                data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'max':
            result, _ = torch_scatter.scatter_max(
                data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'mean':
            result = torch_scatter.scatter_mean(
                data.float(), segment_ids, dim=0, dim_size=num_segments)
        elif operation == 'min':
            result, _ = torch_scatter.scatter_min(
                data.float(), segment_ids, dim=0, dim_size=num_segments)
        else:
            raise Exception('Invalid operation type!')
        result = result.type(data.dtype)
        return result
