import math
from typing import Dict, Tuple

import src.rmp.get_rmp as rmp
import torch
import torch.nn.functional as F
from src import util
from src.migration.meshgraphnet import MeshGraphNet
from src.migration.normalizer import Normalizer
from src.model.abstract_system_model import AbstractSystemModel
from src.util import EdgeSet, MultiGraphWithPos, NodeType, device, MultiGraph
from torch import nn, Tensor

from util.Types import ConfigDict


class CylinderModel(AbstractSystemModel):
    """
    Model for computational fluid dynamics simulation.
    """

    def __init__(self, params: ConfigDict):
        super(CylinderModel, self).__init__(params)
        self.loss_fn = torch.nn.MSELoss()

        self._output_normalizer = Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = Normalizer(size=6, name='node_normalizer')
        self._node_dynamic_normalizer = Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = Normalizer(size=3, name='mesh_edge_normalizer')
        self._intra_edge_normalizer = Normalizer(size=7, name='intra_edge_normalizer')
        self._inter_edge_normalizer = Normalizer(size=7, name='inter_edge_normalizer')
        self._hyper_node_normalizer = Normalizer(size=9, name='hyper_node_normalizer')

        self._model_type = 'plate'
        self._rmp = params.get('rmp').get('clustering') != 'none' and params.get('rmp').get('connector') != 'none'
        self._architecture = params.get('rmp').get('connector') if self._rmp else 'none'
        self._multi = params.get('rmp').get('connector') == 'multigraph' and self._rmp
        self._balancer = params.get('graph_balancer').get('algorithm') != 'none'
        self.message_passing_steps = params.get('message_passing_steps')
        self.message_passing_aggregator = params.get('aggregation')
        self._balance_frequency = params.get('graph_balancer').get('frequency')
        self._rmp_frequency = params.get('rmp').get('frequency')
        self._visualized = False

        self._edge_sets = ['mesh_edges']
        if self._balancer:
            import src.graph_balancer.get_graph_balancer as graph_balancer
            self._graph_balancer = graph_balancer.get_balancer(params)
            self._edge_sets.append('balance')
        if self._rmp:
            self._remote_graph = rmp.get_rmp(params)
            self._edge_sets += self._remote_graph.initialize(
                self._intra_edge_normalizer, self._inter_edge_normalizer, self._hyper_node_normalizer)

        self.learned_model = MeshGraphNet(
            output_size=params.get('size'),
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            architecture=self._architecture,
            edge_sets=self._edge_sets
        ).to(device)

    def build_graph(self, inputs: Dict, is_training: bool) -> MultiGraphWithPos:
        """Builds input graph."""
        node_type = inputs['node_type']
        velocity = inputs['velocity']

        node_types = torch.flatten(node_type[:, 0]).long()
        # TODO: Find dynamic solution
        node_types[node_types == 4] = 1
        node_types[node_types == 5] = 2
        node_types[node_types == 6] = 3
        one_hot_node_type = F.one_hot(node_types)
        node_features = torch.cat((velocity, one_hot_node_type), dim=-1)

        cells = inputs['cells']
        decomposed_cells = util.triangles_to_edges(cells)
        senders, receivers = decomposed_cells['two_way_connectivity']

        mesh_pos = inputs['mesh_pos']
        relative_mesh_pos = (torch.index_select(mesh_pos, 0, senders) -
                             torch.index_select(mesh_pos, 0, receivers))
        edge_features = torch.cat([
            relative_mesh_pos,
            torch.norm(relative_mesh_pos, dim=-1, keepdim=True)], dim=-1)

        mesh_edges = EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)

        return MultiGraphWithPos(node_features=[self._node_normalizer(node_features, is_training)],
                                 edge_sets=[mesh_edges],
                                 mesh_features=mesh_pos,
                                 target_feature=velocity,
                                 model_type=self._model_type,
                                 unnormalized_edges=EdgeSet(
                                     name='mesh_edges',
                                     features=edge_features,
                                     receivers=receivers,
                                     senders=senders
                                 ),
                                 node_dynamic=[], obstacle_nodes=None)

    def expand_graph(self, graph: MultiGraphWithPos, step: int, num_steps: int, is_training: bool) -> MultiGraph:
        if self._balancer:
            if step % math.ceil(num_steps / self._balance_frequency) == 0:
                self._graph_balancer.reset_balancer()
            graph = self._graph_balancer.create_graph(graph, self._mesh_edge_normalizer, is_training)

        if self._rmp:
            if step % math.ceil(num_steps / self._rmp_frequency) == 0:
                self._remote_graph.reset_clusters()
            graph = self._remote_graph.create_graph(graph, is_training)

        return graph

    def forward(self, graph):
        return self.learned_model(graph)

    def training_step(self, graph, data_frame):
        network_output = self(graph)
        target_normalized = self.get_target(data_frame)

        node_type = data_frame['node_type']
        normal_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device).int())
        outflow_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.OUTFLOW.value], device=device).int())
        loss_mask = torch.logical_or(outflow_mask, normal_mask)

        loss = self.loss_fn(target_normalized[loss_mask], network_output[loss_mask])
        # TODO: Differentiate between velocity and pressure loss?

        return loss

    @torch.no_grad()
    def validation_step(self, graph: MultiGraph, data_frame: Dict) -> Tuple[Tensor, Tensor]:
        # TODO: Split vel_los and stress_loss?
        prediction = self(graph)
        target_normalized = self.get_target(data_frame, False)

        node_type = data_frame['node_type']
        normal_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device).int())
        outflow_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.OUTFLOW.value], device=device).int())
        loss_mask = torch.logical_or(outflow_mask, normal_mask)

        vel_loss = self.loss_fn(target_normalized[loss_mask], prediction[loss_mask]).item()

        velocity_update, pressure = self.update(data_frame, prediction)
        pos_error = self.loss_fn(data_frame['target|velocity'][loss_mask], velocity_update[loss_mask]).item()

        return vel_loss, pos_error

    def update(self, inputs: Dict, per_node_network_output: Tensor) -> Tensor:
        """Integrate model outputs."""
        velocity, pressure = torch.split(self._output_normalizer.inverse(per_node_network_output), 2, dim=1)

        # integrate forward
        cur_velocity = inputs['velocity']
        # vel. = cur_pos - prev_pos
        velocity_update = cur_velocity + velocity

        return velocity_update, pressure

    def get_target(self, data_frame, is_training=True):
        cur_velocity = data_frame['velocity']
        target_velocity = data_frame['target|velocity']
        target_pressure = data_frame['pressure']
        target_velocity_change = target_velocity - cur_velocity

        return self._output_normalizer(torch.cat((target_velocity_change, target_pressure), dim=1), is_training).to(device)

    def rollout(self, trajectory: Dict[str, Tensor], num_steps: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """Rolls out a model trajectory."""
        initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}
        num_steps = trajectory['cells'].shape[0]

        # rollout
        node_type = initial_state['node_type']
        mask = torch.logical_or(torch.eq(node_type[:, 0], torch.tensor([util.NodeType.NORMAL.value], device=device)),
                                torch.eq(node_type[:, 0], torch.tensor([util.NodeType.OUTFLOW.value], device=device)))
        mask = torch.stack((mask, mask), dim=1)

        velocity = torch.squeeze(initial_state['velocity'], 0)
        pressure = torch.squeeze(initial_state['pressure'], 0)
        pred_trajectory = []
        pred_pressure = list()
        for step in range(num_steps):
            velocity, pressure, pred_trajectory, pred_pressure = self._step_fn(initial_state, velocity, pressure, pred_trajectory, pred_pressure, step, mask)

        prediction = torch.stack(pred_trajectory)
        pressure = torch.stack(pred_pressure)

        traj_ops = {
            'faces': trajectory['cells'],
            'mesh_pos': trajectory['mesh_pos'],
            'gt_velocity': trajectory['velocity'],
            'gt_pressure': trajectory['pressure'],
            'pred_pressure': pressure,
            'pred_velocity': prediction
        }

        mse_loss_fn = torch.nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fn(trajectory['velocity'][:num_steps], prediction)
        mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss

    @torch.no_grad()
    def _step_fn(self, initial_state, velocity, pressure, trajectory, pressure_trajectory, step, mask):
        input = {**initial_state, 'velocity': velocity, 'pressure': pressure}
        graph = self.build_graph(input, is_training=False)
        if not self._visualized:
            coordinates = graph.target_feature.cpu().detach().numpy()

        graph = self.expand_graph(graph, step, 598, is_training=False)

        if self._rmp and not self._visualized:
            self._remote_graph.visualize_cluster(coordinates)
            self._visualized = True

        prediction, pred_pressure = self.update(input, self(graph))

        # don't update boundary nodes
        next_velocity = torch.where(mask, torch.squeeze(prediction), torch.squeeze(velocity))
        trajectory.append(next_velocity)
        pressure_trajectory.append(pred_pressure)

        return next_velocity, pred_pressure, trajectory, pressure_trajectory

    @torch.no_grad()
    def n_step_computation(self, trajectory: Dict[str, Tensor], n_step: int, num_timesteps=None) -> Tuple[Tensor, Tensor]:
        mse_losses = list()
        last_losses = list()
        num_timesteps = trajectory['cells'].shape[0] if num_timesteps is None else num_timesteps
        for step in range(num_timesteps - n_step):
            # TODO: clusters/balancers are reset when computing n_step loss
            eval_traj = {k: v[step: step + n_step + 1] for k, v in trajectory.items()}
            prediction_trajectory, mse_loss = self.rollout(eval_traj, n_step + 1)
            mse_losses.append(torch.mean(mse_loss).cpu())
            last_losses.append(mse_loss.cpu()[-1])

        return torch.mean(torch.stack(mse_losses)), torch.mean(torch.stack(last_losses))
