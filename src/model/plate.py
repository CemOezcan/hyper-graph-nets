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


class PlateModel(AbstractSystemModel):
    """
    Model for deforming plate simulation.
    """

    def __init__(self, params: ConfigDict):
        super(PlateModel, self).__init__(params)
        self.loss_fn = torch.nn.MSELoss()

        self._output_normalizer = Normalizer(size=3, name='output_normalizer')
        # TODO: Kinematic nodes have a different number of dimensions (Solution: paddingg)
        self._node_normalizer = Normalizer(size=6, name='node_normalizer')
        self._node_dynamic_normalizer = Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = Normalizer(size=8, name='mesh_edge_normalizer')
        self._world_edge_normalizer = Normalizer(size=4, name='world_edge_normalizer')
        self._intra_edge_normalizer = Normalizer(size=8, name='intra_edge_normalizer')
        self._inter_edge_normalizer = Normalizer(size=8, name='intra_edge_normalizer')

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

        self._edge_sets = ['mesh_edges', 'world_edges']
        if self._balancer:
            import src.graph_balancer.get_graph_balancer as graph_balancer
            self._graph_balancer = graph_balancer.get_balancer(params)
            self._edge_sets.append('balance')
        if self._rmp:
            self._remote_graph = rmp.get_rmp(params)
            self._edge_sets += self._remote_graph.initialize(
                self._intra_edge_normalizer, self._inter_edge_normalizer)

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
        world_pos = inputs['world_pos']
        mesh_pos = inputs['mesh_pos']
        target_world_pos = inputs['target|world_pos']

        node_type = inputs['node_type']
        node_types = torch.flatten(node_type[:, 0]).long()
        # TODO: Find dynamic solution
        node_types[node_types == 3] = 2
        one_hot_node_type = F.one_hot(node_types)

        cells = inputs['cells']
        decomposed_cells = util.triangles_to_edges(cells, deform=True)
        senders, receivers = decomposed_cells['two_way_connectivity']

        # find world edges
        radius = 0.03
        world_distance_matrix = torch.cdist(world_pos, world_pos, p=2)
        world_connection_matrix = torch.where(world_distance_matrix < radius, True, False)
        world_connection_matrix = world_connection_matrix.fill_diagonal_(False)

        # remove world edge node pairs that already exist in mesh edge collection
        world_connection_matrix[senders, receivers] = torch.tensor(False, dtype=torch.bool, device=device)

        # Only obstacle nodes as senders and normal nodes as receivers
        # Remove all edges from non-obstacle nodes
        # TODO: Change radius?
        # non_obstacle_nodes = torch.ne(node_type[:, 0], torch.tensor([util.NodeType.OBSTACLE.value], device=device))
        # world_connection_matrix[non_obstacle_nodes, :] = torch.tensor(False, dtype=torch.bool, device=device)

        # Remove all edges to non-normal nodes
        non_normal_nodes = torch.ne(node_type[:, 0], torch.tensor([util.NodeType.NORMAL.value], device=device))
        world_connection_matrix[:, non_normal_nodes] = torch.tensor(False, dtype=torch.bool, device=device)

        # TODO: Only select the closest sender?
        world_senders, world_receivers = torch.nonzero(world_connection_matrix, as_tuple=True)

        relative_world_pos = (
                torch.index_select(input=world_pos, dim=0, index=world_senders) -
                torch.index_select(input=world_pos, dim=0, index=world_receivers)
        )

        # TODO: Encode velocities as edge features?
        """relative_world_velocity = (
                torch.index_select(input=target_world_pos, dim=0, index=world_senders) -
                torch.index_select(input=world_pos, dim=0, index=world_senders)
        )

        world_edge_features = torch.cat(
            (
                relative_world_pos,
                torch.norm(relative_world_pos, dim=-1, keepdim=True),
                relative_world_velocity,
                torch.norm(relative_world_velocity, dim=-1, keepdim=True)
            ),
            dim=-1
        )"""
        # TODO: Test validity
        # TODO: Test Clustering
        # TODO: Test MGN/HGN compatibility
        # TODO: Implement remaining methods
        # TODO: Implement plotting
        world_edge_features = torch.cat(
            (relative_world_pos, torch.norm(relative_world_pos, dim=-1, keepdim=True)),
            dim=-1
        )

        world_edges = EdgeSet(
            name='world_edges',
            features=self._world_edge_normalizer(world_edge_features, is_training),
            receivers=world_receivers,
            senders=world_senders
        )

        relative_mesh_pos = (
                torch.index_select(mesh_pos, 0, senders) -
                torch.index_select(mesh_pos, 0, receivers)
        )
        all_relative_world_pos = (
                torch.index_select(input=world_pos, dim=0, index=senders) -
                torch.index_select(input=world_pos, dim=0, index=receivers)
        )

        mesh_edge_features = torch.cat(
            (
                relative_mesh_pos,
                torch.norm(relative_mesh_pos, dim=-1, keepdim=True),
                all_relative_world_pos,
                torch.norm(all_relative_world_pos, dim=-1, keepdim=True)
            ),
            dim=-1
        )

        mesh_edges = EdgeSet(
            name='mesh_edges',
            features=self._mesh_edge_normalizer(mesh_edge_features, is_training),
            receivers=receivers,
            senders=senders
        )

        # Append velocities to kinematic nodes
        # TODO: Correct?
        obstacle_nodes = torch.eq(node_type[:, 0], torch.tensor([util.NodeType.OBSTACLE.value], device=device))
        obstacle_indices = obstacle_nodes.nonzero().squeeze()

        num_nodes = node_type.shape[0]
        velocities = torch.zeros(num_nodes, 3).to(device)
        velocities[obstacle_nodes] = (
                torch.index_select(input=target_world_pos, dim=0, index=obstacle_indices) -
                torch.index_select(input=world_pos, dim=0, index=obstacle_indices)
        )
        node_features = torch.cat((one_hot_node_type, velocities), dim=-1)

        max_node_dynamic = util.unsorted_segment_operation(
            torch.norm(all_relative_world_pos, dim=-1),
            receivers,
            num_nodes,
            operation='max'
        ).to(device)

        min_node_dynamic = util.unsorted_segment_operation(
            torch.norm(all_relative_world_pos, dim=-1),
            receivers,
            num_nodes,
            operation='min'
        ).to(device)

        node_dynamic = self._node_dynamic_normalizer(max_node_dynamic - min_node_dynamic)

        return MultiGraphWithPos(node_features=[self._node_normalizer(node_features, is_training)],
                                 edge_sets=[mesh_edges, world_edges],
                                 mesh_features=mesh_pos,
                                 target_feature=world_pos,
                                 model_type=self._model_type,
                                 unnormalized_edges=EdgeSet(
                                     name='mesh_edges',
                                     features=mesh_edge_features,
                                     receivers=receivers,
                                     senders=senders
                                 ),
                                 node_dynamic=node_dynamic)

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
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device).int())
        loss = self.loss_fn(target_normalized[loss_mask], network_output[loss_mask])
        # TODO: Differentiate between velocity and stress loss?

        return loss

    @torch.no_grad()
    def validation_step(self, graph: MultiGraph, data_frame: Dict) -> Tuple[Tensor, Tensor]:
        # TODO: Split vel_los and stress_loss?
        prediction = self(graph)
        target_normalized = self.get_target(data_frame, False)

        node_type = data_frame['node_type']
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device).int())
        vel_loss = self.loss_fn(target_normalized[loss_mask], prediction[loss_mask]).item()

        predicted_position, cur_position, velocity = self.update(data_frame, prediction)
        pos_error = self.loss_fn(data_frame['target|world_pos'][loss_mask], predicted_position[loss_mask]).item()

        return vel_loss, pos_error

    def update(self, inputs: Dict, per_node_network_output: Tensor) -> Tensor:
        """Integrate model outputs."""
        velocity = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['world_pos']

        # vel. = next_pos - cur_pos
        position = cur_position + velocity

        return (position, cur_position, velocity)

    def get_target(self, data_frame, is_training=True):
        cur_position = data_frame['world_pos']
        target_position = data_frame['target|world_pos']
        target_velocity = target_position - cur_position

        return self._output_normalizer(target_velocity, is_training).to(device)

    @torch.no_grad()
    def rollout(self, trajectory: Dict[str, Tensor], num_steps: int) -> Tuple[Dict[str, Tensor], Tensor]:
        """Rolls out a model trajectory."""
        num_steps = trajectory['cells'].shape[0] if num_steps is None else num_steps
        initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}

        node_type = initial_state['node_type']
        mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device))
        # TODO: Correct with 4d output?
        mask = torch.stack((mask, mask, mask), dim=1)

        cur_pos = torch.squeeze(initial_state['world_pos'], 0)
        target_pos = trajectory['target|world_pos']
        pred_trajectory = []
        cur_positions = []
        cur_velocities = []
        for step in range(num_steps):
            cur_pos,  pred_trajectory, cur_positions, cur_velocities = \
                self._step_fn(initial_state, cur_pos, pred_trajectory, cur_positions, cur_velocities, target_pos[step], step, mask)

        prediction, cur_positions, cur_velocities = \
            (torch.stack(pred_trajectory), torch.stack(cur_positions), torch.stack(cur_velocities))

        # temp solution for visualization
        faces = trajectory['cells']
        faces_result = []
        for faces_step in faces:
            later = torch.cat((faces_step[:, 2:4], torch.unsqueeze(faces_step[:, 0], 1)), -1)
            faces_step = torch.cat((faces_step[:, 0:3], later), 0)
            faces_result.append(faces_step)

        faces_result = torch.stack(faces_result, 0)
        # trajectory_polygons = to_polygons(trajectory['cells'], trajectory['world_pos'])

        traj_ops = {
            'faces': faces_result,
            'mesh_pos': trajectory['mesh_pos'],
            'mask': torch.eq(node_type[:, 0], torch.tensor([NodeType.OBSTACLE.value], device=device).int()),
            'gt_pos': trajectory['world_pos'],
            'pred_pos': prediction,
            'cur_positions': cur_positions,
            'cur_velocities': cur_velocities
        }

        mse_loss_fn = torch.nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fn(trajectory['world_pos'][:num_steps], prediction)
        mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss

    @torch.no_grad()
    def _step_fn(self, initial_state, cur_pos, trajectory, cur_positions, cur_velocities, target_world_pos, step, mask):
        input = {**initial_state, 'world_pos': cur_pos, 'target|world_pos': target_world_pos}
        graph = self.build_graph(input, is_training=False)
        if not self._visualized:
            coordinates = graph.target_feature.cpu().detach().numpy()

        graph = self.expand_graph(graph, step, 398, is_training=False)

        if self._rmp and not self._visualized:
            self._remote_graph.visualize_cluster(coordinates)
            self._visualized = True

        prediction, cur_position, cur_velocity = self.update(input, self(graph))
        next_pos = torch.where(mask, torch.squeeze(prediction), torch.squeeze(target_world_pos))

        trajectory.append(next_pos)
        cur_positions.append(cur_position)
        cur_velocities.append(cur_velocity)
        return next_pos, trajectory, cur_positions, cur_velocities

    @torch.no_grad()
    def n_step_computation(self, trajectory: Dict[str, Tensor], n_step: int) -> Tensor:
        mse_losses = list()
        for step in range(len(trajectory['world_pos']) - n_step):
            # TODO: clusters/balancers are reset when computing n_step loss
            eval_traj = {k: v[step: step + n_step + 1] for k, v in trajectory.items()}
            prediction_trajectory, mse_loss = self.rollout(eval_traj, n_step + 1)
            mse_losses.append(torch.mean(mse_loss).cpu())

        return torch.mean(torch.stack(mse_losses))
