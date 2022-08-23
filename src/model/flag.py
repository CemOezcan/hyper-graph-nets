"""Model for FlagSimple."""

import torch
import torch_scatter
import torch.nn.functional as F

from torch import nn
import src.rmp.get_rmp as rmp
from src.rmp.remote_message_passing import RemoteMessagePassing
from src.migration.normalizer import Normalizer
from src.migration.meshgraphnet import MeshGraphNet
from src import util
#from src.rmp.ricci import Ricci
from src.util import NodeType, EdgeSet, MultiGraph, device, MultiGraphWithPos


class FlagModel(nn.Module):
    """Model for static cloth simulation."""

    def __init__(self, params):
        super(FlagModel, self).__init__()
        self._params = params
        self.loss_fn = torch.nn.MSELoss()

        self._output_normalizer = Normalizer(size=3, name='output_normalizer')
        self._node_normalizer = Normalizer(size=5, name='node_normalizer')
        self._node_dynamic_normalizer = Normalizer(size=1, name='node_dynamic_normalizer')
        self._mesh_edge_normalizer = Normalizer(size=7, name='mesh_edge_normalizer')
        self._intra_edge_normalizer = Normalizer(size=4, name='intra_edge_normalizer')
        self._inter_edge_normalizer = Normalizer(size=4, name='intra_edge_normalizer')

        self._model_type = 'flag'
        self._rmp = params.get('rmp').get('clustering') != 'none' and params.get('rmp').get('connector') != 'none'
        self._hierarchical = params.get('rmp').get('connector') == 'hierarchical' and self._rmp
        self._multi = params.get('rmp').get(
            'connector') == 'multigraph' and self._rmp
        self._ricci = params.get('rmp').get('ricci')
        self.message_passing_steps = params.get('message_passing_steps')
        self.message_passing_aggregator = params.get('aggregation')

        self._edge_sets = ['mesh_edges']
     #   if self._ricci:
      #      self._ricci_flow = Ricci()
       #     self._edge_sets.append('ricci')
        if self._rmp:
            self._remote_graph = rmp.get_rmp(params)
            self._edge_sets += self._remote_graph.initialize(self._intra_edge_normalizer, self._inter_edge_normalizer)

        self.learned_model = MeshGraphNet(
            output_size=params.get('size'),
            latent_size=128,
            num_layers=2,
            message_passing_steps=self.message_passing_steps,
            message_passing_aggregator=self.message_passing_aggregator,
            hierarchical=self._hierarchical,
            edge_sets=self._edge_sets
        ).to(device)

    def build_graph(self, inputs, is_training):
        """Builds input graph."""
        world_pos = inputs['world_pos']
        prev_world_pos = inputs['prev|world_pos']
        node_type = inputs['node_type']
        velocity = world_pos - prev_world_pos

        node_types = list(map(lambda x: 0 if x == 0 else 1, node_type[:, 0]))
        one_hot_node_type = F.one_hot(torch.tensor(node_types))

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
            features=self._mesh_edge_normalizer(edge_features, is_training),
            receivers=receivers,
            senders=senders)

        # TODO: Add world_edges here? (FlagDynamic instead of FlagSimple)
        # TODO: Change data structure

        num_nodes = node_type.shape[0]
        max_node_dynamic = util.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                           num_nodes,
                                                           operation='max').to(device)
        min_node_dynamic = util.unsorted_segment_operation(torch.norm(relative_world_pos, dim=-1), receivers,
                                                           num_nodes,
                                                           operation='min').to(device)
        node_dynamic = self._node_dynamic_normalizer(max_node_dynamic - min_node_dynamic)

        graph = MultiGraphWithPos(node_features=[self._node_normalizer(node_features, is_training)],
                                  edge_sets=[mesh_edges], target_feature=world_pos, mesh_features=mesh_pos,
                                  model_type=self._model_type, node_dynamic=node_dynamic)

        # No ripples: graph = MultiGraph(node_features=self._node_normalizer(node_features), edge_sets=[mesh_edges])
        # TODO: Normalize hyper nodes
     #   if self._ricci:
      #      graph = self._ricci_flow.run(graph, inputs, self._mesh_edge_normalizer, is_training)
        if self._rmp:
            graph = self._remote_graph.create_graph(graph, is_training)

        return MultiGraph(node_features=graph.node_features, edge_sets=graph.edge_sets)

    def forward(self, graph):
        return self.learned_model(graph)

    def training_step(self, graph, data_frame):
        network_output = self(graph)
        target_normalized = self.get_target(data_frame)

        node_type = data_frame['node_type']
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device).int())
        loss = self.loss_fn(target_normalized[loss_mask], network_output[loss_mask])

        return loss

    @torch.no_grad()
    def validation_step(self, graph, data_frame):
        prediction = self(graph)
        target_normalized = self.get_target(data_frame, False)

        node_type = data_frame['node_type']
        loss_mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device).int())
        acc_loss = self.loss_fn(target_normalized[loss_mask], prediction[loss_mask]).item()

        predicted_position = self.update(data_frame, prediction)
        pos_error = self.loss_fn(data_frame['target|world_pos'][loss_mask], predicted_position[loss_mask]).item()

        return acc_loss, pos_error

    def update(self, inputs, per_node_network_output):
        """Integrate model outputs."""
        acceleration = self._output_normalizer.inverse(per_node_network_output)

        # integrate forward
        cur_position = inputs['world_pos']
        prev_position = inputs['prev|world_pos']

        # vel. = cur_pos - prev_pos
        position = 2 * cur_position + acceleration - prev_position

        return position

    def get_target(self, data_frame, is_training=True):
        cur_position = data_frame['world_pos']
        prev_position = data_frame['prev|world_pos']
        target_position = data_frame['target|world_pos']

        # next_pos = cur_pos + acc + vel <=> acc = next_pos - cur_pos - vel | vel = cur_pos - prev_pos
        target_acceleration = target_position - 2 * cur_position + prev_position

        return self._output_normalizer(target_acceleration, is_training).to(device)

    @torch.no_grad()
    def rollout(self, trajectory, num_steps):
        """Rolls out a model trajectory."""
        num_steps = trajectory['cells'].shape[0] if num_steps is None else num_steps
        initial_state = {k: torch.squeeze(v, 0)[0] for k, v in trajectory.items()}

        node_type = initial_state['node_type']
        mask = torch.eq(node_type[:, 0], torch.tensor([NodeType.NORMAL.value], device=device))
        mask = torch.stack((mask, mask, mask), dim=1)

        prev_pos = torch.squeeze(initial_state['prev|world_pos'], 0)
        cur_pos = torch.squeeze(initial_state['world_pos'], 0)

        pred_trajectory = list()
        for _ in range(num_steps):
            prev_pos, cur_pos, pred_trajectory = self._step_fn(initial_state, prev_pos, cur_pos, pred_trajectory, mask)

        predictions = torch.stack(pred_trajectory)

        traj_ops = {
            'faces': trajectory['cells'],
            'mesh_pos': trajectory['mesh_pos'],
            'gt_pos': trajectory['world_pos'],
            'pred_pos': predictions
        }

        mse_loss_fn = torch.nn.MSELoss(reduction='none')
        mse_loss = mse_loss_fn(trajectory['world_pos'][:num_steps], predictions)
        mse_loss = torch.mean(torch.mean(mse_loss, dim=-1), dim=-1).detach()

        return traj_ops, mse_loss

    @torch.no_grad()
    def _step_fn(self, initial_state, prev_pos, cur_pos, trajectory, mask):
        input = {**initial_state, 'prev|world_pos': prev_pos, 'world_pos': cur_pos}
        graph = self.build_graph(input, is_training=False)
        prediction = self.update(input, self(graph))

        next_pos = torch.where(mask, torch.squeeze(prediction), torch.squeeze(cur_pos))
        trajectory.append(cur_pos)

        return cur_pos, next_pos, trajectory

    @torch.no_grad()
    def n_step_computation(self, trajectory, n_step):
        mse_losses = list()
        for step in range(len(trajectory['world_pos']) - n_step):
            eval_traj = {k: v[step: step + n_step + 1] for k, v in trajectory.items()}
            prediction_trajectory, mse_loss = self.rollout(eval_traj, n_step + 1)
            mse_losses.append(torch.mean(mse_loss).cpu())

        return torch.mean(torch.stack(mse_losses))

    def get_output_normalizer(self):
        return self._output_normalizer

    def reset_remote_graph(self):
        if self._rmp:
            self._remote_graph.reset_clusters()

    def save_model(self, path):
        torch.save(self.learned_model, path + "_learned_model.pth")
        torch.save(self._output_normalizer, path + "_output_normalizer.pth")
        torch.save(self._mesh_edge_normalizer,
                   path + "_mesh_edge_normalizer.pth")
        torch.save(self._world_edge_normalizer,
                   path + "_world_edge_normalizer.pth")
        torch.save(self._node_normalizer, path + "_node_normalizer.pth")
        torch.save(self._node_dynamic_normalizer,
                   path + "_node_dynamic_normalizer.pth")

    def load_model(self, path):
        self.learned_model = torch.load(path + "_learned_model.pth")
        self._output_normalizer = torch.load(path + "_output_normalizer.pth")
        self._mesh_edge_normalizer = torch.load(
            path + "_mesh_edge_normalizer.pth")
        self._world_edge_normalizer = torch.load(
            path + "_world_edge_normalizer.pth")
        self._node_normalizer = torch.load(path + "_node_normalizer.pth")
        self._node_dynamic_normalizer = torch.load(
            path + "_node_dynamic_normalizer.pth")

    def evaluate(self):
        self.eval()
        self.learned_model.eval()
