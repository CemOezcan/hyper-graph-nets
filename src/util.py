
import collections
import enum
import torch
import yaml
import numpy as np


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('MultiGraph', ['node_features', 'edge_sets'])
MultiGraphWithPos = collections.namedtuple('MultiGraph', ['node_features', 'edge_sets', 'target_feature', 'model_type', 'node_dynamic'])


def detach(tensor: torch.Tensor) -> np.array:

    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


class NodeType(enum.IntEnum):
    NORMAL = 0
    OBSTACLE = 1
    AIRFOIL = 2
    HANDLE = 3
    INFLOW = 4
    OUTFLOW = 5
    WALL_BOUNDARY = 6
    SIZE = 9


def read_yaml(config_name: str):
    with open(f'configs/{config_name}.yaml', 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load_all(stream)
            for file in parsed_yaml:
                if file['name'] == 'DEFAULT':
                    return file
        except yaml.YAMLError as e:
            print(e)
            return None


def triangles_to_edges(faces, deform=False):
    """Computes mesh edges from triangles."""
    if not deform:
        # collect edges from triangles
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           torch.stack((faces[:, 2], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
    else:
        edges = torch.cat((faces[:, 0:2],
                           faces[:, 1:3],
                           faces[:, 2:4],
                           torch.stack((faces[:, 3], faces[:, 0]), dim=1)), dim=0)
        # those edges are sometimes duplicated (within the mesh) and sometimes
        # single (at the mesh boundary).
        # sort & pack edges as single tf.int64
        receivers, _ = torch.min(edges, dim=1)
        senders, _ = torch.max(edges, dim=1)

        packed_edges = torch.stack((senders, receivers), dim=1)
        unique_edges = torch.unique(packed_edges, return_inverse=False, return_counts=False, dim=0)
        senders, receivers = torch.unbind(unique_edges, dim=1)
        senders = senders.to(torch.int64)
        receivers = receivers.to(torch.int64)

        two_way_connectivity = (torch.cat((senders, receivers), dim=0), torch.cat((receivers, senders), dim=0))
        return {'two_way_connectivity': two_way_connectivity, 'senders': senders, 'receivers': receivers}
