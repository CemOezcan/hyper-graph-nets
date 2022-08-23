
import collections
import enum
import torch
import torch_scatter
import yaml
import numpy as np


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
EdgeSet = collections.namedtuple('EdgeSet', ['name', 'features', 'senders', 'receivers'])
MultiGraph = collections.namedtuple('MultiGraph', ['node_features', 'edge_sets'])
MultiGraphWithPos = collections.namedtuple('MultiGraph', ['node_features', 'edge_sets', 'target_feature', 'mesh_features', 'model_type', 'node_dynamic'])


def detach(tensor: torch.Tensor) -> np.array:

    if tensor.is_cuda:
        return tensor.cpu().detach().numpy()
    else:
        return tensor.detach().numpy()


class NodeType(enum.IntEnum):
    NORMAL = 0


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


def unsorted_segment_operation(data, segment_ids, num_segments, operation):
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
    data = data.to(device)
    segment_ids = segment_ids.to(device)
    if len(segment_ids.shape) == 1:
        s = torch.prod(torch.tensor(data.shape[1:])).long().to(device)
        segment_ids = segment_ids.repeat_interleave(s).view(
            segment_ids.shape[0], *data.shape[1:]).to(device)

    assert data.shape == segment_ids.shape, "data.shape and segment_ids.shape should be equal"

    shape = [num_segments] + list(data.shape[1:])
    result = torch.zeros(*shape).to(device)
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
    elif operation == 'std':
        result = torch_scatter.scatter_std(
            data.float(), segment_ids, out=result, dim=0, dim_size=num_segments)
    else:
        raise Exception('Invalid operation type!')
    result = result.type(data.dtype)
    return result


