import math

import numpy as np
import torch
import wandb
from numba import cuda
from src.graph_balancer.abstract_graph_balancer import AbstractGraphBalancer
from src.util import MultiGraphWithPos
from torch_geometric.data import Data
from torch_geometric.utils import (from_networkx, remove_self_loops,
                                   to_dense_adj, to_networkx, to_undirected)


class Ricci(AbstractGraphBalancer):
    def __init__(self, params):
        super().__init__()
        self._loops = params.get('ricci').get('loops')
        self._tau = params.get('ricci').get('tau')
        self._wandb = wandb.init(reinit=False)

    def _initialize(self):
        pass

    def run(self, graph: MultiGraphWithPos, inputs, mesh_edge_normalizer, is_training: bool):
        new_graph, added_edges = Ricci.sdrf(data=self.transform_multigraph_to_pyg(
            graph), loops=self._tau, remove_edges=False, tau=self._tau, is_undirected=True)
        new_graph = self.add_graph_balance_edges(
            graph, added_edges, inputs, mesh_edge_normalizer, is_training)
        self._wandb.log({'ricci added edges': len(added_edges['senders'])})
        return new_graph

    def transform_multigraph_to_pyg(self, graph: MultiGraphWithPos) -> Data:
        self.g = graph
        edge_index = torch.stack(
            (graph.edge_sets[0].senders, graph.edge_sets[0].receivers), dim=0)
        node_features = graph.node_features[0]
        pyg = Data(x=node_features, edge_index=edge_index,
                   edge_attr=graph.edge_sets[0].features)
        return pyg

    @staticmethod
    def sdrf(
        data,
        loops=10,
        remove_edges=False,
        removal_bound=0.5,
        tau=1,
        is_undirected=True,
    ):
        edge_index = data.edge_index
        if is_undirected:
            edge_index = to_undirected(edge_index)
        A = to_dense_adj(remove_self_loops(edge_index)[0])[0]
        N = A.shape[0]
        G = to_networkx(data)
        if is_undirected:
            G = G.to_undirected()
        A = A.cuda()
        C = torch.zeros(N, N).cuda()
        added_edges = {'senders': [], 'receivers': []}
        for x in range(loops):
            can_add = True
            Ricci.balanced_forman_curvature(A, C=C)
            ix_min = C.argmin().item()
            x = ix_min // N
            y = ix_min % N

            if is_undirected:
                x_neighbors = list(G.neighbors(x)) + [x]
                y_neighbors = list(G.neighbors(y)) + [y]
            else:
                x_neighbors = list(G.successors(x)) + [x]
                y_neighbors = list(G.predecessors(y)) + [y]
            candidates = []
            for i in x_neighbors:
                for j in y_neighbors:
                    if (i != j) and (not G.has_edge(i, j)):
                        candidates.append((i, j))

            if len(candidates):
                D = Ricci.balanced_forman_post_delta(
                    A, x, y, x_neighbors, y_neighbors)
                improvements = []
                for (i, j) in candidates:
                    improvements.append(
                        (D - C[x, y])[x_neighbors.index(i),
                                      y_neighbors.index(j)].item()
                    )

                k, l = candidates[
                    np.random.choice(
                        range(len(candidates)), p=Ricci.softmax(np.array(improvements), tau=tau)
                    )
                ]
                G.add_edge(k, l)
                added_edges['senders'].append(k)
                added_edges['receivers'].append(l)
                if is_undirected:
                    A[k, l] = A[l, k] = 1
                else:
                    A[k, l] = 1
            else:
                can_add = False
                if not remove_edges:
                    break

            if remove_edges:
                ix_max = C.argmax().item()
                x = ix_max // N
                y = ix_max % N
                if C[x, y] > removal_bound:
                    G.remove_edge(x, y)
                    if is_undirected:
                        A[x, y] = A[y, x] = 0
                    else:
                        A[x, y] = 0
                else:
                    if can_add is False:
                        break

        return from_networkx(G), added_edges

    @staticmethod
    def balanced_forman_curvature(A, C=None):
        N = A.shape[0]
        A2 = torch.matmul(A, A)
        d_in = A.sum(axis=0)
        d_out = A.sum(axis=1)
        if C is None:
            C = torch.zeros(N, N).cuda()

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(N / threadsperblock[0])
        blockspergrid_y = math.ceil(N / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        Ricci._balanced_forman_curvature[blockspergrid, threadsperblock](
            A, A2, d_in, d_out, N, C)
        return C

    @staticmethod
    @cuda.jit(
        "void(float32[:,:], float32[:,:], float32[:], float32[:], int32, float32[:,:])"
    )
    def _balanced_forman_curvature(A, A2, d_in, d_out, N, C):
        i, j = cuda.grid(2)

        if (i < N) and (j < N):
            if A[i, j] == 0:
                C[i, j] = 0
                return

            if d_in[i] > d_out[j]:
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                return

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = (
                (2 / d_max) + (2 / d_min) - 2 +
                (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)

    @staticmethod
    @cuda.jit(
        "void(float32[:,:], float32[:,:], float32, float32, int32, float32[:,:], int32, int32, int32[:], int32[:], int32, int32)"
    )
    def _balanced_forman_post_delta(
        A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
    ):
        I, J = cuda.grid(2)

        if (I < dim_i) and (J < dim_j):
            i = i_neighbors[I]
            j = j_neighbors[J]

            if (i == j) or (A[i, j] != 0):
                D[I, J] = -1000
                return

            # Difference in degree terms
            if j == x:
                d_in_x += 1
            elif i == y:
                d_out_y += 1

            if d_in_x * d_out_y == 0:
                D[I, J] = 0
                return

            if d_in_x > d_out_y:
                d_max = d_in_x
                d_min = d_out_y
            else:
                d_max = d_out_y
                d_min = d_in_x

            # Difference in triangles term
            A2_x_y = A2[x, y]
            if (x == i) and (A[j, y] != 0):
                A2_x_y += A[j, y]
            elif (y == j) and (A[x, i] != 0):
                A2_x_y += A[x, i]

            # Difference in four-cycles term
            sharp_ij = 0
            lambda_ij = 0
            for z in range(N):
                A_z_y = A[z, y] + 0
                A_x_z = A[x, z] + 0
                A2_z_y = A2[z, y] + 0
                A2_x_z = A2[x, z] + 0

                if (z == i) and (y == j):
                    A_z_y += 1
                if (x == i) and (z == j):
                    A_x_z += 1
                if (z == i) and (A[j, y] != 0):
                    A2_z_y += A[j, y]
                if (x == i) and (A[j, z] != 0):
                    A2_x_z += A[j, z]
                if (y == j) and (A[z, i] != 0):
                    A2_z_y += A[z, i]
                if (z == j) and (A[x, i] != 0):
                    A2_x_z += A[x, i]

                TMP = A_z_y * (A2_x_z - A_x_z) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A_x_z * (A2_z_y - A_z_y) * A[x, y]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            D[I, J] = (
                (2 / d_max) + (2 / d_min) - 2 +
                (2 / d_max + 1 / d_min) * A2_x_y * A[x, y]
            )
            if lambda_ij > 0:
                D[I, J] += sharp_ij / (d_max * lambda_ij)

    @staticmethod
    def balanced_forman_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
        N = A.shape[0]
        A2 = torch.matmul(A, A)
        d_in = A[:, x].sum()
        d_out = A[y].sum()
        if D is None:
            D = torch.zeros(len(i_neighbors), len(j_neighbors)).cuda()

        threadsperblock = (16, 16)
        blockspergrid_x = math.ceil(D.shape[0] / threadsperblock[0])
        blockspergrid_y = math.ceil(D.shape[1] / threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        Ricci._balanced_forman_post_delta[blockspergrid, threadsperblock](
            A,
            A2,
            d_in,
            d_out,
            N,
            D,
            x,
            y,
            np.array(i_neighbors),
            np.array(j_neighbors),
            D.shape[0],
            D.shape[1],
        )
        return D

    @staticmethod
    def softmax(a, tau=1):
        exp_a = np.exp(a * tau)
        return exp_a / exp_a.sum()
