# This program is the pipeline for testing expressiveness.
# It includes 4 stages:
#   1. pre-calculation;
#   2. dataset construction;
#   3. model construction;
#   4. evaluation

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


import numpy as np
import torch
import torch_geometric
import torch_geometric.loader
from loguru import logger
import time
#from BRECDataset_v4 import BRECDataset
from BRECDataset_Wrapper import BRECDataset
from tqdm import tqdm
import os
from torch.nn import CosineEmbeddingLoss
import argparse

from torch import nn
import torch.nn.functional as F
from torch_geometric.data import DataLoader, Data
from torch_geometric.utils import degree
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GINConv, GINEConv, global_add_pool
import torch_geometric.transforms as T

from Xent_Loss import nt_bxent_loss
import dejavu_gi
from torch_geometric.utils import *
from torch_geometric.data import Batch

import pickle

from torch_scatter import scatter_add
from copy import deepcopy

import networkx as nx
from collections import Counter
import itertools

import algos

NUM_RELABEL = 32
P_NORM = 2
OUTPUT_DIM = 16
EPSILON_MATRIX = 1e-7
EPSILON_CMP = 1e-6
SAMPLE_NUM = 600
EPOCH = 50
MARGIN = 0.0
LEARNING_RATE = 1e-3
THRESHOLD = 120.12 # with 0.995 (0.995^10 > 0.95) original 72.34
BATCH_SIZE = 16
WEIGHT_DECAY = 1e-5
LOSS_THRESHOLD = 0.05
SEED = 2023

global_var = globals().copy()
HYPERPARAM_DICT = dict()
for k, v in global_var.items():
    if isinstance(v, int) or isinstance(v, float):
        HYPERPARAM_DICT[k] = v

# part_dict: {graph generation type, range}
part_dict = {
    "Basic": (0, 60),
    "Regular": (60, 160),
    "Extension": (160, 260),
    "CFI": (260, 360),
    "4-Vertex_Condition": (360, 380),
    "Distance_Regular": (380, 400),
    "CCoHG": (400, 500),
    "3r2r": (500, 600),
}
parser = argparse.ArgumentParser(description="BREC Test")

parser.add_argument("--P_NORM", type=str, default="2")
parser.add_argument("--EPOCH", type=int, default=EPOCH)
parser.add_argument("--LEARNING_RATE", type=float, default=LEARNING_RATE)
parser.add_argument("--BATCH_SIZE", type=int, default=BATCH_SIZE)
parser.add_argument("--WEIGHT_DECAY", type=float, default=WEIGHT_DECAY)
parser.add_argument("--OUTPUT_DIM", type=int, default=OUTPUT_DIM)
parser.add_argument("--SEED", type=int, default=SEED)
parser.add_argument("--THRESHOLD", type=float, default=THRESHOLD)
parser.add_argument("--MARGIN", type=float, default=MARGIN)
parser.add_argument("--LOSS_THRESHOLD", type=float, default=LOSS_THRESHOLD)
parser.add_argument("--device", type=int, default=0)
parser.add_argument(
    "--augmentation",
    type=str,
    default="none",
    help="Options are ['none', 'ports', 'ids', 'random', 'dropout', 'PSE']",
)
parser.add_argument(
    "--random",
    type=str,
    default="gaussian",
    help="Options are ['gaussian', 'RNI', 'binary', 'IRNI']",#TODO
)
parser.add_argument(
    "--pse",
    type=str,
    default="RWSE",
    help="Options are ['RWSE', 'ElstaticPE', 'HKdiagSE', 'CycleSE', 'LapPE', 'RLapPE']",#TODO
)
parser.add_argument(
    "--loss",
    type=str,
    default="CosineEmbeddingLoss",
    help="Options are ['CosineEmbeddingLoss', 'nt_bxent_loss']",#TODO
)
parser.add_argument("--loss_parameter", type=float, default=1)
parser.add_argument(
    '--parts',
    nargs='+',
    default=[],#list(part_dict.keys()),
    help='Options are a subset of '+str(part_dict.keys())
)
parser.add_argument("--name_tag", type=str, default=None)
parser.add_argument("--prob", type=int, default=-1)
parser.add_argument("--num_runs", type=int, default=50)
parser.add_argument(
    "--num_layers", type=int, default=4
)  # 9 layers were used for skipcircles dataset
parser.add_argument("--use_aux_loss", action="store_true", default=False, help='Not Supported Now!')
parser.add_argument("--hidden_units", type=int, default=32)
parser.add_argument("--added_dimensions", type=int, default=1)
parser.add_argument("--logging", type=str, default="default.log")
parser.add_argument("--root", type=str, default=".")
# General settings.
args = parser.parse_args()

P_NORM = 2 if args.P_NORM == "2" else torch.inf
EPOCH = args.EPOCH
LEARNING_RATE = args.LEARNING_RATE
BATCH_SIZE = args.BATCH_SIZE
WEIGHT_DECAY = args.WEIGHT_DECAY
OUTPUT_DIM = args.OUTPUT_DIM
SEED = args.SEED
THRESHOLD = args.THRESHOLD
MARGIN = args.MARGIN
LOSS_THRESHOLD = args.LOSS_THRESHOLD
torch_geometric.seed_everything(SEED)
torch.backends.cudnn.deterministic = True
# torch.use_deterministic_algorithms(True)
if args.logging == "default.log" and args.name_tag is not None:
    args.logging = f"{args.name_tag}.log"


def mash(input):
    output = torch.sum(input*torch.tensor([2 ** (input.shape[1]-1-i) for i in range(input.shape[1])], device=input.device), dim=1, dtype=torch.int64)
    return output


# Stage 1: pre calculation
# Here is for some calculation without data. e.g. generating all the k-substructures
def pre_calculation(*args, **kwargs):
    time_start = time.process_time()

    # Do something

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"pre-calculation time cost: {time_cost}")


# Stage 2: dataset construction
# Here is for dataset construction, including data processing
def get_dataset(name, device):
    time_start = time.process_time()

    # Do something
    def makefeatures(data):
        if data.x is None:
            data.x = torch.ones((data.num_nodes, 1))
        data.id = torch.tensor(
            np.random.permutation(np.arange(data.num_nodes))
        ).unsqueeze(1)
        return data

    ksteps = list(range(2, 22))
    def RWSE(data):
        # from get_rw_landing_probs in GPSE/graphym/transform/posenc_stats.py
        space_dim = 0
        #if edge_weight is None:
        edge_weight = torch.ones(data.edge_index.size(1), device=data.edge_index.device)
        num_nodes = data.num_nodes
        source, dest = data.edge_index[0], data.edge_index[1]
        deg = scatter_add(edge_weight, source, dim=0, dim_size=num_nodes)  # Out degrees.
        deg_inv = deg.pow(-1.)
        deg_inv.masked_fill_(deg_inv == float('inf'), 0)

        if data.edge_index.numel() == 0:
            P = data.edge_index.new_zeros((1, num_nodes, num_nodes))
        else:
            # P = D^-1 * A
            P = torch.diag(deg_inv) @ to_dense_adj(data.edge_index,
                                                   max_num_nodes=num_nodes)  # 1 x (Num nodes) x (Num nodes)
        rws = []
        if ksteps == list(range(min(ksteps), max(ksteps) + 1)):
            # Efficient way if ksteps are a consecutive sequence (most of the time the case)
            Pk = P.clone().detach().matrix_power(min(ksteps))
            for k in range(min(ksteps), max(ksteps) + 1):
                rws.append(torch.diagonal(Pk, dim1=-2, dim2=-1) * \
                           (k ** (space_dim / 2)))
                Pk = Pk @ P
        else:
            # Explicitly raising P to power k for each k \in ksteps.
            for k in ksteps:
                rws.append(torch.diagonal(P.matrix_power(k), dim1=-2, dim2=-1) * \
                           (k ** (space_dim / 2)))
        rw_landing = torch.cat(rws, dim=0).transpose(0, 1)  # (Num nodes) x (K steps)
        data.x = torch.cat([data.x, rw_landing], dim=1)
        return data

    EPS = 1e-6
    def ElstaticPE(data):
        # from  in GPSE/graphym/transform/posenc_stats.py
        L = to_scipy_sparse_matrix(
            *get_laplacian(data.edge_index, normalization=None, num_nodes=data.num_nodes)
        ).todense()
        L = torch.as_tensor(L)
        tmp = (L.diag() ** -1)
        tmp = torch.where(tmp < torch.inf, tmp, 0)
        Dinv = torch.eye(L.shape[0]) * tmp
        A = deepcopy(L).abs()
        A.fill_diagonal_(0)
        DinvA = Dinv.matmul(A)

        evals, evecs = torch.linalg.eigh(L)
        offset = (evals < EPS).sum().item()
        if offset == data.num_nodes:
            return torch.zeros(data.num_nodes, 7, dtype=torch.float32)

        electrostatic = evecs[:, offset:] / evals[offset:] @ evecs[:, offset:].T
        electrostatic = electrostatic - electrostatic.diag()
        green_encoding = torch.stack([
            electrostatic.min(dim=0)[0],  # Min of Vi -> j
            electrostatic.mean(dim=0),  # Mean of Vi -> j
            electrostatic.std(dim=0),  # Std of Vi -> j
            electrostatic.min(dim=1)[0],  # Min of Vj -> i
            electrostatic.std(dim=1),  # Std of Vj -> i
            (DinvA * electrostatic).sum(dim=0),  # Mean of interaction on direct neighbour
            (DinvA * electrostatic).sum(dim=1),  # Mean of interaction from direct neighbour
        ], dim=1)
        data.x = torch.cat([data.x, green_encoding], dim=1)
        return data

    kernel_times = list(range(1, 21))
    def HKdiagSE(data):
        # from get_heat_kernels_diag in GPSE/graphym/transform/posenc_stats.py

        L_heat = to_scipy_sparse_matrix(
            *get_laplacian(data.edge_index, normalization=None, num_nodes=data.num_nodes)
        )
        evals_heat, evects_heat = np.linalg.eigh(L_heat.toarray())
        evals = torch.from_numpy(evals_heat)
        evects = torch.from_numpy(evects_heat)
        heat_kernels_diag = []
        if len(kernel_times) > 0:
            evects = F.normalize(evects, p=2., dim=0)

            # Remove eigenvalues == 0 from the computation of the heat kernel
            idx_remove = evals < 1e-8
            evals = evals[~idx_remove]
            evects = evects[:, ~idx_remove]

            # Change the shapes for the computations
            evals = evals.unsqueeze(-1)  # lambda_{i, ..., ...}
            evects = evects.transpose(0, 1)  # phi_{i,j}: i-th eigvec X j-th node

            # Compute the heat kernels diagonal only for each time
            eigvec_mul = evects ** 2
            for t in kernel_times:
                # sum_{i>0}(exp(-2 t lambda_i) * phi_{i, j} * phi_{i, j})
                this_kernel = torch.sum(torch.exp(-t * evals) * eigvec_mul,
                                        dim=0, keepdim=False)

                # Multiply by `t` to stabilize the values, since the gaussian height
                # is proportional to `1/t`
                heat_kernels_diag.append(this_kernel * (t ** (0 / 2)))
            heat_kernels_diag = torch.stack(heat_kernels_diag, dim=0).transpose(0, 1)
        data.x = torch.cat([data.x, heat_kernels_diag], dim=1)
        return data

    k_list = list(range(2, 6))#9
    def CycleSE(data):
        graph = to_networkx(data)
        cycles = list(nx.simple_cycles(graph, length_bound=max(k_list)))

        x = torch.zeros((data.x.size()[0], len(k_list)))
        #for cycle in cycles:
        #    size = len(cycle)
        #    if size in k_list:
        #        for node in cycle:
        #            x[node, k_list.index(size)] += 1
        # For efficiency
        cycles.sort(key=lambda c: len(c))
        cycles_len = len(cycles)
        for i, k in enumerate(k_list):
            min_i = next((x for x in range(cycles_len) if len(cycles[x])==k), 0)
            max_i = next((x for x in reversed(range(cycles_len)) if len(cycles[x])==k), 0)
            for node, count in Counter(itertools.chain.from_iterable(cycles[min_i:max_i+1])).items():
                x[node, i] = count/2

        #x[ : , 1: ] /= 2
        data.x = torch.cat([data.x, x], dim=1)
        return data


    frequencies = 7
    def LapPE(data):
        # from  in GPSE/graphym/transform/posenc_stats.py
        L = to_scipy_sparse_matrix(
            *get_laplacian(data.edge_index, normalization=None, num_nodes=data.num_nodes)
        )
        EigVal, EigVec = np.linalg.eigh(L.toarray())
        EigVec = EigVec[:, EigVal.argsort()]  # increasing order
        pos_enc = torch.from_numpy(EigVec[:, :frequencies]).float()
        if pos_enc.size()[1] != frequencies:
            print(EigVec)
            print(pos_enc.size(), data.x.size())
            print(data.edge_index)
        data.x = torch.cat([data.x, pos_enc], dim=1)
        return data


    minlength = 10
    def SPDPE(data):
        dense_adj = torch.squeeze(to_dense_adj(data.edge_index, max_num_nodes=data.num_nodes).type(torch.int))
        shortest_path_result, path = algos.floyd_warshall(dense_adj.numpy())
        node_to_node_shortest_paths = torch.from_numpy(shortest_path_result).long()
        spatial_pos = torch.stack([torch.bincount(_node_to_node_shortest_paths, minlength=minlength)[:minlength]
                                   for _node_to_node_shortest_paths in node_to_node_shortest_paths])
        data.x = torch.cat([data.x, spatial_pos], dim=1)
        return data


    maxlength = 10
    def RDPE(data):
        N = data.num_nodes
        adj = np.zeros((N, N), dtype=np.float32)
        adj[data.edge_index[0, :], data.edge_index[1, :]] = 1.0

        # 2) connected_components
        g = nx.Graph(adj)
        g_components_list = [g.subgraph(c).copy() for c in nx.connected_components(g)]
        g_resistance_matrix = np.zeros((N, N)) - 1.0
        g_index = 0
        for item in g_components_list:
            cur_adj = nx.to_numpy_array(item)
            cur_num_nodes = cur_adj.shape[0]
            cur_res_dis = np.linalg.pinv(
                np.diag(cur_adj.sum(axis=-1)) - cur_adj + np.ones((cur_num_nodes, cur_num_nodes),
                                                                  dtype=np.float32) / cur_num_nodes
            ).astype(np.float32)
            A = np.diag(cur_res_dis)[:, None]
            B = np.diag(cur_res_dis)[None, :]
            cur_res_dis = A + B - 2 * cur_res_dis
            g_resistance_matrix[g_index:g_index + cur_num_nodes, g_index:g_index + cur_num_nodes] = cur_res_dis
            g_index += cur_num_nodes
        g_cur_index = []
        for item in g_components_list:
            g_cur_index.extend(list(item.nodes))
        ori_idx = np.arange(N)
        g_resistance_matrix[g_cur_index, :] = g_resistance_matrix[ori_idx, :]
        g_resistance_matrix[:, g_cur_index] = g_resistance_matrix[:, ori_idx]

        if g_resistance_matrix.max() > N - 1:
            print(f'error: {g_resistance_matrix}')
        g_resistance_matrix[g_resistance_matrix == -1.0] = 512.0
        res_matrix = np.zeros((N, maxlength), dtype=np.float32)
        l = min(maxlength, N)
        res_matrix[:, :l] = g_resistance_matrix[:, :l]
        res_matrix = torch.from_numpy(res_matrix)
        res_matrix, _ = torch.sort(res_matrix, descending=True)
        data.x = torch.cat([data.x, res_matrix], dim=1)
        return data

    def addports(data):
        data.ports = torch.zeros(data.num_edges, 1)
        degs = degree(
            data.edge_index[0], data.num_nodes, dtype=torch.long
        )  # out degree of all nodes
        for n in range(data.num_nodes):
            deg = degs[n]
            ports = np.random.permutation(int(deg))
            for i, neighbor in enumerate(data.edge_index[1][data.edge_index[0] == n]):
                nb = int(neighbor)
                data.ports[
                    torch.logical_and(
                        data.edge_index[0] == n, data.edge_index[1] == nb
                    ),
                    0,
                ] = float(ports[i])
        return data

    if args.augmentation == 'PSE':
        name = args.pse
        if args.pse == "RWSE":
            pre_transform = T.Compose([makefeatures, addports, RWSE])
            args.added_dimensions = len(ksteps)
        if args.pse == "ElstaticPE":
            pre_transform = T.Compose([makefeatures, addports, ElstaticPE])
            args.added_dimensions = 7
        if args.pse == "HKdiagSE":
            pre_transform = T.Compose([makefeatures, addports, HKdiagSE])
            args.added_dimensions = len(kernel_times)
        if args.pse == "CycleSE":
            pre_transform = T.Compose([makefeatures, addports, CycleSE])
            args.added_dimensions = len(k_list)
        if args.pse == "LapPE" or args.pse == "RLapPE":
            pre_transform = T.Compose([makefeatures, addports, LapPE])
            args.added_dimensions = frequencies
        if args.pse == "SPDPE":#RDPE
            pre_transform = T.Compose([makefeatures, addports, SPDPE])
            args.added_dimensions = minlength
        if args.pse == "RDPE":
            pre_transform = T.Compose([makefeatures, addports, RDPE])
            args.added_dimensions = 0
    else:
        pre_transform = T.Compose([makefeatures, addports])

    dataset = BRECDataset(name=name, pre_transform=pre_transform)
    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"dataset construction time cost: {time_cost}")

    return dataset


# Stage 3: model construction
# Here is for model construction.
def get_model(args, num_nodes, num_features, device):
    time_start = time.process_time()
    # Do something

    n = num_nodes
    gamma = n
    p_opt = 2 * 1 / (1 + gamma)
    if args.prob >= 0:
        p = args.prob
    else:
        p = p_opt
    if args.num_runs > 0:
        num_runs = args.num_runs
    else:
        num_runs = gamma

    graph_classification = True
    num_features = num_features
    Conv = GINConv
    if args.augmentation == "ports":
        Conv = GINEConv
    elif args.augmentation == "ids":
        num_features += 1
    elif args.augmentation == "random" or args.augmentation == "PSE":
        num_features += args.added_dimensions
    use_aux_loss = args.use_aux_loss

    class GIN(nn.Module):
        def __init__(self):
            super(GIN, self).__init__()

            dim = args.hidden_units

            self.num_layers = args.num_layers

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(
                Conv(
                    nn.Sequential(
                        nn.Linear(num_features, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Linear(dim, dim),
                    ), train_eps=True
                )
            )
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, OUTPUT_DIM))
            self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

            for i in range(self.num_layers - 1):
                self.convs.append(
                    Conv(
                        nn.Sequential(
                            nn.Linear(dim, dim),
                            nn.BatchNorm1d(dim),
                            nn.ReLU(),
                            nn.Linear(dim, dim),
                        ), train_eps=True
                    )
                )
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()

            if args.augmentation == "ids":
                x = torch.cat([x, data.id.float()], dim=1)
            elif args.augmentation == "random":
                if args.random == "gaussian":
                    x = torch.cat(
                        [x, torch.rand((x.size(0), x.size(1), args.added_dimensions), device=x.device)],
                        dim=2,
                    )
                if args.random == "RNI":
                    x = torch.cat(
                        [x, torch.randint(0, 100, (x.size(0), x.size(1), args.added_dimensions), device=x.device) / 100.0],#torch.randint(0, 2, (x.size(0), x.size(1), 1), device=x.device)],#
                        dim=2,
                    )
                if args.random == "binary": #TODO change this to be more structured and less random
                    x = torch.cat(
                        [x, torch.randint(0, 2, (x.size(0), x.size(1), args.added_dimensions), device=x.device)],
                        dim=2,
                    )
                if args.random == 'IRNI':
                    output = []
                    for datum in data.to_data_list():
                        output.append([])
                        colors = mash(datum.x)  # [mash0(x) for x in data.x]
                        #if self.edge_labels:
                        #    edge_colors = mash(data.edge_attr)  # [mash0(x) for x in data.edge_attr]
                        #else:
                        #    edge_colors = []
                        o = torch.full((num_runs, datum.num_nodes, 1), 0, dtype=datum.x.dtype, device=datum.x.device)
                        for _ in range(num_runs):
                            try:
                                test = dejavu_gi.random_ir_paths(datum.num_nodes, datum.edge_index.T.tolist(), args.added_dimensions,
                                                                 vertex_labels=colors, edge_labels=[],
                                                                 fill_paths=True, directed_dimacs=True)
                            except OSError as e:
                                import traceback
                                traceback.print_exc()
                                print(datum.num_nodes, datum.edge_index.T.tolist(), self.depth, colors)#, edge_colors)
                            k = 0
                            for node in test[0]['base_points']:
                                o[_, node, k] = 1
                                k += 1
                            output[-1].append(torch.cat([datum.x, o[_]], dim=-1))#.to(datum.x.dtype).to(datum.x.device)
                        #output[-1] = torch.stack(output[-1])
                    #print(type(output[0][0]))
                    output = [torch.cat([graph[_] for graph in output]) for _ in range(num_runs)]#[Batch.from_data_list([graph[_] for graph in output]) for _ in range(num_runs)]
                    output = torch.stack(output)
                    x = output
                if args.random == 'tinhofer':
                    x1 = mash(data.x)
                    edge_index = torch_geometric.utils.to_undirected(data.edge_index)
                    edge_index = sort_edge_index(edge_index, num_nodes=data.x.size(0), sort_by_row=False)
                    row, col = edge_index[0], edge_index[1]
                    deg = degree(col, data.x.size(0), dtype=torch.long).tolist()

                    color_classes = None
                    if self.k_weak == 0: color_classes = x1.clone()

                    # break symmetry in orbits of size > 1
                    while True:
                        # color refinement
                        for i_cr in range(1, min(16, x1.shape[0])):
                            out = []
                            for node, neighbors in zip(x1.tolist(), x1[row].split(deg)):
                                hashx = hash(tuple([node] + neighbors.sort()[0].tolist()))
                                out.append(hashx)
                            x1 = torch.tensor(out, device=x1.device)

                            if color_classes == None and (i_cr == self.k_weak): color_classes = x1.clone()
                        if color_classes == None: color_classes = x1.clone()  # for smaller graphs
                        # end color refinement

                        uniq, inv, counts = torch.unique(x1, return_inverse=True, return_counts=True)
                        orbit_size = counts[inv]
                        orbits2_pos = torch.nonzero(orbit_size > 1, as_tuple=True)[0]

                        if orbits2_pos.shape[0] > 0:
                            xs = x1[orbits2_pos]
                            idx = torch.argmin(xs)
                            x1[orbits2_pos[idx]] += 1  # hopefully this is unique
                        else:  # all of size 1
                            break

                    x_out = torch.zeros((x1.shape[0], self.output_dim), device=x1.device)
                    tmp = torch.zeros((x1.shape[0]), dtype=int, device=x1.device)
                    mask = 2 ** torch.arange(self.output_dim - 1, -1, -1).to(x_out.device, int)

                    uniq, counts = torch.unique(color_classes, return_counts=True)
                    for color in uniq:
                        nodes = torch.nonzero(color_classes == color, as_tuple=True)[0]
                        ind_colors = x1[nodes]
                        order = torch.argsort(torch.argsort(ind_colors))
                        order = order % (2 ** self.output_dim)
                        x_out[nodes] = order.unsqueeze(-1).bitwise_and(mask).ne(0).float()
                        tmp[nodes] = order

                    return x_out
            elif args.augmentation == "PSE":
                if args.pse == 'RLapPE':
                    x_shuffle = x[:, :, -args.added_dimensions:]
                    count = torch.bincount(data.batch)
                    x_shuffle = x_shuffle[:,
                                torch.cat([torch.randperm(i, device=x.device)+j
                                           for i, j in zip(count, torch.cat([torch.zeros(1, device=x.device,
                                                                                         dtype=torch.int),
                                                                             torch.cumsum(count, 0)[:-1]]))]),
                                :]
                    signflips = 2*torch.randint(2, (x.size(0), x.size(1), 1), device=x.device)-1
                    x_shuffle = x_shuffle*signflips
                    x_rest = x[:, :, :-args.added_dimensions]
                    x = torch.cat(
                        [x_shuffle, x_rest],
                        dim=2,
                    )

            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(
                num_runs, device=edge_index.device
            ).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):
                if args.augmentation == "ports":
                    x = self.convs[i](x, run_edge_index, data.ports.expand(-1, x.size(-1)))
                else:
                    x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del run_edge_index
            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x)  # No dropout layer in these experiments
                if out is None:
                    out = x
                else:
                    out += x
            return out
            # return F.log_softmax(out, dim=-1), 0

    class DropGIN(nn.Module):
        def __init__(self):
            super(DropGIN, self).__init__()

            dim = args.hidden_units

            self.num_layers = args.num_layers

            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()
            self.fcs = nn.ModuleList()

            self.convs.append(
                Conv(
                    nn.Sequential(
                        nn.Linear(num_features, dim),
                        nn.BatchNorm1d(dim),
                        nn.ReLU(),
                        nn.Linear(dim, dim),
                    )
                )
            )
            self.bns.append(nn.BatchNorm1d(dim))
            self.fcs.append(nn.Linear(num_features, OUTPUT_DIM))
            self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

            for i in range(self.num_layers - 1):
                self.convs.append(
                    Conv(
                        nn.Sequential(
                            nn.Linear(dim, dim),
                            nn.BatchNorm1d(dim),
                            nn.ReLU(),
                            nn.Linear(dim, dim),
                        )
                    )
                )
                self.bns.append(nn.BatchNorm1d(dim))
                self.fcs.append(nn.Linear(dim, OUTPUT_DIM))

            if use_aux_loss:
                self.aux_fcs = nn.ModuleList()
                self.aux_fcs.append(nn.Linear(num_features, OUTPUT_DIM))
                for i in range(self.num_layers):
                    self.aux_fcs.append(nn.Linear(dim, OUTPUT_DIM))

        def reset_parameters(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    m.reset_parameters()
                elif isinstance(m, Conv):
                    m.reset_parameters()
                elif isinstance(m, nn.BatchNorm1d):
                    m.reset_parameters()

        def forward(self, data):
            x = data.x
            edge_index = data.edge_index
            batch = data.batch

            # Do runs in paralel, by repeating the graphs in the batch
            x = x.unsqueeze(0).expand(num_runs, -1, -1).clone()
            drop = torch.bernoulli(
                torch.ones([x.size(0), x.size(1)], device=x.device) * p
            ).bool()
            x[drop] = 0.0
            del drop
            outs = [x]
            x = x.view(-1, x.size(-1))
            run_edge_index = edge_index.repeat(1, num_runs) + torch.arange(
                num_runs, device=edge_index.device
            ).repeat_interleave(edge_index.size(1)) * (edge_index.max() + 1)
            for i in range(self.num_layers):
                x = self.convs[i](x, run_edge_index)
                x = self.bns[i](x)
                x = F.relu(x)
                outs.append(x.view(num_runs, -1, x.size(-1)))
            del run_edge_index

            out = None
            for i, x in enumerate(outs):
                x = x.mean(dim=0)
                if graph_classification:
                    x = global_add_pool(x, batch)
                x = self.fcs[i](x)  # No dropout layer in these experiments
                if out is None:
                    out = x
                else:
                    out += x

            if use_aux_loss:
                aux_out = torch.zeros(
                    num_runs, out.size(0), out.size(1), device=out.device
                )
                run_batch = batch.repeat(num_runs) + torch.arange(
                    num_runs, device=edge_index.device
                ).repeat_interleave(batch.size(0)) * (batch.max() + 1)
                for i, x in enumerate(outs):
                    if graph_classification:
                        x = x.view(-1, x.size(-1))
                        x = global_add_pool(x, run_batch)
                    x = x.view(num_runs, -1, x.size(-1))
                    x = self.aux_fcs[i](x)  # No dropout layer in these experiments
                    aux_out += x

                return out, aux_out
                # return F.log_softmax(out, dim=-1), F.log_softmax(aux_out, dim=-1)
            else:
                return out
                # return F.log_softmax(out, dim=-1), 0

    if args.augmentation == "dropout":
        model = DropGIN().to(device)
    else:
        model = GIN().to(device)
        use_aux_loss = False

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"model construction time cost: {time_cost}")
    return model


# Stage 4: evaluation
# Here is for evaluation.
def evaluation(dataset, device, args):
    """
    When testing on BREC, even on the same graph, the output embedding may be different,
    because numerical precision problem occur on large graphs, and even the same graph is permuted.
    However, if you want to test on some simple graphs without permutation outputting the exact same embedding,
    some modification is needed to avoid computing the inverse matrix of a zero matrix.
    """
    # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use S_epsilon.
    # S_epsilon = torch.diag(
    #     torch.full(size=(OUTPUT_DIM, 1), fill_value=EPSILON_MATRIX).reshape(-1)
    # ).to(device)
    def T2_calculation(dataset, log_flag=False):
        with torch.no_grad():
            loader = torch_geometric.loader.DataLoader(dataset, batch_size=BATCH_SIZE)
            pred_0_list = []
            pred_1_list = []
            for data in loader:
                #torch.set_printoptions(threshold=10_000)
                #print(data.x)
                pred = model(data.to(device)).detach()
                #print(model.state_dict())
                pred_0_list.extend(pred[0::2])
                pred_1_list.extend(pred[1::2])
                #print(pred_0_list, pred_1_list)
            X = torch.cat([x.reshape(1, -1) for x in pred_0_list], dim=0).T
            Y = torch.cat([x.reshape(1, -1) for x in pred_1_list], dim=0).T
            #big = torch.max(torch.max(torch.abs(X)), torch.max(torch.abs(Y)))
            #X/=big
            #Y/=big
            D = X - Y
            if log_flag:
                logger.info(f"X = {X}")
                logger.info(f"X_mean = {torch.mean(X, dim=1)}")
                logger.info(f"Y = {Y}")
                logger.info(f"Y_mean = {torch.mean(Y, dim=1)}")
                logger.info(f"D = {D}")
            D = torch.where(torch.abs(D) < torch.abs(X)/1000, 0, D) # Avoids floating point subtraction errors for similar embeddings
            D_mean = torch.mean(D, dim=1).reshape(-1, 1)
            S = torch.cov(D)
            inv_S = torch.linalg.pinv(S)
            # If you want to test on some simple graphs without permutation outputting the exact same embedding, please use inv_S with S_epsilon.
            # inv_S = torch.linalg.pinv(S + S_epsilon)
            # print(D, D_mean, S, inv_S)
            result = NUM_RELABEL*torch.mm(torch.mm(D_mean.T, inv_S), D_mean)
            if log_flag:
                logger.info(f"result = {result}")
            return result

    time_start = time.process_time()

    # Do something
    #num_nodes_list = np.load('num_node.npy', allow_pickle=True)
    cnt = 0
    correct_list = []
    fail_in_reliability = 0
    loss_func = CosineEmbeddingLoss(margin=MARGIN)
    store = []

    ids = [] #[89, 112, 113, 114, 115, 116, 118, 119, 120, 121, 123, 124, 128, 129, 131, 134, 138,
           #139, 140, 142, 143, 149, 151, 152, 153, 154, 157, 158, 64, 111, 125, 127, 132, 133,
           #136, 137, 141, 144, 146, 150, 159, 117, 145, 148, 155, 156, 126, 130, 135, 147, 122]

    file = f"{args.root}/{args.random}_{args.loss}_{str(args.loss_parameter)}.pkl"
    if args.name_tag is not None:
        file = f"{args.root}/{args.name_tag}.pkl"

    for part_name in args.parts: #for part_name, part_range in part_dict.items():
        part_range = part_dict[part_name]
        logger.info(f"{part_name} part starting ---")

        cnt_part = 0
        fail_in_reliability_part = 0
        start = time.process_time()

        for id in tqdm(range(part_range[0], part_range[1])):
            logger.info(f"ID: {id}")
            for test_count in range(10):
                dataset_traintest = dataset[
                    id * NUM_RELABEL * 2 : (id + 1) * NUM_RELABEL * 2
                ]
                model = get_model(args, dataset_traintest[0].num_nodes, 1, device)  # num_nodes_list[id]
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer)  # StepLR(optimizer, gamma=0.5, step_size=250)
                dataset_reliability = dataset[
                    (id + SAMPLE_NUM)
                    * NUM_RELABEL
                    * 2 : (id + SAMPLE_NUM + 1)
                    * NUM_RELABEL
                    * 2
                ]
                model.train()
                for _ in range(EPOCH):
                    traintest_loader = torch_geometric.loader.DataLoader(
                        dataset_traintest, batch_size=BATCH_SIZE
                    )
                    loss_all = 0
                    for data in traintest_loader:
                        if id in ids:
                            import networkx as nx
                            import matplotlib.pyplot as plt
                            print(id)
                            ids.remove(id)
                            for datum in data.to_data_list():
                                g = torch_geometric.utils.to_networkx(datum, to_undirected=True)
                                nx.draw(g)
                                plt.show()

                        optimizer.zero_grad()
                        #print(data.x)
                        pred = model(data.to(device))
                        #print(pred)
                        apart = loss_func(
                            pred[0::2],
                            pred[1::2],
                            torch.tensor([-1] * (len(pred) // 2)).to(device),
                        )
                        together = loss_func(
                            torch.cat((pred[0::4], pred[1::4])),
                            torch.cat((pred[2::4], pred[3::4])),
                            torch.tensor([1] * (len(pred) // 2)).to(device),
                        )
                        # print(apart, together)
                        a = args.loss_parameter
                        if args.loss == "CosineEmbeddingLoss":
                            loss = a*apart+(1-a)*together
                        elif args.loss == "nt_bxent_loss":
                            loss = nt_bxent_loss(pred,
                                torch.tensor([(2*a, 2*b) for a in range(len(pred)//2) for b in range(len(pred)//2)] +
                                [(2*a+1, 2*b+1) for a in range(len(pred)//2) for b in range(len(pred)//2)]).to(device),
                                                 0.5, device)
                        loss.backward()
                        optimizer.step()
                        loss_all += len(pred) / 2 * loss.item()
                    #if _%5 == 0:
                    #    print(loss_all)  # TODO remove
                    #for name, param in model.named_parameters():
                    #    if param.requires_grad:
                    #        print(name, param.data)
                    #        break
                    loss_all /= NUM_RELABEL
                    logger.info(f"Loss: {loss_all}")
                    if loss_all < LOSS_THRESHOLD:
                        logger.info("Early Stop Here")
                        break
                    scheduler.step(loss_all)

                model.eval()
                T_square_traintest = T2_calculation(dataset_traintest, True)
                T_square_reliability = T2_calculation(dataset_reliability, True)
                print(T_square_traintest, T_square_reliability)

                if (T_square_traintest > THRESHOLD and T_square_reliability < THRESHOLD
                        and not torch.isclose(T_square_traintest, T_square_reliability, atol=EPSILON_CMP)):
                    break

            isomorphic_flag = False
            reliability_flag = False
            if T_square_traintest > THRESHOLD and not torch.isclose(
                T_square_traintest, T_square_reliability, atol=EPSILON_CMP
            ):
                isomorphic_flag = True
            if T_square_reliability < THRESHOLD:
                reliability_flag = True

            if isomorphic_flag:
                cnt += 1
                cnt_part += 1
                correct_list.append(id)
                logger.info(f"Correct num in current part: {cnt_part}")
            if not reliability_flag:
                fail_in_reliability += 1
                fail_in_reliability_part += 1
            logger.info(f"isomorphic: {isomorphic_flag} {T_square_traintest}")
            logger.info(f"reliability: {reliability_flag} {T_square_reliability}")
            #print(isomorphic_flag, reliability_flag, T_square_traintest, T_square_reliability)

            #save to file here
            store.append((part_name, id, isomorphic_flag, T_square_traintest, reliability_flag, T_square_reliability, test_count))

        end = time.process_time()
        time_cost_part = round(end - start, 2)

        logger.info(
            f"{part_name} part costs time {time_cost_part}; Correct in {cnt_part} / {part_range[1] - part_range[0]}"
        )
        logger.info(
            f"Fail in reliability: {fail_in_reliability_part} / {part_range[1] - part_range[0]}"
        )

        store.append(args)
        with open(file, 'ab') as f:
            pickle.dump(store, f)
        store = []

    time_end = time.process_time()
    time_cost = round(time_end - time_start, 2)
    logger.info(f"evaluation time cost: {time_cost}")

    Acc = round(cnt / SAMPLE_NUM, 2)
    logger.info(f"Correct in {cnt} / {SAMPLE_NUM}, Acc = {Acc}")

    logger.info(f"Fail in reliability: {fail_in_reliability} / {SAMPLE_NUM}")
    logger.info(correct_list)

    logger.add(f"{args.root}/{args.name_tag}_show.log", format="{message}", encoding="utf-8")
    logger.info(
        "Real_correct\tCorrect\tFail\tnum_layers\thidden_units\tnum_runs\tOUTPUT_DIM\tBATCH_SIZE\tLEARNING_RATE\tWEIGHT_DECAY\tSEED"
    )
    logger.info(
        f"{cnt-fail_in_reliability}\t{cnt}\t{fail_in_reliability}\t{args.num_layers}\t{args.hidden_units}\t{args.num_runs}\t{OUTPUT_DIM}\t{BATCH_SIZE}\t{LEARNING_RATE}\t{WEIGHT_DECAY}\t{SEED}"
    )

def main():
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    logger.remove(handler_id=None)
    logger.add(f"{args.root}/{args.logging}")
    logger.info(args)

    pre_calculation()
    dataset = get_dataset(name="no_param", device=device)
    # model = get_model(args, device)
    evaluation(dataset, device, args)

if __name__ == "__main__":
    main()
