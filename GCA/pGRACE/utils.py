import math
import numpy as np
import scipy.sparse as sp
from scipy.special import iv
from scipy.sparse.linalg import eigsh
import os.path as osp
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.manifold import SpectralEmbedding
# from libKMCUDA import kmeans_cuda
from tqdm import tqdm
from matplotlib import cm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
from torch.utils.data import random_split
from torch_geometric.nn import GCNConv, SGConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import sort_edge_index, degree, add_remaining_self_loops, remove_self_loops, get_laplacian, \
    to_undirected, to_dense_adj, to_networkx
from torch_scatter import scatter

import networkx as nx
import matplotlib.pyplot as plt
import random
import os


def get_base_model(name: str):
    def gat_wrapper(in_channels, out_channels):
        return GATConv(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            heads=4
        )

    def gin_wrapper(in_channels, out_channels):
        mlp = nn.Sequential(
            nn.Linear(in_channels, 2 * out_channels),
            nn.ELU(),
            nn.Linear(2 * out_channels, out_channels)
        )
        return GINConv(mlp)

    base_models = {
        'GCNConv': GCNConv,
        'SGConv': SGConv,
        'SAGEConv': SAGEConv,
        'GATConv': gat_wrapper,
        'GraphConv': GraphConv,
        'GINConv': gin_wrapper
    }

    return base_models[name]


def get_activation(name: str):
    activations = {
        'relu': F.relu,
        'hardtanh': F.hardtanh,
        'elu': F.elu,
        'leakyrelu': F.leaky_relu,
        'prelu': torch.nn.PReLU(),
        'rrelu': torch.nn.RReLU()
    }

    return activations[name]


def compute_pr(edge_index, damp: float = 0.85, k: int = 10):
    num_nodes = edge_index.max().item() + 1
    deg_out = degree(edge_index[0])
    x = torch.ones((num_nodes, )).to(edge_index.device).to(torch.float32)

    for i in range(k):
        edge_msg = x[edge_index[0]] / deg_out[edge_index[0]]
        agg_msg = scatter(edge_msg, edge_index[1], reduce='sum')

        x = (1 - damp) * x + damp * agg_msg

    return x


def eigenvector_centrality(data):
    graph = to_networkx(data)
    x = nx.eigenvector_centrality_numpy(graph)
    x = [x[i] for i in range(data.num_nodes)]
    return torch.tensor(x, dtype=torch.float32).to(data.edge_index.device)


def generate_split(num_samples: int, train_ratio: float, val_ratio: float):
    train_len = int(num_samples * train_ratio)
    val_len = int(num_samples * val_ratio)
    test_len = num_samples - train_len - val_len

    train_set, test_set, val_set = random_split(torch.arange(0, num_samples), (train_len, test_len, val_len))

    idx_train, idx_test, idx_val = train_set.indices, test_set.indices, val_set.indices
    train_mask = torch.zeros((num_samples,)).to(torch.bool)
    test_mask = torch.zeros((num_samples,)).to(torch.bool)
    val_mask = torch.zeros((num_samples,)).to(torch.bool)

    train_mask[idx_train] = True
    test_mask[idx_test] = True
    val_mask[idx_val] = True

    return train_mask, test_mask, val_mask

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_alpha_beta(l, u, alpha):
    alpha_L= torch.zeros(l.shape,device=l.device)
    alpha_U, beta_L, beta_U = torch.clone(alpha_L), torch.clone(alpha_L), torch.clone(alpha_L)
    pos_mask = l >= 0
    neg_mask = u <= 0
    alpha_L[pos_mask] = 1
    alpha_U[pos_mask] = 1
    alpha_L[neg_mask] = alpha
    alpha_U[neg_mask] = alpha
    not_mask = ~(pos_mask | neg_mask)
    alpha_not_upp = u[not_mask] - alpha * l[not_mask]
    alpha_not = alpha_not_upp / (u[not_mask] - l[not_mask])
    alpha_L[not_mask] = alpha_not
    alpha_U[not_mask] = alpha_not
    beta_U[not_mask] = (alpha - 1) * u[not_mask] * l[not_mask] / alpha_not_upp
    return alpha_L, alpha_U, beta_L, beta_U

def get_crown_weights(l1, u1, l2, u2, alpha, gcn_weights, Wcl):
    alpha_2_L, alpha_2_U, beta_2_L, beta_2_U = get_alpha_beta(l2, u2, alpha) # onehop
    alpha_1_L, alpha_1_U, beta_1_L, beta_1_U = get_alpha_beta(l1, u1, alpha) # twohop
    lambda_2 = torch.where(Wcl >= 0, alpha_2_L, alpha_2_U)
    Delta_2 = torch.where(Wcl >= 0, beta_2_L, beta_2_U)
    Lambda_2 = lambda_2 * Wcl
    W1_tensor, b1_tensor, W2_tensor, b2_tensor = gcn_weights
    W_tilde_2 = Lambda_2 @ W2_tensor.T
    b_tilde_2 = torch.diag(Lambda_2 @ (Delta_2 + b2_tensor).T)
    lambda_1 = torch.where(W_tilde_2 >= 0, alpha_1_L, alpha_1_U)
    Delta_1 = torch.where(W_tilde_2 >= 0, beta_1_L, beta_1_U)
    Lambda_1 = lambda_1 * W_tilde_2
    W_tilde_1 = Lambda_1 @ W1_tensor.T
    b_tilde_1 = torch.diag(Lambda_1 @ (Delta_1 + b1_tensor).T)
    return W_tilde_1, b_tilde_1, W_tilde_2, b_tilde_2

def get_batch(node_list, batch_size, epoch):
    num_nodes = len(node_list)
    num_batches = (num_nodes - 1) // batch_size + 1
    i = epoch % num_batches
    if (i + 1) * batch_size >= len(node_list):
        node_list_batch = node_list[i * batch_size:]
    else:
        node_list_batch = node_list[i * batch_size:(i + 1) * batch_size]
    return node_list_batch

def get_A_bounds(dataset, drop_rate):
    upper_lower_file = osp.join(osp.expanduser('~/datasets'),f"bounds/{dataset}_{drop_rate}_upper_lower.pkl")
    if osp.exists(upper_lower_file):
        A_upper, A_lower = torch.load(upper_lower_file)
    else:
        A_upper, A_lower = None, None
    return A_upper, A_lower