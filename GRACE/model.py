from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.sparse as tsp
from torch.sparse import mm

from torch_geometric.utils import degree, to_undirected
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj
from torch_geometric.nn.conv.gcn_conv import gcn_norm
import numpy as np
import scipy.sparse as sp

import os.path as osp
from time import perf_counter as t
from my_utils import get_alpha_beta, get_crown_weights

class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret


class Encoder(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation,
                 base_model=GCNConv, k: int = 2):
        super(Encoder, self).__init__()
        self.base_model = base_model

        assert k >= 2
        self.k = k
        self.conv = [base_model(in_channels, 2 * out_channels)]
        for _ in range(1, k-1):
            self.conv.append(base_model(2 * out_channels, 2 * out_channels))
        self.conv.append(base_model(2 * out_channels, out_channels))
        self.conv = nn.ModuleList(self.conv)

        self.activation = activation

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index))
        return x


class Model(torch.nn.Module):
    def __init__(self, encoder: Encoder, num_hidden: int, num_proj_hidden: int,
                 tau: float = 0.5, dataset: str = "Cora"):
        super(Model, self).__init__()
        self.encoder: Encoder = encoder
        self.tau: float = tau
        self.dataset = dataset

        self.fc1 = torch.nn.Linear(num_hidden, num_proj_hidden)
        self.fc2 = torch.nn.Linear(num_proj_hidden, num_hidden)
        self.pot_loss_func = nn.BCEWithLogitsLoss()
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, edge_index)

    def projection(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor):
        f = lambda x: torch.exp(x / self.tau)
        refl_sim = f(self.sim(z1, z1))
        between_sim = f(self.sim(z1, z2))

        return -torch.log(
            between_sim.diag()
            / (refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()))

    def batched_semi_loss(self, z1: torch.Tensor, z2: torch.Tensor,
                          batch_size: int):
        # Space complexity: O(BN) (semi_loss: O(N^2))
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))

        return torch.cat(losses)

    def loss(self, z1: torch.Tensor, z2: torch.Tensor,
             mean: bool = True, batch_size: int = None):
        h1 = self.projection(z1)
        h2 = self.projection(z2)

        if batch_size is None:
            l1 = self.semi_loss(h1, h2)
            l2 = self.semi_loss(h2, h1)
        else:
            l1 = self.batched_semi_loss(h1, h2, batch_size)
            l2 = self.batched_semi_loss(h2, h1, batch_size)
        ret = (l1 + l2) * 0.5
        ret = ret.mean() if mean else ret.sum()

        return ret
    def pot_loss(self, z1: torch.Tensor, z2: torch.Tensor, x, edge_index, edge_index_1: torch.Tensor, local_changes=5, node_list = None, A_upper=None, A_lower=None):
        deg = degree(to_undirected(edge_index)[1]).cpu().numpy()
        device = z1.device
        A = to_scipy_sparse_matrix(edge_index).tocsr()
        A_tilde = A + sp.eye(A.shape[0])
        assert self.encoder.k == 2 # only support 2-layer GCN
        conv = self.encoder.conv
        W1, b1 = conv[0].lin.weight.t(), conv[0].bias
        W2, b2 = conv[1].lin.weight.t(), conv[1].bias
        gcn_weights = [W1, b1, W2, b2]
        # load entry-wise bounds, if not exist, calculate
        if A_upper is None:
            degs_tilde = deg + 1
            max_delete = np.maximum(degs_tilde.astype("int") - 2, 0)
            max_delete = np.minimum(max_delete, np.round(local_changes * deg)) # here
            sqrt_degs_tilde_max_delete = 1 / np.sqrt(degs_tilde - max_delete)
            A_upper = sqrt_degs_tilde_max_delete * sqrt_degs_tilde_max_delete[:, None]
            A_upper = np.where(A_tilde.toarray() > 0, A_upper, np.zeros_like(A_upper))
            A_upper = np.float32(A_upper)

            new_edge_index, An = gcn_norm(edge_index, num_nodes=A.shape[0])
            An = to_dense_adj(new_edge_index, edge_attr = An)[0].cpu().numpy()
            A_lower = np.zeros_like(An)
            A_lower[np.diag_indices_from(A_lower)] = np.diag(An)
            A_lower = np.float32(A_lower)
            upper_lower_file = osp.join(osp.expanduser('~/datasets'),f"bounds/{self.dataset}_{local_changes}_upper_lower.pkl")
            if self.dataset == 'ogbn-arxiv':
                torch.save((torch.tensor(A_upper).to_sparse(), torch.tensor(A_lower).to_sparse()), upper_lower_file)
            else:
                torch.save((A_upper, A_lower), upper_lower_file)
        N = len(node_list)
        if self.dataset == 'ogbn-arxiv':
            A_upper_tensor = torch.tensor(A_upper.to_dense()[node_list][:,node_list], device=device).to_sparse()
            A_lower_tensor = torch.tensor(A_lower.to_dense()[node_list][:,node_list], device=device).to_sparse()
        else:
            A_upper_tensor = torch.tensor(A_upper[node_list][:,node_list], device=device).to_sparse()
            A_lower_tensor = torch.tensor(A_lower[node_list][:,node_list], device=device).to_sparse()
        # get pre-activation bounds for each node
        XW = conv[0].lin(x)[node_list]
        H = self.encoder.activation(conv[0](x, edge_index))
        HW = conv[1].lin(H)[node_list]
        W_1 = XW
        b1 = conv[0].bias
        z1_U = mm((A_upper_tensor + A_lower_tensor) / 2, W_1) + mm((A_upper_tensor - A_lower_tensor) / 2, torch.abs(W_1)) + b1
        z1_L = mm((A_upper_tensor + A_lower_tensor) / 2, W_1) - mm((A_upper_tensor - A_lower_tensor) / 2, torch.abs(W_1)) + b1
        W_2 = HW
        b2 = conv[1].bias
        z2_U = mm((A_upper_tensor + A_lower_tensor) / 2, W_2) + mm((A_upper_tensor - A_lower_tensor) / 2, torch.abs(W_2)) + b2
        z2_L = mm((A_upper_tensor + A_lower_tensor) / 2, W_2) - mm((A_upper_tensor - A_lower_tensor) / 2, torch.abs(W_2)) + b2
        # CROWN weights
        activation = self.encoder.activation
        alpha = 0 if activation == F.relu else activation.weight.item()
        # Wcl = torch.stack([z2_norm[i] - (torch.cat([z2_norm[:i], z2_norm[i+1:]], dim=0)).mean(axis=0) for i in range(z2_norm.shape[0])])
        z2_norm = F.normalize(z2)
        z2_sum = z2_norm.sum(axis=0)
        Wcl = z2_norm * (N / (N-1)) - z2_sum / (N - 1)
        W_tilde_1, b_tilde_1, W_tilde_2, b_tilde_2 = get_crown_weights(z1_L, z1_U, z2_L, z2_U, alpha, gcn_weights, Wcl)
        # return the pot_score 
        XW_tilde = (x[node_list,None,:] @ W_tilde_1[:,:,None]).view(-1,1) # N * 1
        edge_index_ptb_sl, An_ptb = gcn_norm(edge_index_1, num_nodes=A.shape[0])
        An_ptb = torch.sparse_coo_tensor(edge_index_ptb_sl, An_ptb, size=(A.shape[0],A.shape[0])).index_select(0,torch.tensor(node_list).to(device)).index_select(1,torch.tensor(node_list).to(device))
        H_tilde = mm(An_ptb, XW_tilde) + b_tilde_1.view(-1,1)
        pot_score = mm(An_ptb, H_tilde) + b_tilde_2.view(-1,1)
        pot_score = pot_score.squeeze()
        target = torch.zeros(pot_score.shape, device=device) + 1
        pot_loss = self.pot_loss_func(pot_score, target)
        return pot_loss
class LogReg(nn.Module):
    def __init__(self, ft_in, nb_classes):
        super(LogReg, self).__init__()
        self.fc = nn.Linear(ft_in, nb_classes)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, seq):
        ret = self.fc(seq)
        return ret
    
def drop_feature(x, drop_prob):
    drop_mask = torch.empty(
        (x.size(1), ),
        dtype=torch.float32,
        device=x.device).uniform_(0, 1) < drop_prob
    x = x.clone()
    x[:, drop_mask] = 0

    return x
    