from model.encoder import GCNEncoder
from model.base import BaseGSSLRunner
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.sparse import mm

import GCL.augmentors as A
from GCL.eval import get_split, LREvaluator, from_predefined_split
from tqdm import tqdm
from util.helper import _similarity
from torch.optim import Adam
from util.data import get_dataset
from torch_geometric.utils import degree, to_undirected
from torch_geometric.utils import to_scipy_sparse_matrix, to_dense_adj
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn.conv import GCNConv
import numpy as np
import scipy.sparse as sp
from model.utils import get_crown_weights, get_batch, get_A_bounds
import os.path as osp
from time import perf_counter as t

class DualBranchContrast(torch.nn.Module):
    def __init__(self, encoder, dataset, loss, mode, intraview_negs=False, x=None, edge_index=None, **kwargs):
        super(DualBranchContrast, self).__init__()
        self.dataset = dataset
        self.encoder = encoder
        self.loss = loss
        self.kwargs = kwargs
        self.x = x
        self.edge_index = edge_index
        self.pot_loss_func = nn.BCEWithLogitsLoss()
    def forward(self, h1=None, h2=None, z1=None, z2=None, kappa=None, pot_batch=None, epoch=None, edge_index_1=None, edge_index_2=None, 
                local_changes_1=None, local_changes_2=None, node_list=None,A_upper_1=None, A_lower_1=None, A_upper_2=None, A_lower_2=None):
        l1 = self.loss(anchor=h1, sample=h2)
        l2 = self.loss(anchor=h2, sample=h1)
        nce_loss = (l1 + l2) * 0.5
        if kappa is None:
            return nce_loss
        else:
            pot_loss_1 = self.pot_loss(z1, z2, self.x, self.edge_index, edge_index_1, local_changes_1, node_list, A_upper_1, A_lower_1)
            pot_loss_2 = self.pot_loss(z2, z1, self.x, self.edge_index, edge_index_2, local_changes_2, node_list, A_upper_2, A_lower_2)
            return (1 - kappa) * nce_loss + kappa * (pot_loss_1 + pot_loss_2) / 2
    def pot_loss(self, z1: torch.Tensor, z2: torch.Tensor, x, edge_index, edge_index_1: torch.Tensor, local_changes=5, node_list=None, A_upper=None, A_lower=None):
        deg = degree(to_undirected(edge_index)[1]).cpu().numpy()
        device = z1.device
        A = to_scipy_sparse_matrix(edge_index).tocsr()
        A_tilde = A + sp.eye(A.shape[0])
        conv = self.encoder.conv
        W1, b1 = conv[0].lin.weight.t(), conv[0].bias
        W2, b2 = conv[1].lin.weight.t(), conv[1].bias
        gcn_weights = [W1, b1, W2, b2]
        z2_norm = F.normalize(z2)
        # load entry-wise bounds, if not exist, calculate
        if A_upper is None:
            degs_tilde = deg + 1
            max_delete = np.maximum(degs_tilde.astype("int") - 2, 0)
            max_delete = np.minimum(max_delete, np.round(local_changes * deg)) # here
            sqrt_degs_tilde_max_delete = 1 / np.sqrt(degs_tilde - max_delete)
            A_upper = sqrt_degs_tilde_max_delete * sqrt_degs_tilde_max_delete[:, None]
            A_upper = np.where(A_tilde.toarray() > 0, A_upper, np.zeros_like(A_upper))
            A_upper = np.float32(A_upper)

            new_edge_index, An = gcn_norm(edge_index)
            An = to_dense_adj(new_edge_index, edge_attr = An)[0].cpu().numpy()
            A_lower = np.zeros_like(An)
            A_lower[np.diag_indices_from(A_lower)] = np.diag(An)
            A_lower = np.float32(A_lower)
            upper_lower_file = osp.join(osp.expanduser('~/datasets'),f"bounds/{self.dataset}_{local_changes}_upper_lower.pkl")
            torch.save((A_upper, A_lower), upper_lower_file)
        
        N = len(node_list)
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

class InfoNCE(object):
    def __init__(self, tau):
        super(InfoNCE, self).__init__()
        self.tau = tau

    def compute(self, anchor, sample):
        sim = _similarity(anchor, sample) / self.tau
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        loss = log_prob.diag()
        return -loss.mean()

    def __call__(self, anchor, sample) -> torch.FloatTensor:
        loss = self.compute(anchor, sample)
        return loss


class COSTA(torch.nn.Module):
    def __init__(self, encoder, hidden_dim, proj_dim, device):
        super(COSTA, self).__init__()
        self.encoder = encoder
        self.device = device
        self.hidden_dim = hidden_dim

        self.fc1 = torch.nn.Linear(hidden_dim, proj_dim)
        self.fc2 = torch.nn.Linear(proj_dim, hidden_dim)

    def forward(self, x, edge_index, edge_weight=None):
        z = self.encoder(x, edge_index, edge_weight)
        return z

    def project(self, z: torch.Tensor) -> torch.Tensor:
        z = F.elu(self.fc1(z))
        return self.fc2(z)
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

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor):
        for i in range(self.k):
            x = self.activation(self.conv[i](x, edge_index, edge_weight))
        return x

class Runner(BaseGSSLRunner):
    def __init__(self, conf, **kwargs):
        super().__init__(conf, **kwargs)
    
    def load_dataset(self):
        self.dataset = get_dataset("/data/yy/datasets", self.config['dataset'])
    # dataset = Planetoid(data_dir, name=args.dataset, transform=T.NormalizeFeatures())
        self.data = self.dataset[0].to(self.device)
        
    def train(self):
        aug1 = A.Compose([A.EdgeRemoving(pe=self.config['drop_edge_rate_1']),
                        A.FeatureMasking(pf=self.config['drop_feature_rate_1'])])
        aug2 = A.Compose([A.EdgeRemoving(pe=self.config['drop_edge_rate_2']),
                        A.FeatureMasking(pf=self.config['drop_feature_rate_2'])])
        gconv = Encoder(self.dataset.num_features, self.config['num_hidden'], activation=F.relu,
                      base_model=GCNConv, k=self.config['num_layers']).to(self.device)

        self.model = COSTA(encoder=gconv, 
                            hidden_dim=self.config['num_hidden'],
                            proj_dim=self.config['num_proj_hidden'],
                            device=self.device)
        self.model = self.model.to(self.device)

        x = self.data.x
        edge_index = self.data.edge_index
        edge_weight = self.data.edge_attr

        contrast_model = DualBranchContrast(loss=InfoNCE(
            tau=self.config['tau']),encoder=gconv, dataset=self.config['dataset'],mode='L2L', intraview_negs=True, x=x, edge_index=edge_index).to(self.device)

        optimizer = Adam(self.model.parameters(), lr=self.config['learning_rate'])

        if self.config['use_pot']:
            A_upper_1, A_lower_1 = get_A_bounds(self.config['dataset'], self.config['drop_edge_rate_1'])
            A_upper_2, A_lower_2 = get_A_bounds(self.config['dataset'], self.config['drop_edge_rate_2'])
        with tqdm(total=self.config['num_epochs'], desc='(T)') as pbar:
            for epoch in range(1, self.config['num_epochs']+1):
                self.model.train()
                optimizer.zero_grad()
                
                x1, edge_index1, edge_weight1 = aug1(x, edge_index, edge_weight)
                x2, edge_index2, edge_weight2 = aug2(x, edge_index, edge_weight)
                z = self.model(x, edge_index, edge_weight)
                z1 = self.model(x1, edge_index1, edge_weight1)
                z2 = self.model(x2, edge_index2, edge_weight2)

                zz1, zz2 = z1, z2
                k = torch.tensor(int(z.shape[0] * 0.5))
                p = (1/torch.sqrt(k))*torch.randn(k, z.shape[0]).to(self.device)
                z1 = p @ z1
                z2 = p @ z2 
                h1, h2 = [self.model.project(x) for x in [z1, z2]]

                node_list = np.arange(z1.shape[0])
                np.random.shuffle(node_list)
                if self.config['dataset'] in ["PubMed"]:
                    batch_size = 4096
                elif self.config['dataset'] in ["Photo"]:
                    batch_size = 2048
                elif self.config['dataset'] in ["Computers", "WikiCS"]:
                    batch_size = 1024
                else:
                    batch_size = None

                if batch_size is not None:
                    node_list_batch = get_batch(node_list, batch_size, epoch)
                    h1 = h1[node_list_batch]
                    h2 = h2[node_list_batch]

                if self.config['use_pot']:
                    if A_upper_1 is None or A_upper_2 is None:
                        A_upper_1, A_lower_1 = get_A_bounds(self.dataset, self.config['drop_edge_rate_1'])
                        A_upper_2, A_lower_2 = get_A_bounds(self.dataset, self.config['drop_edge_rate_2'])
                    pot_batch = self.config['pot_batch']
                    if pot_batch != -1:
                        if batch_size is None:
                            node_list_tmp = get_batch(node_list, pot_batch, epoch)
                        else:
                            node_list_tmp = get_batch(node_list_batch, pot_batch, epoch)
                    else:
                        # full pot batch
                        if batch_size == None:
                            node_list_tmp = node_list
                        else:
                            node_list_tmp = node_list_batch
                    zz1 = zz1[node_list_tmp]
                    zz2 = zz2[node_list_tmp]
                    loss = contrast_model(h1, h2, z1=zz1, z2=zz2, kappa=self.config['kappa'], pot_batch=self.config['pot_batch'], epoch=epoch, 
                                        edge_index_1=edge_index1, edge_index_2=edge_index2, 
                                        local_changes_1=self.config['drop_edge_rate_1'], local_changes_2=self.config['drop_edge_rate_2'], node_list=node_list_tmp,
                                        A_upper_1=A_upper_1, A_lower_1=A_lower_1, A_upper_2=A_upper_2, A_lower_2=A_lower_2)
                else:
                    loss = contrast_model(h1, h2)
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item()})
                pbar.update()

                if epoch % 100 == 0:
                    self.test(t=self.config['test'])

       

    def test(self, t="random"):
        self.model.eval()
        print(t)
        z = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
        if t == 'random':
            split = get_split(num_samples=z.size()[0], train_ratio=0.1, test_ratio=0.8)
        if t == 'public':
            split = from_predefined_split(self.data)
        result = LREvaluator()(z, self.data.y, split)
        print(f"(E): Best test F1Mi={result['micro_f1']:.4f}, F1Ma={result['macro_f1']:.4f}")
        return result
            
