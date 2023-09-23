import argparse
import os.path as osp
import random
import nni
import matplotlib.pyplot as plt
import torch
from torch_geometric.utils import dropout_adj, degree, to_undirected
import numpy as np
import torch.nn.functional as F
from simple_param.sp import SimpleParam
from pGRACE.model import Encoder, GRACE
from pGRACE.functional import drop_feature, drop_edge_weighted, real_weights, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from pGRACE.eval import log_regression, MulticlassEvaluator
from pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality, seed_everything, get_batch, get_A_bounds
from pGRACE.dataset import get_dataset
import sys
from time import perf_counter as t
def train(epoch):
    model.train()
    optimizer.zero_grad()

    def drop_edge(idx: int):
        global drop_weights

        if param['drop_scheme'] == 'uniform':
            return dropout_adj(data.edge_index, p=param[f'drop_edge_rate_{idx}'])[0]
        elif param['drop_scheme'] in ['degree', 'evc', 'pr']:
            return drop_edge_weighted(data.edge_index, drop_weights, p=param[f'drop_edge_rate_{idx}'], threshold=0.7)
        else:
            raise Exception(f'undefined drop scheme: {param["drop_scheme"]}')
    edge_index_1 = drop_edge(1)
    edge_index_2 = drop_edge(2)

    # no feature mask
    x_1 = data.x
    x_2 = data.x
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)
    node_list = np.arange(z1.shape[0])
    np.random.shuffle(node_list)
    if args.dataset in ["PubMed"]:
        batch_size = 4096
    elif args.dataset in ["Computers", "WikiCS"]:
        batch_size = 2048
    else:
        batch_size = None

    if batch_size is not None:
        node_list_batch = get_batch(node_list, batch_size, epoch)

    # nce loss
    if batch_size is not None:
        z11 = z1[node_list_batch]
        z22 = z2[node_list_batch]
        nce_loss = model.loss(z11, z22)
    else:
        nce_loss = model.loss(z1, z2)
    # pot loss
    if use_pot:
        # get node_list_tmp, the nodes to calculate pot_loss
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
        global A_upper_1, A_lower_1, A_upper_2, A_lower_2
        if A_upper_1 is None or A_upper_2 is None:
            A_upper_1, A_lower_1 = get_A_bounds(args.dataset, param['drop_edge_rate_1'])
            A_upper_2, A_lower_2 = get_A_bounds(args.dataset, param['drop_edge_rate_2'])
        z11 = z1[node_list_tmp]
        z22 = z2[node_list_tmp]
        pot_loss_1 = model.pot_loss(z11, z22, data.x, data.edge_index, edge_index_1, local_changes=drop_degree_1, drop_rate=param['drop_edge_rate_1'], 
                                  node_list=node_list_tmp, A_upper=A_upper_1, A_lower=A_lower_1)
        pot_loss_2 = model.pot_loss(z22, z11, data.x, data.edge_index, edge_index_2, local_changes=drop_degree_2, drop_rate=param['drop_edge_rate_2'], 
                                  node_list=node_list_tmp, A_upper=A_upper_2, A_lower=A_lower_2)
        pot_loss = (pot_loss_1 + pot_loss_2) / 2
        loss = (1 - kappa) * nce_loss + kappa * pot_loss
    else:
        loss = nce_loss
    loss.backward()
    optimizer.step()

    return loss.item()


def test(final=False):
    model.eval()
    z = model(data.x, data.edge_index)

    evaluator = MulticlassEvaluator()
    res = log_regression(z, dataset, evaluator, split='preloaded', num_epochs=3000, preload_split=split)

    return res

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:2')
    parser.add_argument('--dataset', type=str, default='WikiCS')
    parser.add_argument('--param', type=str, default='local:wikics.json')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--verbose', type=str, default='eval,final')
    parser.add_argument('--save_split', type=str, nargs='?')
    parser.add_argument('--load_split', type=str, nargs='?')
    parser.add_argument('--use_pot', default=False, action="store_true") # whether to use pot in loss
    parser.add_argument('--kappa', type=float, default=0.5)
    parser.add_argument('--pot_batch', type=int, default=-1)
    parser.add_argument('--save_file', type=str, default=".")
    default_param = {
        'learning_rate': 0.01,
        'num_hidden': 256,
        'num_proj_hidden': 32,
        'activation': 'prelu',
        'base_model': 'GCNConv',
        'num_layers': 2,
        'drop_edge_rate_1': 0.3,
        'drop_edge_rate_2': 0.4,
        'drop_feature_rate_1': 0.1,
        'drop_feature_rate_2': 0.0,
        'tau': 0.4,
        'num_epochs': 1000,
        'weight_decay': 1e-5,
        'drop_scheme': 'degree',
    }

    # add hyper-parameters into parser
    param_keys = default_param.keys()
    for key in param_keys:
        parser.add_argument(f'--{key}', type=type(default_param[key]), nargs='?')
    args = parser.parse_args()

    # parse param
    sp = SimpleParam(default=default_param)
    param = sp(source=args.param, preprocess='nni')

    # merge cli arguments and parsed param
    for key in param_keys:
        if getattr(args, key) is not None:
            param[key] = getattr(args, key)
    print(param)
    use_nni = args.param == 'nni'
    if use_nni and args.device != 'cpu':
        args.device = 'cuda'

    seed_everything(args.seed)

    device = torch.device(args.device)

    path = osp.expanduser('~/datasets')
    dataset = get_dataset(path, args.dataset)

    data = dataset[0]
    data = data.to(device)

    # generate split
    if args.dataset in ["Cora", "CiteSeer", "PubMed"]:
        split = data.train_mask, data.val_mask, data.test_mask
        print("Public Split")
    else:
        split = generate_split(data.num_nodes, train_ratio=0.1, val_ratio=0.1)
        print("Random Split")

    if args.save_split:
        torch.save(split, args.save_split)
    elif args.load_split:
        split = torch.load(args.load_split)
    log = args.verbose.split(',')
    use_pot = args.use_pot
    kappa = args.kappa
    pot_batch = args.pot_batch
    
    # model setup
    encoder = Encoder(dataset.num_features, param['num_hidden'], get_activation(param['activation']),
                      base_model=get_base_model(param['base_model']), k=param['num_layers']).to(device)
    model = GRACE(encoder, param['num_hidden'], param['num_proj_hidden'], param['tau'], args.dataset).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=param['learning_rate'],
        weight_decay=param['weight_decay']
    )

    # calculate drop weights
    if param['drop_scheme'] == 'degree':
        drop_weights = degree_drop_weights(data.edge_index).to(device)
    elif param['drop_scheme'] == 'pr':
        drop_weights = pr_drop_weights(data.edge_index, aggr='sink', k=200).to(device)
    elif param['drop_scheme'] == 'evc':
        drop_weights = evc_drop_weights(data).to(device)
    else:
        drop_weights = None

    real_drop_weights_1 = real_weights(drop_weights, p=param[f'drop_edge_rate_1'], threshold=0.7)
    real_drop_weights_2 = real_weights(drop_weights, p=param[f'drop_edge_rate_2'], threshold=0.7)
    drop_degree_1 = torch.zeros(data.num_nodes, device=real_drop_weights_1.device)
    drop_degree_1.scatter_add_(0, to_undirected(data.edge_index)[1], real_drop_weights_1)
    drop_degree_1 = drop_degree_1.cpu().numpy()
    drop_degree_2 = torch.zeros(data.num_nodes, device=real_drop_weights_2.device)
    drop_degree_2.scatter_add_(0, to_undirected(data.edge_index)[1], real_drop_weights_2)
    drop_degree_2 = drop_degree_2.cpu().numpy()
    
    if param['drop_scheme'] == 'degree':
        edge_index_ = to_undirected(data.edge_index)
        node_deg = degree(edge_index_[1])
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_deg).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_deg).to(device)
    elif param['drop_scheme'] == 'pr':
        node_pr = compute_pr(data.edge_index)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_pr).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_pr).to(device)
    elif param['drop_scheme'] == 'evc':
        node_evc = eigenvector_centrality(data)
        if args.dataset == 'WikiCS':
            feature_weights = feature_drop_weights_dense(data.x, node_c=node_evc).to(device)
        else:
            feature_weights = feature_drop_weights(data.x, node_c=node_evc).to(device)
    else:
        feature_weights = torch.ones((data.x.size(1),)).to(device)

    if use_pot:
        A_upper_1, A_lower_1 = get_A_bounds(args.dataset, param[f'drop_edge_rate_1'])
        A_upper_2, A_lower_2 = get_A_bounds(args.dataset, param[f'drop_edge_rate_2'])

    start = t()
    prev = start
    for epoch in range(1, param['num_epochs'] + 1):
        loss = train(epoch=epoch)
        sys.stdout.flush()
        if 'train' in log:
            print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}')

        if epoch % 100 == 0:
            res = test()

            if 'eval' in log:
                now = t()
                print(f'(E) | Epoch={epoch:04d}, loss={loss:.4f}, this epoch {now - prev:.4f}, total {now - start:.4f}, res:{res}')
                prev = now

    res = test(final=True)

    if 'final' in log:
        print(res)
    res_file = f"res/{args.dataset}_pot_temp.csv" if use_pot else f"res/{args.dataset}_base_temp.csv"
    if args.save_file == '.':
        f = open(res_file,"a+")
    else:
        f = open(args.save_file, "a+")
    if use_pot:
        f.write(f'{param["drop_edge_rate_1"]}, {param["drop_edge_rate_2"]}, {param["tau"]}, {kappa}, {pot_batch}, '
                f'{res["F1Mi"]:.4f}, {res["F1Ma"]:.4f}\n')
    else:
        f.write(f'{param["drop_edge_rate_1"]}, {param["drop_edge_rate_2"]}, {param["tau"]}, '
                f'{res["F1Mi"]:.4f}, {res["F1Ma"]:.4f}\n')
    f.close()