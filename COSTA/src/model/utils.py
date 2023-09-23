import torch
import os.path as osp
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