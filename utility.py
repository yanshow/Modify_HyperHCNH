import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import os
import scipy.io as scio


class dotdict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def seed_everything(seed=616):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def load_data(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    h = data_mat['h']
    X = data_mat['X']
    Y = data_mat['Y']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train_list']
    idx_val_list = data_mat['idx_val_list']
    
    X = normalize_features(X)
    Y = normalize_features(Y)
    
    return h, X, Y, labels, idx_train_list, idx_val_list


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)    

def normalize_hypergraph_tensor(H):

    rowsum = H.sum(1)
    r_inv_sqrt = rowsum.pow(-0.5).flatten()
    r_inv_sqrt[r_inv_sqrt == float('inf')] = 0.
    D = torch.diag(r_inv_sqrt)

    colsum = H.sum(0)
    r_inv_sqrt = colsum.pow(-1).flatten()
    r_inv_sqrt[r_inv_sqrt == float('inf')] = 0.
    B = torch.diag(r_inv_sqrt)

    Omega = torch.eye(B.shape[0]).to('cuda:0').to(torch.float64)

    hx = D.mm(H).mm(Omega).mm(B).mm(H.t()).mm(D)

    return hx

def normalize_sparse_hypergraph_symmetric(H):
    
    rowsum = np.array(H.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    D = sp.diags(r_inv_sqrt)
    
    colsum = np.array(H.sum(0))
    r_inv_sqrt = np.power(colsum, -1).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    B = sp.diags(r_inv_sqrt)
    
    Omega = sp.eye(B.shape[0])
    # 过大容易内存溢出 目前用float32代替尝试 my_array = my_array.astype(np.float32)
    # D=D.astype(np.float32)
    # H = H.astype(np.float32)
    # Omega = Omega.astype(np.float32)
    # B = B.astype(np.float32)
    # B = B.astype(np.float32)
    hx = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)

    return hx
    
 
        
        
        