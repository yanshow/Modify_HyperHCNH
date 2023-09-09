import numpy as np
from utility import load_data, dotdict, seed_everything, accuracy, normalize_sparse_hypergraph_symmetric,normalize_hypergraph_tensor
import sys
import time
from model import HCNH
import torch
import torch.nn.functional as F
from torch import nn, optim
import os
import argparse


"""
run: python run.py --gpu_id 0 --dataname citeseer
"""

# 后续添加
from dataloader.mnist_IBL import dataset_mnist_cifar10_imbalance
from hypergraphConstruct import graph_construct_kNN
from hyperedgeConstruct import hypergraph_propagation

def training(data, args, s = 2021):

    seed_everything(seed = s)
    epochs = args.epochs
    gamma = args.gamma


    model = HCNH(data.X.shape[1], data.Y.shape[1], args.dim_hidden, args.n_class)
    model.cuda()

    criteon = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    for epoch in range(epochs):
        model.train()
        recovered, output, x, y = model(data.X, data.H_trainX_norm, data.Y, data.H_trainY_norm)
        critionCross = nn.CrossEntropyLoss()
        # crossentropy 只能一次softmax
        loss1=critionCross(output[data.idx_train.long()], labels[data.idx_train.long()].long())
        # recover
        loss2=F.binary_cross_entropy(recovered, data.H_trainX.to(torch.float32))

        loss_train = loss1 + args.gamma * loss2
        acc_train = accuracy(output[data.idx_train.long()], labels[data.idx_train.long()].long())
        print(epoch,':',loss_train.item(),acc_train.item())
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()
            
        
    # Test
    with torch.no_grad():
        model.eval()
        recovered, output, x, y = model(data.X, data.H_trainX_norm, data.Y, data.H_trainY_norm)
        loss_test=critionCross(torch.softmax(output[data.idx_test.long()], dim=0), labels[data.idx_test.long()].long())
        acc_test = accuracy(output[data.idx_test.long()], labels[data.idx_test.long()].long())
        print("Test set results:{:.4f}, acc_test: {:.4f}".format(loss_test,acc_test.item()))

    return acc_test.item()



if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='HCNH')
    parser.add_argument('--gpu_id', type=str, nargs='?', default='0', help="device id to run")
    parser.add_argument('--n_label', type=int, default=500, metavar='n_label', help="number of labeled data")
    parser.add_argument('--data_root', default='D:\Code\HCNH\data\data\mnist_imbalance\mnist_imbalance.mat', type=str,
                        metavar='FILE', help='dir of image data')
    parser.add_argument('--n_class', type=int, default=10, help='Number of class.')
    parser.add_argument('--n_sample', type=int, default=3820, metavar='n_label', help="number of sample")
    parser.add_argument('--n_val', type=int, default=500, metavar='n_label', help="number of val data")
    setting = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = setting.gpu_id
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    device = torch.cuda.current_device()
    
    # H, X, Y, labels, idx_train_list, idx_test_list = load_data(setting.dataname) 原始方法

    features, labels, train_index, val_index, test_index=dataset_mnist_cifar10_imbalance(setting).load_data()
    # 构建超图H
    H=graph_construct_kNN(features) # np
    H = torch.tensor(H).to('cuda:0')  # tensor
    H_trainX = H.clone().detach() # np
    H_trainY = H_trainX.clone().detach() #np
    H_trainY=H_trainY.transpose(0,1)


    features=torch.tensor(features).to('cuda:0')
    labels = torch.tensor(labels).to('cuda:0')
    train_index = torch.tensor(train_index).to('cuda:0')
    val_index = torch.tensor(val_index).to('cuda:0')
    test_index = torch.tensor(test_index).to('cuda:0')
    # 构建的超图构超边特征 Y
    Y=hypergraph_propagation(features,H) # tensor

    H_trainX_norm = normalize_hypergraph_tensor(H_trainX)  # 需x
    H_trainY_norm = normalize_hypergraph_tensor(H_trainY)


    acc_test_list = []
    # 不同的trial是不同的trainid mask
    data = dotdict()
    args = dotdict()


    data.X = features
    data.Y = Y
    data.H_trainX = H
    data.H_trainX_norm = H_trainX_norm
    data.H_trainY_norm = H_trainY_norm
    data.labels = labels
    data.idx_train = train_index
    data.idx_val = val_index
    data.idx_test = test_index

    dim_hidden = 128
    epochs = 500 #refix
    seed = 2021

    lr=0.002
    step_size=100
    gamma=0.5
    weight_decay=1e-4
    n_class=10

    args.dim_hidden = dim_hidden
    args.weight_decay = weight_decay
    args.epochs = epochs
    args.learning_rate = lr
    args.gamma = gamma
    args.n_class=n_class
    acc_test = training(data, args, s=seed)
    print(acc_test)




    
    
    
    
    
