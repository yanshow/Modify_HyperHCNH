import os
import shutil

import numpy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



def glorot(shape):
    """Glorot & Bengio (AISTATS 2010) init."""
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = torch.Tensor(shape[0],shape[1]).uniform_(-init_range,init_range)
    return initial





def save_checkpoint(state, is_best, dirpath, epoch):
    # filename = 'checkpoint.{}.ckpt'.format(epoch)
    filename = 'checkpoint.ckpt'
    checkpoint_path = os.path.join(dirpath, filename)
    best_path = os.path.join(dirpath, 'best.ckpt')
    model_path = os.path.join(dirpath, 'best_model.ckpt')
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_path)
        torch.save(state['state_dict'], model_path)



# RGB图像数据归一化、标准化
def img_normalize(img):
    img = img.astype(np.float32) / 255.0    #归一化为[0.0,1.0]
    means, stdevs = [], []      # 均值，方差
    for i in range(3):
        pixels = img[:, :, i, :].ravel()  # 拉成一行
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))
    means = np.array(means)
    stdevs = np.array(stdevs)

    img = (img - means) / stdevs

    return img



# 灰度图像数据归一化、标准化
def img_normalize_mnist(img):
    img = img.astype(np.float32) / 255.0   #归一化为[0.0,1.0]
    means=np.mean(img)
    stdevs=np.std(img)
    img = (img - means) / stdevs
    return img



# 协方差
def torch_cov(X:torch.tensor, Y:torch.tensor):
    n_node = X.shape[0]
    X = X - torch.mean(X, dim=0)
    Y = Y - torch.mean(Y, dim=0)
    X = torch.unsqueeze(X.t(), 0)
    Y = torch.unsqueeze(Y, 1)
    cov_matrix = 1 / (n_node-1) * torch.matmul(Y, X)
    return cov_matrix.squeeze()


def mahalanobis_distance(X:torch.tensor, Y:torch.tensor):
    D = torch_cov(X, Y)  # 求协方差矩阵
    invD = torch.pow(D, -1)     # nxn
    # invD = torch.inverse(D)
    X = torch.unsqueeze(X, 0)
    Y = torch.unsqueeze(Y, 1)
    tp = torch.sub(X, Y).squeeze()  # nxnxd
    result = torch.sqrt(torch.mul(torch.mul(tp.t(),invD),tp))
    return result


def mahalanobis_distance2(X:torch.tensor, Y:torch.tensor):
    X = torch.unsqueeze(X, 0)
    Y = torch.unsqueeze(Y, 1)
    X = X.clone().detach().cpu().numpy()
    Y = Y.clone().detach().cpu().numpy()
    D = numpy.cov(X,Y)
    invD = numpy.linalg.inv(D)
    tp = X - Y
    result = numpy.sqrt(numpy.dot(numpy.dot(tp,invD),tp.T))
    return result



class RelationNetwork(nn.Module):
    """Graph Construction Module"""

    def __init__(self):
        super(RelationNetwork, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, padding=1))

        self.fc3 = nn.Linear(2 * 2, 8)
        self.fc4 = nn.Linear(8, 1)

        self.m0 = nn.MaxPool2d(2)  # max-pool without padding
        self.m1 = nn.MaxPool2d(2, padding=1)  # max-pool with padding

    def forward(self, x):
        x = x.view(-1, 64, 4, 4)

        out = self.layer1(x)
        out = self.layer2(out)
        # flatten
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc3(out))
        out = self.fc4(out)  # no relu

        out = out.view(out.size(0), -1)  # bs*1

        return out



