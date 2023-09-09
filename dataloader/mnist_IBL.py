from __future__ import print_function

import random
import numpy as np
import torch
from scipy.io import loadmat
from utils.utils import img_normalize_mnist
from utils.loss import l2_normalize



class dataset_mnist_cifar10_imbalance(object):
    def __init__(self, args):
        self.args = args
        self.n_label = self.args.n_label
        self.root_dir = self.args.data_root
        self.n_class = self.args.n_class
        self.n_sample = self.args.n_sample
        self.n_val = self.args.n_val

    def l2_norm(self, features, axit=1):
        norm = torch.norm(features, 2, axit, True)
        output = torch.div(features, norm)
        return output

    def load_data(self):
        seed = random.randint(1, 1000)
        data = loadmat(self.root_dir)
        datas = data['featureMat']
        labels = data['labelMat']

        datas = img_normalize_mnist(datas)
        datas = torch.tensor(datas)
        features = l2_normalize(datas, dim=1)
        features = features.cpu().numpy()
        np.random.seed(seed)
        np.random.shuffle(features)

        labels = np.squeeze(labels, axis=None)
        np.random.seed(seed)
        np.random.shuffle(labels)

        train_index = np.array(range(0, self.n_label))
        val_index = np.array(range(self.n_label, self.n_label + self.n_val))
        test_index = np.array(range(self.n_label + self.n_val, self.n_sample))


        print("features", features.shape)  # (10000, 784)
        print("labels", labels.shape)  # (10000,)
        print("train_mask", train_index.shape)  # (10000,)
        print("val_mask", val_index.shape)  # (10000,)
        print("test_mask", test_index.shape)  # (10000,)


        return features, labels, train_index, val_index, test_index