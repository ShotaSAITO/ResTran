import torch
import numpy as np
import sys
from urllib import request
from torch.utils.data import Dataset
import random

sys.path.append("../../../")

import datasets
from graph_manip import Graph

cuda = torch.cuda.is_available()


class GraphDataset(Dataset):
    # initialization
    def __init__(self, pLsq, Y):
        self.pLsq = pLsq
        self.Y = Y
        #self.Y = Y

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):

        y = np.zeros((self.Y.shape[0], self.Y.max() + 1))
        y[np.arange(self.Y.shape[0]), self.Y] = 1
        y = torch.Tensor(y)

        if idx.dtype != 'torch.int64':
            idx = idx.to(torch.int64)

        #print(idx.dtype)

        return self.pLsq[idx], y[idx]
        #return self.X, self.Y


def get_data_krylov(batch_size=64, labels_per_class=20, validation_ratio=0.5, gamma=-1, b=0,
              norm_flag=False, dataset_name='cora', dataset_path='./data', n_workers=0):


    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from sklearn.model_selection import train_test_split

    import pickle
    LX, X, Y = datasets.load_data_krylov(dataset_name, dataset_path)

    n = LX.shape[0]

    '''
    Split into train + validation
    '''

    n_labels = max(Y) + 1
    n_train = n_labels * labels_per_class
    #import math
    #labels_per_class = math.floor(labels_for_train/n_labels)
    print('ntrain:{0}'.format(n_train))

    Y_train = np.array(1)

    while np.unique(Y_train).size != n_labels:
        pLsq_train, pLsq_unlabelled, Y_train, Y_unlabelled = train_test_split(pLsq, Y, train_size=n_train)


    print(pLsq_train.shape)
    print(pLsq_unlabelled.shape)
    pLsq_train_tensor = torch.Tensor(pLsq_train)
    Y_train_tensor = torch.from_numpy(Y_train)
    graph_train_data = GraphDataset(pLsq_train_tensor, Y_train_tensor)

    pLsq_unlabelled_tensor = torch.Tensor(pLsq_unlabelled)
    Y_unlabelled_tensor = torch.from_numpy(Y_unlabelled)
    graph_unlabelled_data = GraphDataset(pLsq_unlabelled_tensor, Y_unlabelled_tensor)


    pLsq_valid, pLsq_test, Y_valid, Y_test = train_test_split(pLsq_unlabelled, Y_unlabelled, train_size=validation_ratio)
    print(pLsq_valid.shape)
    print(pLsq_unlabelled.shape)

    pLsq_valid_tensor = torch.Tensor(pLsq_valid)
    Y_valid_tensor = torch.from_numpy(Y_valid)
    graph_valid_data = GraphDataset(pLsq_valid_tensor, Y_valid_tensor)

    pLsq_test_tensor = torch.Tensor(pLsq_test)
    Y_test_tensor = torch.from_numpy(Y_test)
    graph_test_data = GraphDataset(pLsq_test_tensor, Y_test_tensor)


    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for graph
    labelled = torch.utils.data.DataLoader(graph_train_data, batch_size=batch_size, num_workers=n_workers, pin_memory=cuda,
                                           sampler=get_sampler(graph_train_data.Y.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(graph_unlabelled_data, batch_size=batch_size, num_workers=n_workers, pin_memory=cuda,
                                             sampler=get_sampler(graph_unlabelled_data.Y.numpy()))
    validation = torch.utils.data.DataLoader(graph_valid_data, batch_size=batch_size, num_workers=n_workers, pin_memory=cuda,
                                             sampler=get_sampler(graph_valid_data.Y.numpy()))
    test = torch.utils.data.DataLoader(graph_test_data, batch_size=batch_size, num_workers=n_workers, pin_memory=cuda,
                                             sampler=get_sampler(graph_test_data.Y.numpy()))

    return labelled, unlabelled, validation, test
