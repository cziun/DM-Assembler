import os
from time import time
import numpy as np
import networkx as nx
import torch
from torch.utils.data import DataLoader, Dataset
import json


def dataloader(config, get_graph_list=False):
    start_time = time()

    mols = []
    arr_x = np.load(os.path.join('preprocess', config.data.data.lower(), 'arr_x.npy'))
    arr_adj = np.load(os.path.join('preprocess', config.data.data.lower(), 'arr_adj.npy'))

    for i in range(0, len(arr_x)):
        tmp = (arr_x[i], arr_adj[i])
        mols.append(tmp)

    with open(os.path.join(config.data.dir, config.data.data.lower(), f'valid_idx_{config.data.data.lower()}.json')) as f:
        test_idx = json.load(f)

    train_idx = [i for i in range(len(mols)) if i not in test_idx]
    print(f'Number of training mols: {len(train_idx)} | Number of test mols: {len(test_idx)}')

    train_mols = [mols[i] for i in train_idx]
    test_mols = [mols[i] for i in test_idx]

    train_dataset = MolDataset(train_mols, get_transform_fn(config.data.data, config.data.max_node_num, config.data.max_feat_num))
    test_dataset = MolDataset(test_mols, get_transform_fn(config.data.data, config.data.max_node_num, config.data.max_feat_num))

    if get_graph_list:
        train_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj in train_dataset]
        test_mols_nx = [nx.from_numpy_matrix(np.array(adj)) for x, adj in test_dataset]
        return train_mols_nx, test_mols_nx
    
    train_dataloader = DataLoader(train_dataset, batch_size=config.data.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config.data.batch_size, shuffle=True)

    print(f'{time() - start_time:.2f} sec elapsed for data loading')
    return train_dataloader, test_dataloader


def get_transform_fn(dataset, max_node_num, max_feat_num):
    def transform(data):
            x, adj = data
            x_ = np.zeros((max_node_num, max_feat_num + 1))
            indices = np.where(x >= 1, x - 1, max_feat_num)
            x_[np.arange(max_node_num), indices] = 1
            x = torch.tensor(x_).to(torch.float32)
            x = x[:, :-1]
            adj = torch.tensor(adj).to(torch.float32)
            return x, adj
    return transform


class MolDataset(Dataset):
    def __init__(self, mols, transform):
        self.mols = mols
        self.transform = transform

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        return self.transform(self.mols[idx])
