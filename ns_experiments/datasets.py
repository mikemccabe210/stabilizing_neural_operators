import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np


class KolmDataset(Dataset):
    def __init__(self, dset='train'):
        super().__init__()
        # Remember to fix this eventually - dim order = IC, traj, channel, x, y
        if dset == 'train':
            self.data = np.load('Re1000KolmTrain.npy')
        elif dset == 'valid':
            self.data = np.load('Re1000KolmValid.np.npy')
        elif dset == 'test':
            self.data = np.load('Re1000KolmTest.np.npy')
        self.samples = self.data.shape[0]
        self.T = self.data.shape[1]
    def __len__(self):
        return self.data.shape[0]*self.data.shape[1]

    def __getitem__(self, idx):
        sample = int(idx // self.T)
        time_ind = int(idx % self.T)
        if time_ind == (self.T-1):
            time_ind -= 1
        return self.data[sample, time_ind], self.data[sample, time_ind+1]


