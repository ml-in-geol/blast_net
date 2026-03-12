import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.io import read_image
from torchvision.transforms import ToTensor,ToPILImage,Lambda
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
from sys import argv

class SpectrogramDataset(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.specgram_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.specgram_labels)

    def __getitem__(self, idx):

        #specgram_path = os.path.join(self.specgram_dir, self.specgram_labels.iloc[idx, 0])
        specgram_path = self.specgram_labels.iloc[idx,0]
        specgram = np.load(specgram_path)
        specgram = torch.from_numpy(specgram).float()

        label = self.specgram_labels.iloc[idx, 1]

        if self.target_transform:
            label = self.target_transform(label)

        return specgram, label

class SpectrogramDataset_plus(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.specgram_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.specgram_labels)

    def __getitem__(self, idx):
        specgram_path = self.specgram_labels.iloc[idx,0]
        #print('specgram path: {}'.format(specgram_path))
        specgram = np.load(specgram_path)
        specgram = torch.from_numpy(specgram).float()

        label = self.specgram_labels.iloc[idx, 1]
        #event_name = self.specgram_labels.iloc[idx,2]
        dist_km = self.specgram_labels.iloc[idx,3]
        evlo = self.specgram_labels.iloc[idx,4]
        evla = self.specgram_labels.iloc[idx,5]
        evdp = self.specgram_labels.iloc[idx,6]
        stlo = self.specgram_labels.iloc[idx,7]
        stla = self.specgram_labels.iloc[idx,8]

        if self.target_transform:
            label = self.target_transform(label)

        #return specgram, label, event_name, dist_km, evlo, evla, evdp, stlo, stla
        return specgram, label, dist_km, evlo, evla, evdp, stlo, stla
        #return specgram, label, dist_km, evlo, evla

class cnn_v2(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(3,8,9)
        self.conv2 = nn.Conv2d(8,16,5)
        self.conv3 = nn.Conv2d(16,32,3)
        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear((47*3*32),512)
        self.fc2 = nn.Linear(512,2)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class cnn_v3(nn.Module):
    def __init__(self):
        super().__init__()

        self.pool = nn.MaxPool2d(2,2)
        self.conv1 = nn.Conv2d(3,8,9)
        self.conv2 = nn.Conv2d(8,16,5)
        self.conv3 = nn.Conv2d(16,32,3)
        self.dropout = nn.Dropout(p=0.25)

        self.fc1 = nn.Linear((47*3*32),64)
        self.fc2 = nn.Linear(64,2)

    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
