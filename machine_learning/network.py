import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch.nn.functional as F

class SpectrogramDataset(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.annotations_file = os.path.abspath(annotations_file)
        self.annotations_dir = os.path.dirname(self.annotations_file)
        self.specgram_labels = pd.read_csv(
            self.annotations_file,
            header=None,
            skipinitialspace=True,
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.specgram_labels)

    def __getitem__(self, idx):
        specgram_path = self._resolve_specgram_path(self.specgram_labels.iloc[idx, 0])
        specgram = np.load(specgram_path)
        specgram = torch.from_numpy(specgram).float()

        label = self.specgram_labels.iloc[idx, 1]

        if self.target_transform:
            label = self.target_transform(label)

        return specgram, label

    def _resolve_specgram_path(self, specgram_path):
        specgram_path = str(specgram_path).strip()
        if os.path.isabs(specgram_path):
            return specgram_path
        candidate = os.path.abspath(os.path.join(self.annotations_dir, specgram_path))
        if os.path.exists(candidate):
            return candidate
        return os.path.abspath(specgram_path)

class SpectrogramDataset_plus(Dataset):

    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.annotations_file = os.path.abspath(annotations_file)
        self.annotations_dir = os.path.dirname(self.annotations_file)
        self.specgram_labels = pd.read_csv(
            self.annotations_file,
            header=None,
            skipinitialspace=True,
        )
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.specgram_labels)

    def __getitem__(self, idx):
        specgram_path = self._resolve_specgram_path(self.specgram_labels.iloc[idx,0])
        specgram = np.load(specgram_path)
        specgram = torch.from_numpy(specgram).float()

        label = self.specgram_labels.iloc[idx, 1]
        dist_km = self.specgram_labels.iloc[idx,3]
        evlo = self.specgram_labels.iloc[idx,4]
        evla = self.specgram_labels.iloc[idx,5]
        evdp = self.specgram_labels.iloc[idx,6]
        stlo = self.specgram_labels.iloc[idx,7]
        stla = self.specgram_labels.iloc[idx,8]

        if self.target_transform:
            label = self.target_transform(label)

        return specgram, label, dist_km, evlo, evla, evdp, stlo, stla

    def _resolve_specgram_path(self, specgram_path):
        specgram_path = str(specgram_path).strip()
        if os.path.isabs(specgram_path):
            return specgram_path
        candidate = os.path.abspath(os.path.join(self.annotations_dir, specgram_path))
        if os.path.exists(candidate):
            return candidate
        return os.path.abspath(specgram_path)

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
