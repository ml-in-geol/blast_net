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


class PrototypicalEventCNN(nn.Module):
    """
    Metric-learning CNN for event-level embeddings.

    The model embeds each station-record scalogram, averages embeddings over
    all records belonging to the same event, builds class prototypes from
    support events, and scores query events by distance to those prototypes.
    An additional "unknown" logit is derived from the minimum prototype
    distance so far-away events can be rejected instead of being forced into
    a known class.
    """

    def __init__(
        self,
        embedding_dim=64,
        dropout=0.25,
        normalize_embeddings=True,
        distance_metric='euclidean',
        initial_unknown_radius=1.5,
        initial_temperature=1.0,
        learnable_unknown_radius=True,
        learnable_temperature=False,
    ):
        super().__init__()

        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 8, 9)
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(47 * 3 * 32, 128)
        self.embedding = nn.Linear(128, embedding_dim)

        self.normalize_embeddings = normalize_embeddings
        self.distance_metric = distance_metric

        radius_tensor = torch.tensor(float(initial_unknown_radius))
        if learnable_unknown_radius:
            self.unknown_radius = nn.Parameter(radius_tensor)
        else:
            self.register_buffer('unknown_radius', radius_tensor)

        temperature_tensor = torch.tensor(float(initial_temperature))
        if learnable_temperature:
            self.temperature = nn.Parameter(temperature_tensor)
        else:
            self.register_buffer('temperature', temperature_tensor)

    def forward(self, x):
        return self.encode_records(x)

    def encode_records(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.embedding(x)
        if self.normalize_embeddings:
            x = F.normalize(x, p=2, dim=1)
        return x

    def aggregate_event_embeddings(self, record_embeddings, event_ids, labels=None):
        event_ids = torch.as_tensor(event_ids, device=record_embeddings.device)
        unique_event_ids = torch.unique(event_ids, sorted=False)

        event_embeddings = []
        event_labels = []
        for event_id in unique_event_ids:
            mask = event_ids == event_id
            event_embeddings.append(record_embeddings[mask].mean(dim=0))

            if labels is not None:
                label_values = self._as_class_labels(labels[mask])
                first_label = label_values[0]
                if not torch.all(label_values == first_label):
                    raise ValueError(
                        'Found mixed labels inside event {}. Event-level labels must be consistent.'.format(
                            event_id.item() if hasattr(event_id, 'item') else event_id
                        )
                    )
                event_labels.append(first_label)

        event_embeddings = torch.stack(event_embeddings, dim=0)
        if labels is None:
            return event_embeddings, unique_event_ids

        event_labels = torch.stack(event_labels).to(record_embeddings.device)
        return event_embeddings, event_labels, unique_event_ids

    def build_prototypes(self, event_embeddings, event_labels, unknown_label=-1):
        event_labels = self._as_class_labels(event_labels)
        known_mask = event_labels != unknown_label

        if not torch.any(known_mask):
            raise ValueError('Support set does not contain any known-class events.')

        known_labels = event_labels[known_mask]
        unique_labels = torch.unique(known_labels, sorted=True)

        prototypes = []
        for label in unique_labels:
            mask = event_labels == label
            prototypes.append(event_embeddings[mask].mean(dim=0))

        prototypes = torch.stack(prototypes, dim=0)
        if self.normalize_embeddings:
            prototypes = F.normalize(prototypes, p=2, dim=1)
        return prototypes, unique_labels

    def compute_distances(self, event_embeddings, prototypes):
        if self.distance_metric == 'euclidean':
            return torch.cdist(event_embeddings, prototypes, p=2)
        if self.distance_metric == 'cosine':
            event_norm = F.normalize(event_embeddings, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            cosine_similarity = event_norm @ proto_norm.t()
            return 1.0 - cosine_similarity
        raise ValueError("Unsupported distance_metric '{}'".format(self.distance_metric))

    def compute_logits(self, event_embeddings, prototypes):
        temperature = torch.clamp(self.temperature, min=1.0e-6)
        distances = self.compute_distances(event_embeddings, prototypes)
        known_logits = -distances / temperature

        min_distance, _ = distances.min(dim=1)
        unknown_logit = (min_distance - self.unknown_radius) / temperature
        logits = torch.cat((known_logits, unknown_logit.unsqueeze(1)), dim=1)
        return logits, distances

    def labels_to_targets(self, raw_labels, class_labels, unknown_label=-1):
        raw_labels = self._as_class_labels(raw_labels)
        targets = torch.full_like(raw_labels, fill_value=len(class_labels))

        for class_index, class_label in enumerate(class_labels):
            targets[raw_labels == class_label] = class_index

        known_mask = raw_labels == unknown_label
        targets[known_mask] = len(class_labels)
        return targets.long()

    def episode_loss(
        self,
        support_records,
        support_event_ids,
        support_labels,
        query_records,
        query_event_ids,
        query_labels,
        unknown_label=-1,
    ):
        support_record_embeddings = self.encode_records(support_records)
        query_record_embeddings = self.encode_records(query_records)

        support_event_embeddings, support_event_labels, support_unique_event_ids = self.aggregate_event_embeddings(
            support_record_embeddings, support_event_ids, labels=support_labels
        )
        query_event_embeddings, query_event_labels, query_unique_event_ids = self.aggregate_event_embeddings(
            query_record_embeddings, query_event_ids, labels=query_labels
        )

        prototypes, class_labels = self.build_prototypes(
            support_event_embeddings, support_event_labels, unknown_label=unknown_label
        )
        logits, distances = self.compute_logits(query_event_embeddings, prototypes)
        targets = self.labels_to_targets(query_event_labels, class_labels, unknown_label=unknown_label)
        loss = F.cross_entropy(logits, targets)

        return {
            'loss': loss,
            'logits': logits,
            'targets': targets,
            'distances': distances,
            'prototypes': prototypes,
            'class_labels': class_labels,
            'support_event_ids': support_unique_event_ids,
            'query_event_ids': query_unique_event_ids,
            'support_event_embeddings': support_event_embeddings,
            'query_event_embeddings': query_event_embeddings,
        }

    def predict(
        self,
        support_records,
        support_event_ids,
        support_labels,
        query_records,
        query_event_ids,
        unknown_label=-1,
    ):
        support_record_embeddings = self.encode_records(support_records)
        query_record_embeddings = self.encode_records(query_records)

        support_event_embeddings, support_event_labels, _ = self.aggregate_event_embeddings(
            support_record_embeddings, support_event_ids, labels=support_labels
        )
        query_event_embeddings, query_unique_event_ids = self.aggregate_event_embeddings(
            query_record_embeddings, query_event_ids
        )

        prototypes, class_labels = self.build_prototypes(
            support_event_embeddings, support_event_labels, unknown_label=unknown_label
        )
        logits, distances = self.compute_logits(query_event_embeddings, prototypes)
        probabilities = F.softmax(logits, dim=1)
        predicted_target_indices = logits.argmax(dim=1)

        predicted_labels = torch.full(
            (predicted_target_indices.shape[0],),
            fill_value=unknown_label,
            dtype=class_labels.dtype,
            device=predicted_target_indices.device,
        )
        known_mask = predicted_target_indices < len(class_labels)
        predicted_labels[known_mask] = class_labels[predicted_target_indices[known_mask]]

        return {
            'event_ids': query_unique_event_ids,
            'predicted_labels': predicted_labels,
            'probabilities': probabilities,
            'logits': logits,
            'distances': distances,
            'class_labels': class_labels,
            'prototypes': prototypes,
            'query_event_embeddings': query_event_embeddings,
        }

    def _as_class_labels(self, labels):
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels)
        labels = torch.as_tensor(labels)
        if labels.ndim > 1:
            labels = labels.argmax(dim=-1)
        return labels.long()
