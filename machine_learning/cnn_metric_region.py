import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim

from network import PrototypicalEventCNN


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train an event-level metric CNN and score query events.'
    )
    parser.add_argument('model_dir', help='Path to the regional model directory, e.g. ../models/wvsz_test')
    parser.add_argument(
        '--labels-file',
        default=None,
        help='Optional labels CSV. Defaults to labels_scalogram_<region>.csv in model_dir.',
    )
    parser.add_argument(
        '--support-labels-file',
        default=None,
        help='Optional support labels CSV used to build class prototypes. Defaults to --labels-file.',
    )
    parser.add_argument(
        '--predict-labels-file',
        default=None,
        help='Optional query labels CSV for inference. Defaults to the training labels file.',
    )
    parser.add_argument(
        '--load-model',
        default=None,
        help='Optional trained metric-model checkpoint. Defaults to preferred_metric_model_<region>.pt in model_dir.',
    )
    parser.add_argument(
        '--prediction-output-file',
        default=None,
        help='Optional CSV path for prediction output.',
    )
    parser.add_argument(
        '--predict-only',
        action='store_true',
        help='Skip training, load an existing metric model, and only run inference.',
    )
    parser.add_argument('--unknown-label', type=int, default=-1, help='Label value reserved for unknown events.')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--episodes-per-epoch', type=int, default=40)
    parser.add_argument('--support-events-per-class', type=int, default=3)
    parser.add_argument('--query-events-per-class', type=int, default=2)
    parser.add_argument('--valid-events-per-class', type=int, default=1)
    parser.add_argument('--test-events-per-class', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=1.0e-4)
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--distance-metric', choices=['euclidean', 'cosine'], default='euclidean')
    parser.add_argument('--device', default=None, help='Override torch device, e.g. cpu or cuda')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def infer_region_name(model_dir):
    return Path(model_dir).resolve().name


def find_default_labels_file(model_dir, region):
    candidates = [
        os.path.join(model_dir, 'labels_scalogram_{}.csv'.format(region)),
        os.path.join(model_dir, 'labels_plus_{}.csv'.format(region)),
        os.path.join(model_dir, 'labels_{}.csv'.format(region)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError('Could not find a labels CSV in {}'.format(model_dir))


def resolve_specgram_path(labels_file, specgram_path):
    specgram_path = str(specgram_path).strip()
    if os.path.isabs(specgram_path):
        return specgram_path

    labels_dir = os.path.dirname(os.path.abspath(labels_file))
    candidate = os.path.abspath(os.path.join(labels_dir, specgram_path))
    if os.path.exists(candidate):
        return candidate

    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.abspath(os.path.join(script_dir, specgram_path))
    if os.path.exists(candidate):
        return candidate

    return os.path.abspath(specgram_path)


def load_event_table(labels_file):
    event_table = {}
    with open(labels_file, 'r') as f_in:
        reader = csv.reader(f_in, skipinitialspace=True)
        for row in reader:
            if not row:
                continue
            event_name = row[2].strip()
            label = int(row[1])
            record = {
                'path': resolve_specgram_path(labels_file, row[0]),
                'label': label,
                'event_name': event_name,
                'row': row,
            }

            if event_name not in event_table:
                event_table[event_name] = {
                    'label': label,
                    'records': [],
                }
            elif event_table[event_name]['label'] != label:
                raise ValueError('Mixed labels found inside event {}'.format(event_name))

            event_table[event_name]['records'].append(record)
    return event_table


def split_known_events(event_table, unknown_label, valid_events_per_class, test_events_per_class, seed):
    class_to_events = defaultdict(list)
    for event_name, info in event_table.items():
        if info['label'] == unknown_label:
            continue
        class_to_events[info['label']].append(event_name)

    rng = random.Random(seed)
    split = {'train': [], 'valid': [], 'test': []}

    for label, events in class_to_events.items():
        events = list(events)
        rng.shuffle(events)

        if len(events) < valid_events_per_class + test_events_per_class + 1:
            raise ValueError(
                'Class {} only has {} events; need at least {} to make train/valid/test splits.'.format(
                    label, len(events), valid_events_per_class + test_events_per_class + 1
                )
            )

        valid_events = events[:valid_events_per_class]
        test_events = events[valid_events_per_class:valid_events_per_class + test_events_per_class]
        train_events = events[valid_events_per_class + test_events_per_class:]

        split['train'].extend(train_events)
        split['valid'].extend(valid_events)
        split['test'].extend(test_events)

    return split


class EventTensorCache:
    def __init__(self, event_table):
        self.event_table = event_table
        self.cache = {}

    def get(self, event_name):
        if event_name in self.cache:
            return self.cache[event_name]

        records = self.event_table[event_name]['records']
        tensors = []
        for record in records:
            data = np.load(record['path'])
            tensors.append(torch.from_numpy(data).float())

        record_tensor = torch.stack(tensors, dim=0)
        label_tensor = torch.full(
            (record_tensor.shape[0],),
            fill_value=self.event_table[event_name]['label'],
            dtype=torch.long,
        )
        self.cache[event_name] = (record_tensor, label_tensor)
        return self.cache[event_name]


def build_record_batch(event_names, event_table, cache, device):
    record_tensors = []
    record_event_ids = []
    record_labels = []

    for event_index, event_name in enumerate(event_names):
        event_records, event_labels = cache.get(event_name)
        record_tensors.append(event_records)
        record_event_ids.append(torch.full((event_records.shape[0],), event_index, dtype=torch.long))
        record_labels.append(event_labels)

    records = torch.cat(record_tensors, dim=0).to(device)
    event_ids = torch.cat(record_event_ids, dim=0).to(device)
    labels = torch.cat(record_labels, dim=0).to(device)
    return records, event_ids, labels


def sample_training_episode(split, event_table, support_events_per_class, query_events_per_class, seed=None):
    rng = random.Random(seed)
    class_to_train_events = defaultdict(list)
    for event_name in split['train']:
        class_to_train_events[event_table[event_name]['label']].append(event_name)

    support_events = []
    query_events = []
    for label, events in class_to_train_events.items():
        required = support_events_per_class + query_events_per_class
        if len(events) < required:
            raise ValueError(
                'Class {} has {} train events but requires {} support+query events per episode.'.format(
                    label, len(events), required
                )
            )
        selected = rng.sample(events, required)
        support_events.extend(selected[:support_events_per_class])
        query_events.extend(selected[support_events_per_class:])

    return support_events, query_events


def evaluate_event_set(model, support_events, query_events, event_table, cache, device, unknown_label):
    if not query_events:
        return None

    support_records, support_event_ids, support_labels = build_record_batch(
        support_events, event_table, cache, device
    )
    query_records, query_event_ids, query_labels = build_record_batch(
        query_events, event_table, cache, device
    )

    model.eval()
    with torch.no_grad():
        outputs = model.episode_loss(
            support_records,
            support_event_ids,
            support_labels,
            query_records,
            query_event_ids,
            query_labels,
            unknown_label=unknown_label,
        )
        predictions = model.predict(
            support_records,
            support_event_ids,
            support_labels,
            query_records,
            query_event_ids,
            unknown_label=unknown_label,
        )

    predicted_labels = predictions['predicted_labels'].cpu()
    query_event_labels = outputs['targets'].cpu()
    class_labels = predictions['class_labels'].cpu()

    true_raw_labels = []
    for event_name in query_events:
        true_raw_labels.append(event_table[event_name]['label'])
    true_raw_labels = torch.tensor(true_raw_labels, dtype=torch.long)

    accuracy = float((predicted_labels == true_raw_labels).float().mean().item())
    min_distances = predictions['distances'].min(dim=1).values.cpu()

    return {
        'loss': float(outputs['loss'].item()),
        'accuracy': accuracy,
        'predicted_labels': predicted_labels,
        'true_labels': true_raw_labels,
        'class_labels': class_labels,
        'distances': predictions['distances'].cpu(),
        'min_distances': min_distances,
        'probabilities': predictions['probabilities'].cpu(),
        'event_names': list(query_events),
        'unknown_radius': float(model.unknown_radius.detach().cpu().item()),
    }


def calibrate_unknown_radius(model, support_events, valid_events, event_table, cache, device, unknown_label):
    metrics = evaluate_event_set(model, support_events, valid_events, event_table, cache, device, unknown_label)
    if metrics is None:
        return

    known_min_distances = metrics['min_distances'].numpy()
    if known_min_distances.size == 0:
        return

    calibrated_radius = float(np.quantile(known_min_distances, 0.95))
    with torch.no_grad():
        model.unknown_radius.copy_(torch.tensor(calibrated_radius, device=model.unknown_radius.device))


def write_predictions(output_file, split_name, metrics):
    with open(output_file, 'a') as f_out:
        class_labels = [int(label.item()) for label in metrics['class_labels']]
        for idx, event_name in enumerate(metrics['event_names']):
            predicted_label = int(metrics['predicted_labels'][idx].item())
            true_label = int(metrics['true_labels'][idx].item())
            min_distance = float(metrics['min_distances'][idx].item())
            unknown_probability = float(metrics['probabilities'][idx, -1].item())
            distance_values = [
                '{:.6f}'.format(float(metrics['distances'][idx, class_index].item()))
                for class_index, _ in enumerate(class_labels)
            ]
            f_out.write(
                '{},{},{},{:.6f},{:.6f},{},{}\n'.format(
                    event_name,
                    true_label,
                    predicted_label,
                    min_distance,
                    unknown_probability,
                    ','.join(distance_values),
                    '{:.6f}'.format(float(metrics['unknown_radius'])),
                    split_name,
                )
            )


def build_prediction_header(class_labels):
    distance_columns = ['distance_label_{}'.format(int(label)) for label in class_labels]
    return ','.join(
        [
            'event_name',
            'true_label',
            'predicted_label',
            'min_distance',
            'unknown_probability',
            *distance_columns,
            'unknown_radius',
            'split',
        ]
    )


def write_prediction_outputs(output_file, query_prediction_events, prediction_event_table, prediction_outputs, unknown_radius, split_name):
    class_labels = [int(label.item()) for label in prediction_outputs['class_labels'].cpu()]
    with open(output_file, 'w') as f_out:
        f_out.write(build_prediction_header(class_labels) + '\n')

    with open(output_file, 'a') as f_out:
        for idx, event_name in enumerate(query_prediction_events):
            true_label = prediction_event_table[event_name]['label']
            predicted_label = int(prediction_outputs['predicted_labels'][idx].cpu().item())
            min_distance = float(prediction_outputs['distances'][idx].min().cpu().item())
            unknown_probability = float(prediction_outputs['probabilities'][idx, -1].cpu().item())
            distance_values = [
                '{:.6f}'.format(float(prediction_outputs['distances'][idx, class_index].cpu().item()))
                for class_index, _ in enumerate(class_labels)
            ]
            f_out.write(
                '{},{},{},{:.6f},{:.6f},{},{}\n'.format(
                    event_name,
                    true_label,
                    predicted_label,
                    min_distance,
                    unknown_probability,
                    ','.join(distance_values),
                    '{:.6f}'.format(float(unknown_radius)),
                    split_name,
                )
            )


def run_prediction_only(
    model,
    support_event_table,
    prediction_event_table,
    device,
    unknown_label,
    prediction_output_file,
):
    support_cache = EventTensorCache(support_event_table)
    prediction_cache = EventTensorCache(prediction_event_table)

    support_all_known_events = [
        event_name
        for event_name, info in support_event_table.items()
        if info['label'] != unknown_label
    ]
    query_prediction_events = list(prediction_event_table.keys())

    support_records, support_event_ids, support_labels = build_record_batch(
        support_all_known_events,
        support_event_table,
        support_cache,
        device,
    )
    query_records, query_event_ids, _ = build_record_batch(
        query_prediction_events,
        prediction_event_table,
        prediction_cache,
        device,
    )

    model.eval()
    with torch.no_grad():
        prediction_outputs = model.predict(
            support_records,
            support_event_ids,
            support_labels,
            query_records,
            query_event_ids,
            unknown_label=unknown_label,
        )

    write_prediction_outputs(
        prediction_output_file,
        query_prediction_events,
        prediction_event_table,
        prediction_outputs,
        float(model.unknown_radius.detach().cpu().item()),
        'predict',
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    model_dir = os.path.abspath(args.model_dir)
    region = infer_region_name(model_dir)
    labels_file = args.labels_file or find_default_labels_file(model_dir, region)
    support_labels_file = args.support_labels_file or labels_file
    predict_labels_file = args.predict_labels_file or labels_file
    preferred_model_path = os.path.join(model_dir, 'preferred_metric_model_{}.pt'.format(region))
    load_model_path = args.load_model or preferred_model_path
    prediction_output_file = args.prediction_output_file or os.path.join(
        model_dir, 'metric_event_predictions.csv'
    )

    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = PrototypicalEventCNN(
        embedding_dim=args.embedding_dim,
        dropout=args.dropout,
        distance_metric=args.distance_metric,
    ).to(device)

    if args.predict_only:
        support_event_table = load_event_table(support_labels_file)
        prediction_event_table = load_event_table(predict_labels_file)

        if not os.path.exists(load_model_path):
            raise FileNotFoundError('Could not find metric checkpoint {}'.format(load_model_path))

        model.load_state_dict(torch.load(load_model_path, map_location=device))
        run_prediction_only(
            model,
            support_event_table,
            prediction_event_table,
            device,
            args.unknown_label,
            prediction_output_file,
        )
        print('Loaded model: {}'.format(load_model_path))
        print('Prediction file: {}'.format(prediction_output_file))
        return

    event_table = load_event_table(labels_file)
    prediction_event_table = load_event_table(predict_labels_file)
    split = split_known_events(
        event_table,
        unknown_label=args.unknown_label,
        valid_events_per_class=args.valid_events_per_class,
        test_events_per_class=args.test_events_per_class,
        seed=args.seed,
    )

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    cache = EventTensorCache(event_table)
    saved_models_dir = os.path.join(model_dir, 'metric_saved_models')
    os.makedirs(saved_models_dir, exist_ok=True)

    training_output_file = os.path.join(model_dir, 'metric_training_output.dat')
    best_valid_loss = np.inf
    best_epoch = -1

    train_support_events = list(split['train'])
    valid_query_events = list(split['valid'])
    test_query_events = list(split['test'])

    with open(training_output_file, 'w') as f_train:
        f_train.write('epoch,train_loss,valid_loss,valid_accuracy,unknown_radius\n')

        for epoch in range(args.epochs):
            model.train()
            epoch_losses = []

            for episode_index in range(args.episodes_per_epoch):
                support_events, query_events = sample_training_episode(
                    split,
                    event_table,
                    support_events_per_class=args.support_events_per_class,
                    query_events_per_class=args.query_events_per_class,
                    seed=args.seed + (epoch * args.episodes_per_epoch) + episode_index,
                )

                support_records, support_event_ids, support_labels = build_record_batch(
                    support_events, event_table, cache, device
                )
                query_records, query_event_ids, query_labels = build_record_batch(
                    query_events, event_table, cache, device
                )

                optimizer.zero_grad()
                outputs = model.episode_loss(
                    support_records,
                    support_event_ids,
                    support_labels,
                    query_records,
                    query_event_ids,
                    query_labels,
                    unknown_label=args.unknown_label,
                )
                loss = outputs['loss']
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            valid_metrics = evaluate_event_set(
                model,
                train_support_events,
                valid_query_events,
                event_table,
                cache,
                device,
                args.unknown_label,
            )

            train_loss = float(np.mean(epoch_losses))
            valid_loss = np.nan if valid_metrics is None else valid_metrics['loss']
            valid_accuracy = np.nan if valid_metrics is None else valid_metrics['accuracy']
            unknown_radius = float(model.unknown_radius.detach().cpu().item())

            f_train.write('{},{:.6f},{:.6f},{:.6f},{:.6f}\n'.format(
                epoch + 1,
                train_loss,
                valid_loss,
                valid_accuracy,
                unknown_radius,
            ))

            if valid_metrics is None or valid_loss <= best_valid_loss:
                best_valid_loss = valid_loss
                best_epoch = epoch
                torch.save(model.state_dict(), preferred_model_path)

            torch.save(
                model.state_dict(),
                os.path.join(saved_models_dir, 'metric_model_epoch{}.pt'.format(epoch)),
            )

    model.load_state_dict(torch.load(preferred_model_path, map_location=device))
    calibrate_unknown_radius(
        model,
        train_support_events,
        valid_query_events,
        event_table,
        cache,
        device,
        args.unknown_label,
    )
    torch.save(model.state_dict(), preferred_model_path)

    support_all_known_events = [
        event_name
        for event_name, info in event_table.items()
        if info['label'] != args.unknown_label
    ]

    test_metrics = evaluate_event_set(
        model,
        [event_name for event_name in support_all_known_events if event_name not in test_query_events],
        test_query_events,
        event_table,
        cache,
        device,
        args.unknown_label,
    )

    prediction_cache = EventTensorCache(prediction_event_table)
    query_prediction_events = list(prediction_event_table.keys())
    support_records, support_event_ids, support_labels = build_record_batch(
        support_all_known_events,
        event_table,
        cache,
        device,
    )
    query_records, query_event_ids, _ = build_record_batch(
        query_prediction_events,
        prediction_event_table,
        prediction_cache,
        device,
    )

    model.eval()
    with torch.no_grad():
        prediction_outputs = model.predict(
            support_records,
            support_event_ids,
            support_labels,
            query_records,
            query_event_ids,
            unknown_label=args.unknown_label,
        )

    class_labels = [int(label.item()) for label in prediction_outputs['class_labels'].cpu()]
    with open(prediction_output_file, 'w') as f_out:
        f_out.write(build_prediction_header(class_labels) + '\n')

    if test_metrics is not None:
        write_predictions(prediction_output_file, 'test', test_metrics)

    with open(prediction_output_file, 'a') as f_out:
        for idx, event_name in enumerate(query_prediction_events):
            true_label = prediction_event_table[event_name]['label']
            predicted_label = int(prediction_outputs['predicted_labels'][idx].cpu().item())
            min_distance = float(prediction_outputs['distances'][idx].min().cpu().item())
            unknown_probability = float(prediction_outputs['probabilities'][idx, -1].cpu().item())
            distance_values = [
                '{:.6f}'.format(float(prediction_outputs['distances'][idx, class_index].cpu().item()))
                for class_index, _ in enumerate(class_labels)
            ]
            f_out.write(
                '{},{},{},{:.6f},{:.6f},{},{}\n'.format(
                    event_name,
                    true_label,
                    predicted_label,
                    min_distance,
                    unknown_probability,
                    ','.join(distance_values),
                    '{:.6f}'.format(float(model.unknown_radius.detach().cpu().item())),
                    'predict',
                )
            )

    print('Best epoch: {}'.format(best_epoch + 1))
    if test_metrics is not None:
        print('Test accuracy: {:.2f}%'.format(test_metrics['accuracy'] * 100.0))
    print('Calibrated unknown radius: {:.4f}'.format(float(model.unknown_radius.detach().cpu().item())))
    print('Saved model: {}'.format(preferred_model_path))
    print('Prediction file: {}'.format(prediction_output_file))


if __name__ == '__main__':
    main()
