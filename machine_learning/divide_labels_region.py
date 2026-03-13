import os
import random
from pathlib import Path
from sys import argv
import csv

frac_train = 0.7
frac_valid = 0.15
frac_test = 0.15

def get_region_name(model_dir):
    return Path(model_dir).resolve().name

def find_labels_file(model_dir, region):
    candidates = [
        os.path.join(model_dir, 'labels_scalogram_{}.csv'.format(region)),
        os.path.join(model_dir, 'labels_plus_{}.csv'.format(region)),
        os.path.join(model_dir, 'labels_{}.csv'.format(region)),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError('Could not find labels file in {}'.format(model_dir))


if len(argv) != 2:
    raise SystemExit('Usage: python divide_labels_region.py <model_dir>')

model_dir = os.path.abspath(argv[1])
region = get_region_name(model_dir)
labels_file = find_labels_file(model_dir, region)

print(region)
with open(labels_file, 'r') as f:
    lines = [line for line in f.readlines() if line.strip()]
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    ntot = len(lines)

    n_expl = 0
    n_quake = 0
    n_total = 0

    for line in lines:
        items = next(csv.reader([line], skipinitialspace=True))
        e_type = int(items[1])

        n_total += 1
        if e_type == 0:
            n_quake += 1
        elif e_type == 1:
            n_expl += 1

    n_train = int(ntot*frac_train)
    n_valid = int(ntot*frac_valid)
    n_test = int(ntot*frac_test)

    lines_train = lines[0:n_train]
    lines_valid = lines[n_train:n_train+n_valid]
    lines_test = lines[n_train+n_valid:]

    fout_train = open(os.path.join(model_dir, 'labels_train_{}.csv'.format(region)),'w')
    fout_valid = open(os.path.join(model_dir, 'labels_valid_{}.csv'.format(region)),'w')
    fout_test = open(os.path.join(model_dir, 'labels_test_{}.csv'.format(region)),'w')

    for line in lines_train:
        fout_train.write(line)

    for line in lines_valid:
        fout_valid.write(line)

    for line in lines_test:
        fout_test.write(line)

    fout_train.close()
    fout_valid.close()
    fout_test.close()

    print('{} out of {} records from {} are from earthquakes ({} %)'.format(n_quake,n_total,region,(n_quake/n_total)*100.))
