import glob
import random
import numpy as np

#labels_in = glob.glob('labels*.csv')
regions = ['msh','base','gasc','idor','hlp','spe','ssip','enam','sima','rifsis']

frac_train = 0.7
frac_valid = 0.15
frac_test = 0.15

for region in regions:

    print(region)
    fname = 'labels_{}.csv'.format(region)
    f = open(fname,'r')
    lines = f.readlines()
    random.shuffle(lines)
    random.shuffle(lines)
    random.shuffle(lines)
    ntot = len(lines)

    n_expl = 0
    n_quake = 0
    n_total = 0

    for line in lines:
        items = line.split()
        e_type = int(items[-1])

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
    lines_test = lines[n_train+n_valid:n_train+n_valid+n_test]

    fout_train = open('labels_train_{}.csv'.format(region),'w')
    fout_valid = open('labels_valid_{}.csv'.format(region),'w')
    fout_test = open('labels_test_{}.csv'.format(region),'w')

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
