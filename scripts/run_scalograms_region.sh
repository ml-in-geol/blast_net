#!/bin/bash

dset=$1
multiplier=$2
data_file=$3
python=/Users/rossrm/anaconda3/envs/pytorch_env/bin/python
out_dir=../models/${dset}/training_data

mkdir -p ${out_dir}
code_dir=../processing

echo data_file : ${data_file}
echo dset: ${dset}
echo out_dir: ${out_dir}
echo multiplier: ${multiplier}

$python ${code_dir}/plot_scalograms.py ${data_file} ${dset} ${out_dir} ${multiplier}
mv labels_scalogram_${dset}.csv ../models/${dset}
