# blast_net

Tools for CNN-based seismic source discrimination.

## Environment

Create the conda environment from the repo root:

```bash
conda env create -f environment.yml
conda activate blast_net
```

If you already created the environment and want to update it:

```bash
conda env update -f environment.yml --prune
```

The current workflow scripts use `processing/add_travel_times.py`, which builds ObsPy TauP models directly from the regional `.nd` velocity models in `data/vel_model`.

If you use the legacy `processing/add_travel_times_taup.py` path instead, you will also need the `taup_time` command available on your `PATH`.

## GASC Full Run

The GASC regional example covers the end-to-end processing workflow from raw waveform download through scalogram generation and regional CNN training.

Input files used by the processing step:

- `params/params_gasc.dat`
- `catalogs/GASC_catalog.txt`
- `scripts/run_gasc_processing.sh`

Run the processing workflow from the repo root:

```bash
./scripts/run_gasc_processing.sh
```

To rerun and replace any existing outputs:

```bash
OVERWRITE=1 ./scripts/run_gasc_processing.sh
```

What the processing script does:

1. Uses `params/params_gasc.dat` as the template parameter file.
2. Rewrites the catalog and output paths into a local params file under `data/`.
3. Downloads raw waveform data with `processing/get_data.py`.
4. Preprocesses the waveforms with `processing/pre_process.py`.
5. Adds travel times with `processing/add_travel_times.py`.
6. Computes P/S ratio and SNR with `processing/get_ps_ratio.py`.

Main processing outputs:

- `data/asdf_datasets/gasc_raw.h5`
- `data/asdf_datasets/gasc_processed.h5`
- `data/params_gasc_local.dat`

## GASC Scalograms

After the processed ASDF file is created, generate the GASC scalograms and labels:

```bash
cd scripts
bash do_scalograms_gasc.sh
cd ..
```

This uses `scripts/run_scalograms_region.sh` to create:

- `models/gasc/training_data/`
- `models/gasc/labels_scalogram_gasc.csv`

## GASC CNN Training

Split the labels and train the regional CNN from the `machine_learning` directory:

```bash
cd machine_learning
python divide_labels_region.py ../models/gasc
python cnn_region_plus.py ../models/gasc
cd ..
```

This writes the split label files and model outputs into `models/gasc`, including:

- `models/gasc/labels_train_gasc.csv`
- `models/gasc/labels_valid_gasc.csv`
- `models/gasc/labels_test_gasc.csv`
- `models/gasc/training_output.dat`
- `models/gasc/predict_all_test.dat`
- `models/gasc/preferred_model_plus_gasc.pt`
- `models/gasc/saved_models/`

## WVSZ Example

The WVSZ test example downloads waveform data, preprocesses it, adds P and S travel times, and computes P/S ratio and SNR.

Input files used by the example:

- `params/params_wvsz.dat`
- `catalogs/WVSZ_catalog_test.txt`
- `scripts/run_wvsz_test_processing.sh`

Run the full workflow from the repo root:

```bash
./scripts/run_wvsz_test_processing.sh
```

To rerun and replace any existing outputs:

```bash
OVERWRITE=1 ./scripts/run_wvsz_test_processing.sh
```

What the script does:

1. Uses `params/params_wvsz.dat` as the template parameter file.
2. Rewrites the catalog and output paths into a local params file under `data/`.
3. Downloads raw waveform data with `processing/get_data.py`.
4. Preprocesses the waveforms with `processing/pre_process.py`.
5. Adds travel times with `processing/add_travel_times.py`.
6. Computes P/S ratio and SNR with `processing/get_ps_ratio.py`.

Main outputs:

- `data/asdf_datasets/wvsz_test_raw.h5`
- `data/asdf_datasets/wvsz_test_processed.h5`
- `data/params_wvsz_test_local.dat`

The processed ASDF file contains the derived travel times, SNR values, and P/S ratios added in place during the later workflow steps.
