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

The processing workflow also expects the `taup_time` command to be available on your `PATH`, because `processing/add_travel_times_taup.py` uses the TauP command-line tool.

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
5. Adds travel times with `processing/add_travel_times_taup.py`.
6. Computes P/S ratio and SNR with `processing/get_ps_ratio.py`.

Main outputs:

- `data/asdf_datasets/wvsz_test_raw.h5`
- `data/asdf_datasets/wvsz_test_processed.h5`
- `data/params_wvsz_test_local.dat`

The processed ASDF file contains the derived travel times, SNR values, and P/S ratios added in place during the later workflow steps.
