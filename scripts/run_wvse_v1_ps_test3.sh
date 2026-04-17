#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

PYTHON_BIN="${PYTHON_BIN:-/Users/rossrm/anaconda3/bin/python}"
PROCESSED_FILE="${1:-${ROOT_DIR}/data/asdf_datasets/wvse_v1_processed.h5}"

export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

if [[ ! -f "${PROCESSED_FILE}" ]]; then
    echo "Processed ASDF file not found: ${PROCESSED_FILE}" >&2
    exit 1
fi

echo "Computing WVSE P/S ratios with test3 windowing"
echo "Processed file: ${PROCESSED_FILE}"

"${PYTHON_BIN}" "${ROOT_DIR}/processing/get_ps_ratio.py" "${PROCESSED_FILE}" --window-mode test3

echo "Finished test3 P/S ratio run"
