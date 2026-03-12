#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ROOT_DIR=$(cd -- "${SCRIPT_DIR}/.." && pwd)

REGION="gasc"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PARAMS_TEMPLATE="${ROOT_DIR}/params/params_${REGION}.dat"
GENERATED_PARAMS="${ROOT_DIR}/data/params_${REGION}_local.dat"
CATALOG_FILE="${ROOT_DIR}/catalogs/GASC_catalog.txt"
ASDF_DIR="${ROOT_DIR}/data/asdf_datasets"
RAW_FILE="${ASDF_DIR}/${REGION}_raw.h5"
PROCESSED_FILE="${ASDF_DIR}/${REGION}_processed.h5"
TAUP_MODEL="${ROOT_DIR}/data/vel_model/${REGION}.taup"

require_file() {
    local path="$1"
    if [[ ! -f "${path}" ]]; then
        echo "Required file not found: ${path}" >&2
        exit 1
    fi
}

guard_output() {
    local path="$1"
    if [[ -e "${path}" && "${OVERWRITE:-0}" != "1" ]]; then
        echo "Output already exists: ${path}" >&2
        echo "Set OVERWRITE=1 to remove and rebuild it." >&2
        exit 1
    fi
}

echo "Preparing ${REGION} processing run"

require_file "${PARAMS_TEMPLATE}"
require_file "${CATALOG_FILE}"
require_file "${TAUP_MODEL}"

mkdir -p "${ROOT_DIR}/data"
mkdir -p "${ASDF_DIR}"

guard_output "${RAW_FILE}"
guard_output "${PROCESSED_FILE}"
guard_output "${GENERATED_PARAMS}"

if [[ "${OVERWRITE:-0}" == "1" ]]; then
    rm -f "${RAW_FILE}" "${PROCESSED_FILE}" "${GENERATED_PARAMS}"
fi

sed \
    -e "s|^filename.*|filename        = ${RAW_FILE}|" \
    -e "s|^catalog.*|catalog         = ${CATALOG_FILE}|" \
    "${PARAMS_TEMPLATE}" > "${GENERATED_PARAMS}"

echo "Using params file: ${GENERATED_PARAMS}"
echo "TauP model: ${TAUP_MODEL}"

echo "Step 1/4: download raw waveforms"
"${PYTHON_BIN}" "${ROOT_DIR}/processing/get_data.py" "${GENERATED_PARAMS}"

echo "Step 2/4: preprocess waveforms"
"${PYTHON_BIN}" "${ROOT_DIR}/processing/pre_process.py" "${RAW_FILE}" "${PROCESSED_FILE}"

echo "Step 3/4: add travel times"
"${PYTHON_BIN}" "${ROOT_DIR}/processing/add_travel_times_taup.py" "${PROCESSED_FILE}" "${TAUP_MODEL}"

echo "Step 4/4: compute P/S ratios and SNR"
"${PYTHON_BIN}" "${ROOT_DIR}/processing/get_ps_ratio.py" "${PROCESSED_FILE}"

echo "Finished. Processed dataset: ${PROCESSED_FILE}"
