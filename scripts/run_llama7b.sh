#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKDIR="${WORKDIR:-$(cd -- "${PRETRAIN_DIR}/.." && pwd)}"
MF_DIR="${MF_DIR:-${WORKDIR}/mindformers_r1.3.2}"
CONFIG="${CONFIG:-${PRETRAIN_DIR}/configs/llama2/pretrain_llama2_7b_20steps.yaml}"

cd "${MF_DIR}"

TRAIN_COMMAND="$(printf 'run_mindformer.py --config %q' "${CONFIG}")"
bash scripts/msrun_launcher.sh "${TRAIN_COMMAND}"
