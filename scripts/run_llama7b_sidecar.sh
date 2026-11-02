#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PRETRAIN_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
WORKDIR="${WORKDIR:-$(cd -- "${PRETRAIN_DIR}/.." && pwd)}"
MF_DIR="${MF_DIR:-${WORKDIR}/mindformers_r1.3.2}"
SIDECAR_DIR="${PRETRAIN_DIR}/profiler_sidecar"
CONFIG="${CONFIG:-${PRETRAIN_DIR}/configs/llama2/pretrain_llama2_7b_sidecar_20steps.yaml}"
OUTPUT="${OUTPUT:-${SIDECAR_DIR}/runs/llama7b_$(date -u +%Y%m%dT%H%M%S)_$$}"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
exec /root/miniconda3/envs/mindspore2.4_python3.10/bin/python \
  "${SIDECAR_DIR}/launch.py" \
  --config "${CONFIG}" \
  --output "${OUTPUT}" \
  --mf-dir "${MF_DIR}" \
  "$@"
