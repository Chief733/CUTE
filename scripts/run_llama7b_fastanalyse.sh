#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_llama7b_sidecar.sh" \
  --fast-analyse \
  --no-official-analyse \
  --fast-workers 8 \
  --require-complete-step-trace \
  --dashboard \
  "$@"
