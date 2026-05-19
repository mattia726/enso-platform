#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSO_ROOT="${ENSO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

FOLD="${FOLD:?Set FOLD=0..4}"
RUN_TAG="${RUN_TAG:-tcga_cptac_retrain_$(date -u +%Y%m%dT%H%M%SZ)}"
OUT_DIR="${OUT_DIR:-$ENSO_ROOT/ml/runs/${RUN_TAG}}"
LOG_FILE="${LOG_FILE:-$OUT_DIR/fold${FOLD}.train.log}"

MANIFEST="${MANIFEST:?Set MANIFEST to the combined slide manifest TSV}"
H5_DIR="${H5_DIR:?Set H5_DIR to the unified H5 namespace directory}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR to the fp32 bag cache directory}"

DEVICE="${DEVICE:-cuda}"
BATCH_SIZE="${BATCH_SIZE:-8}"
EFFECTIVE_BATCH_SIZE="${EFFECTIVE_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_EPOCHS="${MAX_EPOCHS:-200}"
PATIENCE="${PATIENCE:-20}"
SEED="${SEED:-42}"

mkdir -p "$OUT_DIR" "$(dirname "$LOG_FILE")"

cd "$ENSO_ROOT"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"
export PYTHONPATH="$ENSO_ROOT/ml:${PYTHONPATH:-}"

"$PYTHON_BIN" -u -m enso_purity_mil.train_cli \
  --manifest "$MANIFEST" \
  --h5-dir "$H5_DIR" \
  --cache-dir "$CACHE_DIR" \
  --out-dir "$OUT_DIR" \
  --device "$DEVICE" \
  --fold "$FOLD" \
  --preassigned-fold-column preassigned_fold \
  --batch-size "$BATCH_SIZE" \
  --effective-batch-size "$EFFECTIVE_BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --max-epochs "$MAX_EPOCHS" \
  --patience "$PATIENCE" \
  --seed "$SEED" \
  2>&1 | tee "$LOG_FILE"
