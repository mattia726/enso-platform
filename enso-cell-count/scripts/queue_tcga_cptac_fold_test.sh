#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENSO_ROOT="${ENSO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
RUN_TAG="${RUN_TAG:?Set RUN_TAG}"
FOLD="${FOLD:?Set FOLD}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MANIFEST="${MANIFEST:?Set MANIFEST to the combined slide manifest TSV}"
H5_DIR="${H5_DIR:?Set H5_DIR to the unified H5 namespace directory}"
CACHE_DIR="${CACHE_DIR:?Set CACHE_DIR to the fp32 bag cache directory}"
OUT_ROOT="${OUT_ROOT:-$ENSO_ROOT/ml/fold_assessments/${RUN_TAG}}"
TEST_SCRIPT="${TEST_SCRIPT:-$ENSO_ROOT/scripts/tcga_cptac_test_eval.py}"
MODEL_PATH="${MODEL_PATH:-$OUT_ROOT/fold${FOLD}/best_model.pth}"
TEST_OUT_DIR="${TEST_OUT_DIR:-$OUT_ROOT/fold${FOLD}_test}"
TEST_LOG="${TEST_LOG:-$OUT_ROOT/fold${FOLD}_test.nohup.log}"
NUM_BAGS="${NUM_BAGS:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"
PREASSIGNED_FOLD_COLUMN="${PREASSIGNED_FOLD_COLUMN:-preassigned_fold}"

mkdir -p "$OUT_ROOT"

while pgrep -af "enso_purity_mil.train_cli.*--fold ${FOLD}" >/dev/null; do
  sleep 60
done

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing model path: $MODEL_PATH" >&2
  exit 1
fi

if [[ -f "$TEST_OUT_DIR/fold${FOLD}_test_metrics.json" ]]; then
  echo "Fold ${FOLD} test already complete at $TEST_OUT_DIR" >&2
  exit 0
fi

mkdir -p "$TEST_OUT_DIR"
export PYTHONPATH="$ENSO_ROOT/ml:${PYTHONPATH:-}"
export HDF5_USE_FILE_LOCKING="${HDF5_USE_FILE_LOCKING:-FALSE}"

exec "$PYTHON_BIN" -u "$TEST_SCRIPT" \
  --manifest "$MANIFEST" \
  --h5-dir "$H5_DIR" \
  --cache-dir "$CACHE_DIR" \
  --model-path "$MODEL_PATH" \
  --out-dir "$TEST_OUT_DIR" \
  --fold "$FOLD" \
  --preassigned-fold-column "$PREASSIGNED_FOLD_COLUMN" \
  --device "$DEVICE" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --num-bags "$NUM_BAGS" \
  >> "$TEST_LOG" 2>&1
