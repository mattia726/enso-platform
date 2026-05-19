#!/usr/bin/env bash
# Run this script ON THE GCP VM (where bucket is mounted and model/cache exist)
# to generate all demo artifacts: gallery (interactive heatmaps), stats, per-cancer stats, scatter.
#
# Prereqs on VM: env activated (e.g. conda enso), repo at ~/enso_workspace (or set ENSO_ROOT),
#   bucket mounted at ~/bucket_embeddings (embeddings_fp32/), data/processed/wedge_mvp_dataset*.xlsx.
#
# After generation: visually review each interactive_*.html (or thumbnails) and add any slide
# that shows a pen marker to data/exclude_markers.txt (one file_uuid_original or barcode per line),
# then re-run the gallery step only.

set -e
ENSO_ROOT="${ENSO_ROOT:-$HOME/enso_workspace}"
cd "$ENSO_ROOT"

# Mount GCS bucket if not already mounted (required for H5 embedding files)
BUCKET_MOUNT="${BUCKET_MOUNT:-$HOME/bucket_embeddings}"
BUCKET_NAME="${BUCKET_NAME:-embeddings-tcga-virchow}"
if [[ ! -d "$BUCKET_MOUNT/embeddings_fp32" ]] || ! ls "$BUCKET_MOUNT"/embeddings_fp32/*.h5 &>/dev/null; then
  echo "=== Mounting GCS bucket $BUCKET_NAME at $BUCKET_MOUNT ==="
  mkdir -p "$BUCKET_MOUNT"
  gcsfuse --implicit-dirs "$BUCKET_NAME" "$BUCKET_MOUNT" || true
fi

H5_DIR="${H5_DIR:-$BUCKET_MOUNT/embeddings_fp32}"
CACHE_DIR="${CACHE_DIR:-$ENSO_ROOT/data/cache}"
MANIFEST="$ENSO_ROOT/data/processed/wedge_mvp_dataset.xlsx"
[[ -f "$ENSO_ROOT/data/processed/wedge_mvp_dataset(1).xlsx" ]] && MANIFEST="$ENSO_ROOT/data/processed/wedge_mvp_dataset(1).xlsx"
EXCLUDE="$ENSO_ROOT/data/exclude_markers.txt"
[[ ! -f "$EXCLUDE" ]] && touch "$EXCLUDE"

echo "=== 1) Statistical tests (global Rho, MAE, scatter) ==="
python -m enso_purity_mil.statistical_tests \
  --model-path "$ENSO_ROOT/ml/runs/fold0/best_model.pth" \
  --manifest "$MANIFEST" \
  --h5-dir "$H5_DIR" \
  --cache-dir "$CACHE_DIR" \
  --fold 0 \
  --out-dir "$ENSO_ROOT/ml/runs/fold0/stats"

echo "=== 2) Per-cancer statistical tests ==="
python -m enso_purity_mil.per_cancer_statistical_tests \
  --model-path "$ENSO_ROOT/ml/runs/fold0/best_model.pth" \
  --manifest "$MANIFEST" \
  --h5-dir "$H5_DIR" \
  --cache-dir "$CACHE_DIR" \
  --fold 0 \
  --out-dir "$ENSO_ROOT/ml/runs/fold0/stats"

echo "=== 3) Gallery: one per cancer, interactive heatmaps (no pen-marker check; review after) ==="
python -m enso_purity_mil.build_demo_gallery \
  --model-path "$ENSO_ROOT/ml/runs/fold0/best_model.pth" \
  --manifest "$MANIFEST" \
  --h5-dir "$H5_DIR" \
  --cache-dir "$CACHE_DIR" \
  --out-dir "$ENSO_ROOT/frontend/gallery" \
  --one-per-cancer \
  --err-limit 0.15 \
  --exclude-markers "$EXCLUDE"

echo "=== 4) Export static cases (base JPG + mask PNG for native viewer / Cloudflare) ==="
mkdir -p "$ENSO_ROOT/frontend/public/cases"
python -m enso_purity_mil.export_static_cases \
  --model-path "$ENSO_ROOT/ml/runs/fold0/best_model.pth" \
  --h5-dir "$H5_DIR" \
  --gallery-csv "$ENSO_ROOT/frontend/gallery/gallery_summary.csv" \
  --out-dir "$ENSO_ROOT/frontend/public/cases"

echo "=== Done. Copy to local: scp -r \$VM:~/enso_workspace/ml/runs/fold0/stats/* frontend/public/data/ ==="
echo "=== And: scp -r \$VM:~/enso_workspace/frontend/gallery/* frontend/public/gallery/ ==="
echo "=== And: scp -r \$VM:~/enso_workspace/frontend/public/cases/* frontend/public/cases/ ==="
echo "=== Then review heatmaps; add pen-marker slide UUIDs/barcodes to data/exclude_markers.txt and re-run step 3 (and 4). ==="
