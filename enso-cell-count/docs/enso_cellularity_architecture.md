# EnsoCellularity Architecture

EnsoCellularity is the tile-level companion model to EnsoPurity. It predicts
total nuclei count, exposed as an estimated nucleated-cell count, from Virchow
tile embeddings. EnsoPurity remains responsible for tumor fraction; the derived
tumor-cell quantity is:

```text
tumor nuclei per tile = total nuclei per tile * local tumor fraction
```

## Training Label Format

The canonical training labels are written as Parquet, one row per Virchow H5
embedding row:

```text
file_uuid_original
barcode
project_id
case_id
embedding_index
tile_y
tile_x
tile_x_level0
tile_y_level0
tile_w_level0
tile_h_level0
mpp_x
mpp_y
target_mpp
tile_area_mm2
tissue_fraction
exposure_mm2
teacher_total_nuclei
teacher_confidence
teacher_disagreement
quality_flags
source
nuclei_density_per_mm2
count_bin
ann_nuclei_count_slide
ann_counted_nuclei_in_embedding_tiles
ann_dicom_path
ann_source_uuid
```

For the Pan-Cancer-Nuclei-Seg labels, `source = pan_cancer_nuclei_seg`,
`teacher_confidence = 1.0`, and `teacher_disagreement = 0.0`. The current H5s
do not include per-tile tissue fraction, so the first Pan-Cancer pass uses
`tissue_fraction = 1.0` for kept tissue-filtered tiles and sets:

```text
exposure_mm2 = tile_area_mm2
```

When future teacher-segmentation runs produce tissue masks, they should replace
this with:

```text
exposure_mm2 = tile_area_mm2 * tissue_fraction
```

## Model

The implemented architecture in `ml/enso_cellularity/model.py` follows
`architecture.txt`:

```text
Inputs
  x9: center + 8 neighbor Virchow embeddings [B, 9, 2560]
  valid9: real-neighbor mask                 [B, 9]
  metadata: normalized tile metadata         [B, 5]
  exposure_mm2                               [B, 1]

Embedding projection
  LayerNorm(2560)
  Linear(2560 -> 512)
  GELU
  Dropout(0.05)

Context
  learned 3x3 positional embeddings
  center-query masked MultiheadAttention
  residual + LayerNorm

Metadata fusion
  metadata MLP: 5 -> 64 -> 128
  concat(context_512, metadata_128)
  Linear(640 -> 512)
  GELU
  LayerNorm

Trunk
  3 residual MLP blocks, each 512 -> 1024 -> 512

Heads
  density_per_mm2: Softplus
  alpha dispersion: Softplus
  ordinal count-bin logits
  non-crossing q05/q50/q95 on log1p scale
  quality/artifact logits
```

The expected count is exposure-normalized:

```text
mu_count = density_per_mm2 * exposure_mm2
```

The Negative Binomial parameterization follows NB2:

```text
Var[Y] = mu + alpha * mu^2
theta = 1 / alpha
nb_logits = log(mu) - log(theta)
```

## Loss

The composite loss in `ml/enso_cellularity/losses.py` is:

```text
L =
  1.00 * NegativeBinomialNLL
+ 0.25 * SmoothL1(log1p(mu), log1p(count))
+ 0.20 * OrdinalBCE
+ 0.10 * QuantilePinball
+ 0.20 * QualityLoss
```

The count losses are weighted by `teacher_confidence` when labels come from a
teacher/ensemble source.

## Pan-Cancer Processing

Pan-Cancer-Nuclei-Seg ANN DICOM files are converted into tile labels by:

1. Reading each ANN DICOM polygon object.
2. Computing fast vertex-mean centroids for each nucleus polygon.
3. Mapping centroids to the non-overlapping Virchow H5 tile grid in base-level
   WSI coordinates.
4. Writing one Parquet file per slide.

For small local checks, the filesystem builder is:

```bash
python scripts/build_pancancer_tile_cellularity_labels.py \
  --manifest data/reports/pancancer_nuclei_seg/pancancer_dx_embedding_source_manifest.csv \
  --h5-dir /mnt/dataset/embeddings_fp32 \
  --ann-root /mnt/dataset/pancancer_nuclei_seg_dicom/raw_ann \
  --out-dir /mnt/dataset/pancancer_nuclei_seg_dicom/tile_cellularity_labels \
  --workers 4
```

Do not use blobfuse or mounted Blob paths for the full Pan-Cancer conversion.
Large runs should use the direct Azure Blob endpoint builder, which downloads
one H5 and one ANN series to local scratch with `azcopy`, writes the Parquet
locally, uploads it directly to Blob, and then removes the scratch files:

```bash
python scripts/build_pancancer_tile_cellularity_labels_blob.py \
  --manifest pancancer_dx_embedding_source_manifest.csv \
  --base-url https://vmshareddisk.blob.core.windows.net/data \
  --h5-prefix embeddings_fp32 \
  --ann-prefix pancancer_nuclei_seg_dicom/raw_ann \
  --out-prefix pancancer_nuclei_seg_dicom/tile_cellularity_labels_direct \
  --scratch-dir scratch/direct_labels \
  --state-dir tile_cellularity_labels_direct_state \
  --workers 4
```

The output directory is resumable through:

```text
state/completed.tsv
state/failed.tsv
state/last_run_summary.json
```

## Training

The full training code lives under `ml/enso_cellularity/`:

```text
model.py       final 3x3 context architecture
losses.py      NB + log + ordinal + quantile + quality composite loss
dataset.py     per-slide Parquet/H5 loader with dynamic 3x3 neighbor gather
folds.py       case-level pan-cancer folds
metrics.py     tile and ROI count metrics
training.py    epoch loop, checkpointing, optimizer helpers
train_cli.py   full training CLI
evaluate_cli.py checkpoint evaluation on train/val/test splits
inference.py   checkpoint loading, H5 prediction, ROI aggregation
predict_cli.py inference CLI
```

The dataset does not duplicate 3x3 embeddings on disk. It stores the original
H5 feature matrix and gathers neighbor rows dynamically:

```text
features:       [num_tiles, 2560]
neighbor_index: computed from tile_y/tile_x at load time
x9 batch:       [batch_tiles, 9, 2560]
valid9 batch:   [batch_tiles, 9]
```

For full Pan-Cancer training, first make sure the H5 embeddings and label
Parquets are on local SSD or a direct-endpoint staging path. Do not train
through blobfuse for the massive H5/Parquet workload.

Example fold-0 training command:

```bash
cd /path/to/repo/ml
export HDF5_USE_FILE_LOCKING=FALSE

python -u -m enso_cellularity.train_cli \
  --label-dir /data/pancancer_nuclei_seg_dicom/tile_cellularity_labels_direct/by_slide \
  --h5-dir /data/embeddings_fp32 \
  --slide-index /data/pancancer_nuclei_seg_dicom/tile_cellularity_labels_direct/slide_index.csv \
  --out-dir runs_cellularity \
  --device cuda \
  --fold 0 \
  --tiles-per-slide 512 \
  --eval-tiles-per-slide 1024 \
  --slide-batch-size 8 \
  --eval-slide-batch-size 4 \
  --num-workers 4 \
  --max-epochs 80
```

For Azure runs where labels/H5s live in Blob, do not use blobfuse. Use the
direct-Blob backend with the completed state file from the preprocessing run:
For high-throughput training, prefer `--blob-transfer-mode sdk`; it uses the
Blob HTTPS endpoint with the VM managed identity without launching one `azcopy`
process per slide.

```bash
cd ~/pancancer_nuclei_ingest
export HDF5_USE_FILE_LOCKING=FALSE

~/pancancer-nuclei-venv/bin/python -u -m enso_cellularity.train_cli \
  --data-backend blob \
  --completed-tsv tile_cellularity_labels_direct_state/completed.tsv \
  --slide-index tile_cellularity_labels_direct_state/slide_index.csv \
  --blob-base-url https://vmshareddisk.blob.core.windows.net/data \
  --blob-h5-prefix embeddings_fp32 \
  --blob-scratch-dir /mnt/cellularity_scratch/cellularity_training_blob \
  --blob-transfer-mode sdk \
  --blob-sdk-max-concurrency 6 \
  --out-dir runs_cellularity \
  --device cuda \
  --fold 0 \
  --tiles-per-slide 8192 \
  --eval-tiles-per-slide 8192 \
  --slide-batch-size 1 \
  --eval-slide-batch-size 1 \
  --num-workers 5 \
  --prefetch-factor 2 \
  --persistent-workers
```

Outputs per fold:

```text
runs_cellularity/fold0/train_slides.csv
runs_cellularity/fold0/val_slides.csv
runs_cellularity/fold0/test_slides.csv
runs_cellularity/fold0/run_config.json
runs_cellularity/fold0/history.json
runs_cellularity/fold0/best_model.pth
runs_cellularity/fold0/latest_checkpoint.pth
```

Standalone evaluation on the validation split:

```bash
python -m enso_cellularity.evaluate_cli \
  --checkpoint runs_cellularity/fold0/best_model.pth \
  --split-csv runs_cellularity/fold0/val_slides.csv \
  --data-backend local \
  --tiles-per-slide 2048 \
  --out-json runs_cellularity/fold0/val_metrics.json \
  --device cuda
```

## Inference

Single-slide inference from an embedding H5:

```bash
python -m enso_cellularity.predict_cli \
  --checkpoint runs_cellularity/fold0/best_model.pth \
  --h5 /data/embeddings_fp32/<file_uuid>.h5 \
  --out /tmp/<file_uuid>.cellularity.parquet \
  --device cuda
```

The prediction table contains:

```text
embedding_index
tile_y
tile_x
tile_x_level0
tile_y_level0
pred_nuclei_count
pred_density_per_mm2
pred_alpha
pred_q05
pred_q50
pred_q95
pred_count_bin
quality_class
quality_good_prob
```

For selected-region aggregation, use `aggregate_roi_from_predictions`.
It sums expected counts and uses the NB variance approximation for ROI
uncertainty:

```text
Var(tile_i) = mu_i + alpha_i * mu_i^2
Var(ROI) = sum(coverage_i^2 * Var(tile_i))
```
