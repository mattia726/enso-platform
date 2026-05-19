# TCGA FS + CPTAC DX Retrain

Run tag: `tcga_cptac_retrain_20260411T023052Z_retry1`

This is the canonical retrain that extends the historical TCGA frozen-section pipeline with CPTAC DX while preserving v3 training semantics.

## Inputs

- TCGA tumour/normal source: `data/processed/wedge_mvp_dataset.xlsx`
- Historical TCGA tumour folds: `logs/v3_allfolds_alltiles_predictions_191512.csv`
- CPTAC tumour linkage: `data/processed/cptac_slides_ngs_purity_final.csv`
- CPTAC normals source: `data/processed/cptac_master_normals.csv`

## Semantics

- TCGA tumour folds are preserved from `logs/v3_allfolds_alltiles_predictions_191512.csv`.
- CPTAC tumour folds are created in the combined manifest with patient-locking, canonical cancer mapping, and purity-bin stratification.
- TCGA tumour bags stay pooled by aliquot.
- TCGA normal bags stay single-slide and train-only.
- CPTAC tumour bags stay single-slide.
- CPTAC normal bags stay single-slide, are fold-assigned for patient consistency, and remain train-only.
- Validation/test remain tumour-only.

## Combined Manifest Counts

- Historical v3 TCGA tumour OOF baseline: `9217` bags
- Filtered direct-comparison TCGA universe after removing exact purity `1.0`: `9126` bags
- Final combined manifest:
  - `tumour_bag_count = 10549`
  - `normal_bag_count = 3899`
  - `expected_cache_bag_count = 14448`
  - `tcga_tumour_bags = 9126`
  - `tcga_normal_bags = 2754`
  - `cptac_tumour_bags = 1423`
  - `cptac_normal_bags = 1145`

## CPTAC Provenance

- `cptac_slides_ngs_purity_final.csv` contains `1773` final linked CPTAC tumour rows.
- `156` CPTAC rows use the audited `legacy_sample_specific_only` fallback linkage.
- `6` CPTAC rows carry `conflicting_ngs_purity_values` audit flags and are retained unless filtered elsewhere by purity or mapping policy.
- `cptac_master_normals.csv` is the canonical processed CPTAC normal-slide source.
- `1145` means CPTAC normal bags retained in the final combined manifest after H5-availability and tumour-case anchoring filters.

## Final Metrics

Canonical merged TCGA+CPTAC comparison universe: `10549` bags

| Model | rho | R^2 | MAE | MedAE |
| --- | ---: | ---: | ---: | ---: |
| Original v3 | `0.7588` | `0.5258` | `0.1028` | `0.0763` |
| Retrained TCGA+CPTAC | `0.7739` | `0.5611` | `0.0996` | `0.0743` |

Canonical CPTAC current-fold test universe: `1423` bags

| Model | rho | R^2 | MAE | MedAE |
| --- | ---: | ---: | ---: | ---: |
| Original v3 on current CPTAC test folds | `0.6071` | `0.0728` | `0.1285` | `0.0994` |
| Retrained TCGA+CPTAC | `0.6457` | `0.3847` | `0.1051` | `0.0814` |

## External Artifacts

- Azure/blob container: `embeddings-tcga-virchow`
- Run root prefix: `models/tcga_cptac_retrain_20260411T023052Z_retry1/artifact_staging`
- Canonical comparison bundle: `models/tcga_cptac_retrain_20260411T023052Z_retry1/artifact_staging/summary_definitive_tcga_cptac_from_excels`

Weights, checkpoints, exported workbooks, and comparison PNGs live in blob storage and are intentionally not committed to git.

## Rebuild Flow

```bash
python -m enso_purity_mil.build_tcga_cptac_retrain_manifest \
  --tcga-h5-dir /path/to/tcga_h5 \
  --cptac-h5-dir /path/to/cptac_h5 \
  --out-dir /path/to/manifests

python -m enso_purity_mil.build_union_h5_namespace \
  --manifest /path/to/manifests/tcga_cptac_combined_slide_manifest_v1.tsv \
  --tcga-h5-dir /path/to/tcga_h5 \
  --cptac-h5-dir /path/to/cptac_h5 \
  --out-dir /path/to/union_h5

python -m enso_purity_mil.build_cache \
  --manifest /path/to/manifests/tcga_cptac_combined_slide_manifest_v1.tsv \
  --h5-dir /path/to/union_h5 \
  --cache-dir /path/to/cache
```

Training and test helper scripts under `scripts/` require explicit `MANIFEST`, `H5_DIR`, and `CACHE_DIR` inputs so they stay environment-neutral.
