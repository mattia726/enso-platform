# External nuclei dataset usability audit

This audit checks the nuclei datasets discussed in `chat.txt` against the local
TCGA raw-slide namespace. A slide is counted as usable only when the dataset
identifier can be joined to a specific SVS in `bucket_physical_scan.csv`.

Generated with:

```bash
python scripts/audit_external_nuclei_datasets.py
```

Detailed outputs:

- `data/reports/external_nuclei_datasets/external_nuclei_dataset_usability_summary.csv`
- `data/reports/external_nuclei_datasets/external_nuclei_dataset_usability_summary.json`
- `data/reports/external_nuclei_datasets/monuseg_bucket_overlap.csv`
- `data/reports/external_nuclei_datasets/nucls_bucket_overlap.csv`
- `data/reports/external_nuclei_datasets/cryonuseg_bucket_overlap.csv`

## Summary

| Dataset | Exact usable raw slides | Exact usable raw files | Dataset records checked | Standalone patch/FOV count | Nuclei labels/counts | Training-use note |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Pan-Cancer-Nuclei-Seg | 6,065 | 6,065 | 6,075 ANN rows | n/a | 6,170,915,689 slide-level nuclei annotations | Best primary source. Exact DX slide match to the physical bucket scan; boundaries are downloadable from DICOM ANN bulk data. |
| MoNuSeg | 51 | 51 | 51 patches/source barcodes | 51 patches | 30,837 instance labels in the parquet mirror | Small but exact TCGA source-slide overlap. Useful as a patch benchmark or auxiliary supervision. |
| NuCLS | 124 | 124 | 125 short DX slide names | 2,168 uncorrected / 1,744 corrected single-rater FOVs | 65,568 uncorrected / 59,485 corrected single-rater nuclei | Good auxiliary source. One listed slide, `TCGA-A2-A0D2-DX1`, is missing from the physical scan. |
| CryoNuSeg | 0 | 0 | 30 selected WSI UUIDs | 30 patches | 8,044 final manual-markup nuclei | Exact GDC UUIDs in `Selected_WSIs.xlsx` do not match the current physical scan. There are 30 case overlaps and 66 raw-slide case-level candidates, but not exact training-ready slide matches. |
| PanNuke | 0 | 0 | 7,904 patches | 7,904 patches | 205,343 nuclei | Useful as standalone patch-level teacher/benchmark data; public metadata checked does not expose TCGA WSI IDs. |
| HoVer-Net / CoNSeP | 0 | 0 | 41 images | 41 images | 24,319 nuclei | Standalone colorectal benchmark, not TCGA-reconnectable. |
| OpenTME | unknown | unknown | 3,634 gated TCGA diagnostic WSIs | n/a | Aggregate count/density readouts for nine cell types | Gated and license-restricted; not usable for training labels or pseudo-labels for this model without a different agreement. |
| CellViT | 0 | 0 | n/a | n/a | n/a | Model/workflow, not a separate reconnectable slide dataset. |
| StarDist | 0 | 0 | n/a | n/a | n/a | Model/package, not a separate reconnectable slide dataset. |
| Cellpose | 0 | 0 | n/a | n/a | n/a | Model/package, not a separate reconnectable slide dataset. |

## Source Notes

- Pan-Cancer-Nuclei-Seg DICOM ANN stores nuclei as closed polygons plus nucleus
  areas and provides IDC manifests for bulk download.
- MoNuSeg Hugging Face mirror exposes 51 rows split into 37 train and 14 test
  rows, with a `patient` field containing TCGA slide barcodes.
- NuCLS single-rater data publishes the slide list and reports both corrected
  and uncorrected FOV/nuclei totals.
- CryoNuSeg publishes `Selected_WSIs.xlsx`, but the UUIDs in its GDC URLs appear
  to be historical or otherwise absent from the current physical scan. Because
  only case IDs reconnect, these should not be used as exact WSI labels.
- OpenTME is a gated TCGA-derived TME profile dataset. It is useful background
  context, but the published terms prohibit using it to create model supervision
  for an Atlas H&E-TME-like task.

