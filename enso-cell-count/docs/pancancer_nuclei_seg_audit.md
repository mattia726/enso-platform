# Pan-Cancer-Nuclei-Seg Audit

Date: 2026-05-13

## Purpose

This audit checks whether Pan-Cancer-Nuclei-Seg can be used as an
EnsoCellularity training source and whether its TCGA image names reconnect to
the frozen-section wedge slides already embedded for the purity model.

## Public Data Source

Pan-Cancer-Nuclei-Seg provides TCGA H&E nuclei segmentations. The DICOM release
in IDC/Zenodo stores the data as DICOM Bulk Simple Annotation (`ANN`) polygon
objects and DICOM Segmentation (`SEG`) raster masks. For count labels, the
`ANN` objects are the preferred source because each nucleus remains an
instance polygon.

Sources:

- Zenodo DICOM release: https://zenodo.org/records/14009675
- Original TCIA analysis result: https://www.cancerimagingarchive.net/analysis-result/pan-cancer-nuclei-seg/
- Scientific Data descriptor: https://www.nature.com/articles/s41597-020-0528-1
- IDC DICOMweb docs: https://learn.canceridc.dev/data/downloading-data/dicomweb-access

## What We Can Download

The dataset exposes both slide-level counts and boundary data:

- `NumberOfAnnotations` in each `ANN` DICOM annotation group gives the
  slide-level nuclei count.
- `PointCoordinatesData` gives polygon coordinates.
- `LongPrimitivePointIndexList` gives polygon boundaries/primitive splits when
  present.
- measurement bulk data gives nucleus area values when present.
- full `ANN` DICOM objects are downloadable from public IDC buckets via the
  `series_aws_url` prefix, and the same metadata exposes DICOMweb bulk-data
  URIs for the coordinate arrays.

A downloaded sample `ANN` object for `TCGA-E2-A14S-01Z-00-DX1` had:

- `Modality = ANN`
- `ContainerIdentifier = TCGA-E2-A14S-01Z-00-DX1`
- `GraphicType = POLYGON`
- `NumberOfAnnotations = 352609`

This confirms the boundary/count source is usable for cell-count training.

## Generated Repo Artifacts

The audit generated a normalized slide-level manifest:

- `data/processed/pancancer_nuclei_seg_ann_manifest.csv`

Key columns:

- `slide_barcode`
- `case_id`
- `sample_vial`
- `sample_type_code`
- `section_type`
- `nuclei_count_slide`
- `series_aws_url`
- `dicomweb_metadata_url`
- `point_coordinates_bulk_uri`
- `primitive_index_bulk_uri`
- `area_measurements_bulk_uri`

Overlap reports were written to:

- `data/reports/pancancer_nuclei_seg/summary.json`
- `data/reports/pancancer_nuclei_seg/exact_slide_overlap.csv`
- `data/reports/pancancer_nuclei_seg/case_overlap.csv`
- `data/reports/pancancer_nuclei_seg/case_sample_type_overlap.csv`

The manifest can be regenerated with:

```bash
python scripts/build_pancancer_nuclei_seg_manifest.py --refresh
```

For quick regeneration from a cached raw metadata CSV:

```bash
python scripts/build_pancancer_nuclei_seg_manifest.py \
  --input-metadata data/interim/pancancer_nuclei_seg/pancancer_ann_metadata.csv
```

Downloaded `ANN` DICOM files can be converted to per-nucleus centroid tables
with:

```python
from enso_cellularity.ann_dicom import read_ann_centroids

centroids = read_ann_centroids("path/to/annotation.dcm")
```

On the sample `TCGA-E2-A14S-01Z-00-DX1` annotation object, this returned
352,609 centroid rows, matching the DICOM `NumberOfAnnotations`.

## Audit Results

Pan-Cancer-Nuclei-Seg `ANN` manifest:

- rows: 6,075
- unique slide barcodes: 6,065
- unique cases: 5,185
- total nuclei annotations: 6,170,915,689
- section types: all `DX`
- sample types: `01` = 5,902, `02` = 2, `06` = 170, `07` = 1

Wedge purity manifest:

- rows: 18,255
- unique slide barcodes: 18,255
- unique cases: 10,892
- section types: `TS`, `BS`, `MS`

Overlap with `data/processed/wedge_mvp_dataset.xlsx`:

- exact slide-barcode overlap: 0
- sample-vial overlap: 0
- case overlap: 5,125
- case + sample-type overlap: 4,928

Overlap with the broader TCGA slide/bucket metadata:

- `data/raw/slides_metadata_report(1).xlsx`: 6,075 / 6,075 Pan-Cancer
  annotation rows matched by exact slide barcode, representing 6,065 unique
  DX slide barcodes and 5,185 cases.
- `C:/Users/zxxx4/embeddings/embeddings/bucket_physical_scan.csv`: 6,075 /
  6,075 Pan-Cancer annotation rows matched by exact slide barcode, representing
  6,065 unique raw SVS file IDs.
- `C:/Users/zxxx4/embeddings/embeddings/slides_metadata_report.csv`: 6,075 /
  6,075 Pan-Cancer annotation rows matched by exact slide barcode.

This means the annotated DX slides are present in the physical TCGA SVS bucket
scan and can be downloaded/embedded. The combined source manifest is:

- `data/reports/pancancer_nuclei_seg/pancancer_dx_embedding_source_manifest.csv`

It contains the Pan-Cancer annotation IDs and count labels plus raw SVS
`file_id`, `file_name`, `full_path`, file size, dimensions, MPP when available,
and DICOM bulk-data links for the polygon boundaries. All 6,075 matched rows
had slide metadata status `OK`; 34 rows lacked MPP fields in the metadata scan
and should be handled by extraction-time metadata fallback. There were 20
annotation rows covering 10 duplicate slide barcodes, so training code should
deduplicate or intentionally keep multiple annotations per slide.

Overlap with slides used in `logs/v3_allfolds_alltiles_predictions_191512.csv`:

- resolved used wedge file UUIDs: 12,572
- used wedge cases: 9,165
- case overlap with Pan-Cancer-Nuclei-Seg: 4,502
- case + sample-type overlap: 4,322

## Interpretation

Pan-Cancer-Nuclei-Seg is a strong first training source for EnsoCellularity,
but it is not a direct label source for the existing frozen-section wedge H5s.
The naming evidence is clear:

- Pan-Cancer-Nuclei-Seg slides are diagnostic `DX` slides.
- The current wedge/purity embeddings are frozen-section `TS`, `BS`, and `MS`
  slides addressed by GDC file UUID.
- The same TCGA cases often overlap, but the exact slide names and sample-vials
  do not.

That is fine for the revised plan: train an initial DX EnsoCellularity model on
Pan-Cancer-Nuclei-Seg, then add domain adaptation and validation on frozen
TS/BS/MS tiles. The DX model can teach the student relationship:

```text
Virchow tile embedding -> nuclei count per tile
```

The frozen wedge data still matters later because it is the production domain.
For that domain, we should generate teacher-segmentation labels directly on
the raw TS/BS/MS tiles and use a small human audit set for calibration.

## Next Technical Step

Build tile-level labels from the `ANN` polygons:

1. Download a small subset of `ANN` DICOM objects and matching TCGA DX slide
   images.
2. Parse `PointCoordinatesData` and polygon primitive indices.
3. Convert each polygon to a centroid in slide pixel coordinates.
4. Join centroids to the embedding tile grid using the H5 `coords`, `tile_size`,
   and `stride`.
5. Count nuclei by centroid-in-tile, producing:

```text
data/processed/tile_cellularity_labels.parquet
```

The initial row schema should include:

```text
slide_barcode
case_id
project_id
tile_x
tile_y
tile_size
stride
mpp_x
mpp_y
tile_area_mm2
embedding_index
teacher_total_nuclei
source
quality_flags
```

For DX training, we also need Virchow embeddings for the DX slides. The existing
wedge H5 files cannot be reused directly because they belong to different
slides, even when the case is shared.
