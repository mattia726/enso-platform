# Dataset linkage spec (slides ↔ purity)

This document is the **source of truth** for the dataset-building agent task.

## Inputs

- `data/raw/slides_metadata_report(1).xlsx`
  - must contain `file_id`, `full_path`, and `base_mpp_x` / `base_mpp_y`
- `data/raw/TCGA_mastercalls.abs_tables_JSedit.fixed.txt`
  - tab-separated table with ABSOLUTE purity values

## Filtering rules

1) Keep only slides where `base_mpp_x` is present and `< 2.0`.
   - (Optionally also require `base_mpp_y` present and `< 2.0`.)
2) Exclude slide suffixes matching **DX*** where `*` is `[0-9A-Z]+`.
   - Examples: `DX1`, `DX2`, `DXA`, `DXB`.
3) Keep slide suffixes starting with: **TS**, **MS**, **BS**.

## Linkage rules

We must compare two strategies:

### A) Naive barcode slicing

- Extract from slide filename:
  - `patient` = `TCGA-XX-YYYY`
  - `sample_vial` = `01A` (2 digits + letter)
  - `portion` = `01` (2 digits)
  - `slide_suffix` = `TS1` / `BS1` / `MS1` / ...

Compute two joins against the ABSOLUTE table:

- **Vial-level**: match on `patient + '-' + sample_vial`
- **Portion-level**: match on `patient + '-' + sample_vial + '-' + portion`

### B) GDC biospecimen relationships

- Use GDC API to map `file_id` → biospecimen chain (slide → portion → aliquot/sample)
- Use the returned aliquot/sample barcode to fetch the correct purity.

## Required outputs

- A summary table of counts at each step:
  - initial rows
  - after MPP filter
  - after suffix filters
  - matched by vial-only (naive)
  - matched by portion (naive)
  - matched by GDC API
  - mismatch rate (naive vs API)

- Plots saved under `data/reports/`:
  - slide suffix histogram
  - match coverage bar chart (vial vs portion vs API)

## Deliverables

- Reusable code under `backend/src/enso_purity/data/`
- Script entrypoint under `backend/scripts/`
- Tests under `backend/tests/`
