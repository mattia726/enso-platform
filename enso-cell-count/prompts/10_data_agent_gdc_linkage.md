# Prompt: DATA agent — TCGA linkage + mismatch report (TDD)

Goal:
Implement a robust pipeline that links TCGA **H&E slides** to **sequencing aliquots** used by ABSOLUTE purity labels.

Inputs (local files):
- `data/raw/slides_metadata_report(1).xlsx` (has `file_id`, `full_path` containing slide barcode + UUID)
- `data/raw/TCGA_mastercalls.abs_tables_JSedit.fixed.txt` (ABSOLUTE purity table)

Tasks:
0) Filter slides FIRST (and report counts at each step):
   - Keep only rows where `base_mpp_x` (and/or `base_mpp_y`) is present and **< 2.0**
   - Exclude every slide whose *slide suffix* matches `DX*` where `*` can be digits or a capital letter
     (examples: `DX1`, `DX2`, `DXA`, `DXB`, ...)
   - Keep slide suffixes starting with: `TS`, `MS`, `BS` (and keep the suffix value for stats)

1) Implement barcode parsers:
   - slide barcode: `TCGA-XX-YYYY-01A-01-TS1`
   - aliquot barcode: `TCGA-XX-YYYY-01A-11D-....`
2) Implement **naive linkage**:
   - vial-level match on `TCGA-XX-YYYY-01A`
   - portion-level match on `TCGA-XX-YYYY-01A-11`
   Output mismatch metrics.
3) Implement **API-verified linkage** using GDC API:
   - For each slide `file_id`, query relationships to portions / analytes / aliquots.
   - Compare the aliquot submitter_id(s) to ABSOLUTE samples.
   - Cache API responses to disk (jsonl) to avoid rate limits.
4) Produce a canonical dataset table:
   - columns: patient_id, sample_vial, portion, slide_id, file_id, purity, purity_source
   - keep TS/BS for same patient in same split (patient-level split)
   - save `outputs/dataset.parquet` and `outputs/mismatch_report.json`
   - save plots under `data/reports/`

TDD:
- Unit tests for parsing and join logic with synthetic examples.
- Integration test is optional and should be skipped by default (`-m integration`).

Acceptance:
- `pytest -q` passes
- Running the script prints counts and saves outputs.

