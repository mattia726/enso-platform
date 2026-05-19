#!/usr/bin/env python3
"""Audit external nuclei datasets against the local TCGA slide namespace.

The output separates exact raw-slide matches from weaker case-level candidates.
Exact matches are usable for EnsoCellularity tile-label generation because the
dataset identifier can be joined to a specific SVS in the physical bucket scan.
Case-only matches are useful for manual follow-up, but are not treated as
training-ready slide matches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPO_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from enso_cellularity.pancancer_nuclei import parse_tcga_slide_barcode  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--bucket-scan",
        type=Path,
        default=None,
        help="bucket_physical_scan.csv. Defaults to ../embeddings/bucket_physical_scan.csv.",
    )
    ap.add_argument(
        "--embedding-root",
        type=Path,
        default=REPO_ROOT.parent,
        help="Root to scan for local .h5 embedding files.",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=REPO_ROOT / "data" / "reports" / "external_nuclei_datasets",
        help="Output report directory.",
    )
    ap.add_argument(
        "--interim-dir",
        type=Path,
        default=REPO_ROOT / "data" / "interim" / "external_nuclei_datasets",
        help="Directory containing downloaded lightweight external manifests.",
    )
    return ap.parse_args()


def _default_bucket_scan() -> Path:
    candidates = [
        REPO_ROOT.parent / "embeddings" / "bucket_physical_scan.csv",
        REPO_ROOT / "data" / "raw" / "bucket_physical_scan.csv",
        REPO_ROOT / "bucket_physical_scan.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def _slide_from_file_name(value: Any) -> str:
    return str(value).split(".")[0].strip().upper()


def _add_slide_fields(df: pd.DataFrame, source_col: str, prefix: str = "") -> pd.DataFrame:
    out = df.copy()
    parsed = out[source_col].map(parse_tcga_slide_barcode)
    out[f"{prefix}slide_barcode"] = [p.barcode for p in parsed]
    out[f"{prefix}case_id"] = [p.case_id for p in parsed]
    out[f"{prefix}sample_vial"] = [p.sample_vial for p in parsed]
    out[f"{prefix}sample_type_code"] = [p.sample_type_code for p in parsed]
    out[f"{prefix}section_type"] = [p.section_type for p in parsed]
    out[f"{prefix}section_number"] = [p.section_number for p in parsed]
    out[f"{prefix}section_key"] = [
        f"{p.case_id}-{p.section_type}-{p.section_number}" for p in parsed
    ]
    return out


def _load_bucket(path: Path) -> pd.DataFrame:
    bucket = pd.read_csv(path)
    bucket["file_id"] = bucket["file_id"].astype(str)
    bucket["file_id_norm"] = bucket["file_id"].str.lower()
    bucket["slide_barcode"] = bucket["file_name"].map(_slide_from_file_name)
    bucket = _add_slide_fields(bucket, "slide_barcode", prefix="")
    return bucket


def _local_h5_keys(root: Path) -> set[str]:
    if not root.exists():
        return set()

    keys: set[str] = set()
    for path in root.rglob("*.h5"):
        stem = path.stem.upper()
        keys.add(stem)
        keys.add(stem.split(".")[0])
    return keys


def _count_local_h5(slides: pd.Series, h5_keys: set[str]) -> int:
    if not h5_keys or slides.empty:
        return 0
    return int(slides.astype(str).str.upper().isin(h5_keys).sum())


def _row(
    *,
    dataset: str,
    annotation_kind: str,
    source_checked: str,
    candidate_records: int,
    unique_dataset_slide_ids: int | None,
    usable_raw_slides_exact: int | None,
    usable_raw_files_exact: int | None,
    exact_matched_records: int | None,
    case_level_candidate_slides: int | None,
    local_h5_matches: int | None,
    standalone_patch_or_fov_count: int | str | None,
    reported_or_observed_nuclei: int | str | None,
    license_or_access_note: str = "",
    notes: str = "",
) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "annotation_kind": annotation_kind,
        "source_checked": source_checked,
        "candidate_records": candidate_records,
        "unique_dataset_slide_ids": unique_dataset_slide_ids,
        "usable_raw_slides_exact": usable_raw_slides_exact,
        "usable_raw_files_exact": usable_raw_files_exact,
        "exact_matched_records": exact_matched_records,
        "case_level_candidate_slides": case_level_candidate_slides,
        "local_h5_matches": local_h5_matches,
        "standalone_patch_or_fov_count": standalone_patch_or_fov_count,
        "reported_or_observed_nuclei": reported_or_observed_nuclei,
        "license_or_access_note": license_or_access_note,
        "notes": notes,
    }


def _audit_pancancer(
    bucket: pd.DataFrame,
    h5_keys: set[str],
    reports_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    manifest_path = REPO_ROOT / "data" / "processed" / "pancancer_nuclei_seg_ann_manifest.csv"
    source_manifest_path = (
        REPO_ROOT
        / "data"
        / "reports"
        / "pancancer_nuclei_seg"
        / "pancancer_dx_embedding_source_manifest.csv"
    )
    if not manifest_path.exists():
        return (
            _row(
                dataset="Pan-Cancer-Nuclei-Seg",
                annotation_kind="WSI-scale DICOM ANN polygons",
                source_checked=str(manifest_path.relative_to(REPO_ROOT)),
                candidate_records=0,
                unique_dataset_slide_ids=0,
                usable_raw_slides_exact=0,
                usable_raw_files_exact=0,
                exact_matched_records=0,
                case_level_candidate_slides=0,
                local_h5_matches=0,
                standalone_patch_or_fov_count=None,
                reported_or_observed_nuclei=None,
                notes="Run scripts/build_pancancer_nuclei_seg_manifest.py first.",
            ),
            None,
        )

    ann = pd.read_csv(manifest_path)
    if source_manifest_path.exists():
        matched = pd.read_csv(source_manifest_path)
    else:
        matched = ann.merge(bucket, on="slide_barcode", how="inner", suffixes=("", "_bucket"))

    return (
        _row(
            dataset="Pan-Cancer-Nuclei-Seg",
            annotation_kind="WSI-scale DICOM ANN polygons; slide-level nuclei counts",
            source_checked=str(source_manifest_path.relative_to(REPO_ROOT))
            if source_manifest_path.exists()
            else str(manifest_path.relative_to(REPO_ROOT)),
            candidate_records=int(len(ann)),
            unique_dataset_slide_ids=int(ann["slide_barcode"].nunique()),
            usable_raw_slides_exact=int(matched["slide_barcode"].nunique()),
            usable_raw_files_exact=int(matched["file_id"].nunique()) if "file_id" in matched else 0,
            exact_matched_records=int(len(matched)),
            case_level_candidate_slides=None,
            local_h5_matches=_count_local_h5(matched["slide_barcode"], h5_keys)
            if "slide_barcode" in matched
            else 0,
            standalone_patch_or_fov_count=None,
            reported_or_observed_nuclei=int(
                pd.to_numeric(ann["nuclei_count_slide"], errors="coerce").sum()
            ),
            notes="Exact DX slide match to the physical bucket scan.",
        ),
        matched,
    )


def _audit_monuseg(
    bucket: pd.DataFrame,
    h5_keys: set[str],
    interim_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    parquet_dir = interim_dir / "monuseg_hf"
    parts = []
    for split in ["train", "test"]:
        path = parquet_dir / f"{split}-00000-of-00001.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["patient", "instances"])
            df["split"] = split
            parts.append(df)

    if not parts:
        return (
            _row(
                dataset="MoNuSeg",
                annotation_kind="Patch-level instance masks/polygons",
                source_checked=str(parquet_dir.relative_to(REPO_ROOT)),
                candidate_records=0,
                unique_dataset_slide_ids=0,
                usable_raw_slides_exact=0,
                usable_raw_files_exact=0,
                exact_matched_records=0,
                case_level_candidate_slides=0,
                local_h5_matches=0,
                standalone_patch_or_fov_count=0,
                reported_or_observed_nuclei=None,
                license_or_access_note="HF mirror card lists CC BY-NC-SA 4.0.",
                notes="Download the lightweight parquet metadata before rerunning.",
            ),
            None,
        )

    monu = pd.concat(parts, ignore_index=True)
    monu["slide_barcode"] = monu["patient"].astype(str).str.upper()
    matched = monu.merge(bucket, on="slide_barcode", how="inner", suffixes=("_monuseg", ""))
    nuclei_count = int(sum(len(instances) for instances in monu["instances"]))

    return (
        _row(
            dataset="MoNuSeg",
            annotation_kind="Patch-level instance masks/polygons; TCGA source slide barcode",
            source_checked="RationAI/MoNuSeg Hugging Face parquet mirror",
            candidate_records=int(len(monu)),
            unique_dataset_slide_ids=int(monu["slide_barcode"].nunique()),
            usable_raw_slides_exact=int(matched["slide_barcode"].nunique()),
            usable_raw_files_exact=int(matched["file_id"].nunique()),
            exact_matched_records=int(len(matched)),
            case_level_candidate_slides=None,
            local_h5_matches=_count_local_h5(matched["slide_barcode"], h5_keys),
            standalone_patch_or_fov_count=int(len(monu)),
            reported_or_observed_nuclei=nuclei_count,
            license_or_access_note="HF mirror card lists CC BY-NC-SA 4.0.",
            notes="Exact barcode match to the physical bucket scan.",
        ),
        matched,
    )


def _audit_nucls(
    bucket: pd.DataFrame,
    h5_keys: set[str],
    interim_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    path = interim_dir / "nucls_slide_names.txt"
    if not path.exists():
        return (
            _row(
                dataset="NuCLS",
                annotation_kind="FOV-level nucleus coordinates, boxes, boundaries, masks",
                source_checked=str(path.relative_to(REPO_ROOT)),
                candidate_records=0,
                unique_dataset_slide_ids=0,
                usable_raw_slides_exact=0,
                usable_raw_files_exact=0,
                exact_matched_records=0,
                case_level_candidate_slides=0,
                local_h5_matches=0,
                standalone_patch_or_fov_count=0,
                reported_or_observed_nuclei=None,
                license_or_access_note="NuCLS site states CC0 1.0.",
                notes="Download nucls_slide_names.txt before rerunning.",
            ),
            None,
        )

    nucls = pd.read_csv(path, header=None, names=["nucls_slide_name"])
    nucls["nucls_slide_name"] = nucls["nucls_slide_name"].astype(str).str.strip().str.upper()
    nucls["case_id"] = nucls["nucls_slide_name"].str.extract(
        r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})-", expand=False
    )
    nucls["section"] = nucls["nucls_slide_name"].str.extract(r"-(DX\d+)$", expand=False)
    nucls["section_type"] = "DX"
    nucls["section_number"] = nucls["section"].str.extract(r"DX(\d+)", expand=False)
    nucls["section_key"] = (
        nucls["case_id"] + "-" + nucls["section_type"] + "-" + nucls["section_number"]
    )

    matched = nucls.merge(bucket, on="section_key", how="inner", suffixes=("_nucls", ""))
    matched_dataset_rows = int(
        nucls["nucls_slide_name"].isin(matched["nucls_slide_name"]).sum()
    )
    missing = nucls[~nucls["nucls_slide_name"].isin(matched["nucls_slide_name"])]

    note = "Official slide list is shortened as TCGA-case-DXn; matched by case + DX section."
    if not missing.empty:
        note += f" Missing from bucket scan: {', '.join(missing['nucls_slide_name'].tolist())}."

    return (
        _row(
            dataset="NuCLS",
            annotation_kind="FOV-level nucleus coordinates, boxes, boundaries, masks",
            source_checked="Official NuCLS nucls_slide_names.txt",
            candidate_records=int(len(nucls)),
            unique_dataset_slide_ids=int(nucls["nucls_slide_name"].nunique()),
            usable_raw_slides_exact=int(matched["slide_barcode"].nunique()),
            usable_raw_files_exact=int(matched["file_id"].nunique()),
            exact_matched_records=matched_dataset_rows,
            case_level_candidate_slides=None,
            local_h5_matches=_count_local_h5(matched["slide_barcode"], h5_keys),
            standalone_patch_or_fov_count="single-rater: 2168 uncorrected FOVs / 1744 corrected FOVs",
            reported_or_observed_nuclei="single-rater: 65568 uncorrected / 59485 corrected nuclei",
            license_or_access_note="NuCLS site states CC0 1.0.",
            notes=note,
        ),
        matched,
    )


def _audit_cryonuseg(
    bucket: pd.DataFrame,
    h5_keys: set[str],
    interim_dir: Path,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    path = interim_dir / "cryonuseg_Selected_WSIs.xlsx"
    if not path.exists():
        return (
            _row(
                dataset="CryoNuSeg",
                annotation_kind="Frozen-section patch-level manual nuclei masks/boundaries",
                source_checked=str(path.relative_to(REPO_ROOT)),
                candidate_records=0,
                unique_dataset_slide_ids=0,
                usable_raw_slides_exact=0,
                usable_raw_files_exact=0,
                exact_matched_records=0,
                case_level_candidate_slides=0,
                local_h5_matches=0,
                standalone_patch_or_fov_count=0,
                reported_or_observed_nuclei=None,
                license_or_access_note="GitHub code MIT; check Kaggle dataset terms.",
                notes="Download Selected_WSIs.xlsx before rerunning.",
            ),
            None,
        )

    cryo = pd.read_excel(path)
    cryo = cryo[cryo["URL"].notna()].copy()
    cryo["file_id_norm"] = (
        cryo["URL"]
        .astype(str)
        .str.extract(r"/files/([0-9a-fA-F-]{36})", expand=False)
        .str.lower()
    )
    cryo["case_id"] = cryo["TCGA"].astype(str).str.upper()
    exact = cryo.merge(bucket, on="file_id_norm", how="inner", suffixes=("_cryo", ""))
    case_matched = cryo.merge(bucket, on="case_id", how="inner", suffixes=("_cryo", ""))

    final_count_col = "Nr Cells\n(biologist) 2nd round of manual mark-ups"
    fallback_count_col = "Nr Cells\n(bioinformatician)"
    count_col = final_count_col if final_count_col in cryo else fallback_count_col
    nuclei_count = int(pd.to_numeric(cryo[count_col], errors="coerce").sum()) if count_col in cryo else None

    return (
        _row(
            dataset="CryoNuSeg",
            annotation_kind="Frozen-section patch-level manual nuclei masks/boundaries",
            source_checked="masih4/CryoNuSeg Selected_WSIs.xlsx",
            candidate_records=int(len(cryo)),
            unique_dataset_slide_ids=int(cryo["file_id_norm"].nunique()),
            usable_raw_slides_exact=int(exact["slide_barcode"].nunique()) if len(exact) else 0,
            usable_raw_files_exact=int(exact["file_id"].nunique()) if len(exact) else 0,
            exact_matched_records=int(len(exact)),
            case_level_candidate_slides=int(case_matched["slide_barcode"].nunique()),
            local_h5_matches=_count_local_h5(exact["slide_barcode"], h5_keys)
            if len(exact)
            else 0,
            standalone_patch_or_fov_count=int(len(cryo)),
            reported_or_observed_nuclei=nuclei_count,
            license_or_access_note="GitHub code MIT; full dataset on Kaggle, check Kaggle terms.",
            notes=(
                "The selected WSI table exposes old GDC UUID URLs that do not match the current "
                "bucket scan. Cases do overlap, but exact source slides are not reconnectable "
                "from this file alone."
            ),
        ),
        exact,
    )


def _static_rows() -> list[dict[str, Any]]:
    return [
        _row(
            dataset="PanNuke",
            annotation_kind="Patch-level instance masks and 5-class nucleus labels",
            source_checked="RationAI/PanNuke Hugging Face card plus Warwick/PanNuke docs",
            candidate_records=7904,
            unique_dataset_slide_ids=0,
            usable_raw_slides_exact=0,
            usable_raw_files_exact=0,
            exact_matched_records=0,
            case_level_candidate_slides=0,
            local_h5_matches=0,
            standalone_patch_or_fov_count=7904,
            reported_or_observed_nuclei=205343,
            license_or_access_note="HF mirror card lists CC BY-NC-SA 4.0.",
            notes="Public metadata checked has image/instances/categories/tissue, not TCGA WSI IDs.",
        ),
        _row(
            dataset="HoVer-Net / CoNSeP",
            annotation_kind="Patch-level nuclei boundaries and phenotype labels",
            source_checked="HoVer-Net/CoNSeP dataset descriptions",
            candidate_records=41,
            unique_dataset_slide_ids=0,
            usable_raw_slides_exact=0,
            usable_raw_files_exact=0,
            exact_matched_records=0,
            case_level_candidate_slides=0,
            local_h5_matches=0,
            standalone_patch_or_fov_count=41,
            reported_or_observed_nuclei=24319,
            notes="CoNSeP is UHCW colorectal data, not TCGA; no bucket reconnect path.",
        ),
        _row(
            dataset="OpenTME",
            annotation_kind="Slide-level aggregate TME/cell readouts; gated access",
            source_checked="Aignostics/OpenTME Hugging Face card",
            candidate_records=3634,
            unique_dataset_slide_ids=3634,
            usable_raw_slides_exact=None,
            usable_raw_files_exact=None,
            exact_matched_records=None,
            case_level_candidate_slides=None,
            local_h5_matches=0,
            standalone_patch_or_fov_count=None,
            reported_or_observed_nuclei="aggregate counts/densities for nine cell types",
            license_or_access_note=(
                "Gated non-commercial academic access; card prohibits training labels or "
                "pseudo-labels that replicate the analysis."
            ),
            notes="Exact barcode overlap cannot be audited without gated files.",
        ),
        _row(
            dataset="CellViT",
            annotation_kind="Model/workflow, not a reconnectable slide dataset",
            source_checked="chat.txt mention",
            candidate_records=0,
            unique_dataset_slide_ids=0,
            usable_raw_slides_exact=0,
            usable_raw_files_exact=0,
            exact_matched_records=0,
            case_level_candidate_slides=0,
            local_h5_matches=0,
            standalone_patch_or_fov_count=0,
            reported_or_observed_nuclei=0,
            notes="Use as model reference/teacher candidate, not as a separate overlap source.",
        ),
        _row(
            dataset="StarDist",
            annotation_kind="Segmentation model/package, not a reconnectable slide dataset",
            source_checked="chat.txt mention",
            candidate_records=0,
            unique_dataset_slide_ids=0,
            usable_raw_slides_exact=0,
            usable_raw_files_exact=0,
            exact_matched_records=0,
            case_level_candidate_slides=0,
            local_h5_matches=0,
            standalone_patch_or_fov_count=0,
            reported_or_observed_nuclei=0,
        ),
        _row(
            dataset="Cellpose",
            annotation_kind="Segmentation model/package, not a reconnectable slide dataset",
            source_checked="chat.txt mention",
            candidate_records=0,
            unique_dataset_slide_ids=0,
            usable_raw_slides_exact=0,
            usable_raw_files_exact=0,
            exact_matched_records=0,
            case_level_candidate_slides=0,
            local_h5_matches=0,
            standalone_patch_or_fov_count=0,
            reported_or_observed_nuclei=0,
        ),
    ]


def _write_match(path: Path, df: pd.DataFrame | None) -> None:
    if df is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main() -> None:
    args = _parse_args()
    bucket_path = args.bucket_scan or _default_bucket_scan()
    if not bucket_path.exists():
        raise FileNotFoundError(f"bucket scan not found: {bucket_path}")

    reports_dir = args.reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)

    bucket = _load_bucket(bucket_path)
    h5_keys = _local_h5_keys(args.embedding_root)

    rows: list[dict[str, Any]] = []

    row, matched = _audit_pancancer(bucket, h5_keys, reports_dir)
    rows.append(row)

    row, matched = _audit_monuseg(bucket, h5_keys, args.interim_dir)
    rows.append(row)
    _write_match(reports_dir / "monuseg_bucket_overlap.csv", matched)

    row, matched = _audit_nucls(bucket, h5_keys, args.interim_dir)
    rows.append(row)
    _write_match(reports_dir / "nucls_bucket_overlap.csv", matched)

    row, matched = _audit_cryonuseg(bucket, h5_keys, args.interim_dir)
    rows.append(row)
    _write_match(reports_dir / "cryonuseg_bucket_overlap.csv", matched)

    rows.extend(_static_rows())

    summary = pd.DataFrame(rows)
    summary.to_csv(reports_dir / "external_nuclei_dataset_usability_summary.csv", index=False)
    (reports_dir / "external_nuclei_dataset_usability_summary.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )

    print(summary.to_string(index=False))
    print(f"\nWrote reports to {reports_dir}")


if __name__ == "__main__":
    main()
