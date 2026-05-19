#!/usr/bin/env python3
"""Build the Pan-Cancer-Nuclei-Seg ANN manifest for EnsoCellularity.

This script can either:
1. fetch IDC DICOMweb metadata for all Pan-Cancer-Nuclei-Seg ANN series, or
2. normalize a previously fetched metadata CSV via ``--input-metadata``.

It writes a slide-level manifest with ``nuclei_count_slide`` and enough DICOM
identifiers/download pointers to fetch polygon boundaries later for tile labels.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPO_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from enso_cellularity.pancancer_nuclei import (  # noqa: E402
    DICOMWEB_BASE_URL,
    PANCANCER_ANALYSIS_RESULT_ID,
    case_sample_type_key,
    extract_ann_metadata_from_dicom_json,
    normalize_ann_manifest,
    overlap_summary,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/pancancer_nuclei_seg_ann_manifest.csv"),
        help="Output normalized manifest CSV.",
    )
    ap.add_argument(
        "--input-metadata",
        type=Path,
        default=None,
        help="Optional raw metadata CSV to normalize instead of fetching IDC metadata.",
    )
    ap.add_argument(
        "--reports-dir",
        type=Path,
        default=Path("data/reports/pancancer_nuclei_seg"),
        help="Directory for overlap reports.",
    )
    ap.add_argument(
        "--wedge-manifest",
        type=Path,
        default=Path("data/processed/wedge_mvp_dataset.xlsx"),
        help="Existing wedge manifest used for overlap auditing.",
    )
    ap.add_argument(
        "--collection",
        action="append",
        default=None,
        help="IDC collection id to include, e.g. tcga_brca. Repeatable.",
    )
    ap.add_argument("--limit", type=int, default=None, help="Optional series limit for smoke tests.")
    ap.add_argument("--workers", type=int, default=12, help="Concurrent DICOMweb metadata fetches.")
    ap.add_argument("--refresh", action="store_true", help="Refetch even if --out already exists.")
    ap.add_argument("--skip-overlap", action="store_true", help="Skip wedge overlap reports.")
    return ap.parse_args()


def _query_idc_ann_series(collections: list[str] | None, limit: int | None) -> pd.DataFrame:
    try:
        from idc_index import IDCClient
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise SystemExit(
            "idc-index is required for live metadata fetches. "
            "Install it with: python -m pip install idc-index"
        ) from exc

    where = [
        f"analysis_result_id = '{PANCANCER_ANALYSIS_RESULT_ID}'",
        "Modality = 'ANN'",
    ]
    if collections:
        quoted = ", ".join(f"'{c.lower()}'" for c in collections)
        where.append(f"collection_id IN ({quoted})")

    limit_sql = f"LIMIT {int(limit)}" if limit else ""
    sql = f"""
    SELECT
        collection_id,
        PatientID,
        StudyInstanceUID,
        SeriesInstanceUID,
        instanceCount,
        series_size_MB,
        series_aws_url
    FROM index
    WHERE {" AND ".join(where)}
    ORDER BY collection_id, PatientID, SeriesInstanceUID
    {limit_sql}
    """
    client = IDCClient.client()
    return client.sql_query(sql)


def _fetch_dicom_json_metadata(url: str) -> list[dict[str, Any]]:
    request = urllib.request.Request(url, headers={"Accept": "application/dicom+json"})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def _fetch_one_series(row: dict[str, Any]) -> dict[str, Any]:
    study_uid = row["StudyInstanceUID"]
    series_uid = row["SeriesInstanceUID"]
    url = f"{DICOMWEB_BASE_URL}/studies/{study_uid}/series/{series_uid}/metadata"
    last_error = None
    for attempt in range(3):
        try:
            metadata = _fetch_dicom_json_metadata(url)
            if not metadata:
                raise RuntimeError("empty metadata response")
            return extract_ann_metadata_from_dicom_json(
                metadata[0],
                collection_id=row.get("collection_id"),
                study_instance_uid=study_uid,
                series_instance_uid=series_uid,
                series_size_mb=row.get("series_size_MB"),
                series_aws_url=row.get("series_aws_url"),
            )
        except Exception as exc:  # pragma: no cover - network retry path
            last_error = repr(exc)
            time.sleep(0.5 * (attempt + 1))
    raise RuntimeError(f"Failed metadata fetch for {series_uid}: {last_error}")


def _fetch_manifest(collections: list[str] | None, limit: int | None, workers: int) -> pd.DataFrame:
    series = _query_idc_ann_series(collections, limit)
    logger.info("IDC ANN series selected: %d", len(series))

    rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_fetch_one_series, row) for row in series.to_dict("records")]
        for idx, future in enumerate(as_completed(futures), 1):
            rows.append(future.result())
            if idx % 250 == 0 or idx == len(futures):
                logger.info("Fetched metadata %d/%d", idx, len(futures))

    return normalize_ann_manifest(pd.DataFrame(rows))


def _write_overlap_reports(
    ann_manifest: pd.DataFrame,
    wedge_manifest_path: Path,
    reports_dir: Path,
) -> None:
    if not wedge_manifest_path.exists():
        logger.warning("Wedge manifest not found; skipping overlap report: %s", wedge_manifest_path)
        return

    reports_dir.mkdir(parents=True, exist_ok=True)
    wedge = pd.read_excel(wedge_manifest_path)
    summary = overlap_summary(ann_manifest, wedge)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    ann = ann_manifest.copy()
    wedge = wedge.copy()
    ann["case_sample_type"] = [
        case_sample_type_key(case_id, sample_type)
        for case_id, sample_type in zip(ann["case_id"], ann["sample_type_code"])
    ]
    wedge["wedge_sample_type_code"] = wedge["sample_type_code"].astype(str).str.zfill(2)
    wedge["case_sample_type"] = [
        case_sample_type_key(case_id, sample_type)
        for case_id, sample_type in zip(wedge["case_id"], wedge["wedge_sample_type_code"])
    ]

    exact = ann.merge(wedge, left_on="slide_barcode", right_on="barcode", how="inner")
    exact.to_csv(reports_dir / "exact_slide_overlap.csv", index=False)

    case = ann.merge(wedge, left_on="case_id", right_on="case_id", how="inner")
    case.to_csv(reports_dir / "case_overlap.csv", index=False)

    case_sample_type = ann.merge(wedge, on="case_sample_type", how="inner")
    case_sample_type.to_csv(reports_dir / "case_sample_type_overlap.csv", index=False)

    logger.info("Overlap summary: %s", summary)
    logger.info("Wrote overlap reports to %s", reports_dir)


def main() -> None:
    args = _parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.out.exists() and not args.refresh and args.input_metadata is None:
        logger.info("Manifest exists and --refresh was not set; reading %s", args.out)
        manifest = pd.read_csv(args.out)
    elif args.input_metadata is not None:
        logger.info("Normalizing existing metadata CSV: %s", args.input_metadata)
        manifest = normalize_ann_manifest(pd.read_csv(args.input_metadata))
    else:
        manifest = _fetch_manifest(args.collection, args.limit, args.workers)

    manifest.to_csv(args.out, index=False)
    logger.info("Wrote %d rows to %s", len(manifest), args.out)
    logger.info("Total nuclei in manifest: %s", int(manifest["nuclei_count_slide"].sum()))

    if not args.skip_overlap:
        _write_overlap_reports(manifest, args.wedge_manifest, args.reports_dir)


if __name__ == "__main__":
    main()

