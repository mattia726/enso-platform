#!/usr/bin/env python3
"""Build tile-level EnsoCellularity labels from Pan-Cancer-Nuclei-Seg ANN files.

The output is partitioned one Parquet file per slide. Each row corresponds to
one existing Virchow embedding row and stores the nucleus count target plus the
geometry needed by the architecture's exposure-normalized count head.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ML_ROOT = REPO_ROOT / "ml"
if str(ML_ROOT) not in sys.path:
    sys.path.insert(0, str(ML_ROOT))

from enso_cellularity.ann_dicom import read_ann_centroids  # noqa: E402
from enso_cellularity.labels import (  # noqa: E402
    count_centroids_in_tiles,
    make_tile_label_frame,
    tile_grid_spec_from_h5_attrs,
)

logger = logging.getLogger("build_pancancer_tile_cellularity_labels")


@dataclass(frozen=True)
class SlideJob:
    row_index: int
    file_id: str
    slide_barcode: str
    project_id: str
    case_id: str
    series_aws_url: str
    nuclei_count_slide: int

    @property
    def source_uuid(self) -> str:
        match = re.search(r"idc-open-data/([0-9a-fA-F-]{36})", self.series_aws_url)
        if not match:
            raise ValueError(f"Could not extract IDC source UUID from {self.series_aws_url!r}")
        return match.group(1).lower()

    @property
    def label_name(self) -> str:
        return f"{self.file_id}__{self.slide_barcode}.parquet"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/reports/pancancer_nuclei_seg/pancancer_dx_embedding_source_manifest.csv"),
        help="Pan-Cancer manifest with file_id, slide_barcode, and series_aws_url.",
    )
    ap.add_argument("--h5-dir", type=Path, required=True, help="Directory containing {file_id}.h5.")
    ap.add_argument(
        "--ann-root",
        type=Path,
        required=True,
        help="Root containing raw_ann/{project_id}/{slide_barcode}/{source_uuid}/*.dcm.",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Output directory for per-slide Parquet labels and state files.",
    )
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def _load_jobs(manifest_path: Path, limit: int | None) -> list[SlideJob]:
    df = pd.read_csv(manifest_path)
    rows = []
    for i, row in df.iterrows():
        rows.append(
            SlideJob(
                row_index=int(i),
                file_id=str(row["file_id"]),
                slide_barcode=str(row["slide_barcode"]),
                project_id=str(row["project_id"]),
                case_id=str(row["case_id"]),
                series_aws_url=str(row["series_aws_url"]),
                nuclei_count_slide=int(row["nuclei_count_slide"]),
            )
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def _ann_path(job: SlideJob, ann_root: Path) -> Path:
    root = ann_root / job.project_id / job.slide_barcode / job.source_uuid
    matches = sorted(root.glob("*.dcm"))
    if not matches:
        raise FileNotFoundError(f"No ANN DICOM found under {root}")
    if len(matches) > 1:
        logger.warning("Multiple DICOM files found for %s; using %s", job.slide_barcode, matches[0])
    return matches[0]


def _state_rows(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as fh:
        return {row["file_id"] for row in csv.DictReader(fh, delimiter="\t") if row.get("file_id")}


def _ensure_tsv(path: Path, fields: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writeheader()


def _append_tsv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, delimiter="\t")
        writer.writerow(row)


def _process_one(job: SlideJob, h5_dir: Path, ann_root: Path, labels_dir: Path, overwrite: bool) -> dict[str, Any]:
    out_path = labels_dir / job.label_name
    if out_path.exists() and not overwrite:
        return {
            "status": "skipped_existing",
            "file_id": job.file_id,
            "slide_barcode": job.slide_barcode,
            "project_id": job.project_id,
            "rows": "",
            "tile_count_sum": "",
            "nuclei_count_slide": job.nuclei_count_slide,
            "out_path": str(out_path),
            "elapsed_s": "0.000",
            "error": "",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }

    h5_path = h5_dir / f"{job.file_id}.h5"
    if not h5_path.exists():
        raise FileNotFoundError(f"Missing H5: {h5_path}")
    ann_path = _ann_path(job, ann_root)

    started = time.time()
    with h5py.File(h5_path, "r") as h5:
        coords = h5["coords"][:]
        coords_level0 = h5["coords_level0"][:]
        spec = tile_grid_spec_from_h5_attrs(h5.attrs)
        n_features = int(h5["features"].shape[0])

    if len(coords) != n_features or len(coords_level0) != n_features:
        raise ValueError(
            f"Coordinate/feature length mismatch for {job.file_id}: "
            f"features={n_features}, coords={len(coords)}, coords_level0={len(coords_level0)}"
        )

    centroids = read_ann_centroids(ann_path, fast_vertex_mean=True)
    counts = count_centroids_in_tiles(
        centroids["centroid_x"].to_numpy(),
        centroids["centroid_y"].to_numpy(),
        coords_level0,
        spec,
    )
    labels = make_tile_label_frame(
        file_id=job.file_id,
        slide_barcode=job.slide_barcode,
        project_id=job.project_id,
        case_id=job.case_id,
        coords=coords,
        coords_level0=coords_level0,
        counts=counts,
        spec=spec,
    )
    labels["ann_nuclei_count_slide"] = job.nuclei_count_slide
    labels["ann_counted_nuclei_in_embedding_tiles"] = int(counts.sum())
    labels["ann_dicom_path"] = str(ann_path)
    labels["ann_source_uuid"] = job.source_uuid

    labels_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".parquet.tmp")
    labels.to_parquet(tmp_path, index=False)
    tmp_path.replace(out_path)

    elapsed = time.time() - started
    return {
        "status": "completed",
        "file_id": job.file_id,
        "slide_barcode": job.slide_barcode,
        "project_id": job.project_id,
        "rows": int(len(labels)),
        "tile_count_sum": int(counts.sum()),
        "nuclei_count_slide": job.nuclei_count_slide,
        "out_path": str(out_path),
        "elapsed_s": f"{elapsed:.3f}",
        "error": "",
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels_dir = args.out_dir / "by_slide"
    state_dir = args.out_dir / "state"
    fields = [
        "status",
        "file_id",
        "slide_barcode",
        "project_id",
        "rows",
        "tile_count_sum",
        "nuclei_count_slide",
        "out_path",
        "elapsed_s",
        "error",
        "finished_at",
    ]
    completed_path = state_dir / "completed.tsv"
    failed_path = state_dir / "failed.tsv"
    _ensure_tsv(completed_path, fields)
    _ensure_tsv(failed_path, fields)

    jobs = _load_jobs(args.manifest, args.limit)
    done = _state_rows(completed_path)
    pending = [job for job in jobs if args.overwrite or job.file_id not in done]
    h5_available = [job for job in pending if (args.h5_dir / f"{job.file_id}.h5").exists()]
    missing_h5 = len(pending) - len(h5_available)
    logger.info(
        "Loaded %d jobs; %d already complete; %d pending with H5; %d missing H5.",
        len(jobs),
        len(done),
        len(h5_available),
        missing_h5,
    )

    run_config = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "h5_dir": str(args.h5_dir),
        "ann_root": str(args.ann_root),
        "out_dir": str(args.out_dir),
        "workers": args.workers,
        "limit": args.limit,
        "overwrite": args.overwrite,
    }
    (args.out_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")

    completed = 0
    failed = 0
    total_rows = 0
    total_counts = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        active = {}
        job_iter = iter(h5_available)

        def submit_next() -> None:
            try:
                job = next(job_iter)
            except StopIteration:
                return
            active[pool.submit(_process_one, job, args.h5_dir, args.ann_root, labels_dir, args.overwrite)] = job

        for _ in range(max(1, args.workers)):
            submit_next()

        while active:
            done_futures, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                job = active.pop(future)
                try:
                    result = future.result()
                    _append_tsv(completed_path, fields, result)
                    completed += 1
                    if result["status"] == "completed":
                        total_rows += int(result["rows"])
                        total_counts += int(result["tile_count_sum"])
                    logger.info(
                        "%s %s rows=%s tile_count_sum=%s elapsed=%ss",
                        result["status"],
                        job.slide_barcode,
                        result["rows"],
                        result["tile_count_sum"],
                        result["elapsed_s"],
                    )
                except Exception as exc:  # noqa: BLE001 - persisted for resume/debug
                    failed += 1
                    row = {
                        "status": "failed",
                        "file_id": job.file_id,
                        "slide_barcode": job.slide_barcode,
                        "project_id": job.project_id,
                        "rows": "",
                        "tile_count_sum": "",
                        "nuclei_count_slide": job.nuclei_count_slide,
                        "out_path": str(labels_dir / job.label_name),
                        "elapsed_s": "",
                        "error": repr(exc),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                    _append_tsv(failed_path, fields, row)
                    logger.exception("Failed %s", job.slide_barcode)
                submit_next()

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "jobs_loaded": len(jobs),
        "already_complete_at_start": len(done),
        "pending_with_h5_at_start": len(h5_available),
        "missing_h5_at_start": missing_h5,
        "completed_this_run": completed,
        "failed_this_run": failed,
        "tile_label_rows_this_run": total_rows,
        "counted_nuclei_this_run": total_counts,
        "elapsed_s": time.time() - started,
    }
    (state_dir / "last_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Summary: %s", summary)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
