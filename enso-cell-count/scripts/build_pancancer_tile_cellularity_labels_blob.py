#!/usr/bin/env python3
"""Build Pan-Cancer tile labels using direct Azure Blob endpoint I/O.

This script is for large remote runs. It avoids blobfuse entirely:

1. download one embedding H5 and one ANN DICOM series to local scratch with azcopy,
2. build the per-tile label Parquet from local files,
3. upload the Parquet and state files back to Azure Blob with azcopy,
4. remove the scratch files.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
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

logger = logging.getLogger("build_pancancer_tile_cellularity_labels_blob")


@dataclass(frozen=True)
class BlobConfig:
    base_url: str
    h5_prefix: str
    ann_prefix: str
    out_prefix: str
    azcopy_bin: str
    azcopy_auto_login_type: str

    def url(self, *parts: str) -> str:
        clean = [self.base_url.rstrip("/")]
        clean.extend(part.strip("/") for part in parts if part)
        return "/".join(clean)


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
    ap.add_argument("--base-url", default="https://vmshareddisk.blob.core.windows.net/data")
    ap.add_argument("--h5-prefix", default="embeddings_fp32")
    ap.add_argument("--ann-prefix", default="pancancer_nuclei_seg_dicom/raw_ann")
    ap.add_argument("--out-prefix", default="pancancer_nuclei_seg_dicom/tile_cellularity_labels_direct")
    ap.add_argument("--scratch-dir", type=Path, default=Path("/mnt/resource/pancancer_tile_labels_direct_scratch"))
    ap.add_argument("--state-dir", type=Path, default=Path("tile_cellularity_labels_direct_state"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-h5-index", action="store_true", help="Do not pre-list H5 blobs; try every manifest row.")
    ap.add_argument("--upload-state-every", type=int, default=10)
    ap.add_argument("--azcopy-bin", default="azcopy")
    ap.add_argument(
        "--azcopy-auto-login-type",
        default="MSI",
        help="Sets AZCOPY_AUTO_LOGIN_TYPE for direct Blob auth. Use '' to disable.",
    )
    ap.add_argument("--log-level", default="INFO")
    return ap.parse_args()


def _azcopy_env(auto_login_type: str) -> dict[str, str]:
    env = os.environ.copy()
    if auto_login_type:
        env["AZCOPY_AUTO_LOGIN_TYPE"] = auto_login_type
    return env


def _run_azcopy(args: list[str], cfg: BlobConfig, *, allow_failure: bool = False) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        [cfg.azcopy_bin, *args],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=_azcopy_env(cfg.azcopy_auto_login_type),
        check=False,
    )
    if proc.returncode and not allow_failure:
        tail = proc.stdout[-4000:] if proc.stdout else ""
        raise RuntimeError(f"azcopy failed ({proc.returncode}) for {args!r}\n{tail}")
    return proc


def _copy_from_blob(url: str, dest: Path, cfg: BlobConfig) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    _run_azcopy(["copy", url, str(dest), "--overwrite=true", "--log-level=ERROR"], cfg)


def _copy_ann_series(url: str, dest_dir: Path, cfg: BlobConfig) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    _run_azcopy(["copy", f"{url.rstrip('/')}/*", str(dest_dir), "--recursive=false", "--overwrite=true", "--log-level=ERROR"], cfg)


def _upload_to_blob(src: Path, url: str, cfg: BlobConfig) -> None:
    _run_azcopy(["copy", str(src), url, "--overwrite=true", "--log-level=ERROR"], cfg)


def _list_h5_file_ids(cfg: BlobConfig) -> set[str]:
    url = cfg.url(cfg.h5_prefix)
    proc = _run_azcopy(["list", url, "--properties=ContentLength"], cfg)
    file_ids: set[str] = set()
    for line in proc.stdout.splitlines():
        name = line.split(";", 1)[0].strip()
        if name.endswith(".h5"):
            file_ids.add(Path(name).stem)
    return file_ids


def _load_jobs(manifest_path: Path, limit: int | None) -> list[SlideJob]:
    df = pd.read_csv(manifest_path)
    jobs = []
    for i, row in df.iterrows():
        jobs.append(
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
        if limit is not None and len(jobs) >= limit:
            break
    return jobs


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
        csv.DictWriter(fh, fieldnames=fields, delimiter="\t").writeheader()


def _append_tsv(path: Path, fields: list[str], row: dict[str, Any]) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        csv.DictWriter(fh, fieldnames=fields, delimiter="\t").writerow(row)


def _upload_state(state_dir: Path, cfg: BlobConfig) -> None:
    for name in ["completed.tsv", "failed.tsv", "last_run_summary.json", "run_config.json"]:
        path = state_dir / name
        if path.exists():
            _upload_to_blob(path, cfg.url(cfg.out_prefix, "state", name), cfg)


def _process_one(job: SlideJob, cfg: BlobConfig, scratch_root: Path, overwrite: bool) -> dict[str, Any]:
    started = time.time()
    job_dir = scratch_root / job.file_id
    if job_dir.exists():
        shutil.rmtree(job_dir)
    ann_dir = job_dir / "ann"
    h5_path = job_dir / f"{job.file_id}.h5"
    parquet_path = job_dir / job.label_name

    h5_url = cfg.url(cfg.h5_prefix, f"{job.file_id}.h5")
    ann_url = cfg.url(cfg.ann_prefix, job.project_id, job.slide_barcode, job.source_uuid)
    out_url = cfg.url(cfg.out_prefix, "by_slide", job.label_name)

    try:
        _copy_from_blob(h5_url, h5_path, cfg)
        _copy_ann_series(ann_url, ann_dir, cfg)
        ann_matches = sorted(ann_dir.glob("*.dcm"))
        if not ann_matches:
            raise FileNotFoundError(f"No ANN DICOM downloaded from {ann_url}")
        ann_path = ann_matches[0]

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
        labels["ann_dicom_path"] = f"{ann_url.rstrip('/')}/{ann_path.name}"
        labels["ann_source_uuid"] = job.source_uuid

        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(parquet_path, index=False)
        _upload_to_blob(parquet_path, out_url, cfg)

        return {
            "status": "completed",
            "file_id": job.file_id,
            "slide_barcode": job.slide_barcode,
            "project_id": job.project_id,
            "rows": int(len(labels)),
            "tile_count_sum": int(counts.sum()),
            "nuclei_count_slide": job.nuclei_count_slide,
            "out_path": out_url,
            "elapsed_s": f"{time.time() - started:.3f}",
            "error": "",
            "finished_at": datetime.now(timezone.utc).isoformat(),
        }
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def main() -> int:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )

    cfg = BlobConfig(
        base_url=args.base_url,
        h5_prefix=args.h5_prefix,
        ann_prefix=args.ann_prefix,
        out_prefix=args.out_prefix,
        azcopy_bin=args.azcopy_bin,
        azcopy_auto_login_type=args.azcopy_auto_login_type,
    )
    args.state_dir.mkdir(parents=True, exist_ok=True)
    args.scratch_dir.mkdir(parents=True, exist_ok=True)

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
    completed_path = args.state_dir / "completed.tsv"
    failed_path = args.state_dir / "failed.tsv"
    _ensure_tsv(completed_path, fields)
    _ensure_tsv(failed_path, fields)

    jobs = _load_jobs(args.manifest, args.limit)
    completed_at_start = _state_rows(completed_path)
    pending = [job for job in jobs if args.overwrite or job.file_id not in completed_at_start]
    missing_h5 = 0
    if args.skip_h5_index:
        runnable = pending
    else:
        logger.info("Listing H5 blobs directly from %s/%s", cfg.base_url, cfg.h5_prefix)
        h5_file_ids = _list_h5_file_ids(cfg)
        runnable = [job for job in pending if job.file_id in h5_file_ids]
        missing_h5 = len(pending) - len(runnable)

    run_config = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(args.manifest),
        "base_url": args.base_url,
        "h5_prefix": args.h5_prefix,
        "ann_prefix": args.ann_prefix,
        "out_prefix": args.out_prefix,
        "scratch_dir": str(args.scratch_dir),
        "state_dir": str(args.state_dir),
        "workers": args.workers,
        "limit": args.limit,
        "overwrite": args.overwrite,
        "skip_h5_index": args.skip_h5_index,
    }
    (args.state_dir / "run_config.json").write_text(json.dumps(run_config, indent=2) + "\n")
    _upload_state(args.state_dir, cfg)

    logger.info(
        "Loaded %d jobs; %d already complete; %d pending with H5; %d missing H5.",
        len(jobs),
        len(completed_at_start),
        len(runnable),
        missing_h5,
    )

    completed = 0
    failed = 0
    total_rows = 0
    total_counts = 0
    started = time.time()

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        active = {}
        job_iter = iter(runnable)

        def submit_next() -> None:
            try:
                job = next(job_iter)
            except StopIteration:
                return
            active[pool.submit(_process_one, job, cfg, args.scratch_dir, args.overwrite)] = job

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
                        "out_path": cfg.url(cfg.out_prefix, "by_slide", job.label_name),
                        "elapsed_s": "",
                        "error": repr(exc),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                    _append_tsv(failed_path, fields, row)
                    logger.exception("Failed %s", job.slide_barcode)

                if (completed + failed) % max(1, args.upload_state_every) == 0:
                    _upload_state(args.state_dir, cfg)
                submit_next()

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "jobs_loaded": len(jobs),
        "already_complete_at_start": len(completed_at_start),
        "pending_with_h5_at_start": len(runnable),
        "missing_h5_at_start": missing_h5,
        "completed_this_run": completed,
        "failed_this_run": failed,
        "tile_label_rows_this_run": total_rows,
        "counted_nuclei_this_run": total_counts,
        "elapsed_s": time.time() - started,
    }
    (args.state_dir / "last_run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _upload_state(args.state_dir, cfg)
    logger.info("Summary: %s", summary)
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
