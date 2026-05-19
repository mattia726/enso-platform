#!/usr/bin/env python3
"""Download Pan-Cancer-Nuclei-Seg ANN DICOM files and upload to Azure Blob.

This is intended to run on the Azure VM. It uses ``idc-index`` to fetch each
public IDC/S3 series into a small local scratch directory, then uploads that
series directly to the Azure Blob endpoint with ``azcopy``. It avoids writing
the large payload through blobfuse and keeps scratch bounded to a few active
series at a time.
"""

from __future__ import annotations

import argparse
import csv
import json
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


DEFAULT_DEST_BASE_URL = (
    "https://vmshareddisk.blob.core.windows.net/data/pancancer_nuclei_seg_dicom"
)


@dataclass(frozen=True)
class SeriesJob:
    row_index: int
    project_id: str
    slide_barcode: str
    source_uuid: str
    source_url: str
    series_size_mb: float | None

    @property
    def dest_url(self) -> str:
        return f"{self.project_id}/{self.slide_barcode}/{self.source_uuid}"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--work-dir", type=Path, default=Path.home() / "pancancer_nuclei_ingest")
    ap.add_argument("--dest-base-url", default=DEFAULT_DEST_BASE_URL)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--downloader", choices=["idc", "s5cmd"], default="idc")
    ap.add_argument("--idc-bin", default=str(Path.home() / "pancancer-nuclei-venv" / "bin" / "idc"))
    ap.add_argument("--s5cmd-bin", default=str(Path.home() / ".local" / "bin" / "s5cmd"))
    ap.add_argument("--azcopy-bin", default="azcopy")
    ap.add_argument(
        "--azcopy-auto-login-type",
        default="MSI",
        help="Sets AZCOPY_AUTO_LOGIN_TYPE for direct Blob endpoint auth. Use '' to disable.",
    )
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--keep-scratch-on-failure", action="store_true")
    return ap.parse_args()


def extract_source_uuid(series_aws_url: str) -> str:
    match = re.search(r"idc-open-data/([0-9a-fA-F-]{36})", series_aws_url)
    if not match:
        raise ValueError(f"Could not extract source UUID from {series_aws_url!r}")
    return match.group(1).lower()


def load_jobs(manifest: Path, limit: int | None) -> list[SeriesJob]:
    jobs: list[SeriesJob] = []
    with manifest.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for idx, row in enumerate(reader):
            source_url = row["series_aws_url"].strip()
            if not source_url:
                continue
            size_text = row.get("series_size_mb", "")
            try:
                series_size_mb = float(size_text) if size_text else None
            except ValueError:
                series_size_mb = None
            jobs.append(
                SeriesJob(
                    row_index=idx,
                    project_id=row["project_id"].strip(),
                    slide_barcode=row["slide_barcode"].strip(),
                    source_uuid=extract_source_uuid(source_url),
                    source_url=source_url,
                    series_size_mb=series_size_mb,
                )
            )
            if limit is not None and len(jobs) >= limit:
                break
    return jobs


def completed_uuids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            if row.get("source_uuid"):
                done.add(row["source_uuid"])
    return done


def ensure_tsv(path: Path, fieldnames: list[str]) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()


def append_tsv(path: Path, fieldnames: list[str], row: dict[str, object]) -> None:
    with path.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, delimiter="\t")
        writer.writerow(row)


def run_command(
    cmd: list[str],
    *,
    log_path: Path,
    cwd: Path | None = None,
    stdin_devnull: bool = False,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab") as log_fh:
        log_fh.write(("$ " + " ".join(cmd) + "\n").encode("utf-8"))
        log_fh.flush()
        stdin = subprocess.DEVNULL if stdin_devnull else None
        proc = subprocess.run(cmd, cwd=cwd, stdout=log_fh, stderr=subprocess.STDOUT, stdin=stdin)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with code {proc.returncode}: {' '.join(cmd)}")


def upload_file(azcopy_bin: str, local_path: Path, dest_url: str, log_path: Path) -> None:
    run_command(
        [
            azcopy_bin,
            "copy",
            str(local_path),
            dest_url,
            "--overwrite=true",
            "--check-length=true",
            "--put-md5=false",
            "--log-level=WARNING",
        ],
        log_path=log_path,
        stdin_devnull=True,
    )


def upload_support_files(args: argparse.Namespace, work_dir: Path, started_at: str) -> None:
    support = work_dir / "support"
    support.mkdir(parents=True, exist_ok=True)
    config_path = support / "ingest_config.json"
    config = {
        "started_at": started_at,
        "manifest": str(args.manifest),
        "dest_base_url": args.dest_base_url,
        "workers": args.workers,
        "limit": args.limit,
        "idc_bin": args.idc_bin,
        "downloader": args.downloader,
        "s5cmd_bin": args.s5cmd_bin,
        "azcopy_bin": args.azcopy_bin,
        "azcopy_auto_login_type": args.azcopy_auto_login_type,
        "overwrite": args.overwrite,
    }
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    base = args.dest_base_url.rstrip("/")
    log_path = work_dir / "logs" / "support_upload.log"
    upload_file(args.azcopy_bin, args.manifest, f"{base}/manifests/{args.manifest.name}", log_path)
    upload_file(args.azcopy_bin, config_path, f"{base}/manifests/{config_path.name}", log_path)


def file_stats(path: Path) -> tuple[int, int]:
    files = [p for p in path.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)


def process_job(args: argparse.Namespace, job: SeriesJob) -> dict[str, object]:
    work_dir = args.work_dir
    scratch = work_dir / "scratch" / f"{job.row_index:05d}_{job.source_uuid}"
    one_manifest = work_dir / "manifests" / f"{job.row_index:05d}_{job.source_uuid}.s5cmd"
    log_path = work_dir / "logs" / f"{job.row_index:05d}_{job.source_uuid}.log"
    base = args.dest_base_url.rstrip("/")
    dest_url = f"{base}/raw_ann/{job.dest_url}"
    started = time.time()

    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    one_manifest.parent.mkdir(parents=True, exist_ok=True)
    one_manifest.write_text(f"cp s3://idc-open-data/{job.source_uuid}/*  .\n", encoding="utf-8")

    try:
        if args.downloader == "s5cmd":
            run_command(
                [
                    args.s5cmd_bin,
                    "--no-sign-request",
                    "cp",
                    f"s3://idc-open-data/{job.source_uuid}/*",
                    str(scratch) + "/",
                ],
                log_path=log_path,
            )
        else:
            run_command(
                [
                    args.idc_bin,
                    "download",
                    str(one_manifest),
                    "--download-dir",
                    str(scratch),
                    "--dir-template",
                    "",
                    "--log-level",
                    "warning",
                ],
                log_path=log_path,
            )
        file_count, byte_count = file_stats(scratch)
        if file_count == 0 or byte_count == 0:
            raise RuntimeError("IDC download produced no files")

        run_command(
            [
                args.azcopy_bin,
                "copy",
                str(scratch / "*"),
                dest_url,
                "--recursive=true",
                "--as-subdir=false",
                f"--overwrite={'true' if args.overwrite else 'false'}",
                "--check-length=true",
                "--put-md5=false",
                "--log-level=WARNING",
            ],
            log_path=log_path,
            stdin_devnull=True,
        )
        status = "completed"
        error = ""
    except Exception as exc:  # noqa: BLE001 - error is persisted for resume/debug
        status = "failed"
        error = repr(exc)
        if not args.keep_scratch_on_failure and scratch.exists():
            shutil.rmtree(scratch)
        raise RuntimeError(error) from exc
    finally:
        if status == "completed":
            shutil.rmtree(scratch, ignore_errors=True)
            one_manifest.unlink(missing_ok=True)

    elapsed = time.time() - started
    return {
        "status": status,
        "row_index": job.row_index,
        "project_id": job.project_id,
        "slide_barcode": job.slide_barcode,
        "source_uuid": job.source_uuid,
        "series_size_mb": job.series_size_mb,
        "file_count": file_count,
        "byte_count": byte_count,
        "elapsed_s": f"{elapsed:.3f}",
        "mbps": f"{byte_count / elapsed / 1_000_000:.3f}" if elapsed > 0 else "",
        "dest_url": dest_url,
        "error": error,
        "finished_at": datetime.now(timezone.utc).isoformat(),
    }


def main() -> int:
    args = parse_args()
    if args.azcopy_auto_login_type:
        os.environ.setdefault("AZCOPY_AUTO_LOGIN_TYPE", args.azcopy_auto_login_type)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()

    completed_path = args.work_dir / "state" / "completed.tsv"
    failed_path = args.work_dir / "state" / "failed.tsv"
    fields = [
        "status",
        "row_index",
        "project_id",
        "slide_barcode",
        "source_uuid",
        "series_size_mb",
        "file_count",
        "byte_count",
        "elapsed_s",
        "mbps",
        "dest_url",
        "error",
        "finished_at",
    ]
    ensure_tsv(completed_path, fields)
    ensure_tsv(failed_path, fields)

    run_log = args.work_dir / "run.log"
    with run_log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{started_at}] starting ingest\n")
        fh.write(json.dumps(vars(args), default=str) + "\n")

    run_command(
        [args.azcopy_bin, "login", "--identity"],
        log_path=args.work_dir / "logs" / "azcopy_login.log",
        stdin_devnull=True,
    )
    upload_support_files(args, args.work_dir, started_at)

    jobs = load_jobs(args.manifest, args.limit)
    done = completed_uuids(completed_path)
    pending = [job for job in jobs if job.source_uuid not in done]
    total_expected_mb = sum(job.series_size_mb or 0.0 for job in jobs)

    print(
        f"Loaded {len(jobs)} jobs ({total_expected_mb:,.1f} MB expected); "
        f"{len(done)} already complete; {len(pending)} pending.",
        flush=True,
    )

    completed_count = 0
    failed_count = 0
    completed_bytes = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        active = {}
        pending_iter = iter(pending)

        def submit_next() -> bool:
            try:
                job = next(pending_iter)
            except StopIteration:
                return False
            active[pool.submit(process_job, args, job)] = job
            return True

        for _ in range(args.workers):
            if not submit_next():
                break

        while active:
            done_futures, _ = wait(active.keys(), return_when=FIRST_COMPLETED)
            for future in done_futures:
                job = active.pop(future)
                try:
                    result = future.result()
                    append_tsv(completed_path, fields, result)
                    completed_count += 1
                    completed_bytes += int(result["byte_count"])
                    msg = (
                        f"OK {completed_count}/{len(pending)} "
                        f"{job.project_id} {job.slide_barcode} "
                        f"{int(result['byte_count']) / 1_000_000:.1f} MB "
                        f"{result['mbps']} MB/s"
                    )
                    print(msg, flush=True)
                    with run_log.open("a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")
                except Exception as exc:  # noqa: BLE001 - persisted for resume/debug
                    failed_count += 1
                    row = {
                        "status": "failed",
                        "row_index": job.row_index,
                        "project_id": job.project_id,
                        "slide_barcode": job.slide_barcode,
                        "source_uuid": job.source_uuid,
                        "series_size_mb": job.series_size_mb,
                        "file_count": "",
                        "byte_count": "",
                        "elapsed_s": "",
                        "mbps": "",
                        "dest_url": f"{args.dest_base_url.rstrip('/')}/raw_ann/{job.dest_url}",
                        "error": repr(exc),
                        "finished_at": datetime.now(timezone.utc).isoformat(),
                    }
                    append_tsv(failed_path, fields, row)
                    msg = f"FAIL {job.project_id} {job.slide_barcode} {job.source_uuid}: {exc!r}"
                    print(msg, flush=True)
                    with run_log.open("a", encoding="utf-8") as fh:
                        fh.write(msg + "\n")
                submit_next()

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "jobs_loaded": len(jobs),
        "jobs_pending_at_start": len(pending),
        "completed_this_run": completed_count,
        "failed_this_run": failed_count,
        "completed_bytes_this_run": completed_bytes,
    }
    (args.work_dir / "state" / "last_run_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if failed_count == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
