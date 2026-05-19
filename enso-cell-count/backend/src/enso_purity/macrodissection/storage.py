"""File-backed append-only ROI storage.

The pilot environment has no database; the AGENTS.md guidance is to keep the
backend file-based. We use a JSON-lines file per case under
``backend/.runtime/rois/`` so that every persisted ROI annotation is a single
self-contained line in chronological order. Locking an ROI re-writes a new
line with ``locked: True``; the previous draft line is *not* deleted —
auditors can replay the entire ROI history by reading the file top to
bottom.

The store is intentionally minimal: no transactions, no concurrent-writer
guards beyond the obvious POSIX append-write semantics. The pilot serves at
most a handful of users on a single host. For multi-host deployments a
proper database (Postgres) is a drop-in replacement; only this module would
change.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


_FILE_LOCK = threading.Lock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _slug_token() -> str:
    return uuid.uuid4().hex[:8]


def new_roi_id() -> str:
    """Return a new globally-unique ROI identifier."""

    return f"roi_{_utc_now().replace(':', '-')}_{_slug_token()}"


@dataclass
class ROIRecord:
    """One persisted ROI annotation row.

    The fields mirror what the frontend POSTs except ``metrics_snapshot``
    which is filled in by the server-side recompute when an ROI is locked.
    """

    roi_id: str
    case_id: int
    user_id: str
    label: str
    geometry_thumb_px: dict
    thresholds: dict
    created_at: str
    updated_at: str
    locked: bool
    model_run: dict
    metrics_snapshot: dict | None = None
    notes: str = ""
    revision: int = 1

    def to_json_line(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"))

    @classmethod
    def from_json_line(cls, line: str) -> "ROIRecord":
        return cls(**json.loads(line))


class ROIStore:
    """Append-only JSON-lines store backed by one file per case."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, case_id: int) -> Path:
        return self.root / f"case_{case_id}.jsonl"

    # ---- write paths ------------------------------------------------------

    def append(self, record: ROIRecord) -> ROIRecord:
        """Append a new line for ``record`` to the case file.

        Returns the same record (caller may have mutated revision before).
        """

        path = self._path(record.case_id)
        line = record.to_json_line()
        with _FILE_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        return record

    def create_draft(
        self,
        *,
        case_id: int,
        user_id: str,
        label: str,
        geometry_thumb_px: dict,
        thresholds: dict,
        model_run: dict,
        metrics_snapshot: dict | None = None,
        notes: str = "",
    ) -> ROIRecord:
        record = ROIRecord(
            roi_id=new_roi_id(),
            case_id=int(case_id),
            user_id=user_id,
            label=label,
            geometry_thumb_px=geometry_thumb_px,
            thresholds=thresholds,
            created_at=_utc_now(),
            updated_at=_utc_now(),
            locked=False,
            model_run=model_run,
            metrics_snapshot=metrics_snapshot,
            notes=notes,
            revision=1,
        )
        return self.append(record)

    def update_geometry(
        self,
        *,
        case_id: int,
        roi_id: str,
        geometry_thumb_px: dict,
        label: str | None = None,
        thresholds: dict | None = None,
        metrics_snapshot: dict | None = None,
        notes: str | None = None,
    ) -> ROIRecord:
        latest = self.get_latest(case_id, roi_id)
        if latest is None:
            raise KeyError(f"ROI {roi_id} not found in case {case_id}")
        if latest.locked:
            raise PermissionError(
                f"ROI {roi_id} is locked; create a new ROI to edit"
            )
        new_record = ROIRecord(
            roi_id=latest.roi_id,
            case_id=latest.case_id,
            user_id=latest.user_id,
            label=label if label is not None else latest.label,
            geometry_thumb_px=geometry_thumb_px,
            thresholds=thresholds if thresholds is not None else latest.thresholds,
            created_at=latest.created_at,
            updated_at=_utc_now(),
            locked=False,
            model_run=latest.model_run,
            metrics_snapshot=(
                metrics_snapshot
                if metrics_snapshot is not None
                else latest.metrics_snapshot
            ),
            notes=notes if notes is not None else latest.notes,
            revision=latest.revision + 1,
        )
        return self.append(new_record)

    def lock(
        self,
        *,
        case_id: int,
        roi_id: str,
        metrics_snapshot: dict,
    ) -> ROIRecord:
        latest = self.get_latest(case_id, roi_id)
        if latest is None:
            raise KeyError(f"ROI {roi_id} not found in case {case_id}")
        if latest.locked:
            return latest
        locked_record = ROIRecord(
            roi_id=latest.roi_id,
            case_id=latest.case_id,
            user_id=latest.user_id,
            label=latest.label,
            geometry_thumb_px=latest.geometry_thumb_px,
            thresholds=latest.thresholds,
            created_at=latest.created_at,
            updated_at=_utc_now(),
            locked=True,
            model_run=latest.model_run,
            metrics_snapshot=metrics_snapshot,
            notes=latest.notes,
            revision=latest.revision + 1,
        )
        return self.append(locked_record)

    def delete(self, *, case_id: int, roi_id: str) -> None:
        """Remove all records for an ROI; refuses to delete locked records."""

        latest = self.get_latest(case_id, roi_id)
        if latest is None:
            return
        if latest.locked:
            raise PermissionError(
                f"Cannot delete locked ROI {roi_id}; duplicate-and-edit it instead."
            )
        path = self._path(case_id)
        with _FILE_LOCK:
            if not path.exists():
                return
            kept: list[str] = []
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    rec = ROIRecord.from_json_line(line)
                    if rec.roi_id != roi_id:
                        kept.append(line.rstrip("\n"))
            with path.open("w", encoding="utf-8") as f:
                if kept:
                    f.write("\n".join(kept) + "\n")
                f.flush()
                os.fsync(f.fileno())

    # ---- read paths -------------------------------------------------------

    def _iter_records(self, case_id: int):
        path = self._path(case_id)
        if not path.exists():
            return
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield ROIRecord.from_json_line(line)

    def get_latest(self, case_id: int, roi_id: str) -> ROIRecord | None:
        latest: ROIRecord | None = None
        for rec in self._iter_records(case_id):
            if rec.roi_id == roi_id:
                latest = rec
        return latest

    def list_latest(self, case_id: int) -> list[ROIRecord]:
        """Return the latest revision of each ROI on a case, in append order."""

        latest_by_id: dict[str, ROIRecord] = {}
        first_seen: dict[str, int] = {}
        for i, rec in enumerate(self._iter_records(case_id)):
            if rec.roi_id not in first_seen:
                first_seen[rec.roi_id] = i
            latest_by_id[rec.roi_id] = rec
        return [
            latest_by_id[rid]
            for rid in sorted(latest_by_id, key=lambda r: first_seen[r])
        ]

    def list_history(self, case_id: int, roi_id: str) -> list[ROIRecord]:
        return [rec for rec in self._iter_records(case_id) if rec.roi_id == roi_id]
