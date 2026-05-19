"""FastAPI router for the macrodissection workbench.

All endpoints are mounted under ``/api/macrodissection`` by
:mod:`enso_purity.api.main`. Request/response models live next to the router
to keep the contract obvious; persistent storage uses the JSON-lines store in
:mod:`enso_purity.macrodissection.storage`.

The router is intentionally backend-agnostic: it does *not* touch the
matplotlib/PyTorch tile-generation paths at request time. All heavy lifting
happens once at startup via the artifact builder; serving an ROI preview is
pure numpy arithmetic on the in-memory tile grid.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, field_validator

from .adequacy import label_adequacy
from .cases import (
    CaseMeta,
    clear_case_cache,
    discover_cases,
    load_tile_arrays,
)
from .candidates import suggest_candidates
from .report import build_report_payload
from .roi import compute_roi_metrics
from .storage import ROIRecord, ROIStore
from .thresholds import (
    PROFILES,
    ThresholdOverride,
    list_profiles,
    resolve_thresholds,
)


LOG = logging.getLogger(__name__)


# --------- Pydantic request/response models --------------------------------


class PolygonPayload(BaseModel):
    """GeoJSON-style polygon in *thumbnail-pixel* coordinates."""

    type: Literal["Polygon"] = "Polygon"
    coordinates: list[list[list[float]]] = Field(
        ..., description="GeoJSON polygon coordinates: [[ [x, y], ... ]]"
    )

    @field_validator("coordinates")
    @classmethod
    def _validate_coords(cls, value):
        if not value:
            raise ValueError("polygon coordinates must contain at least one ring")
        ring = value[0]
        if len(ring) < 3:
            raise ValueError("polygon ring must have at least 3 vertices")
        return value

    def as_vertex_list(self) -> list[tuple[float, float]]:
        ring = self.coordinates[0]
        return [(float(p[0]), float(p[1])) for p in ring]


class ThresholdsPayload(BaseModel):
    profile: str = "humanitas_ngs"
    purity_min: float | None = None
    tumor_cells_min: int | None = None
    borderline_purity_band: float | None = None
    borderline_tumor_cells_band: int | None = None
    pass_probability: float | None = None
    borderline_probability: float | None = None

    def to_override(self) -> ThresholdOverride:
        return ThresholdOverride(
            purity_min=self.purity_min,
            tumor_cells_min=self.tumor_cells_min,
            borderline_purity_band=self.borderline_purity_band,
            borderline_tumor_cells_band=self.borderline_tumor_cells_band,
            pass_probability=self.pass_probability,
            borderline_probability=self.borderline_probability,
        )


class PreviewRequest(BaseModel):
    polygon: PolygonPayload
    thresholds: ThresholdsPayload = ThresholdsPayload()
    n_samples: int = Field(400, ge=20, le=4000)
    seed: int | None = None


class SaveROIRequest(BaseModel):
    polygon: PolygonPayload
    label: str = "ROI"
    user_id: str = "demo-pathologist"
    thresholds: ThresholdsPayload = ThresholdsPayload()
    notes: str = ""


class UpdateROIRequest(BaseModel):
    polygon: PolygonPayload | None = None
    label: str | None = None
    thresholds: ThresholdsPayload | None = None
    notes: str | None = None


# --------- Router factory ---------------------------------------------------


def _ensure_dirs(cases_dir: Path, rois_dir: Path) -> None:
    if not cases_dir.exists():
        LOG.warning("cases dir %s does not exist", cases_dir)
    rois_dir.mkdir(parents=True, exist_ok=True)


def _format_record(rec: ROIRecord) -> dict[str, Any]:
    return {
        "roi_id": rec.roi_id,
        "case_id": rec.case_id,
        "user_id": rec.user_id,
        "label": rec.label,
        "geometry_thumb_px": rec.geometry_thumb_px,
        "thresholds": rec.thresholds,
        "created_at": rec.created_at,
        "updated_at": rec.updated_at,
        "locked": rec.locked,
        "model_run": rec.model_run,
        "metrics_snapshot": rec.metrics_snapshot,
        "notes": rec.notes,
        "revision": rec.revision,
    }


def build_router(
    *,
    cases_dir: Path | str | None = None,
    rois_dir: Path | str | None = None,
) -> APIRouter:
    """Return a fully-wired ``APIRouter`` for the macrodissection workbench.

    Parameters
    ----------
    cases_dir
        Directory holding ``case_N_*.{jpg,png,json,bin}`` artifacts (the
        Next.js ``frontend/public/cases`` folder by default).
    rois_dir
        Directory where the append-only ROI store keeps its JSONL files.
    """

    cases_path = Path(
        cases_dir
        or os.environ.get("ENSO_CASES_DIR")
        or Path("frontend/public/cases")
    ).resolve()
    rois_path = Path(
        rois_dir
        or os.environ.get("ENSO_ROIS_DIR")
        or Path("backend/.runtime/rois")
    ).resolve()
    _ensure_dirs(cases_path, rois_path)

    store = ROIStore(rois_path)
    router = APIRouter(prefix="/api/macrodissection", tags=["macrodissection"])

    @router.get("/healthz")
    def healthz() -> dict[str, Any]:
        cases = discover_cases(cases_path)
        return {
            "status": "ok",
            "cases_dir": str(cases_path),
            "rois_dir": str(rois_path),
            "n_cases": len(cases),
        }

    @router.get("/threshold-profiles")
    def threshold_profiles() -> list[dict[str, Any]]:
        return list_profiles()

    @router.get("/cases")
    def get_cases() -> list[dict[str, Any]]:
        return [c.to_dict() for c in discover_cases(cases_path)]

    @router.get("/cases/{case_id}")
    def get_case(case_id: int) -> dict[str, Any]:
        cases = {c.case_id: c for c in discover_cases(cases_path)}
        if case_id not in cases:
            raise HTTPException(status_code=404, detail=f"case {case_id} not found")
        return cases[case_id].to_dict()

    @router.post("/cases/{case_id}/rois/preview")
    def preview_metrics(case_id: int, body: PreviewRequest) -> dict[str, Any]:
        try:
            tiles = load_tile_arrays(str(cases_path), case_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail=f"case {case_id} artifacts missing")
        try:
            profile = resolve_thresholds(
                body.thresholds.profile, body.thresholds.to_override()
            )
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        metrics = compute_roi_metrics(
            body.polygon.as_vertex_list(),
            tiles,
            thresholds_purity_min=profile.purity_min,
            thresholds_tumor_cells_min=profile.tumor_cells_min,
            n_samples=body.n_samples,
            seed=body.seed,
        )
        verdict = label_adequacy(metrics, profile)
        return {"metrics": metrics.to_dict(), "verdict": verdict.to_dict()}

    @router.post("/cases/{case_id}/rois")
    def create_roi(case_id: int, body: SaveROIRequest) -> dict[str, Any]:
        if not (cases_path / f"case_{case_id}_tiles.json").exists():
            raise HTTPException(status_code=404, detail=f"case {case_id} artifacts missing")
        tiles = load_tile_arrays(str(cases_path), case_id)
        profile = resolve_thresholds(
            body.thresholds.profile, body.thresholds.to_override()
        )
        metrics = compute_roi_metrics(
            body.polygon.as_vertex_list(),
            tiles,
            thresholds_purity_min=profile.purity_min,
            thresholds_tumor_cells_min=profile.tumor_cells_min,
        )
        verdict = label_adequacy(metrics, profile)
        # Stamp the model versions from the case JSON for audit.
        cases = {c.case_id: c for c in discover_cases(cases_path)}
        case = cases[case_id]
        rec = store.create_draft(
            case_id=case_id,
            user_id=body.user_id,
            label=body.label,
            geometry_thumb_px=body.polygon.model_dump(),
            thresholds=profile.to_dict(),
            model_run={
                "purity_model_version": case.purity_model_version,
                "cellularity_model_version": case.cellularity_model_version,
                "tile_encoder_version": case.tile_encoder_version,
            },
            metrics_snapshot={
                "metrics": metrics.to_dict(),
                "verdict": verdict.to_dict(),
            },
            notes=body.notes,
        )
        return _format_record(rec)

    @router.get("/cases/{case_id}/rois")
    def list_rois(case_id: int) -> list[dict[str, Any]]:
        return [_format_record(r) for r in store.list_latest(case_id)]

    @router.patch("/cases/{case_id}/rois/{roi_id}")
    def update_roi(case_id: int, roi_id: str, body: UpdateROIRequest) -> dict[str, Any]:
        latest = store.get_latest(case_id, roi_id)
        if latest is None:
            raise HTTPException(status_code=404, detail=f"ROI {roi_id} not found")
        if latest.locked:
            raise HTTPException(status_code=409, detail="ROI is locked")
        new_geom = body.polygon.model_dump() if body.polygon else latest.geometry_thumb_px
        new_thresh_payload = body.thresholds or ThresholdsPayload(**{
            k: v
            for k, v in latest.thresholds.items()
            if k
            in (
                "purity_min",
                "tumor_cells_min",
                "borderline_purity_band",
                "borderline_tumor_cells_band",
                "pass_probability",
                "borderline_probability",
            )
        } | {"profile": latest.thresholds.get("name", "humanitas_ngs")})
        profile = resolve_thresholds(
            new_thresh_payload.profile, new_thresh_payload.to_override()
        )
        tiles = load_tile_arrays(str(cases_path), case_id)
        polygon_points = (
            body.polygon.as_vertex_list()
            if body.polygon
            else [
                (float(p[0]), float(p[1]))
                for p in latest.geometry_thumb_px["coordinates"][0]
            ]
        )
        metrics = compute_roi_metrics(
            polygon_points,
            tiles,
            thresholds_purity_min=profile.purity_min,
            thresholds_tumor_cells_min=profile.tumor_cells_min,
        )
        verdict = label_adequacy(metrics, profile)
        updated = store.update_geometry(
            case_id=case_id,
            roi_id=roi_id,
            geometry_thumb_px=new_geom,
            label=body.label,
            thresholds=profile.to_dict(),
            metrics_snapshot={
                "metrics": metrics.to_dict(),
                "verdict": verdict.to_dict(),
            },
            notes=body.notes,
        )
        return _format_record(updated)

    @router.post("/cases/{case_id}/rois/{roi_id}/lock")
    def lock_roi(case_id: int, roi_id: str) -> dict[str, Any]:
        latest = store.get_latest(case_id, roi_id)
        if latest is None:
            raise HTTPException(status_code=404, detail=f"ROI {roi_id} not found")
        # Recompute with authoritative defaults so the locked snapshot is
        # always consistent with the latest geometry.
        if not latest.thresholds:
            raise HTTPException(status_code=400, detail="ROI has no threshold profile")
        profile_name = latest.thresholds.get("name", "humanitas_ngs")
        profile = resolve_thresholds(profile_name, ThresholdOverride(
            purity_min=latest.thresholds.get("purity_min"),
            tumor_cells_min=latest.thresholds.get("tumor_cells_min"),
            borderline_purity_band=latest.thresholds.get("borderline_purity_band"),
            borderline_tumor_cells_band=latest.thresholds.get("borderline_tumor_cells_band"),
            pass_probability=latest.thresholds.get("pass_probability"),
            borderline_probability=latest.thresholds.get("borderline_probability"),
        ))
        tiles = load_tile_arrays(str(cases_path), case_id)
        polygon_points = [
            (float(p[0]), float(p[1]))
            for p in latest.geometry_thumb_px["coordinates"][0]
        ]
        metrics = compute_roi_metrics(
            polygon_points,
            tiles,
            thresholds_purity_min=profile.purity_min,
            thresholds_tumor_cells_min=profile.tumor_cells_min,
        )
        verdict = label_adequacy(metrics, profile)
        rec = store.lock(
            case_id=case_id,
            roi_id=roi_id,
            metrics_snapshot={
                "metrics": metrics.to_dict(),
                "verdict": verdict.to_dict(),
            },
        )
        return _format_record(rec)

    @router.delete("/cases/{case_id}/rois/{roi_id}")
    def delete_roi(case_id: int, roi_id: str) -> dict[str, Any]:
        try:
            store.delete(case_id=case_id, roi_id=roi_id)
        except PermissionError as e:
            raise HTTPException(status_code=409, detail=str(e))
        return {"status": "deleted", "roi_id": roi_id}

    @router.get("/cases/{case_id}/rois/{roi_id}/report")
    def roi_report(case_id: int, roi_id: str) -> dict[str, Any]:
        latest = store.get_latest(case_id, roi_id)
        if latest is None:
            raise HTTPException(status_code=404, detail="ROI not found")
        cases = {c.case_id: c for c in discover_cases(cases_path)}
        if case_id not in cases:
            raise HTTPException(status_code=404, detail="case not found")
        snapshot = latest.metrics_snapshot or {}
        verdict_dict = snapshot.get("verdict")
        metrics_dict = snapshot.get("metrics")
        if not verdict_dict or not metrics_dict:
            raise HTTPException(
                status_code=400,
                detail="ROI has no metrics snapshot; lock the ROI first",
            )
        # Rebuild the verdict object for the payload helper.
        from .adequacy import AdequacyVerdict
        from .roi import MetricsCI, ROIMetrics

        def _ci(d: dict[str, float]) -> MetricsCI:
            return MetricsCI(median=d["median"], low=d["low"], high=d["high"])

        metrics = ROIMetrics(
            n_tiles=int(metrics_dict["n_tiles"]),
            tiles_with_data=int(metrics_dict["tiles_with_data"]),
            area_thumbpx2=float(metrics_dict["area_thumbpx2"]),
            area_mm2=float(metrics_dict["area_mm2"]),
            tissue_fraction_mean=float(metrics_dict["tissue_fraction_mean"]),
            purity=_ci(metrics_dict["purity"]),
            total_nuclei=_ci(metrics_dict["total_nuclei"]),
            tumor_nuclei=_ci(metrics_dict["tumor_nuclei"]),
            adequacy_probability=float(metrics_dict["adequacy_probability"]),
            purity_point=float(metrics_dict["purity_point"]),
            total_nuclei_point=float(metrics_dict["total_nuclei_point"]),
            tumor_nuclei_point=float(metrics_dict["tumor_nuclei_point"]),
        )
        verdict = AdequacyVerdict(
            label=verdict_dict["label"],
            confidence=float(verdict_dict["confidence"]),
            reasons=list(verdict_dict["reasons"]),
            thresholds=verdict_dict["thresholds"],
            metrics_snapshot=verdict_dict["metrics_snapshot"],
        )
        profile_name = latest.thresholds.get("name", "humanitas_ngs")
        profile = resolve_thresholds(profile_name)
        payload = build_report_payload(
            case_meta=cases[case_id],
            roi_record=latest,
            verdict=verdict,
            threshold=profile,
        )
        return payload

    @router.get("/cases/{case_id}/candidates")
    def candidates_endpoint(
        case_id: int,
        k: int = Query(5, ge=1, le=20),
        profile: str = "humanitas_ngs",
        window_tiles: int = Query(5, ge=2, le=40),
        nms_iou: float = Query(0.3, ge=0.0, le=0.95),
    ) -> list[dict[str, Any]]:
        try:
            tiles = load_tile_arrays(str(cases_path), case_id)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="case not found")
        try:
            threshold = resolve_thresholds(profile)
        except KeyError as e:
            raise HTTPException(status_code=400, detail=str(e))
        cands = suggest_candidates(
            tiles,
            purity_min=threshold.purity_min,
            tumor_cells_min=threshold.tumor_cells_min,
            window_tiles=window_tiles,
            top_k=k,
            nms_iou=nms_iou,
        )
        return [
            {
                "rank": c.rank,
                "score": c.score,
                "bbox_thumb_px": list(c.bbox_thumb_px),
                "polygon": {
                    "type": "Polygon",
                    "coordinates": [list([list(p) for p in c.polygon]) + [list(c.polygon[0])]],
                },
                "purity_point": c.purity_point,
                "total_nuclei_point": c.total_nuclei_point,
                "tumor_nuclei_point": c.tumor_nuclei_point,
                "adequacy_probability": c.adequacy_probability,
            }
            for c in cands
        ]

    @router.post("/_debug/clear-cache")
    def _clear_cache() -> dict[str, str]:
        """Drop the LRU tile-array cache (used by tests)."""

        clear_case_cache()
        return {"status": "cleared"}

    return router
