"""End-to-end API tests for the macrodissection workbench router."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from enso_purity.macrodissection.cases import CASE_TILES_BIN_CHANNELS
from enso_purity.macrodissection.router import build_router


def _write_synthetic_case(cases_dir: Path, case_id: int = 1) -> None:
    """Write a synthetic case_N_tiles.json + case_N_grid.bin pair.

    The grid is 6 × 4 tiles, 20 thumbnail-pixels per side. The left half is
    a hotspot with purity 0.6 and 200 nuclei/tile; the right half is low-
    purity background (0.05 / 30 nuclei).
    """

    grid_nx, grid_ny = 6, 4
    purity = np.full((grid_ny, grid_nx), 0.05, dtype=np.float32)
    nuclei = np.full((grid_ny, grid_nx), 30.0, dtype=np.float32)
    purity[:, : grid_nx // 2] = 0.6
    nuclei[:, : grid_nx // 2] = 200.0
    purity_sd = np.full((grid_ny, grid_nx), 0.05, dtype=np.float32)
    nuclei_sd = np.full((grid_ny, grid_nx), 5.0, dtype=np.float32)
    tumor = purity * nuclei
    tissue = np.full((grid_ny, grid_nx), 1.0, dtype=np.float32)
    packed = np.stack(
        [purity, purity_sd, nuclei, nuclei_sd, tumor, tissue],
        axis=-1,
    ).astype(np.float32)
    bin_path = cases_dir / f"case_{case_id}_grid.bin"
    packed.tofile(bin_path)
    meta = {
        "schema_version": 1,
        "case_id": case_id,
        "barcode": "TCGA-SYN-0001",
        "project_id": "TCGA-TEST",
        "file_uuid": "synthetic-0001",
        "base_width": grid_nx * 20,
        "base_height": grid_ny * 20,
        "tile_pix_w": 20,
        "tile_pix_h": 20,
        "offset_x": 0,
        "offset_y": 0,
        "tile_size_um": 112.0,
        "tile_area_mm2": 0.012544,
        "mpp_thumb_x": 5.6,
        "mpp_thumb_y": 5.6,
        "grid_nx": grid_nx,
        "grid_ny": grid_ny,
        "n_tiles_tissue": int((tissue > 0).sum()),
        "purity_model_version": "synthetic-v0",
        "cellularity_model_version": "synthetic-v0",
        "tile_encoder_version": "synthetic-encoder",
        "thresholds_default": {
            "purity_min": 0.2,
            "tumor_cells_min": 1000,
        },
        "tiles_bin": f"case_{case_id}_grid.bin",
        "tiles_bin_layout": list(CASE_TILES_BIN_CHANNELS),
    }
    (cases_dir / f"case_{case_id}_tiles.json").write_text(json.dumps(meta), encoding="utf-8")
    # Add lightweight placeholders for image URLs (so URLs come back populated).
    for suffix in ("base.jpg", "mask.png", "cell_count_mask.png"):
        (cases_dir / f"case_{case_id}_{suffix}").write_bytes(b"placeholder")


@pytest.fixture()
def client(tmp_path: Path):
    cases_dir = tmp_path / "cases"
    rois_dir = tmp_path / "rois"
    cases_dir.mkdir()
    rois_dir.mkdir()
    _write_synthetic_case(cases_dir, case_id=1)

    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(build_router(cases_dir=cases_dir, rois_dir=rois_dir))
    with TestClient(app) as c:
        c.post("/api/macrodissection/_debug/clear-cache")
        yield c


# ---------- discovery ------------------------------------------------------


def test_healthz_lists_cases(client: TestClient):
    res = client.get("/api/macrodissection/healthz")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["n_cases"] == 1


def test_threshold_profiles_endpoint(client: TestClient):
    res = client.get("/api/macrodissection/threshold-profiles")
    assert res.status_code == 200
    names = {p["name"] for p in res.json()}
    assert "humanitas_ngs" in names


def test_cases_endpoint_returns_case(client: TestClient):
    res = client.get("/api/macrodissection/cases")
    assert res.status_code == 200
    body = res.json()
    assert len(body) == 1
    assert body[0]["case_id"] == 1
    assert body[0]["has_purity"]
    assert body[0]["has_cellularity"]


def test_case_detail(client: TestClient):
    res = client.get("/api/macrodissection/cases/1")
    assert res.status_code == 200
    body = res.json()
    assert body["barcode"] == "TCGA-SYN-0001"


def test_case_detail_404(client: TestClient):
    res = client.get("/api/macrodissection/cases/999")
    assert res.status_code == 404


# ---------- preview --------------------------------------------------------


def _hotspot_polygon() -> dict:
    """Polygon covering the entire left (hotspot) half of the synthetic case."""

    return {
        "type": "Polygon",
        "coordinates": [
            [[0.0, 0.0], [60.0, 0.0], [60.0, 80.0], [0.0, 80.0]],
        ],
    }


def _background_polygon() -> dict:
    """Polygon over the right (low-purity) half."""

    return {
        "type": "Polygon",
        "coordinates": [
            [[60.0, 0.0], [120.0, 0.0], [120.0, 80.0], [60.0, 80.0]],
        ],
    }


def test_preview_hotspot_returns_pass(client: TestClient):
    res = client.post(
        "/api/macrodissection/cases/1/rois/preview",
        json={"polygon": _hotspot_polygon(), "thresholds": {"profile": "humanitas_ngs"}},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["metrics"]["purity"]["median"] > 0.5
    assert body["metrics"]["tumor_nuclei"]["median"] > 1000
    assert body["verdict"]["label"] in ("pass", "borderline")


def test_preview_background_returns_fail(client: TestClient):
    res = client.post(
        "/api/macrodissection/cases/1/rois/preview",
        json={"polygon": _background_polygon()},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["verdict"]["label"] == "fail"


def test_preview_is_deterministic(client: TestClient):
    payload = {"polygon": _hotspot_polygon(), "seed": 42}
    a = client.post("/api/macrodissection/cases/1/rois/preview", json=payload).json()
    b = client.post("/api/macrodissection/cases/1/rois/preview", json=payload).json()
    assert a["metrics"]["purity"]["median"] == b["metrics"]["purity"]["median"]


# ---------- save / lock / list / delete ------------------------------------


def test_save_lock_list_delete_roundtrip(client: TestClient):
    create = client.post(
        "/api/macrodissection/cases/1/rois",
        json={
            "polygon": _hotspot_polygon(),
            "label": "Hot",
            "user_id": "tester",
        },
    )
    assert create.status_code == 200, create.text
    rec = create.json()
    assert rec["locked"] is False
    roi_id = rec["roi_id"]

    listed = client.get("/api/macrodissection/cases/1/rois").json()
    assert any(r["roi_id"] == roi_id for r in listed)

    locked = client.post(f"/api/macrodissection/cases/1/rois/{roi_id}/lock").json()
    assert locked["locked"] is True
    assert locked["metrics_snapshot"]["metrics"]["purity"]["median"] > 0.5

    # Delete should refuse after lock.
    deleted = client.delete(f"/api/macrodissection/cases/1/rois/{roi_id}")
    assert deleted.status_code == 409


def test_delete_draft_succeeds(client: TestClient):
    create = client.post(
        "/api/macrodissection/cases/1/rois",
        json={"polygon": _hotspot_polygon(), "label": "Draft"},
    ).json()
    roi_id = create["roi_id"]
    deleted = client.delete(f"/api/macrodissection/cases/1/rois/{roi_id}")
    assert deleted.status_code == 200
    listed = client.get("/api/macrodissection/cases/1/rois").json()
    assert not any(r["roi_id"] == roi_id for r in listed)


def test_update_geometry_changes_metrics(client: TestClient):
    create = client.post(
        "/api/macrodissection/cases/1/rois",
        json={"polygon": _background_polygon(), "label": "X"},
    ).json()
    roi_id = create["roi_id"]
    updated = client.patch(
        f"/api/macrodissection/cases/1/rois/{roi_id}",
        json={"polygon": _hotspot_polygon()},
    ).json()
    assert updated["revision"] == 2
    assert (
        updated["metrics_snapshot"]["metrics"]["purity"]["median"]
        > create["metrics_snapshot"]["metrics"]["purity"]["median"]
    )


def test_report_endpoint(client: TestClient):
    create = client.post(
        "/api/macrodissection/cases/1/rois",
        json={"polygon": _hotspot_polygon(), "label": "R"},
    ).json()
    roi_id = create["roi_id"]
    client.post(f"/api/macrodissection/cases/1/rois/{roi_id}/lock")
    report = client.get(
        f"/api/macrodissection/cases/1/rois/{roi_id}/report"
    ).json()
    assert "case" in report
    assert "verdict" in report
    assert report["roi"]["roi_id"] == roi_id
    assert report["case"]["case_id"] == 1
    assert "disclaimer" in report


def test_candidates_endpoint(client: TestClient):
    res = client.get(
        "/api/macrodissection/cases/1/candidates?k=3&window_tiles=2"
    )
    assert res.status_code == 200
    body = res.json()
    assert 0 < len(body) <= 3
    top = body[0]
    assert top["adequacy_probability"] >= 0.0
    assert len(top["polygon"]["coordinates"][0]) >= 4


def test_preview_invalid_polygon(client: TestClient):
    res = client.post(
        "/api/macrodissection/cases/1/rois/preview",
        json={"polygon": {"type": "Polygon", "coordinates": [[[0, 0], [1, 0]]]}},
    )
    # Pydantic validation should return 422.
    assert res.status_code == 422


def test_lock_unknown_roi(client: TestClient):
    res = client.post("/api/macrodissection/cases/1/rois/roi_nope/lock")
    assert res.status_code == 404


def test_preview_unknown_profile(client: TestClient):
    res = client.post(
        "/api/macrodissection/cases/1/rois/preview",
        json={
            "polygon": _hotspot_polygon(),
            "thresholds": {"profile": "not_a_profile"},
        },
    )
    assert res.status_code == 400
