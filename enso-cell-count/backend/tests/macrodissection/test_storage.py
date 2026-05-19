"""Unit tests for the JSON-lines ROI store."""

from __future__ import annotations

import pytest

from enso_purity.macrodissection.storage import ROIRecord, ROIStore


def _draft_kwargs():
    return dict(
        case_id=12,
        user_id="demo",
        label="ROI 1",
        geometry_thumb_px={
            "type": "Polygon",
            "coordinates": [[[0, 0], [10, 0], [10, 10], [0, 10]]],
        },
        thresholds={"purity_min": 0.2, "tumor_cells_min": 1000},
        model_run={"purity_model_version": "v3_fold0"},
        metrics_snapshot=None,
        notes="initial draft",
    )


def test_append_and_list_latest(tmp_path):
    store = ROIStore(tmp_path)
    draft = store.create_draft(**_draft_kwargs())
    rois = store.list_latest(12)
    assert len(rois) == 1
    assert rois[0].roi_id == draft.roi_id
    assert rois[0].locked is False


def test_update_geometry_increments_revision(tmp_path):
    store = ROIStore(tmp_path)
    draft = store.create_draft(**_draft_kwargs())
    updated = store.update_geometry(
        case_id=12,
        roi_id=draft.roi_id,
        geometry_thumb_px={
            "type": "Polygon",
            "coordinates": [[[5, 5], [12, 5], [12, 12], [5, 12]]],
        },
    )
    assert updated.revision == 2
    history = store.list_history(12, draft.roi_id)
    assert len(history) == 2
    assert history[-1].geometry_thumb_px["coordinates"][0][0] == [5, 5]


def test_lock_records_metrics_snapshot(tmp_path):
    store = ROIStore(tmp_path)
    draft = store.create_draft(**_draft_kwargs())
    locked = store.lock(
        case_id=12,
        roi_id=draft.roi_id,
        metrics_snapshot={"purity": 0.42, "tumor_nuclei": 2500},
    )
    assert locked.locked is True
    assert locked.metrics_snapshot["purity"] == 0.42
    # Re-lock is a no-op
    locked_again = store.lock(case_id=12, roi_id=draft.roi_id, metrics_snapshot={})
    assert locked_again.metrics_snapshot["purity"] == 0.42


def test_locked_roi_cannot_be_edited(tmp_path):
    store = ROIStore(tmp_path)
    draft = store.create_draft(**_draft_kwargs())
    store.lock(case_id=12, roi_id=draft.roi_id, metrics_snapshot={"x": 1})
    with pytest.raises(PermissionError):
        store.update_geometry(
            case_id=12,
            roi_id=draft.roi_id,
            geometry_thumb_px={"type": "Polygon", "coordinates": [[[0, 0]]]},
        )


def test_delete_draft_removes_record(tmp_path):
    store = ROIStore(tmp_path)
    draft = store.create_draft(**_draft_kwargs())
    store.delete(case_id=12, roi_id=draft.roi_id)
    assert store.list_latest(12) == []


def test_delete_locked_refuses(tmp_path):
    store = ROIStore(tmp_path)
    draft = store.create_draft(**_draft_kwargs())
    store.lock(case_id=12, roi_id=draft.roi_id, metrics_snapshot={})
    with pytest.raises(PermissionError):
        store.delete(case_id=12, roi_id=draft.roi_id)


def test_history_persists_across_store_instances(tmp_path):
    store = ROIStore(tmp_path)
    draft = store.create_draft(**_draft_kwargs())
    store.update_geometry(
        case_id=12,
        roi_id=draft.roi_id,
        geometry_thumb_px={"type": "Polygon", "coordinates": [[[1, 1]]]},
    )
    other = ROIStore(tmp_path)
    history = other.list_history(12, draft.roi_id)
    assert len(history) == 2
