"""Unit tests for adequacy labelling and threshold resolution."""

from __future__ import annotations

import pytest

from enso_purity.macrodissection.adequacy import label_adequacy
from enso_purity.macrodissection.roi import MetricsCI, ROIMetrics
from enso_purity.macrodissection.thresholds import (
    PROFILES,
    ThresholdOverride,
    list_profiles,
    resolve_thresholds,
)


def _metrics(
    *,
    purity_median: float,
    tumor_nuclei_median: float,
    adequacy_probability: float,
    n_tiles: int = 12,
    tiles_with_data: int = 10,
    total_nuclei_median: float = 1000.0,
) -> ROIMetrics:
    return ROIMetrics(
        n_tiles=n_tiles,
        tiles_with_data=tiles_with_data,
        area_thumbpx2=2400.0,
        area_mm2=0.15,
        tissue_fraction_mean=1.0,
        purity=MetricsCI(
            median=purity_median,
            low=purity_median - 0.05,
            high=purity_median + 0.05,
        ),
        total_nuclei=MetricsCI(
            median=total_nuclei_median,
            low=total_nuclei_median * 0.9,
            high=total_nuclei_median * 1.1,
        ),
        tumor_nuclei=MetricsCI(
            median=tumor_nuclei_median,
            low=tumor_nuclei_median * 0.9,
            high=tumor_nuclei_median * 1.1,
        ),
        adequacy_probability=adequacy_probability,
        purity_point=purity_median,
        total_nuclei_point=total_nuclei_median,
        tumor_nuclei_point=tumor_nuclei_median,
    )


def test_pass_label_when_well_above_thresholds():
    profile = PROFILES["humanitas_ngs"]
    metrics = _metrics(
        purity_median=0.50,
        tumor_nuclei_median=3000.0,
        adequacy_probability=0.97,
    )
    verdict = label_adequacy(metrics, profile)
    assert verdict.label == "pass"


def test_fail_label_when_far_below_thresholds():
    profile = PROFILES["humanitas_ngs"]
    metrics = _metrics(
        purity_median=0.05,
        tumor_nuclei_median=100.0,
        adequacy_probability=0.03,
    )
    verdict = label_adequacy(metrics, profile)
    assert verdict.label == "fail"


def test_borderline_label_inside_purity_band():
    profile = PROFILES["humanitas_ngs"]
    # Median purity just above threshold, well inside the 5pp band.
    metrics = _metrics(
        purity_median=0.22,
        tumor_nuclei_median=1500.0,
        adequacy_probability=0.93,
    )
    verdict = label_adequacy(metrics, profile)
    assert verdict.label == "borderline"


def test_not_quantifiable_when_no_tiles_with_data():
    profile = PROFILES["humanitas_ngs"]
    metrics = _metrics(
        purity_median=0.5,
        tumor_nuclei_median=1000.0,
        adequacy_probability=0.6,
        n_tiles=0,
        tiles_with_data=0,
    )
    verdict = label_adequacy(metrics, profile)
    assert verdict.label == "not_quantifiable"


def test_resolve_thresholds_returns_base_when_no_override():
    profile = resolve_thresholds("humanitas_ngs")
    assert profile.name == "humanitas_ngs"


def test_resolve_thresholds_applies_partial_override():
    profile = resolve_thresholds(
        "humanitas_ngs",
        ThresholdOverride(purity_min=0.40, tumor_cells_min=500),
    )
    assert profile.purity_min == 0.40
    assert profile.tumor_cells_min == 500
    # Untouched fields preserved.
    assert profile.borderline_purity_band == PROFILES["humanitas_ngs"].borderline_purity_band


def test_resolve_thresholds_rejects_unknown_profile():
    with pytest.raises(KeyError):
        resolve_thresholds("nonexistent")


def test_list_profiles_includes_known_names():
    names = {p["name"] for p in list_profiles()}
    assert {"humanitas_ngs", "research", "strict_solid_tumor"} <= names
