"""Tests for GDC biospecimen-based slide ↔ purity matching (Method 2).

All tests are **offline** — no live HTTP calls.  We mock GDC responses.
"""
from __future__ import annotations

import json
from unittest.mock import patch, MagicMock

import pandas as pd
import pytest

from enso_purity.data.gdc_linkage import (
    fetch_biospecimen_for_cases,
    parse_biospecimen_to_slide_aliquot_map,
    match_slides_to_purity_via_gdc,
    _build_gdc_cases_payload,
)


# ---------------------------------------------------------------------------
# Fixture: a minimal GDC /cases response
# ---------------------------------------------------------------------------
@pytest.fixture()
def gdc_cases_response() -> list[dict]:
    """Simulate the 'hits' array from a GDC /cases response with expanded biospecimen."""
    return [
        {
            "submitter_id": "TCGA-AA-0001",
            "case_id": "uuid-case-aa0001",
            "samples": [
                {
                    "submitter_id": "TCGA-AA-0001-01A",
                    "sample_id": "uuid-sample-01a",
                    "portions": [
                        {
                            "submitter_id": "TCGA-AA-0001-01A-11",
                            "portion_id": "uuid-portion-11",
                            "slides": [
                                {
                                    "submitter_id": "TCGA-AA-0001-01A-01-TS1",
                                    "slide_id": "uuid-slide-ts1",
                                    "section_location": "TOP",
                                },
                                {
                                    "submitter_id": "TCGA-AA-0001-01A-01-BS1",
                                    "slide_id": "uuid-slide-bs1",
                                    "section_location": "BOTTOM",
                                },
                            ],
                            "analytes": [
                                {
                                    "submitter_id": "TCGA-AA-0001-01A-11D",
                                    "analyte_id": "uuid-analyte-d",
                                    "aliquots": [
                                        {
                                            "submitter_id": "TCGA-AA-0001-01A-11D-A111-01",
                                            "aliquot_id": "uuid-aliquot-1",
                                        }
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ],
        },
        {
            "submitter_id": "TCGA-BB-0002",
            "case_id": "uuid-case-bb0002",
            "samples": [
                {
                    "submitter_id": "TCGA-BB-0002-01A",
                    "sample_id": "uuid-sample-bb-01a",
                    "portions": [
                        {
                            "submitter_id": "TCGA-BB-0002-01A-21",
                            "portion_id": "uuid-portion-bb-21",
                            "slides": [
                                {
                                    "submitter_id": "TCGA-BB-0002-01A-02-TS1",
                                    "slide_id": "uuid-slide-bb-ts1",
                                    "section_location": "TOP",
                                }
                            ],
                            "analytes": [],
                        },
                        {
                            "submitter_id": "TCGA-BB-0002-01A-11",
                            "portion_id": "uuid-portion-bb-11",
                            "slides": [],
                            "analytes": [
                                {
                                    "submitter_id": "TCGA-BB-0002-01A-11D",
                                    "analyte_id": "uuid-analyte-bb-d",
                                    "aliquots": [
                                        {
                                            "submitter_id": "TCGA-BB-0002-01A-11D-B222-01",
                                            "aliquot_id": "uuid-aliquot-bb-1",
                                        }
                                    ],
                                }
                            ],
                        },
                    ],
                }
            ],
        },
    ]


@pytest.fixture()
def abs_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "array": ["TCGA-AA-0001-01", "TCGA-BB-0002-01"],
            "sample": [
                "TCGA-AA-0001-01A-11D-A111-01",
                "TCGA-BB-0002-01A-11D-B222-01",
            ],
            "call status": ["called", "called"],
            "purity": [0.8, 0.65],
            "ploidy": [2.0, 2.0],
            "solution": ["new", "new"],
        }
    )


@pytest.fixture()
def slides_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "file_id": ["aaa-111", "bbb-222", "ccc-333"],
            "full_path": [
                "aaa-111/TCGA-AA-0001-01A-01-TS1.uuid1.svs",
                "bbb-222/TCGA-AA-0001-01A-01-BS1.uuid2.svs",
                "ccc-333/TCGA-BB-0002-01A-02-TS1.uuid3.svs",
            ],
            "base_mpp_x": [0.5, 0.5, 0.5],
            "base_mpp_y": [0.5, 0.5, 0.5],
            "barcode": [
                "TCGA-AA-0001-01A-01-TS1",
                "TCGA-AA-0001-01A-01-BS1",
                "TCGA-BB-0002-01A-02-TS1",
            ],
            "case_id": ["TCGA-AA-0001", "TCGA-AA-0001", "TCGA-BB-0002"],
            "sample_vial": [
                "TCGA-AA-0001-01A",
                "TCGA-AA-0001-01A",
                "TCGA-BB-0002-01A",
            ],
            "portion": ["01", "01", "02"],
            "section_type": ["TS", "BS", "TS"],
        }
    )


# ---------------------------------------------------------------------------
# _build_gdc_cases_payload
# ---------------------------------------------------------------------------
class TestBuildPayload:
    def test_contains_case_ids(self):
        payload = _build_gdc_cases_payload(["TCGA-AA-0001", "TCGA-BB-0002"], size=10)
        filters = payload["filters"]
        assert filters["content"]["field"] == "submitter_id"
        assert set(filters["content"]["value"]) == {"TCGA-AA-0001", "TCGA-BB-0002"}

    def test_expand_fields(self):
        payload = _build_gdc_cases_payload(["X"], size=10)
        assert "samples.portions.slides" in payload["expand"]
        assert "samples.portions.analytes.aliquots" in payload["expand"]


# ---------------------------------------------------------------------------
# parse_biospecimen_to_slide_aliquot_map
# ---------------------------------------------------------------------------
class TestParseBiospecimen:
    def test_same_portion_mapping(self, gdc_cases_response):
        mapping = parse_biospecimen_to_slide_aliquot_map(gdc_cases_response)

        # AA-0001: slide TS1 + BS1 both in portion-11 which has aliquot 11D-A111-01
        assert "TCGA-AA-0001-01A-01-TS1" in mapping
        assert "TCGA-AA-0001-01A-01-BS1" in mapping

        ts1_info = mapping["TCGA-AA-0001-01A-01-TS1"]
        assert ts1_info["portion_submitter_id"] == "TCGA-AA-0001-01A-11"
        assert "TCGA-AA-0001-01A-11D-A111-01" in ts1_info["same_portion_aliquots"]

    def test_cross_portion_mapping(self, gdc_cases_response):
        mapping = parse_biospecimen_to_slide_aliquot_map(gdc_cases_response)

        # BB-0002: slide TS1 is in portion-21, aliquot is in portion-11
        # same_portion_aliquots should be empty, same_sample_aliquots should have the aliquot
        bb_ts1 = mapping["TCGA-BB-0002-01A-02-TS1"]
        assert len(bb_ts1["same_portion_aliquots"]) == 0
        assert "TCGA-BB-0002-01A-11D-B222-01" in bb_ts1["same_sample_aliquots"]

    def test_all_slides_present(self, gdc_cases_response):
        mapping = parse_biospecimen_to_slide_aliquot_map(gdc_cases_response)
        assert len(mapping) == 3  # TS1+BS1 from AA, TS1 from BB


# ---------------------------------------------------------------------------
# match_slides_to_purity_via_gdc (mocked API)
# ---------------------------------------------------------------------------
class TestMatchViaGDC:
    def test_match_with_mock_api(self, slides_df, abs_df, gdc_cases_response):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"hits": gdc_cases_response, "pagination": {"pages": 1}}
        }

        with patch("enso_purity.data.gdc_linkage.requests.post", return_value=mock_response):
            merged = match_slides_to_purity_via_gdc(slides_df, abs_df)

        # All 3 slides should match (AA slides via same-portion, BB via same-sample)
        assert len(merged) == 3
        assert "purity" in merged.columns
        assert merged["purity"].notna().all()

    def test_match_type_column(self, slides_df, abs_df, gdc_cases_response):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "data": {"hits": gdc_cases_response, "pagination": {"pages": 1}}
        }

        with patch("enso_purity.data.gdc_linkage.requests.post", return_value=mock_response):
            merged = match_slides_to_purity_via_gdc(slides_df, abs_df)

        # AA slides: same_portion, BB slide: same_sample
        aa_rows = merged[merged["case_id"] == "TCGA-AA-0001"]
        assert (aa_rows["gdc_match_type"] == "same_portion").all()

        bb_rows = merged[merged["case_id"] == "TCGA-BB-0002"]
        assert (bb_rows["gdc_match_type"] == "same_sample").all()
