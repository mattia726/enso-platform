"""Tests for barcode-based slide ↔ ABSOLUTE purity matching (Method 1)."""
from __future__ import annotations

import pandas as pd
import pytest

from enso_purity.data.slide_purity_matching import (
    extract_barcode_from_path,
    load_and_filter_slides,
    load_and_filter_absolute,
    match_at_sample_vial_level,
    match_at_portion_level,
    build_merged_dataset,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture()
def slides_df() -> pd.DataFrame:
    """Minimal slides dataframe mimicking the real xlsx."""
    return pd.DataFrame(
        {
            "file_id": [
                "aaa-111",
                "bbb-222",
                "ccc-333",
                "ddd-444",
                "eee-555",
                "fff-666",
            ],
            "full_path": [
                "aaa-111/TCGA-AA-0001-01A-01-TS1.uuid1.svs",
                "bbb-222/TCGA-AA-0001-01A-01-BS1.uuid2.svs",
                "ccc-333/TCGA-BB-0002-01A-02-TS1.uuid3.svs",
                "ddd-444/TCGA-CC-0003-01A-01-DX1.uuid4.svs",  # DX → excluded
                "eee-555/TCGA-DD-0004-01A-01-TS1.uuid5.svs",
                "fff-666/TCGA-EE-0005-01A-01-MS1.uuid6.svs",
            ],
            "base_mpp_x": [0.5, 0.5, 0.5, 0.5, 3.0, 0.5],  # DD has mpp>2
            "base_mpp_y": [0.5, 0.5, 0.5, 0.5, 3.0, 0.5],
        }
    )


@pytest.fixture()
def abs_df() -> pd.DataFrame:
    """Minimal ABSOLUTE dataframe mimicking the real txt."""
    return pd.DataFrame(
        {
            "array": ["TCGA-AA-0001-01", "TCGA-BB-0002-01", "TCGA-EE-0005-01", "TCGA-FF-0006-01"],
            "sample": [
                "TCGA-AA-0001-01A-01D-A111-01",  # portion=01, matches slide portion
                "TCGA-BB-0002-01A-11D-B222-01",  # portion=11, does NOT match slide portion 02
                "TCGA-EE-0005-01A-01R-E555-01",  # portion=01, matches slide portion
                "TCGA-FF-0006-01A-11D-F666-01",  # no matching slide at all
            ],
            "call status": ["called", "called", "called", "called"],
            "purity": [0.8, 0.65, 0.9, 0.5],
            "ploidy": [2.0, 2.0, 2.0, 2.0],
            "solution": ["new", "new", "new", "new"],
        }
    )


# ---------------------------------------------------------------------------
# extract_barcode_from_path
# ---------------------------------------------------------------------------
class TestExtractBarcode:
    def test_standard_path(self):
        path = "aaa-111/TCGA-CS-5394-01A-01-BS1.dbd677c6-30a2-4ba5-a81f-af72aff30dcf.svs"
        assert extract_barcode_from_path(path) == "TCGA-CS-5394-01A-01-BS1"

    def test_no_uuid_suffix(self):
        path = "aaa-111/TCGA-CS-5394-01A-01-BS1.svs"
        assert extract_barcode_from_path(path) == "TCGA-CS-5394-01A-01-BS1"


# ---------------------------------------------------------------------------
# load_and_filter_slides
# ---------------------------------------------------------------------------
class TestLoadAndFilterSlides:
    def test_filters_dx_and_high_mpp(self, slides_df):
        result = load_and_filter_slides(slides_df)
        # Should keep: TS1 (AA-0001), BS1 (AA-0001), TS1 (BB-0002), MS1 (EE-0005)
        # Excludes: DX1 (CC-0003), mpp>2 (DD-0004)
        assert len(result) == 4
        section_types = set(result["section_type"])
        assert "DX" not in section_types
        assert section_types == {"TS", "BS", "MS"}

    def test_barcode_columns_present(self, slides_df):
        result = load_and_filter_slides(slides_df)
        for col in ["barcode", "case_id", "sample_vial", "portion", "section_type"]:
            assert col in result.columns


# ---------------------------------------------------------------------------
# load_and_filter_absolute
# ---------------------------------------------------------------------------
class TestLoadAndFilterAbsolute:
    def test_filters_called_and_new(self, abs_df):
        result = load_and_filter_absolute(abs_df)
        assert len(result) == 4  # all pass in our fixture
        assert "sample_vial" in result.columns
        assert "portion" in result.columns

    def test_filters_out_non_called(self, abs_df):
        abs_df = abs_df.copy()
        abs_df.loc[0, "call status"] = "legacy_call"
        result = load_and_filter_absolute(abs_df)
        assert len(result) == 3

    def test_filters_out_non_new(self, abs_df):
        abs_df = abs_df.copy()
        abs_df.loc[0, "solution"] = "old"
        result = load_and_filter_absolute(abs_df)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# match_at_sample_vial_level
# ---------------------------------------------------------------------------
class TestSampleVialMatch:
    def test_matches_correct_count(self, slides_df, abs_df):
        slides = load_and_filter_slides(slides_df)
        absolute = load_and_filter_absolute(abs_df)
        merged = match_at_sample_vial_level(slides, absolute)

        # AA-0001: 2 slides (TS1+BS1) × 1 aliquot = 2 rows
        # BB-0002: 1 slide × 1 aliquot = 1 row
        # EE-0005: 1 slide × 1 aliquot = 1 row
        # FF-0006: no matching slide
        assert len(merged) == 4

    def test_purity_column_present(self, slides_df, abs_df):
        slides = load_and_filter_slides(slides_df)
        absolute = load_and_filter_absolute(abs_df)
        merged = match_at_sample_vial_level(slides, absolute)
        assert "purity" in merged.columns
        assert merged["purity"].notna().all()

    def test_multi_slide_preserved(self, slides_df, abs_df):
        slides = load_and_filter_slides(slides_df)
        absolute = load_and_filter_absolute(abs_df)
        merged = match_at_sample_vial_level(slides, absolute)
        aa_rows = merged[merged["case_id"] == "TCGA-AA-0001"]
        assert len(aa_rows) == 2
        assert set(aa_rows["section_type"]) == {"TS", "BS"}


# ---------------------------------------------------------------------------
# match_at_portion_level
# ---------------------------------------------------------------------------
class TestPortionMatch:
    def test_matches_only_portion_exact(self, slides_df, abs_df):
        slides = load_and_filter_slides(slides_df)
        absolute = load_and_filter_absolute(abs_df)
        merged = match_at_portion_level(slides, absolute)

        # AA-0001 slide portion=01, aliquot portion=01 → match (2 slides)
        # BB-0002 slide portion=02, aliquot portion=11 → NO match
        # EE-0005 slide portion=01, aliquot portion=01 → match
        assert len(merged) == 3

    def test_no_mismatched_portions(self, slides_df, abs_df):
        slides = load_and_filter_slides(slides_df)
        absolute = load_and_filter_absolute(abs_df)
        merged = match_at_portion_level(slides, absolute)
        # BB-0002 should not be present (portion mismatch)
        assert "TCGA-BB-0002" not in merged["case_id"].values


# ---------------------------------------------------------------------------
# build_merged_dataset
# ---------------------------------------------------------------------------
class TestBuildMergedDataset:
    def test_returns_both_methods(self, slides_df, abs_df):
        result = build_merged_dataset(slides_df, abs_df)
        assert "vial" in result
        assert "portion" in result
        assert isinstance(result["vial"], pd.DataFrame)
        assert isinstance(result["portion"], pd.DataFrame)

    def test_vial_superset_of_portion(self, slides_df, abs_df):
        result = build_merged_dataset(slides_df, abs_df)
        assert len(result["vial"]) >= len(result["portion"])

    def test_stats_dict(self, slides_df, abs_df):
        result = build_merged_dataset(slides_df, abs_df)
        assert "stats" in result
        stats = result["stats"]
        assert "n_slides_total" in stats
        assert "n_frozen_mpp_ok" in stats
        assert "n_matched_vial" in stats
        assert "n_matched_portion" in stats
