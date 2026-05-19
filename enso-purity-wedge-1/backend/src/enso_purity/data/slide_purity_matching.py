"""Method 1 — Barcode-based slide ↔ ABSOLUTE purity matching.

Two join strategies:
  * **sample_vial level** — match on case+sample+vial (ignores portion).
    This mirrors the SRTPMs approach and yields ~97 % ABSOLUTE coverage.
  * **portion level** — additionally require the slide barcode portion to
    equal the aliquot barcode portion.  Yields far fewer matches (~16 %)
    because slide and aliquot portion numbers follow different conventions
    in TCGA (see ``docs/dataset_linkage_spec.md``).
"""
from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from enso_purity.data.tcga_barcode import parse_slide_barcode, parse_aliquot_barcode

logger = logging.getLogger(__name__)

_FROZEN_SECTION_TYPES = {"TS", "MS", "BS"}


# ------------------------------------------------------------------
# helpers
# ------------------------------------------------------------------
def extract_barcode_from_path(full_path: str) -> str:
    """Return the TCGA slide barcode from a GCS-style path.

    Expected formats:
        ``<file_uuid>/<BARCODE>.<suffix_uuid>.svs``
        ``<file_uuid>/<BARCODE>.svs``
    """
    filename = str(full_path).rsplit("/", maxsplit=1)[-1]
    return filename.split(".")[0]


# ------------------------------------------------------------------
# loading / filtering
# ------------------------------------------------------------------
def load_and_filter_slides(
    slides_raw: pd.DataFrame,
    *,
    max_mpp: float = 2.0,
) -> pd.DataFrame:
    """Filter slides to processed frozen-section slides and add barcode columns.

    Steps:
        1. Drop rows with missing ``full_path``.
        2. Keep rows where ``base_mpp_x < max_mpp``.
        3. Extract barcode, parse it.
        4. Keep only TS / MS / BS section types.
    """
    df = slides_raw.dropna(subset=["full_path"]).copy()
    df = df[df["base_mpp_x"].notna() & (df["base_mpp_x"] < max_mpp)].copy()

    df["barcode"] = df["full_path"].apply(extract_barcode_from_path)

    parsed = df["barcode"].apply(parse_slide_barcode)
    df["case_id"] = parsed.apply(lambda p: p.case_id)
    df["sample_vial"] = parsed.apply(lambda p: p.sample_vial)
    df["portion"] = parsed.apply(lambda p: p.portion)
    df["section"] = parsed.apply(lambda p: p.section)
    df["section_type"] = parsed.apply(lambda p: p.section_type)

    df = df[df["section_type"].isin(_FROZEN_SECTION_TYPES)].copy()
    logger.info("Filtered slides: %d frozen-section rows (mpp < %.1f)", len(df), max_mpp)
    return df.reset_index(drop=True)


def load_and_filter_absolute(
    abs_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Filter ABSOLUTE table to *called* + *new* and add barcode keys."""
    df = abs_raw[
        (abs_raw["call status"] == "called") & (abs_raw["solution"] == "new")
    ].copy()

    parsed = df["sample"].astype(str).apply(parse_aliquot_barcode)
    df["patient_id"] = parsed.apply(lambda p: p.patient_id)
    df["sample_vial"] = parsed.apply(lambda p: p.sample_vial)
    df["portion"] = parsed.apply(lambda p: p.portion)
    df["analyte"] = parsed.apply(lambda p: p.analyte)

    logger.info("Filtered ABSOLUTE: %d rows (called + new)", len(df))
    return df.reset_index(drop=True)


# ------------------------------------------------------------------
# matching
# ------------------------------------------------------------------
def match_at_sample_vial_level(
    slides: pd.DataFrame,
    absolute: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join slides and ABSOLUTE on ``sample_vial``."""
    merged = slides.merge(
        absolute[["sample", "sample_vial", "purity", "ploidy"]],
        on="sample_vial",
        how="inner",
    )
    logger.info("Sample-vial join: %d slide-purity rows", len(merged))
    return merged


def match_at_portion_level(
    slides: pd.DataFrame,
    absolute: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join slides and ABSOLUTE on ``sample_vial`` + ``portion``."""
    merged = slides.merge(
        absolute[["sample", "sample_vial", "portion", "purity", "ploidy"]],
        on=["sample_vial", "portion"],
        how="inner",
    )
    logger.info("Portion-level join: %d slide-purity rows", len(merged))
    return merged


# ------------------------------------------------------------------
# top-level builder
# ------------------------------------------------------------------
def build_merged_dataset(
    slides_raw: pd.DataFrame,
    abs_raw: pd.DataFrame,
) -> dict[str, Any]:
    """Run both matching strategies and return results + stats."""
    slides = load_and_filter_slides(slides_raw)
    absolute = load_and_filter_absolute(abs_raw)

    vial_merged = match_at_sample_vial_level(slides, absolute)
    portion_merged = match_at_portion_level(slides, absolute)

    n_slides_total = len(slides_raw.dropna(subset=["full_path"]))
    n_frozen_mpp_ok = len(slides)

    stats: dict[str, Any] = {
        "n_slides_total": n_slides_total,
        "n_frozen_mpp_ok": n_frozen_mpp_ok,
        "n_abs_called_new": len(absolute),
        "n_matched_vial": len(vial_merged),
        "n_unique_slides_vial": vial_merged["barcode"].nunique() if len(vial_merged) else 0,
        "n_matched_portion": len(portion_merged),
        "n_unique_slides_portion": portion_merged["barcode"].nunique() if len(portion_merged) else 0,
        "n_multi_slide_portions_vial": int(
            (vial_merged.groupby("sample_vial").size() > 1).sum()
        ) if len(vial_merged) else 0,
    }
    logger.info("Stats: %s", stats)
    return {"vial": vial_merged, "portion": portion_merged, "stats": stats}
