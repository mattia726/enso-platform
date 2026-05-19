from __future__ import annotations

import pandas as pd

from .tcga_barcode import parse_slide_barcode, parse_aliquot_barcode


def build_naive_keys_from_slides(slides: pd.DataFrame, slide_col: str = "slide_id") -> pd.DataFrame:
    """Add case/sample/portion keys to a slides table."""
    out = slides.copy()
    parsed = out[slide_col].apply(parse_slide_barcode)
    out["case_id"] = parsed.apply(lambda x: x.case_id)
    out["sample_vial"] = parsed.apply(lambda x: x.sample_vial)
    out["portion"] = parsed.apply(lambda x: x.portion)
    out["section_type"] = parsed.apply(lambda x: x.section_type)
    out["is_dx"] = parsed.apply(lambda x: x.is_dx)
    return out


def build_naive_keys_from_abs(abs_df: pd.DataFrame, aliquot_col: str = "sample") -> pd.DataFrame:
    """Add patient/sample/portion keys to ABSOLUTE table."""
    out = abs_df.copy()
    parsed = out[aliquot_col].astype(str).apply(parse_aliquot_barcode)
    out["patient_id"] = parsed.apply(lambda x: x.patient_id)
    out["sample_vial"] = parsed.apply(lambda x: x.sample_vial)
    out["portion"] = parsed.apply(lambda x: x.portion)
    return out


def naive_match_rates(slides: pd.DataFrame, abs_df: pd.DataFrame) -> dict:
    """Compute match rates at vial-level and portion-level."""
    slide_vials = set(slides["sample_vial"].unique())
    slide_portions = set((slides["sample_vial"] + "-" + slides["portion"]).unique())

    abs_vials = abs_df["sample_vial"].astype(str)
    abs_portions = abs_df["sample_vial"].astype(str) + "-" + abs_df["portion"].astype(str)

    return {
        "abs_to_slides_vial_match_rate": float(abs_vials.isin(slide_vials).mean()),
        "abs_to_slides_portion_match_rate": float(abs_portions.isin(slide_portions).mean()),
    }
