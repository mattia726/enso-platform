"""Utilities for stats computation from precomputed prediction CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def resolve_manifest_path(manifest_path: Path) -> Path:
    """Resolve manifest path, with fallback to wedge_mvp_dataset(1).xlsx."""
    resolved = manifest_path.expanduser().resolve()
    if resolved.exists():
        return resolved
    alt = resolved.parent / "wedge_mvp_dataset(1).xlsx"
    if alt.exists():
        return alt
    raise FileNotFoundError(f"Manifest not found: {resolved} (tried {alt})")


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    """Load wedge manifest and keep rows with genomic purity labels."""
    manifest = pd.read_excel(resolve_manifest_path(manifest_path))
    manifest = manifest[manifest["purity"].notna()].copy()
    return manifest


def load_rows_from_predictions(
    predictions_csv: Path,
    manifest: pd.DataFrame,
    *,
    pred_fold: Optional[int] = None,
) -> pd.DataFrame:
    """Build per-aliquot rows with genomic, MIL, PTN, and project_id.

    Returns columns:
      - ``project_id``
      - ``genomic`` (true purity)
      - ``mil`` (predicted purity)
      - ``ptn`` (pathologist PTN normalized to 0-1)
      - ``aliquot_barcode``
      - ``fold`` (if present in source CSV)
    """
    pred_path = predictions_csv.expanduser().resolve()
    pred = pd.read_csv(pred_path)

    required = {"true_purity", "pred_purity", "aliquot_barcode"}
    missing = required.difference(pred.columns)
    if missing:
        raise ValueError(f"Predictions CSV missing required columns: {sorted(missing)}")

    if pred_fold is not None and "fold" in pred.columns:
        pred = pred[pred["fold"] == pred_fold].copy()

    pred = pred.rename(columns={"true_purity": "genomic", "pred_purity": "mil"})

    tumour_manifest = manifest[manifest["gdc_match_type"] != "normal_tissue"].copy()
    agg_kwargs: dict[str, tuple[str, str]] = {
        "ptn": ("percent_tumor_nuclei", "mean"),
    }
    has_project_id_manifest = "project_id" in tumour_manifest.columns
    if has_project_id_manifest:
        agg_kwargs["project_id_manifest"] = ("project_id", "first")

    aliquot_stats = (
        tumour_manifest.dropna(subset=["aliquot_barcode"])
        .groupby("aliquot_barcode", as_index=False)
        .agg(**agg_kwargs)
    )
    aliquot_stats["ptn"] = aliquot_stats["ptn"] / 100.0

    merged = pred.merge(aliquot_stats, on="aliquot_barcode", how="left")

    if has_project_id_manifest:
        if "project_id" in merged.columns:
            merged["project_id"] = merged["project_id"].fillna(merged["project_id_manifest"])
        else:
            merged["project_id"] = merged["project_id_manifest"]
    elif "project_id" not in merged.columns:
        merged["project_id"] = ""

    keep_cols = ["project_id", "genomic", "mil", "ptn", "aliquot_barcode"]
    if "fold" in merged.columns:
        keep_cols.append("fold")

    rows = merged[keep_cols].copy()
    rows = rows[rows["genomic"].notna() & rows["mil"].notna() & rows["ptn"].notna()].copy()
    return rows
