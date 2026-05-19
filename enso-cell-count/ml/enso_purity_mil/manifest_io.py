"""Helpers for loading training manifests in multiple table formats."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_manifest_table(path: Path | str) -> pd.DataFrame:
    """Load a manifest table from ``.xlsx``, ``.csv``, or ``.tsv``."""
    manifest_path = Path(path)
    suffix = manifest_path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(manifest_path)
    if suffix == ".csv":
        return pd.read_csv(manifest_path)
    if suffix == ".tsv":
        return pd.read_csv(manifest_path, sep="\t")

    raise ValueError(
        f"Unsupported manifest format for {manifest_path}. "
        "Expected one of: .xlsx, .xls, .csv, .tsv"
    )
