"""Stratified 5-fold cross-validation, stratified by patient + cancer type + purity bin.

Ensures no patient leaks across folds and every fold has balanced
representation of cancer types across purity ranges.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_stratified_folds(
    manifest: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    *,
    case_col: str = "case_id",
    cancer_col: str = "cancer_type",
    purity_col: str = "purity",
    purity_bin_width: float = 0.1,
) -> list[list[int]]:
    """Generate patient-level stratified folds.

    Stratification key = ``(cancer_type, purity_bin)`` where
    ``purity_bin = floor(purity / purity_bin_width)``.

    Returns a list of ``n_folds`` lists, each containing row indices
    into ``manifest``.
    """
    rng = np.random.RandomState(seed)

    df = manifest.copy()
    df["_purity_bin"] = (df[purity_col] / purity_bin_width).astype(int).clip(upper=9)
    df["_strat_key"] = df[cancer_col].astype(str) + "_" + df["_purity_bin"].astype(str)

    # De-duplicate to patient level (one row per patient)
    patient_df = df.groupby(case_col).first().reset_index()

    folds_patient: list[list[str]] = [[] for _ in range(n_folds)]
    for _key, group in patient_df.groupby("_strat_key"):
        patients = group[case_col].tolist()
        rng.shuffle(patients)
        for j, p in enumerate(patients):
            folds_patient[j % n_folds].append(p)

    # Map back to row indices
    folds_indices: list[list[int]] = []
    for fold_patients in folds_patient:
        patient_set = set(fold_patients)
        indices = df.index[df[case_col].isin(patient_set)].tolist()
        folds_indices.append(sorted(indices))

    return folds_indices
