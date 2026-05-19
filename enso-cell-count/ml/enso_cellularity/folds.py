"""Case-level folds for EnsoCellularity.

Splits are by ``case_id`` so tile or slide leakage cannot inflate validation
metrics. Cases are shuffled within project and assigned round-robin to keep the
pan-cancer mix roughly balanced.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FoldSplit:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def assign_case_folds(
    slide_index: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    project_col: str = "project_id",
    case_col: str = "case_id",
) -> pd.DataFrame:
    """Add a deterministic ``fold`` column to a slide index."""

    if case_col not in slide_index.columns:
        raise ValueError(f"Missing case column: {case_col}")
    if project_col not in slide_index.columns:
        raise ValueError(f"Missing project column: {project_col}")

    rng = np.random.default_rng(seed)
    case_table = (
        slide_index[[case_col, project_col]]
        .drop_duplicates(subset=[case_col])
        .sort_values([project_col, case_col])
        .reset_index(drop=True)
    )
    case_to_fold: dict[str, int] = {}
    for _, sub in case_table.groupby(project_col, sort=True):
        cases = sub[case_col].astype(str).to_numpy()
        rng.shuffle(cases)
        for i, case_id in enumerate(cases.tolist()):
            case_to_fold[case_id] = int(i % n_folds)

    out = slide_index.copy()
    out["fold"] = out[case_col].astype(str).map(case_to_fold).astype(int)
    return out


def apply_preassigned_folds(
    slide_index: pd.DataFrame,
    fold_file: str,
    *,
    policy: str = "drop",
    seed: int = 42,
    n_folds: int = 5,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply purity-model fold assignments to a cellularity slide index.

    The purity prediction file contains one row per purity bag with ``case_id``,
    ``fold``, and semicolon-separated ``file_ids``. Pan-Cancer DX slides usually
    do not share file UUIDs with the purity training slides, so the primary
    matching key is ``case_id``. File-ID matching is still supported when it
    exists.

    ``policy`` controls cases absent from the fold file:
    * ``drop``: remove unassigned rows, preserving only exact purity folds.
    * ``assign``: assign absent cases with a deterministic case-level fallback.
    * ``error``: raise if any rows are absent.
    """

    fold_df = pd.read_csv(fold_file)
    if "case_id" not in fold_df.columns or "fold" not in fold_df.columns:
        raise ValueError("preassigned fold file must contain case_id and fold columns.")
    case_fold_counts = fold_df.groupby("case_id")["fold"].nunique(dropna=True)
    leaking = case_fold_counts[case_fold_counts > 1]
    if not leaking.empty:
        raise ValueError(
            "Preassigned fold file has cases in multiple folds. "
            f"Examples: {leaking.index.tolist()[:10]}"
        )

    case_to_fold = fold_df.drop_duplicates("case_id").set_index("case_id")["fold"].astype(int)
    out = slide_index.copy()
    out["fold"] = out["case_id"].astype(str).map(case_to_fold)

    if "file_ids" in fold_df.columns:
        file_to_fold: dict[str, int] = {}
        for row in fold_df[["fold", "file_ids"]].itertuples(index=False):
            for file_id in str(row.file_ids).split(";"):
                if file_id:
                    file_to_fold[file_id] = int(row.fold)
        missing = out["fold"].isna()
        if missing.any():
            out.loc[missing, "fold"] = out.loc[missing, "file_uuid_original"].astype(str).map(
                file_to_fold
            )

    unassigned = out["fold"].isna()
    report = {
        "input_rows": int(len(out)),
        "assigned_rows": int((~unassigned).sum()),
        "unassigned_rows": int(unassigned.sum()),
    }
    if unassigned.any():
        if policy == "error":
            examples = out.loc[unassigned, ["case_id", "barcode", "project_id"]].head(10)
            raise ValueError(
                "Found slides absent from preassigned fold file. "
                f"Examples:\n{examples.to_string(index=False)}"
            )
        if policy == "drop":
            out = out.loc[~unassigned].copy()
        elif policy == "assign":
            assigned = assign_case_folds(
                out.loc[unassigned].drop(columns=["fold"]),
                n_folds=n_folds,
                seed=seed,
            )
            out.loc[unassigned, "fold"] = assigned["fold"].to_numpy()
        else:
            raise ValueError(f"Unknown unassigned fold policy: {policy}")

    out["fold"] = out["fold"].astype(int)
    return out.reset_index(drop=True), report


def split_for_fold(
    slide_index: pd.DataFrame,
    *,
    fold: int,
    n_folds: int = 5,
    val_fold: int | None = None,
) -> FoldSplit:
    """Return train/val/test DataFrames for one fold."""

    if "fold" not in slide_index.columns:
        raise ValueError("slide_index must contain a fold column. Call assign_case_folds first.")
    test_fold = int(fold)
    val_fold = int((fold + 1) % n_folds if val_fold is None else val_fold)
    if test_fold == val_fold:
        raise ValueError("Validation fold must differ from test fold.")

    test = slide_index[slide_index["fold"] == test_fold].reset_index(drop=True)
    val = slide_index[slide_index["fold"] == val_fold].reset_index(drop=True)
    train = slide_index[~slide_index["fold"].isin([test_fold, val_fold])].reset_index(drop=True)
    _assert_no_case_leakage(train, val, test)
    return FoldSplit(train=train, val=val, test=test)


def _assert_no_case_leakage(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    train_cases = set(train["case_id"].astype(str))
    val_cases = set(val["case_id"].astype(str))
    test_cases = set(test["case_id"].astype(str))
    if train_cases & val_cases:
        raise ValueError("Case leakage between train and val splits.")
    if train_cases & test_cases:
        raise ValueError("Case leakage between train and test splits.")
    if val_cases & test_cases:
        raise ValueError("Case leakage between val and test splits.")
