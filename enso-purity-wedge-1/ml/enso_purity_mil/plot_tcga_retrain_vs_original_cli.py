from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from enso_purity_mil.plot_tcga_cptac_cv_test_eval import _r2_score, _rolling_mae


def _format_threshold(value: float) -> str:
    text = f"{value:.3f}".rstrip("0").rstrip(".")
    if "." not in text:
        text += ".0"
    return text


def _load_prediction_table(path: Path, *, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        read_kwargs = {}
        if sheet_name is not None:
            read_kwargs["sheet_name"] = sheet_name
        df = pd.read_excel(path, **read_kwargs)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t")
    else:
        raise ValueError(f"Unsupported predictions table format: {path}")

    rename_map = {}
    if "predicted_purity" in df.columns and "pred_purity" not in df.columns:
        rename_map["predicted_purity"] = "pred_purity"
    if "label" in df.columns and "true_purity" not in df.columns:
        rename_map["label"] = "true_purity"
    if "pred" in df.columns and "pred_purity" not in df.columns:
        rename_map["pred"] = "pred_purity"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


def _compute_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    true_vals = df["true_purity"].to_numpy(dtype=np.float64)
    pred_vals = df["pred_purity"].to_numpy(dtype=np.float64)
    abs_err = np.abs(pred_vals - true_vals)
    rolling = _rolling_mae(df)
    return {
        "n_items": int(len(df)),
        "rho_spearman": float(spearmanr(true_vals, pred_vals).statistic),
        "r2": _r2_score(true_vals, pred_vals),
        "mae": float(np.mean(abs_err)),
        "medae": float(np.median(abs_err)),
        "rolling_window": int(rolling.attrs["window"]),
        "rolling_mae_mean": float(rolling["rolling_mae"].mean()) if len(rolling) else float("nan"),
    }


def _load_rows_for_audit(
    path: Path,
    folds: list[int],
    *,
    sheet_name: str | None = None,
    bag_prefixes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    df = _load_prediction_table(path, sheet_name=sheet_name)
    needed = {"fold", "bag_id", "true_purity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df[df["fold"].isin(folds)].copy()
    if bag_prefixes:
        mask = pd.Series(False, index=df.index)
        bag_ids = df["bag_id"].astype(str)
        for prefix in bag_prefixes:
            mask = mask | bag_ids.str.startswith(prefix)
        df = df[mask].copy()
    return df.reset_index(drop=True)


def _load_subset_rows(
    path: Path,
    folds: list[int],
    exclude_purity_ge: float,
    *,
    sheet_name: str | None = None,
    bag_prefixes: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    df = _load_prediction_table(path, sheet_name=sheet_name)
    needed = {"fold", "bag_id", "true_purity", "pred_purity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df[df["fold"].isin(folds)].copy()
    if bag_prefixes:
        mask = pd.Series(False, index=df.index)
        bag_ids = df["bag_id"].astype(str)
        for prefix in bag_prefixes:
            mask = mask | bag_ids.str.startswith(prefix)
        df = df[mask].copy()
    df = df[df["true_purity"] < exclude_purity_ge].copy()
    return df[["fold", "bag_id", "true_purity", "pred_purity"]].reset_index(drop=True)


def _load_new_rows(
    path: Path,
    folds: list[int],
    exclude_purity_ge: float,
    *,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    return _load_subset_rows(
        path,
        folds,
        exclude_purity_ge,
        sheet_name=sheet_name,
        bag_prefixes=("tumor_TCGA-",),
    )


def _load_new_rows_combined(
    path: Path,
    folds: list[int],
    exclude_purity_ge: float,
    *,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    return _load_subset_rows(
        path,
        folds,
        exclude_purity_ge,
        sheet_name=sheet_name,
        bag_prefixes=None,
    )


def _load_cptac_rows(
    path: Path,
    folds: list[int],
    exclude_purity_ge: float,
    *,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    return _load_subset_rows(
        path,
        folds,
        exclude_purity_ge,
        sheet_name=sheet_name,
        bag_prefixes=("tumor_CPTAC__",),
    )


def _load_original_rows(
    path: Path,
    folds: list[int],
    exclude_purity_ge: float,
    *,
    sheet_name: str | None = None,
) -> pd.DataFrame:
    df = _load_prediction_table(path, sheet_name=sheet_name)
    needed = {"fold", "bag_id", "true_purity", "pred_purity"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    df = df[["fold", "bag_id", "true_purity", "pred_purity"]].copy()
    df = df[df["fold"].isin(folds)].copy()
    df = df[df["true_purity"] < exclude_purity_ge].copy()
    return df.reset_index(drop=True)


def _align_rows(new_df: pd.DataFrame, orig_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    merged = orig_df.merge(
        new_df,
        on="bag_id",
        how="inner",
        suffixes=("_orig", "_new"),
    )
    if merged.empty:
        raise ValueError("No overlapping bag IDs between original and retrain predictions")

    if not np.allclose(
        merged["true_purity_orig"].to_numpy(dtype=np.float64),
        merged["true_purity_new"].to_numpy(dtype=np.float64),
        atol=1e-8,
    ):
        raise ValueError("True purity mismatch between original and retrain rows after bag alignment")

    if not np.array_equal(
        merged["fold_orig"].to_numpy(dtype=np.int64),
        merged["fold_new"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("Fold mismatch between original and retrain rows after bag alignment")

    merged = merged.rename(
        columns={
            "fold_orig": "fold",
            "true_purity_orig": "true_purity",
            "pred_purity_orig": "pred_purity_original",
            "pred_purity_new": "pred_purity_retrain",
        }
    )
    merged = merged[
        ["fold", "bag_id", "true_purity", "pred_purity_original", "pred_purity_retrain"]
    ].sort_values(["fold", "bag_id"]).reset_index(drop=True)

    orig_aligned = merged[["fold", "bag_id", "true_purity", "pred_purity_original"]].rename(
        columns={"pred_purity_original": "pred_purity"}
    )
    retrain_aligned = merged[["fold", "bag_id", "true_purity", "pred_purity_retrain"]].rename(
        columns={"pred_purity_retrain": "pred_purity"}
    )
    return orig_aligned, retrain_aligned, merged


def _validate_same_bag_universe(
    reference_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    *,
    reference_label: str,
    candidate_label: str,
) -> None:
    ref_bags = set(reference_df["bag_id"].astype(str))
    cand_bags = set(candidate_df["bag_id"].astype(str))
    if ref_bags != cand_bags:
        only_ref = sorted(ref_bags - cand_bags)[:5]
        only_cand = sorted(cand_bags - ref_bags)[:5]
        raise ValueError(
            f"{reference_label} and {candidate_label} do not share the same bag universe. "
            f"{reference_label}-only examples={only_ref}, {candidate_label}-only examples={only_cand}"
        )

    merged = reference_df.merge(
        candidate_df,
        on="bag_id",
        how="inner",
        suffixes=("_ref", "_cand"),
    )
    if len(merged) != len(reference_df) or len(merged) != len(candidate_df):
        raise ValueError(
            f"{reference_label} and {candidate_label} contain duplicate bag IDs after alignment."
        )

    if not np.allclose(
        merged["true_purity_ref"].to_numpy(dtype=np.float64),
        merged["true_purity_cand"].to_numpy(dtype=np.float64),
        atol=1e-8,
    ):
        raise ValueError(
            f"True purity mismatch between {reference_label} and {candidate_label} after bag alignment."
        )

    if not np.array_equal(
        merged["fold_ref"].to_numpy(dtype=np.int64),
        merged["fold_cand"].to_numpy(dtype=np.int64),
    ):
        raise ValueError(
            f"Fold mismatch between {reference_label} and {candidate_label} after bag alignment."
        )


def _concat_disjoint_rows(*frames: pd.DataFrame, label: str) -> pd.DataFrame:
    non_empty = [frame.copy() for frame in frames if frame is not None and not frame.empty]
    if not non_empty:
        raise ValueError(f"No rows available to build {label}")
    merged = pd.concat(non_empty, ignore_index=True)
    dup_mask = merged["bag_id"].astype(str).duplicated(keep=False)
    if dup_mask.any():
        dup_examples = merged.loc[dup_mask, "bag_id"].astype(str).unique().tolist()[:5]
        raise ValueError(f"{label} contains duplicate bag IDs after merge. Examples: {dup_examples}")
    return merged.sort_values(["fold", "bag_id"]).reset_index(drop=True)


def _build_paired_error_frame(
    original_df: pd.DataFrame,
    retrain_df: pd.DataFrame,
    *,
    label: str,
) -> pd.DataFrame:
    _validate_same_bag_universe(
        original_df,
        retrain_df,
        reference_label=f"{label} original rows",
        candidate_label=f"{label} retrain rows",
    )
    merged = original_df.merge(
        retrain_df,
        on="bag_id",
        how="inner",
        suffixes=("_original", "_retrain"),
    )
    merged = merged.rename(
        columns={
            "fold_original": "fold",
            "true_purity_original": "true_purity",
            "pred_purity_original": "pred_purity_original",
            "pred_purity_retrain": "pred_purity_retrain",
        }
    )
    merged["abs_err_original"] = np.abs(
        merged["pred_purity_original"].to_numpy(dtype=np.float64)
        - merged["true_purity"].to_numpy(dtype=np.float64)
    )
    merged["abs_err_retrain"] = np.abs(
        merged["pred_purity_retrain"].to_numpy(dtype=np.float64)
        - merged["true_purity"].to_numpy(dtype=np.float64)
    )
    return merged[
        [
            "fold",
            "bag_id",
            "true_purity",
            "pred_purity_original",
            "pred_purity_retrain",
            "abs_err_original",
            "abs_err_retrain",
        ]
    ].sort_values(["fold", "bag_id"]).reset_index(drop=True)


def _binned_mae(
    df: pd.DataFrame,
    *,
    bin_width: float = 0.1,
) -> pd.DataFrame:
    if bin_width <= 0.0:
        raise ValueError("bin_width must be > 0")
    if df.empty:
        out = pd.DataFrame(
            columns=["bin_left", "bin_right", "bin_center", "mae", "count"]
        )
        out.attrs["bin_width"] = float(bin_width)
        return out

    work = df.copy()
    work["abs_error"] = np.abs(
        work["pred_purity"].to_numpy(dtype=np.float64)
        - work["true_purity"].to_numpy(dtype=np.float64)
    )
    edges = np.arange(0.0, 1.0 + bin_width, bin_width, dtype=np.float64)
    if not np.isclose(edges[-1], 1.0):
        edges = np.append(edges, 1.0)
    # Use left-closed bins so purity 0.9 falls into [0.9, 1.0).
    work["purity_bin"] = pd.cut(
        work["true_purity"],
        bins=edges,
        include_lowest=True,
        right=False,
    )
    grouped = (
        work.dropna(subset=["purity_bin"])
        .groupby("purity_bin", observed=True)["abs_error"]
        .agg(["mean", "count"])
        .reset_index()
    )
    if grouped.empty:
        out = pd.DataFrame(
            columns=["bin_left", "bin_right", "bin_center", "mae", "count"]
        )
        out.attrs["bin_width"] = float(bin_width)
        return out

    grouped["bin_left"] = grouped["purity_bin"].map(lambda iv: float(iv.left)).astype(float)
    grouped["bin_right"] = grouped["purity_bin"].map(lambda iv: float(iv.right)).astype(float)
    grouped["bin_center"] = (grouped["bin_left"] + grouped["bin_right"]) / 2.0
    out = grouped.rename(columns={"mean": "mae"})[
        ["bin_left", "bin_right", "bin_center", "mae", "count"]
    ].sort_values("bin_left").reset_index(drop=True)
    out.attrs["bin_width"] = float(bin_width)
    return out


def _count_unique_or_none(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    return int(df[column].nunique())


def _fold_bag_counts(df: pd.DataFrame) -> dict[str, int]:
    if "fold" not in df.columns or "bag_id" not in df.columns:
        return {}
    counts = df.groupby("fold")["bag_id"].nunique().sort_index()
    return {str(int(fold)): int(count) for fold, count in counts.items()}


def _validate_expected_count(name: str, actual: int | None, expected: int | None) -> None:
    if expected is None or actual is None:
        return
    if actual != expected:
        raise ValueError(f"Audit count mismatch for {name}: expected {expected}, got {actual}")


def _build_audit_payload(
    *,
    comparison_contract: str,
    folds: list[int],
    exclude_purity_ge: float,
    raw_original_df: pd.DataFrame,
    filtered_original_df: pd.DataFrame,
    retrain_tcga_df: pd.DataFrame,
    matched_df: pd.DataFrame | None,
    cptac_df: pd.DataFrame | None,
    original_cptac_df: pd.DataFrame | None,
) -> dict[str, object]:
    return {
        "comparison_contract": comparison_contract,
        "folds_used": [int(fold) for fold in folds],
        "exclude_purity_ge": float(exclude_purity_ge),
        "raw_original_bags": int(raw_original_df["bag_id"].nunique()),
        "raw_original_cases": _count_unique_or_none(raw_original_df, "case_id"),
        "filtered_original_bags": int(filtered_original_df["bag_id"].nunique()),
        "filtered_original_cases": _count_unique_or_none(filtered_original_df, "case_id"),
        "retrain_tcga_bags": int(retrain_tcga_df["bag_id"].nunique()),
        "matched_bags": int(matched_df["bag_id"].nunique()) if matched_df is not None else None,
        "cptac_retrain_bags": int(cptac_df["bag_id"].nunique()) if cptac_df is not None else None,
        "cptac_original_bags": (
            int(original_cptac_df["bag_id"].nunique()) if original_cptac_df is not None else None
        ),
        "raw_original_fold_bag_counts": _fold_bag_counts(raw_original_df),
        "filtered_original_fold_bag_counts": _fold_bag_counts(filtered_original_df),
        "retrain_tcga_fold_bag_counts": _fold_bag_counts(retrain_tcga_df),
        "matched_fold_bag_counts": _fold_bag_counts(matched_df) if matched_df is not None else {},
        "cptac_retrain_fold_bag_counts": _fold_bag_counts(cptac_df) if cptac_df is not None else {},
        "cptac_original_fold_bag_counts": (
            _fold_bag_counts(original_cptac_df) if original_cptac_df is not None else {}
        ),
    }


def _format_audit_note(
    audit_payload: dict[str, object],
    *,
    include_original_cptac_row: bool,
) -> str:
    filtered_cases = audit_payload.get("filtered_original_cases")
    filtered_cases_text = f"{filtered_cases}" if filtered_cases is not None else "unknown"
    lines = [
        f"Comparison contract: {audit_payload['comparison_contract']}",
        (
            "This artifact is a direct bag-matched TCGA comparison on bags with "
            f"true purity < {_format_threshold(float(audit_payload['exclude_purity_ge']))}."
        ),
        (
            "Original v3 raw TCGA source over the selected folds contains "
            f"{audit_payload['raw_original_bags']} bags and {audit_payload['raw_original_cases']} cases."
        ),
        (
            "After excluding exact purity == 1.0, the direct-comparison TCGA universe contains "
            f"{audit_payload['filtered_original_bags']} bags and {filtered_cases_text} cases."
        ),
        (
            "The retrained TCGA evaluation universe contains "
            f"{audit_payload['retrain_tcga_bags']} bags, and the aligned direct overlap is "
            f"{audit_payload['matched_bags']} bags."
        ),
    ]
    if include_original_cptac_row:
        lines.extend(
            [
                (
                    "The CPTAC original-v3 row uses fold-specific original-v3 checkpoints on the same "
                    "current CPTAC test bags as the retrained row."
                ),
                (
                    "The earlier CPTAC result near rho ~0.3 belongs to a different evaluation universe "
                    "unless a saved artifact proves otherwise."
                ),
            ]
        )
    return "\n".join(lines) + "\n"


def _build_comparison_frames(
    mode: str,
    new_predictions_csv: Path,
    original_predictions_csv: Path,
    folds: list[int],
    exclude_purity_ge: float,
    *,
    new_sheet: str | None = None,
    original_sheet: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int | str], str, str, pd.DataFrame | None]:
    orig_df = _load_original_rows(
        original_predictions_csv,
        folds,
        exclude_purity_ge,
        sheet_name=original_sheet,
    )

    if mode == "bag_matched_tcga":
        new_df = _load_new_rows(
            new_predictions_csv,
            folds,
            exclude_purity_ge,
            sheet_name=new_sheet,
        )
        orig_plot_df, retrain_plot_df, merged = _align_rows(new_df, orig_df)
        return (
            orig_plot_df,
            retrain_plot_df,
            {
                "comparison_contract": "bag_matched_tcga_purity_lt_1",
                "unit_label": "bags",
                "n_bags_common": int(len(merged)),
                "exclude_purity_ge": float(exclude_purity_ge),
                "folds": [int(fold) for fold in folds],
                "mode": mode,
                "original_predictions_path": str(original_predictions_csv),
                "original_sheet": original_sheet or "",
                "new_predictions_path": str(new_predictions_csv),
                "new_sheet": new_sheet or "",
            },
            "original_v3",
            "retrain_tcga_subset",
            merged,
        )

    if mode == "original_vs_combined":
        retrain_plot_df = _load_new_rows_combined(
            new_predictions_csv,
            folds,
            exclude_purity_ge,
            sheet_name=new_sheet,
        )
        return (
            orig_df,
            retrain_plot_df,
            {
                "comparison_contract": "original_vs_combined_test_cohorts",
                "unit_label": "bags",
                "n_original_bags": int(len(orig_df)),
                "n_retrain_bags": int(len(retrain_plot_df)),
                "exclude_purity_ge": float(exclude_purity_ge),
                "folds": [int(fold) for fold in folds],
                "mode": mode,
                "original_predictions_path": str(original_predictions_csv),
                "original_sheet": original_sheet or "",
                "new_predictions_path": str(new_predictions_csv),
                "new_sheet": new_sheet or "",
            },
            "original_v3_tcga",
            "retrain_combined_tcga_cptac",
            None,
        )

    raise ValueError(f"Unsupported comparison mode: {mode}")


def _plot_one_row(
    axes: list[plt.Axes],
    df: pd.DataFrame,
    metrics: dict[str, float | int],
    *,
    row_title: str,
    scatter_color: str,
    rolling_color: str,
    residual_color: str,
) -> None:
    scatter_ax, rolling_ax, residual_ax = axes

    true_vals = df["true_purity"].to_numpy(dtype=np.float64)
    pred_vals = df["pred_purity"].to_numpy(dtype=np.float64)
    residuals = pred_vals - true_vals
    rolling = _rolling_mae(df)

    scatter_ax.scatter(true_vals, pred_vals, s=14, alpha=0.72, color=scatter_color, edgecolors="none")
    scatter_ax.plot([0, 1], [0, 1], linestyle="--", color="#111827", linewidth=1.0)
    scatter_ax.set_xlim(0.0, 1.0)
    scatter_ax.set_ylim(0.0, 1.0)
    scatter_ax.set_xlabel("True purity")
    scatter_ax.set_ylabel("Predicted purity")
    scatter_ax.set_title(f"{row_title}\nScatter", fontsize=10.5, pad=8)
    scatter_ax.text(
        0.03,
        0.97,
        (
            f"bags={metrics['n_items']}\n"
            f"rho={metrics['rho_spearman']:.3f}\n"
            f"R2={metrics['r2']:.3f}\n"
            f"MAE={metrics['mae']:.3f}\n"
            f"MedAE={metrics['medae']:.3f}"
        ),
        transform=scatter_ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.5,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db"},
    )

    rolling_ax.scatter(
        true_vals,
        np.abs(residuals),
        s=10,
        alpha=0.30,
        color="#9ca3af",
        edgecolors="none",
    )
    if len(rolling):
        rolling_ax.plot(
            rolling["true_purity"].to_numpy(),
            rolling["rolling_mae"].to_numpy(),
            color=rolling_color,
            linewidth=2.0,
        )
    rolling_ax.set_xlim(0.0, 1.0)
    rolling_ax.set_xlabel("True purity")
    rolling_ax.set_ylabel("|Error| / rolling MAE")
    rolling_ax.set_title(
        f"{row_title}\nRolling MAE (window={metrics['rolling_window']})",
        fontsize=10.5,
        pad=8,
    )

    residual_ax.hist(residuals, bins=24, color=residual_color, edgecolor="white", linewidth=0.8)
    residual_ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    residual_ax.set_xlabel("Prediction error")
    residual_ax.set_ylabel("Bag count")
    residual_ax.set_title(
        f"{row_title}\nResiduals (mean rolling MAE={metrics['rolling_mae_mean']:.3f})",
        fontsize=10.5,
        pad=8,
    )


def _plot_dual(
    original_df: pd.DataFrame,
    retrain_df: pd.DataFrame,
    original_metrics: dict[str, float | int],
    retrain_metrics: dict[str, float | int],
    *,
    original_title: str,
    retrain_title: str,
    figure_title: str,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), constrained_layout=True)
    _plot_one_row(
        list(axes[0]),
        original_df,
        original_metrics,
        row_title=original_title,
        scatter_color="#2563eb",
        rolling_color="#dc2626",
        residual_color="#10b981",
    )
    _plot_one_row(
        list(axes[1]),
        retrain_df,
        retrain_metrics,
        row_title=retrain_title,
        scatter_color="#7c3aed",
        rolling_color="#ea580c",
        residual_color="#0891b2",
    )
    fig.suptitle(figure_title, fontsize=14)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_rows(
    row_specs: list[dict[str, object]],
    *,
    figure_title: str,
    out_path: Path,
) -> None:
    n_rows = len(row_specs)
    if n_rows < 1:
        raise ValueError("Expected at least one row spec to plot")
    fig, axes = plt.subplots(n_rows, 3, figsize=(17.0, 4.15 * n_rows), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.05, h_pad=0.08, hspace=0.07, wspace=0.04)
    if n_rows == 1:
        axes = np.array([axes])
    for ax_row, row_spec in zip(axes, row_specs, strict=True):
        _plot_one_row(
            list(ax_row),
            row_spec["df"],
            row_spec["metrics"],
            row_title=row_spec["title"],
            scatter_color=row_spec["scatter_color"],
            rolling_color=row_spec["rolling_color"],
            residual_color=row_spec["residual_color"],
        )
    if figure_title.strip():
        fig.suptitle(figure_title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overlay_mae_pairs(
    paired_specs: list[dict[str, object]],
    *,
    out_path: Path,
) -> None:
    if not paired_specs:
        raise ValueError("Expected at least one paired spec for overlay MAE plotting")
    fig, axes = plt.subplots(1, len(paired_specs), figsize=(6.0 * len(paired_specs), 5.2), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.06, h_pad=0.05, hspace=0.02, wspace=0.08)
    if len(paired_specs) == 1:
        axes = np.array([axes])

    for ax, spec in zip(axes, paired_specs, strict=True):
        df = spec["paired_df"]
        title = str(spec["title"])
        orig_color = str(spec["original_color"])
        retrain_color = str(spec["retrain_color"])

        true_vals = df["true_purity"].to_numpy(dtype=np.float64)
        orig_err = df["abs_err_original"].to_numpy(dtype=np.float64)
        retrain_err = df["abs_err_retrain"].to_numpy(dtype=np.float64)

        for x, y0, y1 in zip(true_vals, orig_err, retrain_err, strict=True):
            ax.plot([x, x], [y0, y1], color="#cbd5e1", alpha=0.07, linewidth=0.5, zorder=1)

        ax.scatter(
            true_vals,
            orig_err,
            s=10,
            alpha=0.28,
            color=orig_color,
            edgecolors="none",
            label="Original",
            zorder=2,
        )
        ax.scatter(
            true_vals,
            retrain_err,
            s=10,
            alpha=0.28,
            color=retrain_color,
            edgecolors="none",
            label="Retrain",
            zorder=3,
        )

        original_rows = df[["fold", "bag_id", "true_purity", "pred_purity_original"]].rename(
            columns={"pred_purity_original": "pred_purity"}
        )
        retrain_rows = df[["fold", "bag_id", "true_purity", "pred_purity_retrain"]].rename(
            columns={"pred_purity_retrain": "pred_purity"}
        )
        rolling_original = _rolling_mae(original_rows)
        rolling_retrain = _rolling_mae(retrain_rows)
        if len(rolling_original):
            ax.plot(
                rolling_original["true_purity"].to_numpy(),
                rolling_original["rolling_mae"].to_numpy(),
                color=orig_color,
                linewidth=2.0,
                zorder=4,
            )
        if len(rolling_retrain):
            ax.plot(
                rolling_retrain["true_purity"].to_numpy(),
                rolling_retrain["rolling_mae"].to_numpy(),
                color=retrain_color,
                linewidth=2.0,
                zorder=5,
            )

        orig_mae = float(orig_err.mean())
        retrain_mae = float(retrain_err.mean())
        orig_medae = float(np.median(orig_err))
        retrain_medae = float(np.median(retrain_err))

        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("True purity")
        ax.set_ylabel("Absolute error")
        ax.set_title(f"{title}\nOriginal vs retrain MAE pairs", fontsize=11, pad=9)
        ax.text(
            0.03,
            0.97,
            (
                f"bags={len(df)}\n"
                f"orig MAE={orig_mae:.3f}\n"
                f"retr MAE={retrain_mae:.3f}\n"
                f"orig MedAE={orig_medae:.3f}\n"
                f"retr MedAE={retrain_medae:.3f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.3,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#d1d5db"},
        )
        ax.legend(loc="upper right", fontsize=9, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_overlay_mae_pairs_binned(
    paired_specs: list[dict[str, object]],
    *,
    out_path: Path,
    bin_width: float = 0.1,
) -> pd.DataFrame:
    if not paired_specs:
        raise ValueError("Expected at least one paired spec for binned MAE plotting")
    fig, axes = plt.subplots(1, len(paired_specs), figsize=(6.0 * len(paired_specs), 5.2), constrained_layout=True)
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None:
        layout_engine.set(w_pad=0.06, h_pad=0.05, hspace=0.02, wspace=0.08)
    if len(paired_specs) == 1:
        axes = np.array([axes])

    export_rows: list[pd.DataFrame] = []
    for ax, spec in zip(axes, paired_specs, strict=True):
        df = spec["paired_df"]
        title = str(spec["title"])
        dataset_key = str(spec["dataset_key"])
        orig_color = str(spec["original_color"])
        retrain_color = str(spec["retrain_color"])

        true_vals = df["true_purity"].to_numpy(dtype=np.float64)
        orig_err = df["abs_err_original"].to_numpy(dtype=np.float64)
        retrain_err = df["abs_err_retrain"].to_numpy(dtype=np.float64)

        for x, y0, y1 in zip(true_vals, orig_err, retrain_err, strict=True):
            ax.plot([x, x], [y0, y1], color="#cbd5e1", alpha=0.07, linewidth=0.5, zorder=1)

        ax.scatter(
            true_vals,
            orig_err,
            s=10,
            alpha=0.14,
            color=orig_color,
            edgecolors="none",
            label="Original bags",
            zorder=2,
        )
        ax.scatter(
            true_vals,
            retrain_err,
            s=10,
            alpha=0.14,
            color=retrain_color,
            edgecolors="none",
            label="Retrain bags",
            zorder=3,
        )

        original_rows = df[["fold", "bag_id", "true_purity", "pred_purity_original"]].rename(
            columns={"pred_purity_original": "pred_purity"}
        )
        retrain_rows = df[["fold", "bag_id", "true_purity", "pred_purity_retrain"]].rename(
            columns={"pred_purity_retrain": "pred_purity"}
        )
        original_bins = _binned_mae(original_rows, bin_width=bin_width)
        retrain_bins = _binned_mae(retrain_rows, bin_width=bin_width)

        if len(original_bins):
            ax.plot(
                original_bins["bin_center"].to_numpy(dtype=np.float64),
                original_bins["mae"].to_numpy(dtype=np.float64),
                color=orig_color,
                linewidth=2.2,
                marker="o",
                markersize=4.0,
                zorder=4,
                label="Original bin mean",
            )
        if len(retrain_bins):
            ax.plot(
                retrain_bins["bin_center"].to_numpy(dtype=np.float64),
                retrain_bins["mae"].to_numpy(dtype=np.float64),
                color=retrain_color,
                linewidth=2.2,
                marker="o",
                markersize=4.0,
                zorder=5,
                label="Retrain bin mean",
            )

        export_rows.append(
            original_bins.assign(dataset=dataset_key, method="original").rename(columns={"mae": "bin_mae"})
        )
        export_rows.append(
            retrain_bins.assign(dataset=dataset_key, method="retrain").rename(columns={"mae": "bin_mae"})
        )

        orig_mae = float(orig_err.mean())
        retrain_mae = float(retrain_err.mean())
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("True purity")
        ax.set_ylabel("Absolute error")
        ax.set_title(
            f"{title}\nOriginal vs retrain | fixed-bin MAE ({bin_width:.1f})",
            fontsize=11,
            pad=9,
        )
        ax.text(
            0.03,
            0.97,
            (
                f"bags={len(df)}\n"
                f"orig MAE={orig_mae:.3f}\n"
                f"retr MAE={retrain_mae:.3f}\n"
                f"bin width={bin_width:.1f}"
            ),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.3,
            bbox={"facecolor": "white", "alpha": 0.92, "edgecolor": "#d1d5db"},
        )
        ax.legend(loc="upper right", fontsize=8.5, frameon=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    if export_rows:
        return pd.concat(export_rows, ignore_index=True)
    return pd.DataFrame(columns=["dataset", "method", "bin_left", "bin_right", "bin_center", "bin_mae", "count"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Compare bag-level retrained and original v3 evaluation cohorts with explicit audit outputs."
    )
    ap.add_argument("--new-predictions-csv", type=Path, required=True,
                    help="Path to retrain predictions table (.csv/.tsv/.xlsx).")
    ap.add_argument("--original-predictions-csv", type=Path, required=True,
                    help="Path to original-v3 predictions table (.csv/.tsv/.xlsx).")
    ap.add_argument("--new-sheet", type=str, default=None,
                    help="Excel sheet name for retrain predictions when using .xlsx/.xls.")
    ap.add_argument("--original-sheet", type=str, default=None,
                    help="Excel sheet name for original-v3 predictions when using .xlsx/.xls.")
    ap.add_argument("--include-cptac-row", action="store_true",
                    help="Append a third row with CPTAC-only test predictions from the retrained model.")
    ap.add_argument("--cptac-predictions-csv", type=Path, default=None,
                    help="Optional CPTAC row source table; defaults to --new-predictions-csv.")
    ap.add_argument("--cptac-sheet", type=str, default=None,
                    help="Excel sheet name for the CPTAC row source when using .xlsx/.xls.")
    ap.add_argument("--include-original-cptac-row", action="store_true",
                    help="Append an additional CPTAC-only row from original-v3 checkpoints on the current CPTAC test folds.")
    ap.add_argument("--original-cptac-predictions-csv", type=Path, default=None,
                    help="Source table for the original-v3 CPTAC row; required when --include-original-cptac-row is set.")
    ap.add_argument("--original-cptac-sheet", type=str, default=None,
                    help="Excel sheet name for the original-v3 CPTAC row source when using .xlsx/.xls.")
    ap.add_argument("--include-merged-test-rows", action="store_true",
                    help="Append two extra rows with merged TCGA+CPTAC test bags for original-v3 and retrained checkpoints.")
    ap.add_argument("--write-overlay-mae-pairs", action="store_true",
                    help="Write a companion PNG with overlaid original/retrain MAE-vs-purity pairs for TCGA, CPTAC, and merged cohorts.")
    ap.add_argument("--audit-raw-original-predictions-csv", type=Path, default=None,
                    help="Optional raw original-v3 source used to audit pre-filter TCGA bag/case counts.")
    ap.add_argument("--audit-raw-original-sheet", type=str, default=None,
                    help="Excel sheet name for --audit-raw-original-predictions-csv when using .xlsx/.xls.")
    ap.add_argument("--expected-raw-original-bags", type=int, default=None)
    ap.add_argument("--expected-raw-original-cases", type=int, default=None)
    ap.add_argument("--expected-filtered-original-bags", type=int, default=None)
    ap.add_argument("--expected-filtered-original-cases", type=int, default=None)
    ap.add_argument("--expected-retrain-tcga-bags", type=int, default=None)
    ap.add_argument("--expected-matched-bags", type=int, default=None)
    ap.add_argument("--expected-cptac-bags", type=int, default=None)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--exclude-purity-ge", type=float, default=1.0)
    ap.add_argument("--artifact-stem", type=str, default="tcga_folds_0_3_retrain_vs_original")
    ap.add_argument(
        "--comparison-mode",
        type=str,
        choices=["bag_matched_tcga", "original_vs_combined"],
        default="bag_matched_tcga",
    )
    args = ap.parse_args()

    orig_plot_df, retrain_plot_df, metadata, orig_label, retrain_label, merged = _build_comparison_frames(
        args.comparison_mode,
        args.new_predictions_csv,
        args.original_predictions_csv,
        args.folds,
        args.exclude_purity_ge,
        new_sheet=args.new_sheet,
        original_sheet=args.original_sheet,
    )

    orig_metrics = _compute_metrics(orig_plot_df)
    retrain_metrics = _compute_metrics(retrain_plot_df)
    retrain_tcga_df = _load_rows_for_audit(
        args.new_predictions_csv,
        args.folds,
        sheet_name=args.new_sheet,
        bag_prefixes=("tumor_TCGA-",),
    )
    retrain_tcga_df = retrain_tcga_df[
        retrain_tcga_df["true_purity"] < args.exclude_purity_ge
    ].copy().reset_index(drop=True)
    raw_original_source = args.audit_raw_original_predictions_csv or args.original_predictions_csv
    raw_original_source_df = _load_rows_for_audit(
        raw_original_source,
        args.folds,
        sheet_name=args.audit_raw_original_sheet or args.original_sheet,
    )
    filtered_original_source_df = raw_original_source_df[
        raw_original_source_df["true_purity"] < args.exclude_purity_ge
    ].copy().reset_index(drop=True)
    cptac_plot_df = None
    cptac_metrics = None
    original_cptac_plot_df = None
    original_cptac_metrics = None
    merged_original_plot_df = None
    merged_original_metrics = None
    merged_retrain_plot_df = None
    merged_retrain_metrics = None
    if args.include_cptac_row:
        cptac_plot_df = _load_cptac_rows(
            args.cptac_predictions_csv or args.new_predictions_csv,
            args.folds,
            args.exclude_purity_ge,
            sheet_name=args.cptac_sheet or args.new_sheet,
        )
        cptac_metrics = _compute_metrics(cptac_plot_df)
    if args.include_original_cptac_row:
        if args.original_cptac_predictions_csv is None:
            raise ValueError("--original-cptac-predictions-csv is required when --include-original-cptac-row is set")
        original_cptac_plot_df = _load_cptac_rows(
            args.original_cptac_predictions_csv,
            args.folds,
            args.exclude_purity_ge,
            sheet_name=args.original_cptac_sheet,
        )
        original_cptac_metrics = _compute_metrics(original_cptac_plot_df)

    if args.include_cptac_row and args.include_original_cptac_row:
        _validate_same_bag_universe(
            cptac_plot_df,
            original_cptac_plot_df,
            reference_label="retrain CPTAC test rows",
            candidate_label="original-v3 CPTAC test rows",
        )
    if args.include_merged_test_rows:
        if cptac_plot_df is None or original_cptac_plot_df is None:
            raise ValueError(
                "--include-merged-test-rows requires both --include-cptac-row and --include-original-cptac-row"
            )
        merged_original_plot_df = _concat_disjoint_rows(
            orig_plot_df,
            original_cptac_plot_df,
            label="original-v3 merged TCGA+CPTAC test rows",
        )
        merged_original_metrics = _compute_metrics(merged_original_plot_df)
        merged_retrain_plot_df = _concat_disjoint_rows(
            retrain_plot_df,
            cptac_plot_df,
            label="retrained merged TCGA+CPTAC test rows",
        )
        merged_retrain_metrics = _compute_metrics(merged_retrain_plot_df)

    audit_payload = _build_audit_payload(
        comparison_contract=str(metadata["comparison_contract"]),
        folds=args.folds,
        exclude_purity_ge=args.exclude_purity_ge,
        raw_original_df=raw_original_source_df,
        filtered_original_df=filtered_original_source_df,
        retrain_tcga_df=retrain_tcga_df,
        matched_df=merged,
        cptac_df=cptac_plot_df,
        original_cptac_df=original_cptac_plot_df,
    )
    _validate_expected_count("raw_original_bags", audit_payload["raw_original_bags"], args.expected_raw_original_bags)
    _validate_expected_count("raw_original_cases", audit_payload["raw_original_cases"], args.expected_raw_original_cases)
    _validate_expected_count(
        "filtered_original_bags",
        audit_payload["filtered_original_bags"],
        args.expected_filtered_original_bags,
    )
    _validate_expected_count(
        "filtered_original_cases",
        audit_payload["filtered_original_cases"],
        args.expected_filtered_original_cases,
    )
    _validate_expected_count(
        "retrain_tcga_bags",
        audit_payload["retrain_tcga_bags"],
        args.expected_retrain_tcga_bags,
    )
    _validate_expected_count("matched_bags", audit_payload["matched_bags"], args.expected_matched_bags)
    if args.include_cptac_row or args.include_original_cptac_row:
        _validate_expected_count("cptac_retrain_bags", audit_payload["cptac_retrain_bags"], args.expected_cptac_bags)
        _validate_expected_count(
            "cptac_original_bags",
            audit_payload["cptac_original_bags"],
            args.expected_cptac_bags,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_path = args.out_dir / f"{args.artifact_stem}_bag_matched.csv"
    metrics_path = args.out_dir / f"{args.artifact_stem}_metrics.json"
    audit_json_path = args.out_dir / f"{args.artifact_stem}_audit.json"
    audit_tsv_path = args.out_dir / f"{args.artifact_stem}_audit.tsv"
    audit_note_path = args.out_dir / f"{args.artifact_stem}_audit_note.txt"
    plot_path = args.out_dir / f"{args.artifact_stem}_composite.png"
    overlay_plot_path = args.out_dir / f"{args.artifact_stem}_overlay_mae_pairs.png"
    overlay_pairs_path = args.out_dir / f"{args.artifact_stem}_overlay_mae_pairs.csv"
    overlay_binned_plot_path = args.out_dir / f"{args.artifact_stem}_overlay_mae_pairs_fixed_bins.png"
    overlay_binned_path = args.out_dir / f"{args.artifact_stem}_overlay_mae_pairs_fixed_bins.csv"
    metrics_tsv_path = args.out_dir / f"{args.artifact_stem}_metrics.tsv"
    orig_rows_path = args.out_dir / f"{args.artifact_stem}_original_rows_used.csv"
    retrain_rows_path = args.out_dir / f"{args.artifact_stem}_retrain_rows_used.csv"
    cptac_rows_path = args.out_dir / f"{args.artifact_stem}_cptac_rows_used.csv"
    original_cptac_rows_path = args.out_dir / f"{args.artifact_stem}_original_cptac_rows_used.csv"

    if merged is not None:
        merged.to_csv(merged_path, index=False)
    orig_plot_df.to_csv(orig_rows_path, index=False)
    retrain_plot_df.to_csv(retrain_rows_path, index=False)
    if cptac_plot_df is not None:
        cptac_plot_df.to_csv(cptac_rows_path, index=False)
    if original_cptac_plot_df is not None:
        original_cptac_plot_df.to_csv(original_cptac_rows_path, index=False)
    metrics_payload = {
        **metadata,
        "audit": audit_payload,
        orig_label: {
            "comparison_contract": str(metadata["comparison_contract"]),
            **orig_metrics,
        },
        retrain_label: {
            "comparison_contract": str(metadata["comparison_contract"]),
            **retrain_metrics,
        },
    }
    if cptac_metrics is not None:
        metrics_payload["retrain_cptac_test_only"] = {
            "comparison_contract": "current_fold_specific_cptac_tests",
            **cptac_metrics,
        }
    if original_cptac_metrics is not None:
        metrics_payload["original_v3_cptac_test_only"] = {
            "comparison_contract": "current_fold_specific_cptac_tests",
            **original_cptac_metrics,
        }
    if merged_original_metrics is not None:
        metrics_payload["original_v3_tcga_cptac_test_only"] = {
            "comparison_contract": "merged_tcga_cptac_test_bags",
            **merged_original_metrics,
        }
    if merged_retrain_metrics is not None:
        metrics_payload["retrain_tcga_cptac_test_only"] = {
            "comparison_contract": "merged_tcga_cptac_test_bags",
            **merged_retrain_metrics,
        }
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")
    audit_json_path.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")
    pd.DataFrame([audit_payload]).to_csv(audit_tsv_path, sep="\t", index=False)
    audit_note_path.write_text(
        _format_audit_note(
            audit_payload,
            include_original_cptac_row=args.include_original_cptac_row,
        ),
        encoding="utf-8",
    )
    metrics_rows = [
        {"method": orig_label, "comparison_contract": str(metadata["comparison_contract"]), **orig_metrics},
        {"method": retrain_label, "comparison_contract": str(metadata["comparison_contract"]), **retrain_metrics},
    ]
    if cptac_metrics is not None:
        metrics_rows.append(
            {
                "method": "retrain_cptac_test_only",
                "comparison_contract": "current_fold_specific_cptac_tests",
                **cptac_metrics,
            }
        )
    if original_cptac_metrics is not None:
        metrics_rows.append(
            {
                "method": "original_v3_cptac_test_only",
                "comparison_contract": "current_fold_specific_cptac_tests",
                **original_cptac_metrics,
            }
        )
    if merged_original_metrics is not None:
        metrics_rows.append(
            {
                "method": "original_v3_tcga_cptac_test_only",
                "comparison_contract": "merged_tcga_cptac_test_bags",
                **merged_original_metrics,
            }
        )
    if merged_retrain_metrics is not None:
        metrics_rows.append(
            {
                "method": "retrain_tcga_cptac_test_only",
                "comparison_contract": "merged_tcga_cptac_test_bags",
                **merged_retrain_metrics,
            }
        )
    pd.DataFrame(metrics_rows).to_csv(metrics_tsv_path, sep="\t", index=False)

    fold_label = ",".join(str(fold) for fold in args.folds)
    purity_label = _format_threshold(args.exclude_purity_ge)

    if args.comparison_mode == "bag_matched_tcga":
        original_title = (
            f"Original v3 | TCGA matched bags\nfolds {fold_label}, true purity < {purity_label}"
        )
        retrain_title = (
            f"Retrain | TCGA matched bags\nfolds {fold_label}, true purity < {purity_label}"
        )
        figure_title = ""
    else:
        original_title = f"Original v3 | TCGA test bags\nfolds {fold_label}"
        retrain_title = f"Retrain | combined TCGA + CPTAC test bags\nfolds {fold_label}"
        figure_title = ""

    row_specs: list[dict[str, object]] = [
        {
            "df": orig_plot_df,
            "metrics": orig_metrics,
            "title": original_title,
            "scatter_color": "#2563eb",
            "rolling_color": "#dc2626",
            "residual_color": "#10b981",
        },
        {
            "df": retrain_plot_df,
            "metrics": retrain_metrics,
            "title": retrain_title,
            "scatter_color": "#7c3aed",
            "rolling_color": "#ea580c",
            "residual_color": "#0891b2",
        },
    ]
    if original_cptac_plot_df is not None and original_cptac_metrics is not None:
        row_specs.append(
            {
                "df": original_cptac_plot_df,
                "metrics": original_cptac_metrics,
                "title": f"Original v3 | CPTAC current test bags\nfolds {fold_label}",
                "scatter_color": "#1d4ed8",
                "rolling_color": "#991b1b",
                "residual_color": "#047857",
            }
        )
    if cptac_plot_df is not None and cptac_metrics is not None:
        row_specs.append(
            {
                "df": cptac_plot_df,
                "metrics": cptac_metrics,
                "title": f"Retrain | CPTAC current test bags\nfolds {fold_label}",
                "scatter_color": "#b45309",
                "rolling_color": "#be123c",
                "residual_color": "#0f766e",
            }
        )
    if merged_original_plot_df is not None and merged_original_metrics is not None:
        row_specs.append(
            {
                "df": merged_original_plot_df,
                "metrics": merged_original_metrics,
                "title": f"Original v3 | merged TCGA + CPTAC\ntest bags, folds {fold_label}",
                "scatter_color": "#1e40af",
                "rolling_color": "#b91c1c",
                "residual_color": "#0f766e",
            }
        )
    if merged_retrain_plot_df is not None and merged_retrain_metrics is not None:
        row_specs.append(
            {
                "df": merged_retrain_plot_df,
                "metrics": merged_retrain_metrics,
                "title": f"Retrain | merged TCGA + CPTAC\ntest bags, folds {fold_label}",
                "scatter_color": "#6d28d9",
                "rolling_color": "#c2410c",
                "residual_color": "#0d9488",
            }
        )

    _plot_rows(
        row_specs,
        figure_title=figure_title,
        out_path=plot_path,
    )

    if args.write_overlay_mae_pairs:
        if cptac_plot_df is None or original_cptac_plot_df is None:
            raise ValueError("--write-overlay-mae-pairs requires both CPTAC rows to be present")
        if merged_original_plot_df is None or merged_retrain_plot_df is None:
            raise ValueError("--write-overlay-mae-pairs requires --include-merged-test-rows")
        paired_tcga = _build_paired_error_frame(orig_plot_df, retrain_plot_df, label="TCGA matched")
        paired_cptac = _build_paired_error_frame(
            original_cptac_plot_df,
            cptac_plot_df,
            label="CPTAC current test",
        )
        paired_merged = _build_paired_error_frame(
            merged_original_plot_df,
            merged_retrain_plot_df,
            label="Merged TCGA+CPTAC",
        )
        overlay_specs = [
            {
                "paired_df": paired_tcga,
                "dataset_key": "tcga_matched",
                "title": f"TCGA matched bags\nfolds {fold_label}, true purity < {purity_label}",
                "original_color": "#2563eb",
                "retrain_color": "#7c3aed",
            },
            {
                "paired_df": paired_cptac,
                "dataset_key": "cptac_current_test",
                "title": f"CPTAC current test bags\nfolds {fold_label}",
                "original_color": "#1d4ed8",
                "retrain_color": "#b45309",
            },
            {
                "paired_df": paired_merged,
                "dataset_key": "merged_tcga_cptac",
                "title": f"Merged TCGA + CPTAC\ntest bags, folds {fold_label}",
                "original_color": "#1e40af",
                "retrain_color": "#6d28d9",
            },
        ]
        _plot_overlay_mae_pairs(overlay_specs, out_path=overlay_plot_path)
        overlay_binned_export = _plot_overlay_mae_pairs_binned(
            overlay_specs,
            out_path=overlay_binned_plot_path,
            bin_width=0.1,
        )
        overlay_export = pd.concat(
            [
                paired_tcga.assign(dataset="tcga_matched"),
                paired_cptac.assign(dataset="cptac_current_test"),
                paired_merged.assign(dataset="merged_tcga_cptac"),
            ],
            ignore_index=True,
        )
        overlay_export.to_csv(overlay_pairs_path, index=False)
        overlay_binned_export.to_csv(overlay_binned_path, index=False)


if __name__ == "__main__":
    main()
