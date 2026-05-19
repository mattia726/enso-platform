"""Plot held-out cellularity predictions against true slide cellularity."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

FONT_STACK = ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
HEADER_TEXT = "#0f172a"
META_TEXT = "#64748b"
POINT_BLUE = "#2563eb"
GRID = "#e5e7eb"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("logs/cellularity_allfolds_test_predictions.csv"),
    )
    ap.add_argument(
        "--out-png",
        type=Path,
        default=Path("logs/cellularity_true_vs_pred_scatter.png"),
    )
    ap.add_argument("--out-json", type=Path, default=None)
    return ap.parse_args()


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0.0:
        return float("nan")
    numer = float(np.sum((y_pred - y_true) ** 2))
    return float(1.0 - numer / denom)


def _safe_float(value: float) -> float | None:
    value = float(value)
    if math.isfinite(value):
        return value
    return None


def _compute_metrics(df: pd.DataFrame) -> dict[str, float | int | None]:
    true_vals = df["true_cell_count"].to_numpy(dtype=np.float64)
    pred_vals = df["pred_cell_count"].to_numpy(dtype=np.float64)
    abs_err = np.abs(pred_vals - true_vals)
    rho = spearmanr(true_vals, pred_vals).statistic
    return {
        "slides": int(len(df)),
        "spearman_rho": _safe_float(rho),
        "r2": _safe_float(_r2_score(true_vals, pred_vals)),
        "slide_mae_count": float(np.mean(abs_err)),
        "slide_medae_count": float(np.median(abs_err)),
        "mean_true_cell_count": float(np.mean(true_vals)),
        "mean_pred_cell_count": float(np.mean(pred_vals)),
        "tile_mae_count": (
            float((df["tile_mae_count"] * df["n_tiles"]).sum() / df["n_tiles"].sum())
            if {"tile_mae_count", "n_tiles"}.issubset(df.columns)
            else None
        ),
    }


def _axis_limit(values: np.ndarray) -> float:
    max_value = float(np.nanmax(values))
    if max_value <= 0.0:
        return 1.0
    step = 25.0 if max_value <= 250.0 else 50.0
    return max(step, math.ceil((max_value * 1.06) / step) * step)


def _plot(df: pd.DataFrame, metrics: dict[str, float | int | None], out_png: Path) -> None:
    true_vals = df["true_cell_count"].to_numpy(dtype=np.float64)
    pred_vals = df["pred_cell_count"].to_numpy(dtype=np.float64)
    limit = _axis_limit(np.concatenate([true_vals, pred_vals]))
    ticks = np.arange(0.0, limit + 0.5, 25.0 if limit <= 250.0 else 50.0)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": FONT_STACK,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )

    fig = plt.figure(figsize=(6.8, 7.25))
    header_ax = fig.add_axes([0.11, 0.86, 0.84, 0.11])
    ax = fig.add_axes([0.12, 0.11, 0.82, 0.73])

    header_ax.set_axis_off()
    header_ax.text(
        0.5,
        0.68,
        "EnsoCellularity",
        ha="center",
        va="center",
        fontsize=17,
        fontweight="bold",
        color=HEADER_TEXT,
    )
    header_ax.text(
        0.5,
        0.22,
        (
            f"slides = {metrics['slides']:,}   "
            f"rho = {metrics['spearman_rho']:.3f}   "
            f"R2 = {metrics['r2']:.3f}   "
            f"MAE = {metrics['slide_mae_count']:.2f}"
        ),
        ha="center",
        va="center",
        fontsize=9.5,
        color=META_TEXT,
    )

    ax.scatter(true_vals, pred_vals, alpha=0.32, s=13, color=POINT_BLUE, edgecolors="none")
    ax.plot([0.0, limit], [0.0, limit], color="#111827", linestyle="--", linewidth=1.0, alpha=0.55)
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_aspect("equal")
    ax.grid(True, color=GRID, linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)
    ax.set_xlabel("True cellularity (mean cells per tile)", labelpad=10)
    ax.set_ylabel("Predicted cellularity (mean cells per tile)", labelpad=10)

    ax.text(
        0.04,
        0.96,
        (
            f"Slide MAE: {metrics['slide_mae_count']:.2f}\n"
            f"Median AE: {metrics['slide_medae_count']:.2f}\n"
            f"Tile MAE: {metrics['tile_mae_count']:.2f}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        color=HEADER_TEXT,
        bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#d1d5db", "boxstyle": "round,pad=0.35"},
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def main() -> None:
    args = _parse_args()
    out_json = args.out_json or args.out_png.with_suffix(".json")
    df = pd.read_csv(args.predictions_csv)
    required = {"true_cell_count", "pred_cell_count"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {args.predictions_csv}: {sorted(missing)}")

    metrics = _compute_metrics(df)
    _plot(df, metrics, args.out_png)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.out_png}")
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
