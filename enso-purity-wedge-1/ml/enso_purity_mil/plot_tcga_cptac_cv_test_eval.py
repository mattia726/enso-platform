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


def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    if denom <= 0.0:
        return float("nan")
    numer = float(np.sum((y_pred - y_true) ** 2))
    return float(1.0 - (numer / denom))


def _rolling_mae(df: pd.DataFrame, *, window_frac: float = 0.15) -> pd.DataFrame:
    work = df.sort_values("true_purity").reset_index(drop=True).copy()
    n = len(work)
    window = max(5, int(round(window_frac * n)))
    if window % 2 == 0:
        window += 1
    work["abs_error"] = np.abs(work["pred_purity"] - work["true_purity"])
    rolling = (
        work["abs_error"]
        .rolling(window=window, center=True, min_periods=max(3, window // 3))
        .mean()
    )
    out = pd.DataFrame(
        {
            "true_purity": work["true_purity"],
            "rolling_mae": rolling,
        }
    ).dropna()
    out.attrs["window"] = window
    return out


def _load_rows(run_dir: Path, folds: list[int]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        path = run_dir / f"fold{fold}_test" / f"fold{fold}_test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing fold predictions CSV: {path}")
        df = pd.read_csv(path)
        if "label" not in df.columns or "pred" not in df.columns:
            raise ValueError(f"Unexpected columns in {path}")
        df = df.copy()
        df["fold"] = fold
        df["true_purity"] = df["label"].astype(float)
        df["pred_purity"] = df["pred"].astype(float)
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


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


def _plot_eval(df: pd.DataFrame, metrics: dict[str, float | int], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    scatter_ax, rolling_ax, residual_ax = axes

    true_vals = df["true_purity"].to_numpy(dtype=np.float64)
    pred_vals = df["pred_purity"].to_numpy(dtype=np.float64)
    residuals = pred_vals - true_vals
    rolling = _rolling_mae(df)

    scatter_ax.scatter(true_vals, pred_vals, s=18, alpha=0.75, color="#2563eb", edgecolors="none")
    scatter_ax.plot([0, 1], [0, 1], linestyle="--", color="#111827", linewidth=1.0)
    scatter_ax.set_xlim(0.0, 1.0)
    scatter_ax.set_ylim(0.0, 1.0)
    scatter_ax.set_xlabel("True purity")
    scatter_ax.set_ylabel("Predicted purity")
    scatter_ax.set_title("TCGA FS + CPTAC DX | folds 0-3 test")
    scatter_ax.text(
        0.03,
        0.97,
        (
            f"n={metrics['n_items']}\n"
            f"rho={metrics['rho_spearman']:.3f}\n"
            f"R2={metrics['r2']:.3f}\n"
            f"MAE={metrics['mae']:.3f}\n"
            f"MedAE={metrics['medae']:.3f}"
        ),
        transform=scatter_ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": "#d1d5db"},
    )

    rolling_ax.scatter(
        true_vals,
        np.abs(residuals),
        s=12,
        alpha=0.35,
        color="#9ca3af",
        edgecolors="none",
    )
    if len(rolling):
        rolling_ax.plot(
            rolling["true_purity"].to_numpy(),
            rolling["rolling_mae"].to_numpy(),
            color="#dc2626",
            linewidth=2.0,
        )
    rolling_ax.set_xlim(0.0, 1.0)
    rolling_ax.set_xlabel("True purity")
    rolling_ax.set_ylabel("|Error| / rolling MAE")
    rolling_ax.set_title(f"Rolling MAE (window={metrics['rolling_window']})")

    residual_ax.hist(residuals, bins=24, color="#10b981", edgecolor="white", linewidth=0.8)
    residual_ax.axvline(0.0, color="#111827", linestyle="--", linewidth=1.0)
    residual_ax.set_xlabel("Prediction error")
    residual_ax.set_ylabel("Bag count")
    residual_ax.set_title(f"Residuals | mean rolling MAE={metrics['rolling_mae_mean']:.3f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot combined CV test evaluation for folds 0-3.")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    df = _load_rows(args.run_dir, args.folds)
    metrics = _compute_metrics(df)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pred_path = args.out_dir / "folds_0_3_test_predictions_merged.csv"
    metrics_path = args.out_dir / "folds_0_3_test_metrics.json"
    plot_path = args.out_dir / "folds_0_3_test_composite.png"
    per_fold_path = args.out_dir / "folds_0_3_test_per_fold.tsv"

    df.to_csv(pred_path, index=False)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    per_fold_rows = []
    for fold, sub in df.groupby("fold"):
        per_fold_rows.append(
            {
                "fold": int(fold),
                "n_items": int(len(sub)),
                "mae": float(np.mean(np.abs(sub["pred_purity"] - sub["true_purity"]))),
                "medae": float(np.median(np.abs(sub["pred_purity"] - sub["true_purity"]))),
                "r2": _r2_score(
                    sub["true_purity"].to_numpy(dtype=np.float64),
                    sub["pred_purity"].to_numpy(dtype=np.float64),
                ),
                "rho_spearman": float(
                    spearmanr(
                        sub["true_purity"].to_numpy(dtype=np.float64),
                        sub["pred_purity"].to_numpy(dtype=np.float64),
                    ).statistic
                ),
            }
        )
    pd.DataFrame(per_fold_rows).sort_values("fold").to_csv(per_fold_path, sep="\t", index=False)

    _plot_eval(df, metrics, plot_path)


if __name__ == "__main__":
    main()
