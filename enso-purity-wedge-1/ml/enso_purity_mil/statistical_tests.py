"""Global statistical tests: Enso MIL vs pathologist PTN vs genomic purity.

Supports two data sources:
1. On-the-fly model inference over a fold test split (legacy behavior).
2. Precomputed predictions CSV (e.g. all-fold/all-tiles outputs).

Outputs:
  - statistical_tests.json
  - scatter_mil_vs_ptn.png
  - scatter_data.json
  - error_difference_histogram.png
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
import torch
from matplotlib.ticker import PercentFormatter
from scipy import stats
from scipy.stats import norm, wilcoxon

from enso_purity_mil.dataset import EmbeddingBagDataset, custom_collate_fn
from enso_purity_mil.folds import generate_stratified_folds
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig
from enso_purity_mil.predictions_utils import load_manifest, load_rows_from_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

SCATTER_FONT_STACK = ["Segoe UI", "Arial", "Helvetica", "DejaVu Sans", "sans-serif"]
SCATTER_TICKS = np.linspace(0.0, 1.0, 6)
HEADER_TEXT = "#0f172a"
META_TEXT = "#64748b"


def _safe_float(value: float, default: float = 0.0) -> float:
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return default
    return v


def _bootstrap_spearman(x: np.ndarray, y: np.ndarray, n_boots: int = 2000, seed: int = 42) -> tuple[float, float]:
    if len(x) < 3:
        return 0.0, 0.0
    rng = np.random.RandomState(seed)
    scores: list[float] = []
    for _ in range(n_boots):
        idx = rng.randint(0, len(x), len(x))
        rho, _ = stats.spearmanr(x[idx], y[idx])
        scores.append(_safe_float(rho, default=0.0))
    scores = np.sort(np.array(scores, dtype=np.float64))
    lo = float(scores[int(0.025 * len(scores))])
    hi = float(scores[int(0.975 * len(scores))])
    return lo, hi


def _center_y_ticklabels(ax: plt.Axes, fig: plt.Figure) -> None:
    labels = [label for label in ax.get_yticklabels() if label.get_text()]
    if not labels:
        return

    renderer = fig.canvas.get_renderer()
    max_width_px = max(label.get_window_extent(renderer=renderer).width for label in labels)
    shift_inches = -(max_width_px / fig.dpi) / 2.0

    for label in labels:
        label.set_horizontalalignment("center")
        label.set_transform(
            label.get_transform()
            + mtransforms.ScaledTranslation(shift_inches, 0.0, fig.dpi_scale_trans)
        )


def _meng_z_test(rho1: float, rho2: float, rho12: float, n: int) -> tuple[float, float]:
    """Meng et al. (1992) test for comparing two dependent correlations."""
    if n <= 3:
        return 0.0, 1.0

    rho1 = max(min(_safe_float(rho1), 0.999999), -0.999999)
    rho2 = max(min(_safe_float(rho2), 0.999999), -0.999999)
    rho12 = max(min(_safe_float(rho12), 0.999999), -0.999999)

    z1 = 0.5 * math.log((1 + rho1) / (1 - rho1))
    z2 = 0.5 * math.log((1 + rho2) / (1 - rho2))
    mean_rho_sq = (rho1**2 + rho2**2) / 2
    denom_base = max(1e-8, 1 - mean_rho_sq)
    f = min((1 - rho12) / (2 * denom_base), 1.0)
    h = (1 - f * mean_rho_sq) / denom_base
    denom = 2 * (1 - rho12) * h
    if denom <= 0:
        return 0.0, 1.0
    z_obs = (z1 - z2) * math.sqrt((n - 3) / denom)
    p_val = 2 * (1 - norm.cdf(abs(z_obs)))
    return _safe_float(z_obs), _safe_float(p_val, default=1.0)


def _build_rows_from_model(args: argparse.Namespace, manifest: pd.DataFrame) -> pd.DataFrame:
    if args.model_path is None or args.h5_dir is None:
        raise ValueError("--model-path and --h5-dir are required when --predictions-csv is not set.")

    model_path = args.model_path.expanduser().resolve()
    h5_dir = args.h5_dir.expanduser().resolve()
    cache_dir = args.cache_dir.expanduser().resolve() if args.cache_dir is not None else None

    tumour_df = manifest[manifest["gdc_match_type"] != "normal_tissue"].copy().reset_index(drop=True)
    folds = generate_stratified_folds(tumour_df, n_folds=5, seed=args.seed, cancer_col="project_id")
    test_indices = folds[args.fold]
    test_df = tumour_df.iloc[test_indices].reset_index(drop=True)

    has_ptn = test_df["percent_tumor_nuclei"].notna()
    test_ptn = test_df[has_ptn].copy()
    logger.info("Test set: %d samples, %d with PTN annotations", len(test_df), len(test_ptn))

    ckpt = torch.load(model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    model = EnsoMILModel(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = EmbeddingBagDataset(test_ptn, h5_dir, num_instances=4096, cache_dir=cache_dir)
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=False,
    )

    all_preds: list[float] = []
    all_labels: list[float] = []
    with torch.no_grad():
        for feats, labels, _is_tumor in loader:
            feats = feats.to(args.device)
            preds = model(feats).squeeze(-1)
            preds = torch.clamp(preds, 0, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    rows: list[dict[str, float | str]] = []
    for i, group in enumerate(ds.groups):
        if i >= len(all_preds):
            break
        bag_id = str(group["bag_id"])
        if not bag_id.startswith("tumor_"):
            continue
        aliquot = bag_id[len("tumor_") :]
        sub = test_ptn[test_ptn["aliquot_barcode"] == aliquot]
        if sub.empty:
            continue
        ptn_vals = sub["percent_tumor_nuclei"].dropna().values
        if len(ptn_vals) == 0:
            continue
        rows.append(
            {
                "project_id": str(sub["project_id"].iloc[0]),
                "genomic": float(all_labels[i]),
                "mil": float(all_preds[i]),
                "ptn": float(np.mean(ptn_vals) / 100.0),
                "aliquot_barcode": aliquot,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, default=None)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/wedge_mvp_dataset.xlsx"))
    ap.add_argument("--h5-dir", type=Path, default=None)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--predictions-csv", type=Path, default=None,
                    help="Optional precomputed predictions CSV with true_purity/pred_purity.")
    ap.add_argument("--pred-fold", type=int, default=None,
                    help="Optional fold filter when using --predictions-csv.")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out-dir", type=Path, default=Path("ml/runs/fold0/stats"))
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    manifest = load_manifest(args.manifest)
    if args.predictions_csv is not None:
        rows_df = load_rows_from_predictions(args.predictions_csv, manifest, pred_fold=args.pred_fold)
        mode_label = "predictions-csv"
    else:
        rows_df = _build_rows_from_model(args, manifest)
        mode_label = f"fold-{args.fold}"

    if rows_df.empty:
        raise ValueError("No rows available with MIL + PTN + genomic purity.")

    y_genomic = rows_df["genomic"].to_numpy(dtype=np.float64)
    y_mil = rows_df["mil"].to_numpy(dtype=np.float64)
    y_ptn = rows_df["ptn"].to_numpy(dtype=np.float64)
    n = len(rows_df)
    logger.info("Valid samples with MIL + PTN + genomic: %d (%s)", n, mode_label)

    rho_mil, pval_mil = stats.spearmanr(y_mil, y_genomic)
    rho_mil = _safe_float(rho_mil, default=0.0)
    pval_mil = _safe_float(pval_mil, default=1.0)
    ci_mil_lo, ci_mil_hi = _bootstrap_spearman(y_mil, y_genomic, seed=args.seed)

    rho_ptn, pval_ptn = stats.spearmanr(y_ptn, y_genomic)
    rho_ptn = _safe_float(rho_ptn, default=0.0)
    pval_ptn = _safe_float(pval_ptn, default=1.0)
    ci_ptn_lo, ci_ptn_hi = _bootstrap_spearman(y_ptn, y_genomic, seed=args.seed)

    rho_cross, _ = stats.spearmanr(y_mil, y_ptn)
    rho_cross = _safe_float(rho_cross, default=0.0)

    z_obs, p_meng = _meng_z_test(rho_mil, rho_ptn, rho_cross, n)

    err_mil = np.abs(y_genomic - y_mil)
    err_ptn = np.abs(y_genomic - y_ptn)
    try:
        wilcox_stat, wilcox_p = wilcoxon(err_mil, err_ptn)
        wilcox_stat = _safe_float(wilcox_stat, default=0.0)
        wilcox_p = _safe_float(wilcox_p, default=1.0)
    except Exception:
        wilcox_stat, wilcox_p = 0.0, 1.0

    results = {
        "n_samples": int(n),
        "source": mode_label,
        "rho_mil": rho_mil,
        "pval_mil": pval_mil,
        "ci_mil": [ci_mil_lo, ci_mil_hi],
        "rho_ptn": rho_ptn,
        "pval_ptn": pval_ptn,
        "ci_ptn": [ci_ptn_lo, ci_ptn_hi],
        "rho_cross": rho_cross,
        "meng_z": z_obs,
        "meng_p": p_meng,
        "mae_mil": _safe_float(np.mean(err_mil), default=0.0),
        "median_ae_mil": _safe_float(np.median(err_mil), default=0.0),
        "mae_ptn": _safe_float(np.mean(err_ptn), default=0.0),
        "median_ae_ptn": _safe_float(np.median(err_ptn), default=0.0),
        "wilcoxon_stat": wilcox_stat,
        "wilcoxon_p": wilcox_p,
    }

    print("\n" + "=" * 70)
    print(f"STATISTICAL TESTS - {mode_label} ({n} samples with MIL + PTN + genomic)")
    print("=" * 70)
    print(
        f"\n  Spearman rho (MIL vs genomic):  {rho_mil:.4f} "
        f"(95% CI: {ci_mil_lo:.4f}-{ci_mil_hi:.4f}, P={pval_mil:.1e})"
    )
    print(
        f"  Spearman rho (PTN vs genomic):  {rho_ptn:.4f} "
        f"(95% CI: {ci_ptn_lo:.4f}-{ci_ptn_hi:.4f}, P={pval_ptn:.1e})"
    )
    print(f"\n  Meng et al. z-test:             z={z_obs:.3f}, P={p_meng:.1e}")
    print(f"  Wilcoxon signed-rank test:      stat={wilcox_stat:.0f}, P={wilcox_p:.1e}")
    print("=" * 70)

    (args.out_dir / "statistical_tests.json").write_text(json.dumps(results, indent=2))

    scatter_data = {
        "genomic_purity": y_genomic.tolist(),
        "enso_mil": y_mil.tolist(),
        "pathologist_ptn": y_ptn.tolist(),
    }
    (args.out_dir / "scatter_data.json").write_text(json.dumps(scatter_data, indent=2))

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": SCATTER_FONT_STACK,
            "axes.labelsize": 11,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
        }
    )

    fig = plt.figure(figsize=(10, 5.0))
    header_gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.805, top=0.915, wspace=0.22)
    plot_gs = fig.add_gridspec(1, 2, left=0.07, right=0.985, bottom=0.14, top=0.77, wspace=0.22)

    header_axes = [fig.add_subplot(header_gs[0, i]) for i in range(2)]
    axes = [fig.add_subplot(plot_gs[0, i]) for i in range(2)]

    for header_ax, title, rho, ci_lo, ci_hi in [
        (header_axes[0], "EnsoPurity", rho_mil, ci_mil_lo, ci_mil_hi),
        (header_axes[1], "Pathologist", rho_ptn, ci_ptn_lo, ci_ptn_hi),
    ]:
        header_ax.set_axis_off()
        header_ax.text(
            0.5,
            0.67,
            title,
            ha="center",
            va="center",
            fontsize=15,
            fontweight="bold",
            color=HEADER_TEXT,
        )
        header_ax.text(
            0.5,
            0.21,
            f"ρ = {rho:.3f}   95% CI {ci_lo:.3f}–{ci_hi:.3f}",
            ha="center",
            va="center",
            fontsize=9.25,
            color=META_TEXT,
        )

    for ax, y_pred, y_label in [
        (axes[0], y_mil, "Prediction"),
        (axes[1], y_ptn, ""),
    ]:
        ax.scatter(y_genomic, y_pred, alpha=0.3, s=12, edgecolors="none")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, linewidth=1)
        ax.set_xlabel("Actual Tumor Purity", labelpad=10)
        ax.set_ylabel(y_label, labelpad=10 if y_label else 0)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(SCATTER_TICKS)
        ax.set_yticks(SCATTER_TICKS)
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.tick_params(axis="x", which="major", pad=5)
        ax.tick_params(axis="y", which="major", pad=6)
        ax.set_aspect("equal")

    fig.canvas.draw()
    for ax in axes:
        _center_y_ticklabels(ax, fig)

    fig.savefig(args.out_dir / "scatter_mil_vs_ptn.png", dpi=150, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    diff = err_mil - err_ptn
    ax.hist(diff, bins=40, color="#4c72b0", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("|MIL error| - |PTN error|")
    ax.set_ylabel("Count")
    ax.set_title(f"Error difference (Wilcoxon P={wilcox_p:.1e})\n<0 means MIL is better")
    fig.tight_layout()
    fig.savefig(args.out_dir / "error_difference_histogram.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("Results saved to %s", args.out_dir)


if __name__ == "__main__":
    main()
