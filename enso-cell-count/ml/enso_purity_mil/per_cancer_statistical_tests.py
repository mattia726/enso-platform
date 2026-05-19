"""Per-cancer Spearman rho and MAE: Enso MIL vs pathologist PTN.

Supports two data sources:
1. On-the-fly model inference over a fold test split (legacy behavior).
2. Precomputed predictions CSV (e.g. all-fold/all-tiles outputs).

Writes per_cancer_stats.json for frontend/performance table.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from scipy.stats import norm, wilcoxon

from enso_purity_mil.dataset import EmbeddingBagDataset, custom_collate_fn
from enso_purity_mil.folds import generate_stratified_folds
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig
from enso_purity_mil.predictions_utils import load_manifest, load_rows_from_predictions

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _safe_float(value: float, default: float = 0.0) -> float:
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        return default
    return v


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
    logger.info("Test set with PTN: %d samples", len(test_ptn))

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
            preds = model(feats.to(args.device)).squeeze(-1)
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
    ap = argparse.ArgumentParser(description=__doc__)
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
    ap.add_argument("--min-n", type=int, default=5, help="Minimum samples per cancer type to include")
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
        df = load_rows_from_predictions(args.predictions_csv, manifest, pred_fold=args.pred_fold)
        mode_label = "predictions-csv"
    else:
        df = _build_rows_from_model(args, manifest)
        mode_label = f"fold-{args.fold}"

    logger.info("Aliquots with MIL + PTN + genomic: %d (%s)", len(df), mode_label)
    if df.empty:
        raise ValueError("No rows available with MIL + PTN + genomic purity.")

    out: dict[str, dict[str, float | int | bool]] = {}
    for project_id, grp in df.groupby("project_id"):
        n = int(len(grp))
        if n < args.min_n:
            continue

        y_g = grp["genomic"].to_numpy(dtype=np.float64)
        y_mil = grp["mil"].to_numpy(dtype=np.float64)
        y_ptn = grp["ptn"].to_numpy(dtype=np.float64)

        rho_mil, _ = stats.spearmanr(y_mil, y_g)
        rho_ptn, _ = stats.spearmanr(y_ptn, y_g)
        rho_cross, _ = stats.spearmanr(y_mil, y_ptn)
        rho_mil = _safe_float(rho_mil, default=0.0)
        rho_ptn = _safe_float(rho_ptn, default=0.0)
        rho_cross = _safe_float(rho_cross, default=0.0)

        mae_mil = _safe_float(np.mean(np.abs(y_g - y_mil)), default=0.0)
        mae_ptn = _safe_float(np.mean(np.abs(y_g - y_ptn)), default=0.0)
        err_mil = np.abs(y_g - y_mil)
        err_ptn = np.abs(y_g - y_ptn)

        _, p_rho = _meng_z_test(rho_mil, rho_ptn, rho_cross, n)
        try:
            _, p_mae = wilcoxon(err_mil, err_ptn, alternative="two-sided")
            p_mae = _safe_float(p_mae, default=1.0)
        except Exception:
            p_mae = 1.0

        alpha = 0.05
        sig_rho_mil = p_rho < alpha and rho_mil > rho_ptn
        sig_rho_ptn = p_rho < alpha and rho_ptn > rho_mil
        sig_mae_mil = p_mae < alpha and mae_mil < mae_ptn
        sig_mae_ptn = p_mae < alpha and mae_ptn < mae_mil

        out[str(project_id)] = {
            "n": n,
            "rho_mil": rho_mil,
            "rho_ptn": rho_ptn,
            "mae_mil": mae_mil,
            "mae_ptn": mae_ptn,
            "p_rho": _safe_float(p_rho, default=1.0),
            "p_mae": p_mae,
            "sig_rho_mil": bool(sig_rho_mil),
            "sig_rho_ptn": bool(sig_rho_ptn),
            "sig_mae_mil": bool(sig_mae_mil),
            "sig_mae_ptn": bool(sig_mae_ptn),
            "improvement": bool((rho_mil > rho_ptn) or (mae_mil < mae_ptn)),
        }

    out_path = args.out_dir / "per_cancer_stats.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("Wrote %s (%d cancer types)", out_path, len(out))


if __name__ == "__main__":
    main()
