"""CLI entry point for hold-out test evaluation.

Reproduces the exact fold split from train_cli.py to extract the test set,
loads the best checkpoint, and reports L1 loss, R², and Spearman ρ.

Usage:
    python -m enso_purity_mil.test_cli \
        --model-path ml/runs/fold0/best_model.pth \
        --manifest data/processed/wedge_mvp_dataset.xlsx \
        --h5-dir /path/to/embeddings_fp32 \
        --cache-dir /path/to/cache \
        --fold 0 --device cuda
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as scipy_beta

from enso_purity_mil.dataset import EmbeddingBagDataset, custom_collate_fn
from enso_purity_mil.manifest_io import load_manifest_table
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig
from enso_purity_mil.train_cli import _build_tumour_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate Enso MIL on hold-out test fold")
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/wedge_mvp_dataset.xlsx"))
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument(
        "--preassigned-fold-column",
        type=str,
        default=None,
        help="Optional tumour manifest column with preassigned fold IDs (0-4).",
    )
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-bags", type=int, default=10,
                    help="Number of random bag samples per case for inference bagging.")
    ap.add_argument("--decision-threshold", type=float, default=0.20,
                    help="Purity threshold for P(purity > threshold) reporting.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load manifest (same logic as train_cli) ──────────────────
    logger.info("Loading manifest: %s", args.manifest)
    manifest = load_manifest_table(args.manifest)
    manifest = manifest[manifest["purity"].notna()].copy()
    logger.info("Slides with purity: %d", len(manifest))

    tumour_df = manifest[manifest["gdc_match_type"] != "normal_tissue"].copy().reset_index(drop=True)
    normal_df = manifest[manifest["gdc_match_type"] == "normal_tissue"].copy().reset_index(drop=True)
    logger.info("Tumour slides: %d, Normal slides: %d", len(tumour_df), len(normal_df))

    # ── Reproduce fold split ─────────────────────────────────────
    folds = _build_tumour_folds(
        tumour_df,
        seed=args.seed,
        preassigned_fold_col=args.preassigned_fold_column,
        cancer_col="project_id",
    )
    if args.preassigned_fold_column is None:
        logger.info("Tumour folds: generated stratified folds from manifest project/purity.")
    else:
        logger.info(
            "Tumour folds: using preassigned fold column %s.",
            args.preassigned_fold_column,
        )

    test_indices = folds[args.fold]
    test_df = tumour_df.iloc[test_indices].reset_index(drop=True)
    logger.info("Fold %d test set: %d tumour slides", args.fold, len(test_df))

    # ── Build dataset ────────────────────────────────────────────
    test_ds = EmbeddingBagDataset(
        test_df, args.h5_dir, num_instances=4096, cache_dir=args.cache_dir,
    )
    logger.info("Test bags: %d", len(test_ds))

    # ── Load model ───────────────────────────────────────────────
    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    model = EnsoMILModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device)
    logger.info("Loaded model from %s (epoch %d, val_loss=%.4f)",
                args.model_path, ckpt.get("epoch", "?"), ckpt.get("val_loss", float("nan")))

    # ── Evaluate with inference bagging ──────────────────────────
    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    y_true: np.ndarray | None = None

    model.eval()
    with torch.no_grad():
        for bag_idx in range(args.num_bags):
            loader = torch.utils.data.DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, collate_fn=custom_collate_fn,
                pin_memory=False, drop_last=False,
            )
            preds_pass: list[float] = []
            labels_pass: list[float] = []
            probs_pass: list[float] = []

            for feats, labels, _is_tumor in loader:
                feats = feats.to(args.device)
                if hasattr(model, "forward_outputs"):
                    out = model.forward_outputs(feats)
                    mu = out["mu"].squeeze(-1)
                    alpha = out["alpha"].squeeze(-1).detach().cpu().numpy()
                    beta = out["beta"].squeeze(-1).detach().cpu().numpy()
                    prob = 1.0 - scipy_beta.cdf(args.decision_threshold, alpha, beta)
                    probs_pass.extend(prob.tolist())
                else:
                    mu = model(feats).squeeze(-1)
                preds_pass.extend(mu.detach().cpu().tolist())
                labels_pass.extend(labels.tolist())

            if y_true is None:
                y_true = np.array(labels_pass, dtype=np.float64)
            all_preds.append(np.array(preds_pass, dtype=np.float64))
            if probs_pass:
                all_probs.append(np.array(probs_pass, dtype=np.float64))

            logger.info("Inference bag %d/%d complete", bag_idx + 1, args.num_bags)

    preds = np.mean(np.stack(all_preds, axis=0), axis=0)
    pred_std = np.std(np.stack(all_preds, axis=0), axis=0)
    probs = np.mean(np.stack(all_probs, axis=0), axis=0) if all_probs else None

    mae = float(np.mean(np.abs(preds - y_true)))
    try:
        from sklearn.metrics import r2_score
        from scipy.stats import spearmanr
        r2 = float(r2_score(y_true, preds))
        sp_corr, _ = spearmanr(y_true, preds)
        spearman = float(sp_corr)
    except Exception:
        r2 = float("nan")
        spearman = float("nan")

    print("\n" + "=" * 60)
    print("TEST RESULTS — Fold %d" % args.fold)
    print("=" * 60)
    print(f"  Test L1 Loss:     {mae:.4f}")
    print(f"  Test R²:          {r2:.4f}")
    print(f"  Test Spearman ρ:  {spearman:.4f}")
    print(f"  Test bags:        {len(test_ds)}")
    print(f"  Inference bags:   {args.num_bags}")
    print(f"  Mean bagging σ:   {float(pred_std.mean()):.4f}")
    if probs is not None:
        print(f"  Mean P(purity>{args.decision_threshold:.2f}): {float(probs.mean()):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
