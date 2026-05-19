#!/usr/bin/env python3
"""Evaluate a trained fold on the held-out tumour test split.

This mirrors the retrain setup by honoring a preassigned tumour fold column
and keeping normals out of the test set.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import beta as scipy_beta
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score

from enso_purity_mil.dataset import EmbeddingBagDataset, custom_collate_fn
from enso_purity_mil.manifest_io import load_manifest_table
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig
from enso_purity_mil.train_cli import _build_tumour_folds

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate a retrained fold on its held-out test split")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--preassigned-fold-column", type=str, default="preassigned_fold")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--num-bags", type=int, default=10)
    ap.add_argument("--decision-threshold", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    logger.info("Loading manifest: %s", args.manifest)
    manifest = load_manifest_table(args.manifest)
    manifest = manifest[manifest["purity"].notna()].copy()
    tumour_df = manifest[manifest["gdc_match_type"] != "normal_tissue"].copy().reset_index(drop=True)
    logger.info("Tumour slides with purity: %d", len(tumour_df))

    folds = _build_tumour_folds(
        tumour_df,
        seed=args.seed,
        preassigned_fold_col=args.preassigned_fold_column,
        cancer_col="project_id",
    )
    test_indices = folds[args.fold]
    test_df = tumour_df.iloc[test_indices].reset_index(drop=True)
    logger.info("Fold %d test tumour slides: %d", args.fold, len(test_df))

    test_ds = EmbeddingBagDataset(
        test_df,
        args.h5_dir,
        num_instances=4096,
        cache_dir=args.cache_dir,
    )
    logger.info("Fold %d test bags: %d", args.fold, len(test_ds))

    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    model = EnsoMILModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(args.device)
    model.eval()
    logger.info(
        "Loaded model from %s (epoch=%s, val_loss=%s)",
        args.model_path,
        ckpt.get("epoch", "?"),
        ckpt.get("val_loss", "nan"),
    )

    all_preds: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    y_true: np.ndarray | None = None

    with torch.no_grad():
        for bag_idx in range(args.num_bags):
            loader = torch.utils.data.DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=custom_collate_fn,
                pin_memory=False,
                drop_last=False,
            )
            preds_pass: list[float] = []
            labels_pass: list[float] = []
            probs_pass: list[float] = []
            for feats, labels, _is_tumor in loader:
                feats = feats.to(args.device)
                out = model.forward_outputs(feats) if hasattr(model, "forward_outputs") else None
                if out is not None:
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

    preds_stack = np.stack(all_preds, axis=0)
    preds = np.mean(preds_stack, axis=0)
    pred_std = np.std(preds_stack, axis=0)
    probs = np.mean(np.stack(all_probs, axis=0), axis=0) if all_probs else None

    assert y_true is not None
    mae = float(mean_absolute_error(y_true, preds))
    r2 = float(r2_score(y_true, preds))
    rho = float(spearmanr(y_true, preds).statistic)

    bag_groups = test_ds.groups
    rows = []
    for group, true_value, pred_value, pred_sigma in zip(bag_groups, y_true, preds, pred_std):
        rows.append(
            {
                "fold": args.fold,
                "bag_id": group["bag_id"],
                "label": group["label"],
                "pred": pred_value,
                "pred_std": pred_sigma,
                "is_tumor": group["is_tumor"],
                "file_ids": ";".join(group["file_ids"]),
            }
        )
    rows_df = pd.DataFrame(rows)
    rows_path = args.out_dir / f"fold{args.fold}_test_predictions.csv"
    rows_df.to_csv(rows_path, index=False)

    metrics = {
        "fold": args.fold,
        "test_bags": int(len(test_ds)),
        "num_bags": int(args.num_bags),
        "mae": mae,
        "r2": r2,
        "spearman_rho": rho,
        "mean_bagging_sigma": float(pred_std.mean()),
        "mean_prob_above_threshold": float(probs.mean()) if probs is not None else None,
        "model_path": str(args.model_path),
        "rows_path": str(rows_path),
    }
    metrics_path = args.out_dir / f"fold{args.fold}_test_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    logger.info("Fold %d test MAE=%.4f R2=%.4f rho=%.4f", args.fold, mae, r2, rho)
    logger.info("Wrote predictions: %s", rows_path)
    logger.info("Wrote metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
