"""CLI entry point: ``python -m enso_purity_mil.train_cli``.

5-fold CV training with early stopping, ReduceLROnPlateau scheduler,
and regularized EnsoPurity-v2 objectives.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from enso_purity_mil.dataset import EmbeddingBagDataset, custom_collate_fn
from enso_purity_mil.folds import generate_stratified_folds
from enso_purity_mil.manifest_io import load_manifest_table
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig
from enso_purity_mil.training import EarlyStopping, run_one_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _build_tumour_folds(
    tumour_df: pd.DataFrame,
    *,
    seed: int,
    preassigned_fold_col: str | None = None,
    cancer_col: str = "project_id",
) -> list[list[int]]:
    """Return tumour row indices partitioned into 5 folds."""
    if preassigned_fold_col is None:
        return generate_stratified_folds(
            tumour_df,
            n_folds=5,
            seed=seed,
            cancer_col=cancer_col,
        )

    if preassigned_fold_col not in tumour_df.columns:
        raise ValueError(f"Missing preassigned fold column: {preassigned_fold_col}")

    fold_values = tumour_df[preassigned_fold_col]
    if fold_values.isna().any():
        missing = tumour_df.loc[fold_values.isna(), ["case_id", "aliquot_barcode"]].head(10)
        raise ValueError(
            f"Found tumour rows with missing preassigned folds in {preassigned_fold_col}. "
            f"Examples:\n{missing.to_string(index=False)}"
        )

    fold_values = fold_values.astype(int)
    unexpected = sorted(set(fold_values.tolist()) - {0, 1, 2, 3, 4})
    if unexpected:
        raise ValueError(
            f"Unexpected values in {preassigned_fold_col}: {unexpected}. Expected folds 0-4."
        )

    case_fold_counts = tumour_df.assign(_fold=fold_values).groupby("case_id")["_fold"].nunique()
    leaking_cases = case_fold_counts[case_fold_counts > 1]
    if not leaking_cases.empty:
        examples = leaking_cases.index.tolist()[:10]
        raise ValueError(
            "Patient leakage across preassigned folds detected for tumour rows. "
            f"Example cases: {examples}"
        )

    folds: list[list[int]] = []
    for fold_id in range(5):
        indices = tumour_df.index[fold_values == fold_id].tolist()
        folds.append(sorted(indices))
    return folds


def build_adamw_with_decay_exclusions(
    model: nn.Module,
    *,
    lr: float,
    weight_decay: float,
) -> optim.AdamW:
    """Apply weight decay only to true weights (exclude norms + biases)."""
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias") or "norm" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = []
    if decay_params:
        groups.append({"params": decay_params, "weight_decay": weight_decay})
    if no_decay_params:
        groups.append({"params": no_decay_params, "weight_decay": 0.0})

    logger.info(
        "AdamW param groups: decay=%d, no_decay=%d, weight_decay=%.2e",
        len(decay_params), len(no_decay_params), weight_decay
    )
    return optim.AdamW(groups, lr=lr)


def main() -> None:
    ap = argparse.ArgumentParser(description="Train Enso MIL purity regressor (5-fold CV)")
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/wedge_mvp_dataset.xlsx"))
    ap.add_argument("--h5-dir", type=Path, required=True,
                    help="Directory containing <file_uuid>.h5 embedding files")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="Local bag cache dir (from build_cache.py). New pool cache re-samples each epoch.")
    ap.add_argument("--out-dir", type=Path, default=Path("ml/runs"))
    ap.add_argument("--num-instances", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=3e-3)
    ap.add_argument("--max-epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--num-workers", type=int, default=14)
    ap.add_argument("--effective-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--include-train-normals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include matched normal_tissue samples in training split.",
    )
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--fold", type=int, default=None,
                    help="Train only this fold (0-4). If None, train all 5.")
    ap.add_argument(
        "--preassigned-fold-column",
        type=str,
        default=None,
        help="Optional tumour manifest column with preassigned fold IDs (0-4).",
    )
    ap.add_argument("--resume", type=Path, default=None,
                    help="Path to a .pth checkpoint to resume from.")
    ap.add_argument("--prototype-k", type=int, default=64)
    ap.add_argument("--prototype-temp", type=float, default=0.10)
    ap.add_argument("--adapter-dropout", type=float, default=0.20)
    ap.add_argument("--head-dropout", type=float, default=0.50)
    ap.add_argument("--instance-dropout", type=float, default=0.20)
    ap.add_argument("--feature-noise-std", type=float, default=0.01)
    ap.add_argument(
        "--sigmas",
        type=float,
        nargs="+",
        default=[0.02, 0.05, 0.10],
        help="Multi-scale KDE sigma values.",
    )
    ap.add_argument("--aux-bce-weight", type=float, default=0.05)
    ap.add_argument("--beta-nll-weight", type=float, default=0.0)
    ap.add_argument("--proto-entropy-weight", type=float, default=0.001)
    ap.add_argument("--consistency-weight", type=float, default=0.05)
    ap.add_argument(
        "--val-use-all-tiles",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use all tiles per validation bag (no tile sub-sampling).",
    )
    ap.add_argument(
        "--deterministic-val",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use deterministic per-bag sampling for validation when sub-sampling is enabled.",
    )
    ap.add_argument(
        "--deterministic-val-seed",
        type=int,
        default=None,
        help="Seed for deterministic validation sampling (defaults to --seed).",
    )
    ap.add_argument(
        "--max-train-log-lines",
        type=int,
        default=8,
        help="Maximum number of batch-progress log lines per training epoch.",
    )
    ap.add_argument(
        "--max-val-log-lines",
        type=int,
        default=6,
        help="Maximum number of batch-progress log lines per validation epoch.",
    )
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load manifest ────────────────────────────────────────────
    logger.info("Loading manifest: %s", args.manifest)
    manifest = load_manifest_table(args.manifest)
    manifest = manifest[manifest["purity"].notna()].copy()
    logger.info("Slides with purity: %d", len(manifest))

    # ── Separate tumours and normals ─────────────────────────────
    tumour_df = manifest[manifest["gdc_match_type"] != "normal_tissue"].copy().reset_index(drop=True)
    normal_df = manifest[manifest["gdc_match_type"] == "normal_tissue"].copy().reset_index(drop=True)
    logger.info("Tumour slides: %d, Normal slides: %d", len(tumour_df), len(normal_df))

    # ── Generate folds on tumour samples only ────────────────────
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
    fold_range = [args.fold] if args.fold is not None else range(5)

    for fold_id in fold_range:
        logger.info("=" * 60)
        logger.info("FOLD %d", fold_id)
        logger.info("=" * 60)

        test_indices = folds[fold_id]
        val_indices = folds[(fold_id + 1) % 5]
        train_indices = []
        for f in range(5):
            if f != fold_id and f != (fold_id + 1) % 5:
                train_indices.extend(folds[f])

        train_df = tumour_df.iloc[train_indices].reset_index(drop=True)
        val_df = tumour_df.iloc[val_indices].reset_index(drop=True)
        test_df = tumour_df.iloc[test_indices].reset_index(drop=True)

        train_tumour_count = len(train_df)
        if args.include_train_normals:
            train_cases = set(train_df["case_id"])
            train_normals = normal_df[normal_df["case_id"].isin(train_cases)]
            train_df = pd.concat([train_df, train_normals], ignore_index=True)
        else:
            train_normals = normal_df.iloc[0:0].copy()

        logger.info(
            "Train: %d (+ %d normals, include_train_normals=%s), Val: %d (0 normals), Test: %d",
            train_tumour_count,
            len(train_normals),
            args.include_train_normals,
            len(val_df),
            len(test_df),
        )

        train_ds = EmbeddingBagDataset(
            train_df, args.h5_dir, num_instances=args.num_instances,
            cache_dir=args.cache_dir,
        )
        val_ds = EmbeddingBagDataset(
            val_df, args.h5_dir, num_instances=args.num_instances,
            cache_dir=args.cache_dir,
            use_all_tiles=args.val_use_all_tiles,
            deterministic=args.deterministic_val,
            deterministic_seed=(
                args.seed if args.deterministic_val_seed is None else args.deterministic_val_seed
            ),
        )

        val_batch_size = 1 if args.val_use_all_tiles else args.batch_size
        logger.info(
            "Train bags: %d, Val bags: %d (cache_dir=%s, val_use_all_tiles=%s, val_batch_size=%d)",
            len(train_ds), len(val_ds), args.cache_dir, args.val_use_all_tiles, val_batch_size
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=custom_collate_fn,
            pin_memory=False, drop_last=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_ds, batch_size=val_batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=custom_collate_fn,
            pin_memory=False,
        )

        cfg = EnsoModelConfig(
            input_dim=2560,
            adapter_hidden_dim=256,
            num_features=128,
            num_bins=21,
            sigmas=tuple(args.sigmas),
            prototype_k=args.prototype_k,
            prototype_temp=args.prototype_temp,
            use_moments=True,
            adapter_dropout=args.adapter_dropout,
            head_dropout=args.head_dropout,
            instance_dropout=args.instance_dropout,
            feature_noise_std=args.feature_noise_std,
            tau_init=1.0,
            learnable_tau=True,
        )
        model = EnsoMILModel(cfg).to(args.device)
        optimizer = build_adamw_with_decay_exclusions(
            model, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        criterion = nn.L1Loss()
        early_stop = EarlyStopping(patience=args.patience)
        grad_accum_steps = max(1, (args.effective_batch_size + args.batch_size - 1) // args.batch_size)
        effective_batch_size = args.batch_size * grad_accum_steps
        logger.info(
            "Gradient accumulation: micro_batch=%d, grad_accum_steps=%d, effective_batch=%d",
            args.batch_size, grad_accum_steps, effective_batch_size
        )
        logger.info(
            "Logging caps per epoch: train<=%d lines, val<=%d lines",
            args.max_train_log_lines, args.max_val_log_lines
        )

        start_epoch = 1
        best_val_loss = float("inf")
        history: list[dict] = []

        # ── Resume from checkpoint ───────────────────────────────
        if args.resume and args.resume.exists():
            ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except Exception as exc:
                logger.warning("Could not load optimizer state from checkpoint: %s", exc)
            if "scheduler_state_dict" in ckpt:
                try:
                    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                except Exception as exc:
                    logger.warning("Could not load scheduler state from checkpoint: %s", exc)
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            logger.info("Resumed from %s (epoch %d, val_loss=%.4f)",
                        args.resume, start_epoch - 1, best_val_loss)

        fold_dir = args.out_dir / f"fold{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(start_epoch, args.max_epochs + 1):
            t0 = time.time()
            train_loss = run_one_epoch(
                model, train_loader, criterion, optimizer,
                device=args.device, train=True,
                max_log_lines=args.max_train_log_lines,
                aux_bce_weight=args.aux_bce_weight,
                beta_nll_weight=args.beta_nll_weight,
                proto_entropy_weight=args.proto_entropy_weight,
                consistency_weight=args.consistency_weight,
                grad_accum_steps=grad_accum_steps,
            )
            val_result = run_one_epoch(model, val_loader, criterion, optimizer=None,
                                       device=args.device, train=False,
                                       max_log_lines=args.max_val_log_lines,
                                       aux_bce_weight=args.aux_bce_weight,
                                       beta_nll_weight=args.beta_nll_weight,
                                       proto_entropy_weight=args.proto_entropy_weight,
                                       consistency_weight=0.0)
            val_loss = val_result["loss"]
            r2 = val_result["r2"]
            spearman = val_result["spearman"]
            scheduler.step(val_loss)
            elapsed = time.time() - t0
            lr_now = optimizer.param_groups[0]["lr"]

            logger.info("Epoch %3d  train=%.4f  val=%.4f  R²=%.4f  ρ=%.4f  lr=%.2e  (%.1fs)",
                        epoch, train_loss, val_loss, r2, spearman, lr_now, elapsed)

            history.append({"epoch": epoch, "train_loss": train_loss,
                            "val_loss": val_loss, "r2": r2, "spearman": spearman,
                            "lr": lr_now})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "r2": r2,
                    "spearman": spearman,
                    "config": cfg.__dict__,
                }, fold_dir / "best_model.pth")
                logger.info("  → saved best model (val_loss=%.4f)", val_loss)

            # Also save latest checkpoint for resume
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "config": cfg.__dict__,
            }, fold_dir / "latest_checkpoint.pth")

            if early_stop(val_loss):
                logger.info("Early stopping at epoch %d (best val_loss=%.4f)", epoch, best_val_loss)
                break

        hist_path = fold_dir / "history.json"
        hist_path.write_text(json.dumps(history, indent=2))
        logger.info("Fold %d done. Best val_loss=%.4f. History: %s", fold_id, best_val_loss, hist_path)


if __name__ == "__main__":
    main()
