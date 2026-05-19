"""CLI entry point for EnsoCellularity training.

Example:

```bash
python -u -m enso_cellularity.train_cli \
  --label-dir /data/pancancer_nuclei_seg_dicom/tile_cellularity_labels_direct/by_slide \
  --h5-dir /data/embeddings_fp32 \
  --out-dir ml/runs_cellularity \
  --device cuda \
  --fold 0
```
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from torch.optim.lr_scheduler import ReduceLROnPlateau

from enso_cellularity.dataset import (
    CellularityBlobSlideTileDataset,
    CellularitySlideTileDataset,
    build_slide_index,
    build_slide_index_from_completed_tsv,
    cellularity_collate,
    load_slide_index,
    write_slide_index,
)
from enso_cellularity.folds import assign_case_folds, apply_preassigned_folds, split_for_fold
from enso_cellularity.losses import CellularityLossWeights, EnsoCellularityCompositeLoss
from enso_cellularity.model import EnsoCellularityConfig, EnsoCellularityModel
from enso_cellularity.training import (
    EarlyStopping,
    TrainConfig,
    build_adamw_with_decay_exclusions,
    run_one_epoch,
    save_checkpoint,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("urllib3.connectionpool").setLevel(logging.ERROR)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train EnsoCellularity tile-count model")
    ap.add_argument("--data-backend", choices=["local", "blob"], default="local")
    ap.add_argument("--label-dir", type=Path, default=None, help="Directory of per-slide label Parquets.")
    ap.add_argument("--h5-dir", type=Path, default=None, help="Directory containing {file_id}.h5 embeddings.")
    ap.add_argument("--completed-tsv", type=Path, default=None, help="Direct-Blob completed.tsv from label processing.")
    ap.add_argument("--blob-base-url", default="https://vmshareddisk.blob.core.windows.net/data")
    ap.add_argument("--blob-h5-prefix", default="embeddings_fp32")
    ap.add_argument("--blob-scratch-dir", type=Path, default=Path("scratch/cellularity_training_blob"))
    ap.add_argument("--blob-keep-cache", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--blob-transfer-mode", choices=["azcopy", "sdk"], default="azcopy")
    ap.add_argument("--blob-sdk-max-concurrency", type=int, default=12)
    ap.add_argument("--azcopy-bin", default="azcopy")
    ap.add_argument("--azcopy-auto-login-type", default="MSI")
    ap.add_argument("--slide-index", type=Path, default=None, help="Existing or output slide index CSV/Parquet.")
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--out-dir", type=Path, default=Path("ml/runs_cellularity"))
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--val-fold", type=int, default=None)
    ap.add_argument(
        "--preassigned-fold-file",
        type=Path,
        default=None,
        help="Purity prediction/fold CSV with case_id, fold, and optional file_ids.",
    )
    ap.add_argument(
        "--unassigned-fold-policy",
        choices=["drop", "assign", "error"],
        default="drop",
        help="How to handle cellularity cases absent from --preassigned-fold-file.",
    )
    ap.add_argument("--max-slides", type=int, default=None, help="Debug/smoke limit before fold split.")

    ap.add_argument("--tiles-per-slide", type=int, default=8192)
    ap.add_argument(
        "--eval-tiles-per-slide",
        type=int,
        default=0,
        help="Validation tiles per slide; <=0 evaluates every labeled tile.",
    )
    ap.add_argument(
        "--eval-tile-chunk-size",
        type=int,
        default=8192,
        help="Chunk size used when --eval-tiles-per-slide <= 0, so full validation does not materialize huge slides.",
    )
    ap.add_argument("--sample-strategy", choices=["uniform", "balanced_bins"], default="balanced_bins")
    ap.add_argument("--slide-batch-size", type=int, default=1)
    ap.add_argument("--eval-slide-batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--prefetch-factor", type=int, default=1)
    ap.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--label-cache-size", type=int, default=2)

    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=3e-3)
    ap.add_argument("--max-epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--min-delta", type=float, default=0.0)
    ap.add_argument(
        "--early-stop-metric",
        choices=["val_loss", "val_mae_count", "val_mae_log1p"],
        default="val_mae_count",
        help="Validation metric used by patience-based early stopping.",
    )
    ap.add_argument("--grad-accum-steps", type=int, default=1)
    ap.add_argument("--grad-clip-norm", type=float, default=1.0)
    ap.add_argument("--scheduler-patience", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--resume", type=Path, default=None)
    ap.add_argument(
        "--reset-optimizer-on-resume",
        action="store_true",
        help="Load model weights from --resume but rebuild optimizer/scheduler with current CLI settings.",
    )
    ap.add_argument(
        "--truncate-history-on-resume",
        action="store_true",
        help="When resuming from epoch N, keep only history records with epoch <= N before appending.",
    )
    ap.add_argument("--log-every", type=int, default=10)

    ap.add_argument("--input-dim", type=int, default=2560)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--attention-heads", type=int, default=4)
    ap.add_argument("--trunk-depth", type=int, default=3)
    ap.add_argument("--trunk-hidden-dim", type=int, default=1024)
    ap.add_argument("--embed-dropout", type=float, default=0.05)
    ap.add_argument("--trunk-dropout", type=float, default=0.10)
    ap.add_argument("--num-quality-classes", type=int, default=3)

    ap.add_argument("--nb-weight", type=float, default=0.30)
    ap.add_argument("--smooth-l1-weight", type=float, default=1.00)
    ap.add_argument("--ordinal-weight", type=float, default=0.05)
    ap.add_argument("--quantile-weight", type=float, default=0.02)
    ap.add_argument("--quality-weight", type=float, default=0.0)
    return ap.parse_args()


def _best_history_metric(history: list[dict], key: str, default: float = float("inf")) -> float:
    values: list[float] = []
    for record in history:
        if key not in record:
            continue
        value = float(record[key])
        if np.isfinite(value):
            values.append(value)
    return min(values, default=default)


def _load_or_build_index(args: argparse.Namespace):
    if args.slide_index is None:
        args.slide_index = args.out_dir / "slide_index.csv"
    if args.slide_index.exists() and not args.rebuild_index:
        logger.info("Loading slide index: %s", args.slide_index)
        index = load_slide_index(args.slide_index)
    else:
        if args.data_backend == "blob":
            if args.completed_tsv is None:
                raise ValueError("--completed-tsv is required when --data-backend blob and no index exists.")
            logger.info("Building direct-Blob slide index from %s", args.completed_tsv)
            index = build_slide_index_from_completed_tsv(
                args.completed_tsv,
                base_url=args.blob_base_url,
                h5_prefix=args.blob_h5_prefix,
            )
        else:
            if args.label_dir is None or args.h5_dir is None:
                raise ValueError("--label-dir and --h5-dir are required for --data-backend local.")
            logger.info("Building slide index from %s", args.label_dir)
            index = build_slide_index(args.label_dir, args.h5_dir, max_slides=args.max_slides)
        write_slide_index(index, args.slide_index)
        logger.info("Wrote slide index: %s", args.slide_index)

    if args.max_slides is not None and len(index) > args.max_slides:
        index = index.iloc[: args.max_slides].copy()

    if args.data_backend == "local" and "has_h5" in index.columns:
        has_h5 = index["has_h5"].astype(bool)
        missing_h5 = int((~has_h5).sum())
        index = index[has_h5].reset_index(drop=True)
    else:
        missing_h5 = 0
    logger.info("Usable slides with H5: %d (missing_h5=%d)", len(index), missing_h5)
    if args.preassigned_fold_file is not None:
        index, report = apply_preassigned_folds(
            index,
            str(args.preassigned_fold_file),
            policy=args.unassigned_fold_policy,
            seed=args.seed,
            n_folds=args.n_folds,
        )
        logger.info("Applied preassigned purity folds: %s", report)
        write_slide_index(index, args.slide_index)
        logger.info("Updated index with preassigned folds: %s", args.slide_index)
    elif "fold" not in index.columns or args.rebuild_index:
        index = assign_case_folds(index, n_folds=args.n_folds, seed=args.seed)
        write_slide_index(index, args.slide_index)
        logger.info("Assigned case-level folds and updated index: %s", args.slide_index)
    return index, None


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    slide_index, _ = _load_or_build_index(args)
    split = split_for_fold(
        slide_index,
        fold=args.fold,
        n_folds=args.n_folds,
        val_fold=args.val_fold,
    )
    fold_dir = args.out_dir / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    split.train.to_csv(fold_dir / "train_slides.csv", index=False)
    split.val.to_csv(fold_dir / "val_slides.csv", index=False)
    split.test.to_csv(fold_dir / "test_slides.csv", index=False)
    logger.info(
        "Fold %d split: train_slides=%d val_slides=%d test_slides=%d",
        args.fold,
        len(split.train),
        len(split.val),
        len(split.test),
    )
    if args.eval_tiles_per_slide <= 0:
        logger.info("Validation will evaluate all labeled tiles for each slide.")

    train_cfg = TrainConfig(
        tiles_per_slide=args.tiles_per_slide,
        eval_tiles_per_slide=args.eval_tiles_per_slide,
        eval_tile_chunk_size=args.eval_tile_chunk_size,
        slide_batch_size=args.slide_batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        early_stop_metric=args.early_stop_metric,
        num_workers=args.num_workers,
        grad_accum_steps=args.grad_accum_steps,
        grad_clip_norm=args.grad_clip_norm,
        scheduler_patience=args.scheduler_patience,
        seed=args.seed,
    )
    model_cfg = EnsoCellularityConfig(
        input_dim=args.input_dim,
        d_model=args.d_model,
        attention_heads=args.attention_heads,
        trunk_depth=args.trunk_depth,
        trunk_hidden_dim=args.trunk_hidden_dim,
        embed_dropout=args.embed_dropout,
        trunk_dropout=args.trunk_dropout,
        num_quality_classes=args.num_quality_classes,
    )
    loss_weights = CellularityLossWeights(
        nb_nll=args.nb_weight,
        smooth_l1_log=args.smooth_l1_weight,
        ordinal_bce=args.ordinal_weight,
        quantile_pinball=args.quantile_weight,
        quality=args.quality_weight,
    )

    if args.data_backend == "blob":
        logger.info(
            "Using direct-Blob training backend with scratch_dir=%s keep_cache=%s transfer_mode=%s",
            args.blob_scratch_dir,
            args.blob_keep_cache,
            args.blob_transfer_mode,
        )
        train_ds = CellularityBlobSlideTileDataset(
            split.train,
            scratch_dir=args.blob_scratch_dir / f"fold{args.fold}" / "train",
            tiles_per_slide=args.tiles_per_slide,
            sample_strategy=args.sample_strategy,
            training=True,
            seed=args.seed,
            azcopy_bin=args.azcopy_bin,
            azcopy_auto_login_type=args.azcopy_auto_login_type,
            transfer_mode=args.blob_transfer_mode,
            sdk_max_concurrency=args.blob_sdk_max_concurrency,
            keep_cache=args.blob_keep_cache,
        )
        val_ds = CellularityBlobSlideTileDataset(
            split.val,
            scratch_dir=args.blob_scratch_dir / f"fold{args.fold}" / "val",
            tiles_per_slide=args.eval_tiles_per_slide,
            sample_strategy="uniform",
            training=False,
            seed=args.seed,
            azcopy_bin=args.azcopy_bin,
            azcopy_auto_login_type=args.azcopy_auto_login_type,
            transfer_mode=args.blob_transfer_mode,
            sdk_max_concurrency=args.blob_sdk_max_concurrency,
            keep_cache=args.blob_keep_cache,
        )
    else:
        train_ds = CellularitySlideTileDataset(
            split.train,
            tiles_per_slide=args.tiles_per_slide,
            sample_strategy=args.sample_strategy,
            training=True,
            seed=args.seed,
            label_cache_size=args.label_cache_size,
        )
        val_ds = CellularitySlideTileDataset(
            split.val,
            tiles_per_slide=args.eval_tiles_per_slide,
            all_tiles_chunk_size=args.eval_tile_chunk_size,
            sample_strategy="uniform",
            training=False,
            seed=args.seed,
            label_cache_size=args.label_cache_size,
        )

    loader_kwargs = {}
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = max(1, int(args.prefetch_factor))
        loader_kwargs["persistent_workers"] = bool(args.persistent_workers)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=args.slide_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=cellularity_collate,
        pin_memory=str(args.device).startswith("cuda"),
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=args.eval_slide_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cellularity_collate,
        pin_memory=str(args.device).startswith("cuda"),
        **loader_kwargs,
    )

    model = EnsoCellularityModel(model_cfg).to(args.device)
    optimizer = build_adamw_with_decay_exclusions(
        model,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=max(0, int(args.scheduler_patience)),
        min_lr=1e-6,
    )
    criterion = EnsoCellularityCompositeLoss(weights=loss_weights)
    early_stop = EarlyStopping(patience=args.patience, min_delta=args.min_delta)
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and str(args.device).startswith("cuda")))

    start_epoch = 1
    best_val_loss = float("inf")
    best_val_mae_count = float("inf")
    best_val_mae_log1p = float("inf")
    history: list[dict] = []
    history_path = fold_dir / "history.json"
    if args.resume is not None and args.resume.exists():
        logger.info("Resuming from checkpoint: %s", args.resume)
        ckpt = torch.load(args.resume, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if args.reset_optimizer_on_resume:
            logger.info("Resetting optimizer/scheduler state after loading model weights.")
        elif "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if not args.reset_optimizer_on_resume and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        resume_epoch = int(ckpt.get("epoch", 0))
        start_epoch = resume_epoch + 1
        metrics = ckpt.get("metrics", {})
        best_val_loss = float(metrics.get("val_loss", best_val_loss))
        best_val_mae_count = float(metrics.get("mae_count", best_val_mae_count))
        best_val_mae_log1p = float(metrics.get("mae_log1p", best_val_mae_log1p))
        if history_path.exists():
            history = json.loads(history_path.read_text())
            if args.truncate_history_on_resume:
                original_len = len(history)
                history = [record for record in history if int(record.get("epoch", 0)) <= resume_epoch]
                logger.info(
                    "Truncated history on resume: kept %d/%d records through epoch %d",
                    len(history),
                    original_len,
                    resume_epoch,
                )
            logger.info("Loaded %d existing history records from %s", len(history), history_path)
            best_val_loss = _best_history_metric(history, "val_loss", best_val_loss)
            best_val_mae_count = _best_history_metric(history, "val_mae_count", best_val_mae_count)
            best_val_mae_log1p = _best_history_metric(history, "val_mae_log1p", best_val_mae_log1p)
        early_stop_best = {
            "val_loss": best_val_loss,
            "val_mae_count": best_val_mae_count,
            "val_mae_log1p": best_val_mae_log1p,
        }[args.early_stop_metric]
        if np.isfinite(early_stop_best):
            early_stop.best = early_stop_best

    run_config = {
        "args": vars(args),
        "model_config": model_cfg.__dict__,
        "train_config": train_cfg.__dict__,
        "loss_weights": loss_weights.__dict__,
    }
    (fold_dir / "run_config.json").write_text(json.dumps(run_config, indent=2, default=str) + "\n")

    for epoch in range(start_epoch, args.max_epochs + 1):
        t0 = time.time()
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(0)
        train_metrics = run_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=args.device,
            train=True,
            scaler=scaler,
            grad_accum_steps=args.grad_accum_steps,
            grad_clip_norm=args.grad_clip_norm,
            log_every=args.log_every,
        )
        val_metrics = run_one_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=args.device,
            train=False,
            scaler=None,
            grad_accum_steps=1,
            grad_clip_norm=None,
            log_every=max(1, args.log_every),
        )
        val_loss = float(val_metrics["loss"])
        val_mae_count = float(val_metrics.get("mae_count", float("inf")))
        val_mae_log1p = float(val_metrics.get("mae_log1p", float("inf")))
        early_stop_value = {
            "val_loss": val_loss,
            "val_mae_count": val_mae_count,
            "val_mae_log1p": val_mae_log1p,
        }[args.early_stop_metric]
        scheduler.step(val_loss)
        lr_now = float(optimizer.param_groups[0]["lr"])
        epoch_record = {
            "epoch": epoch,
            "lr": lr_now,
            "elapsed_s": time.time() - t0,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_record)
        history_path.write_text(json.dumps(history, indent=2) + "\n")

        logger.info(
            "Epoch %03d train_loss=%.4f train_mae=%.3f val_loss=%.4f val_mae=%.3f val_mae_log1p=%.4f "
            "val_factor2=%.3f val_bin_acc=%.3f lr=%.2e elapsed=%.1fs",
            epoch,
            train_metrics["loss"],
            train_metrics.get("mae_count", float("nan")),
            val_loss,
            val_mae_count,
            val_mae_log1p,
            val_metrics.get("within_factor2", float("nan")),
            val_metrics.get("bin_accuracy", float("nan")),
            lr_now,
            epoch_record["elapsed_s"],
        )

        checkpoint_metrics = {"train_loss": train_metrics["loss"], "val_loss": val_loss, **val_metrics}
        checkpoint_extra = {
            "loss_weights": loss_weights.__dict__,
            "fold": args.fold,
            "best_metrics": {
                "val_loss": best_val_loss,
                "val_mae_count": best_val_mae_count,
                "val_mae_log1p": best_val_mae_log1p,
            },
        }
        def _save_best(filename: str, selection_metric: str, selection_value: float) -> None:
            save_checkpoint(
                fold_dir / filename,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                model_config=model_cfg,
                train_config=train_cfg,
                metrics={
                    **checkpoint_metrics,
                    "selection_metric": selection_metric,
                    "selection_value": selection_value,
                },
                extra=checkpoint_extra,
            )

        if np.isfinite(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_extra["best_metrics"]["val_loss"] = best_val_loss
            _save_best("best_model.pth", "val_loss", val_loss)
            _save_best("best_by_loss.pth", "val_loss", val_loss)
            logger.info("  saved best by val_loss: %.4f", val_loss)

        if np.isfinite(val_mae_count) and val_mae_count < best_val_mae_count:
            best_val_mae_count = val_mae_count
            checkpoint_extra["best_metrics"]["val_mae_count"] = best_val_mae_count
            _save_best("best_by_mae_count.pth", "val_mae_count", val_mae_count)
            logger.info("  saved best by val_mae_count: %.4f", val_mae_count)

        if np.isfinite(val_mae_log1p) and val_mae_log1p < best_val_mae_log1p:
            best_val_mae_log1p = val_mae_log1p
            checkpoint_extra["best_metrics"]["val_mae_log1p"] = best_val_mae_log1p
            _save_best("best_by_mae_log1p.pth", "val_mae_log1p", val_mae_log1p)
            logger.info("  saved best by val_mae_log1p: %.4f", val_mae_log1p)

        save_checkpoint(
            fold_dir / "latest_checkpoint.pth",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            model_config=model_cfg,
            train_config=train_cfg,
            metrics=checkpoint_metrics,
            extra=checkpoint_extra,
        )

        if early_stop(early_stop_value):
            logger.info(
                "Early stopping at epoch %d; best_%s=%.4f",
                epoch,
                args.early_stop_metric,
                early_stop.best if early_stop.best is not None else early_stop_value,
            )
            break

    logger.info(
        "Training complete. Best val_loss=%.4f best_val_mae_count=%.4f "
        "best_val_mae_log1p=%.4f fold_dir=%s",
        best_val_loss,
        best_val_mae_count,
        best_val_mae_log1p,
        fold_dir,
    )


if __name__ == "__main__":
    main()
