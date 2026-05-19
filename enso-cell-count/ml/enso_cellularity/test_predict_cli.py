"""Write held-out EnsoCellularity test predictions for one fold.

The output mirrors the purity all-fold predictions table: one row per held-out
prediction unit, with model/fold identifiers, slide metadata, true target, and
prediction. For cellularity the prediction unit is a slide and the scalar target
is mean nuclei count per labeled tile; total counts and tile-level error
summaries are included as additional columns.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data

from enso_cellularity.dataset import (
    CellularitySlideTileDataset,
    cellularity_collate,
)
from enso_cellularity.inference import load_cellularity_model
from enso_cellularity.metrics import ordinal_count_bins
from enso_cellularity.training import move_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, required=True, help="Fold test_slides.csv from training.")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--model-name", default="cellularity_ssd_fullval_q0_mae_es5")
    ap.add_argument("--out-csv", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tile-chunk-size", type=int, default=8192)
    ap.add_argument("--slide-batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--log-every", type=int, default=50)
    return ap.parse_args()


def _new_stats() -> dict[str, float]:
    return defaultdict(float)


def _safe_float(value: float) -> float | None:
    if math.isfinite(float(value)):
        return float(value)
    return None


def _r2_from_sums(n: float, sum_y: float, sum_y2: float, sse: float) -> float | None:
    if n <= 1:
        return None
    sst = float(sum_y2) - (float(sum_y) * float(sum_y) / float(n))
    if sst <= 0:
        return None
    return _safe_float(1.0 - float(sse) / sst)


def _metadata_by_file(slides: pd.DataFrame) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in slides.itertuples(index=False):
        file_id = str(getattr(row, "file_uuid_original"))
        out[file_id] = {
            "barcode": str(getattr(row, "barcode", "")),
            "case_id": str(getattr(row, "case_id", "")),
            "project_id": str(getattr(row, "project_id", "")),
            "num_tiles_index": int(getattr(row, "num_tiles", 0)),
            "label_path": str(getattr(row, "label_path", "")),
            "h5_path": str(getattr(row, "h5_path", "")),
        }
    return out


def _row_from_stats(
    *,
    model_name: str,
    fold: int,
    file_id: str,
    meta: dict[str, Any],
    stats: dict[str, float],
    checkpoint: Path,
    ckpt: dict[str, Any],
) -> dict[str, Any]:
    n = float(stats["n_tiles"])
    true_sum = float(stats["true_sum"])
    pred_sum = float(stats["pred_sum"])
    abs_err_sum = float(stats["abs_err_sum"])
    log_abs_err_sum = float(stats["log_abs_err_sum"])
    sse = float(stats["sse"])
    pred_var_total = float(stats["pred_var_total"])

    true_mean = true_sum / n
    pred_mean = pred_sum / n
    total_abs_error = abs(pred_sum - true_sum)
    tile_mae = abs_err_sum / n
    tile_mae_log1p = log_abs_err_sum / n
    q05_mean = float(stats["q05_sum"]) / n
    q50_mean = float(stats["q50_sum"]) / n
    q95_mean = float(stats["q95_sum"]) / n
    pred_alpha_mean = float(stats["alpha_sum"]) / n
    pred_density_mean = float(stats["density_sum"]) / n
    bin_acc = float(stats["bin_correct"]) / n if stats["bin_total"] else None
    tile_r2 = _r2_from_sums(n, true_sum, float(stats["true_sq_sum"]), sse)

    metrics = ckpt.get("metrics", {})
    return {
        "model": model_name,
        "fold": fold,
        "bag_id": f"slide_{file_id}",
        "slide_barcode": meta.get("barcode", ""),
        "case_id": meta.get("case_id", ""),
        "project_id": meta.get("project_id", ""),
        "file_ids": file_id,
        "true_cell_count": true_mean,
        "pred_cell_count": pred_mean,
        "true_mean_tile_cell_count": true_mean,
        "pred_mean_tile_cell_count": pred_mean,
        "true_total_cell_count": true_sum,
        "pred_total_cell_count": pred_sum,
        "total_abs_error": total_abs_error,
        "tile_mae_count": tile_mae,
        "mean_tile_mae_log1p": tile_mae_log1p,
        "tile_r2_count": tile_r2,
        "n_tiles": int(n),
        "num_tiles_index": int(meta.get("num_tiles_index", 0)),
        "pred_total_sd_nb": math.sqrt(max(pred_var_total, 0.0)),
        "pred_total_q05_normal": max(0.0, pred_sum - 1.645 * math.sqrt(max(pred_var_total, 0.0))),
        "pred_total_q95_normal": max(0.0, pred_sum + 1.645 * math.sqrt(max(pred_var_total, 0.0))),
        "pred_q05_mean": q05_mean,
        "pred_q50_mean": q50_mean,
        "pred_q95_mean": q95_mean,
        "pred_alpha_mean": pred_alpha_mean,
        "pred_density_per_mm2_mean": pred_density_mean,
        "bin_accuracy": bin_acc,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection_metric": metrics.get("selection_metric"),
        "checkpoint_selection_value": metrics.get("selection_value"),
        "checkpoint_path": str(checkpoint),
    }


def main() -> None:
    args = _parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json = args.out_json or args.out_csv.with_suffix(".metrics.json")

    slides = pd.read_csv(args.split_csv)
    meta = _metadata_by_file(slides)
    logger.info("Loaded %d test slides from %s", len(slides), args.split_csv)

    model, ckpt = load_cellularity_model(args.checkpoint, device=args.device)
    logger.info(
        "Loaded checkpoint %s epoch=%s selection=%s %.4f",
        args.checkpoint,
        ckpt.get("epoch"),
        ckpt.get("metrics", {}).get("selection_metric"),
        float(ckpt.get("metrics", {}).get("selection_value", float("nan"))),
    )

    dataset = CellularitySlideTileDataset(
        slides,
        tiles_per_slide=0,
        all_tiles_chunk_size=args.tile_chunk_size,
        sample_strategy="uniform",
        training=False,
        seed=123,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.slide_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cellularity_collate,
        pin_memory=str(args.device).startswith("cuda"),
    )
    logger.info("Evaluating %d chunks with tile_chunk_size=%d", len(loader), args.tile_chunk_size)

    by_file: dict[str, dict[str, float]] = defaultdict(_new_stats)
    overall = _new_stats()
    model.eval()
    with torch.no_grad():
        for batch_idx, raw_batch in enumerate(loader, start=1):
            batch = move_batch(raw_batch, args.device)
            outputs = model.forward_outputs(
                batch["x9"],
                batch["valid9"],
                batch["metadata"],
                batch["exposure_mm2"],
            )

            y = batch["y_count"].detach().float().reshape(-1).cpu().numpy().astype(np.float64)
            mu = outputs["mu"].detach().float().reshape(-1).cpu().numpy().astype(np.float64)
            alpha = outputs["alpha"].detach().float().reshape(-1).cpu().numpy().astype(np.float64)
            density = (
                outputs["density_per_mm2"].detach().float().reshape(-1).cpu().numpy().astype(np.float64)
            )
            quantiles = outputs["quantiles"].detach().float().cpu().numpy().astype(np.float64)
            pred_bin = ordinal_count_bins(outputs["ordinal_logits"]).detach().cpu().numpy()
            true_bin = batch["count_bin"].detach().reshape(-1).cpu().numpy()
            file_ids = np.asarray(raw_batch["file_id"], dtype=object)

            abs_err = np.abs(mu - y)
            log_abs_err = np.abs(np.log1p(np.clip(mu, 0.0, None)) - np.log1p(np.clip(y, 0.0, None)))
            sq_err = (mu - y) ** 2
            pred_var = mu + alpha * (mu**2)

            def update_stats(stats: dict[str, float], mask: np.ndarray) -> None:
                yy = y[mask]
                mm = mu[mask]
                stats["n_tiles"] += float(mask.sum())
                stats["true_sum"] += float(yy.sum())
                stats["pred_sum"] += float(mm.sum())
                stats["true_sq_sum"] += float((yy**2).sum())
                stats["abs_err_sum"] += float(abs_err[mask].sum())
                stats["log_abs_err_sum"] += float(log_abs_err[mask].sum())
                stats["sse"] += float(sq_err[mask].sum())
                stats["pred_var_total"] += float(pred_var[mask].sum())
                stats["q05_sum"] += float(quantiles[mask, 0].sum())
                stats["q50_sum"] += float(quantiles[mask, 1].sum())
                stats["q95_sum"] += float(quantiles[mask, 2].sum())
                stats["alpha_sum"] += float(alpha[mask].sum())
                stats["density_sum"] += float(density[mask].sum())
                stats["bin_correct"] += float((pred_bin[mask] == true_bin[mask]).sum())
                stats["bin_total"] += float(mask.sum())

            update_stats(overall, np.ones(len(y), dtype=bool))
            for file_id in np.unique(file_ids):
                update_stats(by_file[str(file_id)], file_ids == file_id)

            if args.log_every > 0 and batch_idx % args.log_every == 0:
                logger.info(
                    "Evaluated chunk %d/%d tiles=%d slide_rows=%d",
                    batch_idx,
                    len(loader),
                    int(overall["n_tiles"]),
                    len(by_file),
                )

    rows = [
        _row_from_stats(
            model_name=args.model_name,
            fold=args.fold,
            file_id=file_id,
            meta=meta.get(file_id, {}),
            stats=stats,
            checkpoint=args.checkpoint,
            ckpt=ckpt,
        )
        for file_id, stats in sorted(by_file.items(), key=lambda kv: (meta.get(kv[0], {}).get("case_id", ""), kv[0]))
    ]
    rows_df = pd.DataFrame(rows)
    rows_df.to_csv(args.out_csv, index=False)

    slide_errors = rows_df["pred_cell_count"] - rows_df["true_cell_count"]
    slide_true = rows_df["true_cell_count"]
    slide_sse = float((slide_errors**2).sum())
    slide_sst = float(((slide_true - slide_true.mean()) ** 2).sum()) if len(rows_df) else 0.0
    metrics = {
        "model": args.model_name,
        "fold": args.fold,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_selection_metric": ckpt.get("metrics", {}).get("selection_metric"),
        "checkpoint_selection_value": ckpt.get("metrics", {}).get("selection_value"),
        "slides": int(len(rows_df)),
        "tiles": int(overall["n_tiles"]),
        "tile_mae_count": float(overall["abs_err_sum"] / overall["n_tiles"]),
        "tile_mae_log1p": float(overall["log_abs_err_sum"] / overall["n_tiles"]),
        "tile_r2_count": _r2_from_sums(
            overall["n_tiles"],
            overall["true_sum"],
            overall["true_sq_sum"],
            overall["sse"],
        ),
        "mean_true_count": float(overall["true_sum"] / overall["n_tiles"]),
        "mean_pred_count": float(overall["pred_sum"] / overall["n_tiles"]),
        "slide_mean_abs_error": float(slide_errors.abs().mean()) if len(rows_df) else None,
        "slide_r2_mean_count": _safe_float(1.0 - slide_sse / slide_sst) if slide_sst > 0 else None,
        "out_csv": str(args.out_csv),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote %d slide predictions to %s", len(rows_df), args.out_csv)
    logger.info("Wrote metrics to %s", out_json)
    logger.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
