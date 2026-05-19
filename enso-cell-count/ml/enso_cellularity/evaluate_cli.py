"""Evaluate an EnsoCellularity checkpoint on a slide split."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import torch
import torch.utils.data

from enso_cellularity.dataset import (
    CellularityBlobSlideTileDataset,
    CellularitySlideTileDataset,
    build_slide_index_from_completed_tsv,
    cellularity_collate,
    load_slide_index,
)
from enso_cellularity.inference import load_cellularity_model
from enso_cellularity.losses import CellularityLossWeights, EnsoCellularityCompositeLoss
from enso_cellularity.training import run_one_epoch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--split-csv", type=Path, default=None, help="train/val/test slide CSV from train_cli.")
    ap.add_argument("--slide-index", type=Path, default=None)
    ap.add_argument("--completed-tsv", type=Path, default=None)
    ap.add_argument("--data-backend", choices=["local", "blob"], default="local")
    ap.add_argument("--blob-base-url", default="https://vmshareddisk.blob.core.windows.net/data")
    ap.add_argument("--blob-h5-prefix", default="embeddings_fp32")
    ap.add_argument("--blob-scratch-dir", type=Path, default=Path("scratch/cellularity_eval_blob"))
    ap.add_argument("--blob-keep-cache", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--azcopy-bin", default="azcopy")
    ap.add_argument("--azcopy-auto-login-type", default="MSI")
    ap.add_argument(
        "--tiles-per-slide",
        type=int,
        default=0,
        help="Tiles per slide to evaluate; <=0 evaluates every labeled tile.",
    )
    ap.add_argument(
        "--tile-chunk-size",
        type=int,
        default=8192,
        help="Chunk size used when --tiles-per-slide <= 0.",
    )
    ap.add_argument("--slide-batch-size", type=int, default=1)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument(
        "--quality-weight",
        type=float,
        default=0.0,
        help="Quality loss weight used when reporting eval loss.",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out-json", type=Path, required=True)
    return ap.parse_args()


def _load_split(args: argparse.Namespace) -> pd.DataFrame:
    if args.split_csv is not None:
        return pd.read_csv(args.split_csv)
    if args.slide_index is not None:
        return load_slide_index(args.slide_index)
    if args.completed_tsv is not None:
        return build_slide_index_from_completed_tsv(
            args.completed_tsv,
            base_url=args.blob_base_url,
            h5_prefix=args.blob_h5_prefix,
        )
    raise ValueError("Provide --split-csv, --slide-index, or --completed-tsv.")


def main() -> None:
    args = _parse_args()
    model, ckpt = load_cellularity_model(args.checkpoint, device=args.device)
    slides = _load_split(args)
    logger.info("Loaded checkpoint epoch=%s; evaluating %d slides", ckpt.get("epoch"), len(slides))

    if args.data_backend == "blob":
        ds = CellularityBlobSlideTileDataset(
            slides,
            scratch_dir=args.blob_scratch_dir,
            tiles_per_slide=args.tiles_per_slide,
            sample_strategy="uniform",
            training=False,
            seed=123,
            azcopy_bin=args.azcopy_bin,
            azcopy_auto_login_type=args.azcopy_auto_login_type,
            keep_cache=args.blob_keep_cache,
        )
    else:
        ds = CellularitySlideTileDataset(
            slides,
            tiles_per_slide=args.tiles_per_slide,
            all_tiles_chunk_size=args.tile_chunk_size,
            sample_strategy="uniform",
            training=False,
            seed=123,
        )
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=args.slide_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=cellularity_collate,
        pin_memory=str(args.device).startswith("cuda"),
    )
    weight_kwargs = dict(ckpt.get("loss_weights", {}))
    weight_kwargs["quality"] = args.quality_weight
    weights = CellularityLossWeights(**weight_kwargs)
    metrics = run_one_epoch(
        model,
        loader,
        EnsoCellularityCompositeLoss(weights=weights),
        optimizer=None,
        device=args.device,
        train=False,
        scaler=None,
        log_every=10,
    )
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info("Metrics: %s", metrics)
    logger.info("Wrote %s", args.out_json)


if __name__ == "__main__":
    main()
