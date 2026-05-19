"""CLI for EnsoCellularity tile-level inference."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from enso_cellularity.inference import (
    aggregate_roi_from_predictions,
    load_cellularity_model,
    predict_h5,
    save_prediction_frame,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run EnsoCellularity inference on H5 embeddings")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--h5", type=Path, default=None, help="Single H5 file.")
    ap.add_argument("--h5-dir", type=Path, default=None, help="Directory of H5 files for batch inference.")
    ap.add_argument("--out", type=Path, required=True, help="Output Parquet/CSV for single H5 or dir for batch.")
    ap.add_argument("--batch-size", type=int, default=8192)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--summary-json", type=Path, default=None)
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    model, ckpt = load_cellularity_model(args.checkpoint, device=args.device)
    logger.info(
        "Loaded checkpoint %s (epoch=%s)",
        args.checkpoint,
        ckpt.get("epoch", "unknown"),
    )

    if args.h5 is None and args.h5_dir is None:
        raise SystemExit("Provide --h5 or --h5-dir.")
    if args.h5 is not None and args.h5_dir is not None:
        raise SystemExit("Use only one of --h5 or --h5-dir.")

    summaries: dict[str, dict[str, float]] = {}
    if args.h5 is not None:
        pred = predict_h5(model, args.h5, device=args.device, batch_size=args.batch_size)
        save_prediction_frame(pred, args.out)
        summaries[args.h5.stem] = aggregate_roi_from_predictions(pred)
        logger.info("Wrote %d tile predictions to %s", len(pred), args.out)
    else:
        args.out.mkdir(parents=True, exist_ok=True)
        h5_paths = sorted(args.h5_dir.glob("*.h5"))
        if not h5_paths:
            raise FileNotFoundError(f"No H5 files found in {args.h5_dir}")
        for i, h5_path in enumerate(h5_paths, start=1):
            logger.info("[%d/%d] Predicting %s", i, len(h5_paths), h5_path.name)
            pred = predict_h5(model, h5_path, device=args.device, batch_size=args.batch_size)
            out_path = args.out / f"{h5_path.stem}.parquet"
            save_prediction_frame(pred, out_path)
            summaries[h5_path.stem] = aggregate_roi_from_predictions(pred)

    if args.summary_json is not None:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(json.dumps(summaries, indent=2) + "\n")
        logger.info("Wrote summary JSON: %s", args.summary_json)


if __name__ == "__main__":
    main()

