"""Export static cell-count heatmap masks for the demo case viewer.

This mirrors ``enso_purity_mil.export_static_cases`` but uses the
EnsoCellularity checkpoint and writes count overlays for the same gallery rows.
The frontend already has ``case_N_base.jpg`` files, so this command can align
new cell-count masks to those base images without re-fetching thumbnails.

Example:

    python -m enso_cellularity.export_static_cases \
      --checkpoint runs_cellularity_ssd/fold1/best_model.pth \
      --h5-dir /mnt/dataset/embeddings_fp32 \
      --gallery-csv frontend/gallery/gallery_summary.csv \
      --base-dir frontend/public/cases \
      --out-dir frontend/public/cases
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from enso_cellularity.inference import load_cellularity_model, predict_h5

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _get_colormap(name: str) -> mcolors.Colormap:
    if name == "purity":
        return mcolors.LinearSegmentedColormap.from_list(
            "purity_no_white",
            [
                "#313695",  # deep blue
                "#4575b4",
                "#74add1",
                "#fee08b",  # warm yellow, avoids a white midpoint
                "#fdae61",
                "#f46d43",
                "#a50026",  # deep red
            ],
            N=256,
        )
    return plt.colormaps[name]


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export case_N_cell_count_mask.png overlays")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--gallery-csv", type=Path, required=True)
    ap.add_argument("--base-dir", type=Path, default=None, help="Directory with case_N_base.jpg files.")
    ap.add_argument("--out-dir", type=Path, default=Path("frontend/public/cases"))
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cmap", default="purity")
    ap.add_argument("--count-vmax", type=float, default=180.0)
    ap.add_argument(
        "--scale-gamma",
        type=float,
        default=0.65,
        help="Power transform on normalized counts; values <1 expand lower/mid counts out of blue.",
    )
    ap.add_argument(
        "--write-website-mask",
        action="store_true",
        help="Also overwrite case_N_mask.png so the existing frontend shows cell-count masks.",
    )
    return ap.parse_args()


def _resolve_h5(row: pd.Series, h5_dir: Path) -> Path | None:
    candidates = [str(row["file_uuid_original"]).strip()]
    file_uuid_new = row.get("file_uuid_new")
    if pd.notna(file_uuid_new) and str(file_uuid_new).strip():
        candidates.append(str(file_uuid_new).strip())
    for uuid in candidates:
        path = h5_dir / f"{uuid}.h5"
        if path.exists():
            return path
    return None


def _grid_shape_from_h5(h5_path: Path) -> tuple[int, int, int]:
    with h5py.File(h5_path, "r") as h5:
        tile_size = int(h5.attrs.get("tile_size", 224))
        stride = int(h5.attrs.get("stride", tile_size))
        grid_nx = int(h5.attrs.get("grid_nx", 0))
        grid_ny = int(h5.attrs.get("grid_ny", 0))
        coords = h5["coords"][:]
    if grid_nx <= 0 or grid_ny <= 0:
        grid_nx = int(coords[:, 1].max() // stride) + 1
        grid_ny = int(coords[:, 0].max() // stride) + 1
    return grid_nx, grid_ny, stride


def _predictions_to_rgba(
    predictions: pd.DataFrame,
    *,
    grid_nx: int,
    grid_ny: int,
    stride: int,
    cmap_name: str,
    count_vmax: float,
    scale_gamma: float,
) -> Image.Image:
    heatmap = np.full((grid_ny, grid_nx), np.nan, dtype=np.float32)
    y_idx = (predictions["tile_y"].to_numpy() // stride).astype(int)
    x_idx = (predictions["tile_x"].to_numpy() // stride).astype(int)
    counts = predictions["pred_nuclei_count"].to_numpy(dtype=np.float32)
    for yi, xi, count in zip(y_idx, x_idx, counts, strict=False):
        if 0 <= yi < grid_ny and 0 <= xi < grid_nx:
            heatmap[yi, xi] = count

    rgba = np.zeros((grid_ny, grid_nx, 4), dtype=np.uint8)
    valid = np.isfinite(heatmap)
    cmap = _get_colormap(cmap_name)
    norm = mcolors.Normalize(vmin=0.0, vmax=float(count_vmax), clip=True)
    scaled = norm(heatmap[valid])
    scaled = np.power(scaled, max(float(scale_gamma), 1e-6))
    colors = cmap(scaled)
    rgba[valid, :3] = (colors[:, :3] * 255).astype(np.uint8)
    rgba[valid, 3] = 255
    return Image.fromarray(rgba, "RGBA")


def _summary_row(case_id: int, row: pd.Series, h5_path: Path, pred: pd.DataFrame) -> dict[str, object]:
    counts = pred["pred_nuclei_count"].to_numpy(dtype=np.float64)
    density = pred["pred_density_per_mm2"].to_numpy(dtype=np.float64)
    return {
        "case_id": case_id,
        "file_uuid": h5_path.stem,
        "barcode": row.get("barcode", ""),
        "project_id": row.get("project_id", ""),
        "n_tiles": int(len(pred)),
        "mean_pred_nuclei_count": float(np.mean(counts)),
        "median_pred_nuclei_count": float(np.median(counts)),
        "p05_pred_nuclei_count": float(np.percentile(counts, 5)),
        "p95_pred_nuclei_count": float(np.percentile(counts, 95)),
        "mean_pred_density_per_mm2": float(np.mean(density)),
        "mask_file": f"case_{case_id}_cell_count_mask.png",
    }


def main() -> None:
    args = _parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.h5_dir = args.h5_dir.expanduser().resolve()
    args.gallery_csv = args.gallery_csv.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    if args.base_dir is not None:
        args.base_dir = args.base_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gallery = pd.read_csv(args.gallery_csv)
    logger.info("Gallery: %d cases from %s", len(gallery), args.gallery_csv)
    logger.info(
        "Color scale: %s, 0..%.1f nuclei/tile, gamma=%.3f",
        args.cmap,
        args.count_vmax,
        args.scale_gamma,
    )

    model, ckpt = load_cellularity_model(args.checkpoint, device=args.device)
    logger.info("Loaded checkpoint %s (epoch=%s)", args.checkpoint, ckpt.get("epoch", "unknown"))

    summaries: list[dict[str, object]] = []
    missing: list[dict[str, object]] = []
    for i, row in gallery.iterrows():
        case_id = i + 1
        h5_path = _resolve_h5(row, args.h5_dir)
        if h5_path is None:
            logger.warning("[%d/%d] missing H5 for %s", case_id, len(gallery), row.get("barcode", ""))
            missing.append({"case_id": case_id, "barcode": row.get("barcode", "")})
            continue

        logger.info("[%d/%d] case_%d %s %s", case_id, len(gallery), case_id, row.get("project_id", ""), h5_path.name)
        grid_nx, grid_ny, stride = _grid_shape_from_h5(h5_path)
        pred = predict_h5(model, h5_path, device=args.device, batch_size=args.batch_size)
        overlay = _predictions_to_rgba(
            pred,
            grid_nx=grid_nx,
            grid_ny=grid_ny,
            stride=stride,
            cmap_name=args.cmap,
            count_vmax=args.count_vmax,
            scale_gamma=args.scale_gamma,
        )

        if args.base_dir is not None:
            base_path = args.base_dir / f"case_{case_id}_base.jpg"
            if base_path.exists():
                with Image.open(base_path) as base:
                    overlay = overlay.resize(base.size, Image.NEAREST)
            else:
                logger.warning("  base image missing, keeping tile-grid mask size: %s", base_path)

        mask_path = args.out_dir / f"case_{case_id}_cell_count_mask.png"
        overlay.save(mask_path, "PNG")
        if args.write_website_mask:
            overlay.save(args.out_dir / f"case_{case_id}_mask.png", "PNG")

        summary = _summary_row(case_id, row, h5_path, pred)
        summary["mask_width"] = int(overlay.width)
        summary["mask_height"] = int(overlay.height)
        summaries.append(summary)
        logger.info(
            "  mean=%.1f median=%.1f p95=%.1f tiles=%d -> %s",
            summary["mean_pred_nuclei_count"],
            summary["median_pred_nuclei_count"],
            summary["p95_pred_nuclei_count"],
            summary["n_tiles"],
            mask_path.name,
        )

    summary_df = pd.DataFrame(summaries)
    summary_csv = args.out_dir / "cell_count_case_summary.csv"
    summary_json = args.out_dir / "cell_count_case_summary.json"
    summary_df.to_csv(summary_csv, index=False)
    summary_json.write_text(
        json.dumps(
            {
                "checkpoint": str(args.checkpoint),
                "checkpoint_epoch": ckpt.get("epoch", "unknown"),
                "color_scale": {
                    "cmap": args.cmap,
                    "vmin": 0.0,
                    "vmax": args.count_vmax,
                    "gamma": args.scale_gamma,
                },
                "n_cases": len(summaries),
                "missing": missing,
                "cases": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    logger.info("Wrote %d masks to %s", len(summaries), args.out_dir)
    logger.info("Summary: %s", summary_csv)
    if missing:
        logger.warning("Missing H5 for %d cases; see %s", len(missing), summary_json)


if __name__ == "__main__":
    main()
