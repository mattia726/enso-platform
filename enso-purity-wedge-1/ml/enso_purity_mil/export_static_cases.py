"""Export static case assets for Cloudflare / native viewer: base JPG + mask PNG per gallery row.

Reads gallery_summary.csv (from build_demo_gallery), and for each row in order writes:
  case_{i+1}_base.jpg  — H&E thumbnail (or synthetic placeholder)
  case_{i+1}_mask.png  — Purity heatmap overlay (RGBA, same size as base)

Masks are 100% solid (alpha=255 for every tile) so the React opacity slider has full control.
All tiles are included, including 0.0 scores (painted at low end of colormap).

Order matches the frontend gallery (CSV row order = case_1, case_2, ...).

Usage:
  python -m enso_purity_mil.export_static_cases \\
    --model-path ml/runs/fold0/best_model.pth \\
    --h5-dir ~/bucket_embeddings/embeddings_fp32 \\
    --gallery-csv frontend/gallery/gallery_summary.csv \\
    --out-dir frontend/public/cases
"""
from __future__ import annotations

import argparse
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

from enso_purity_mil.heatmap import predict_tile_scores
from enso_purity_mil.interactive_viewer import _fetch_thumbnail
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _scores_to_solid_rgba_image(
    scores: np.ndarray,
    coords: np.ndarray,
    grid_nx: int,
    grid_ny: int,
    stride: int,
) -> Image.Image:
    """Build a 100% solid RGBA mask: every tile painted with alpha=255, including 0.0 scores.

    React opacity slider then has full control; no baked-in transparency.
    coords[:, 0] = row (Y), coords[:, 1] = col (X).
    """
    heatmap = np.full((grid_ny, grid_nx), np.nan, dtype=np.float32)
    y_idx = (coords[:, 0] // stride).astype(int)
    x_idx = (coords[:, 1] // stride).astype(int)
    for i in range(len(scores)):
        yi, xi = int(y_idx[i]), int(x_idx[i])
        if 0 <= yi < grid_ny and 0 <= xi < grid_nx:
            heatmap[yi, xi] = scores[i]

    cmap = plt.colormaps["RdYlBu_r"]
    norm = mcolors.Normalize(vmin=0, vmax=1)
    rgba = np.zeros((grid_ny, grid_nx, 4), dtype=np.uint8)
    for y in range(grid_ny):
        for x in range(grid_nx):
            v = heatmap[y, x]
            if np.isnan(v):
                continue
            r, g, b, _ = cmap(norm(v))
            # Solid: alpha=255 for every tile so the frontend slider controls opacity.
            rgba[y, x] = [int(r * 255), int(g * 255), int(b * 255), 255]
    return Image.fromarray(rgba, "RGBA")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export static case_N_base.jpg and case_N_mask.png from gallery CSV")
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--gallery-csv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("frontend/public/cases"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    args.model_path = Path(args.model_path).expanduser().resolve()
    args.h5_dir = Path(args.h5_dir).expanduser().resolve()
    args.gallery_csv = Path(args.gallery_csv).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()

    if not args.gallery_csv.exists():
        raise FileNotFoundError(f"Gallery CSV not found: {args.gallery_csv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gallery = pd.read_csv(args.gallery_csv)
    logger.info("Gallery: %d cases from %s", len(gallery), args.gallery_csv)

    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    model = EnsoMILModel(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    for i in range(len(gallery)):
        row = gallery.iloc[i]
        case_id = i + 1
        file_uuid_orig = str(row["file_uuid_original"]).strip()
        file_uuid_new = row.get("file_uuid_new")
        if pd.isna(file_uuid_new) or file_uuid_new == "":
            file_uuid_new = None
        else:
            file_uuid_new = str(file_uuid_new).strip()

        orig_h5 = args.h5_dir / f"{file_uuid_orig}.h5"
        new_h5 = args.h5_dir / f"{file_uuid_new}.h5" if file_uuid_new else None
        if orig_h5.exists():
            h5_path = orig_h5
        elif new_h5 and new_h5.exists():
            h5_path = new_h5
        else:
            logger.warning("[%d/%d] Skip case_%d: no H5 at %s or %s", i + 1, len(gallery), case_id, orig_h5, new_h5)
            continue

        logger.info("[%d/%d] case_%d %s", i + 1, len(gallery), case_id, row.get("project_id", ""))

        with h5py.File(h5_path, "r") as f:
            tile_size = int(f.attrs.get("tile_size", 224))
            stride = int(f.attrs.get("stride", tile_size))
            grid_nx = int(f.attrs.get("grid_nx", 0))
            grid_ny = int(f.attrs.get("grid_ny", 0))
            raw_coords = f["coords"][:]

        if grid_nx == 0 or grid_ny == 0:
            grid_nx = int(raw_coords[:, 1].max() // stride) + 1
            grid_ny = int(raw_coords[:, 0].max() // stride) + 1

        with torch.no_grad():
            scores, coords = predict_tile_scores(model, h5_path, k=81, batch_size=1024, device=args.device)
        overlay_img = _scores_to_solid_rgba_image(scores, coords, grid_nx, grid_ny, stride)

        thumb = None
        for uuid in (file_uuid_orig, file_uuid_new):
            if not uuid:
                continue
            thumb = _fetch_thumbnail(uuid)
            if thumb is not None:
                break
        if thumb is None:
            logger.warning("  No thumbnail — synthetic background for case_%d", case_id)
            thumb = Image.new("RGB", (grid_nx * 4, grid_ny * 4), (240, 230, 220))

        overlay_resized = overlay_img.resize(thumb.size, Image.NEAREST)

        base_path = args.out_dir / f"case_{case_id}_base.jpg"
        mask_path = args.out_dir / f"case_{case_id}_mask.png"
        thumb.save(base_path, "JPEG", quality=90)
        overlay_resized.save(mask_path, "PNG")
        logger.info("  %s  %s", base_path.name, mask_path.name)

    logger.info("Done. Wrote %d cases to %s", len(gallery), args.out_dir)


if __name__ == "__main__":
    main()
