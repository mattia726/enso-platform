"""CLI entry point for spatial heatmap inference.

Usage:
    python -m enso_purity_mil.heatmap_cli \
        --model-path ml/runs/fold0/best_model.pth \
        --h5-path /path/to/<file_uuid>.h5 \
        --out-dir heatmaps/
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from enso_purity_mil.heatmap import predict_tile_scores
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

import torch


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate tile-level purity heatmap")
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--h5-path", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("heatmaps"))
    # K=81 (9x9 grid) provides a ~1 mm² physical context window because our
    # upstream Virchow extraction pipeline strictly normalizes all embeddings
    # to 0.5 mpp, making each tile 224 px × 0.5 µm/px = 112 µm.
    # 9 tiles × 112 µm ≈ 1008 µm ≈ 1 mm per side.
    ap.add_argument("--k", type=int, default=81, help="Neighbourhood size for local KDE (81 = 9x9 ≈ 1mm²)")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(args.model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**checkpoint["config"])
    model = EnsoMILModel(cfg)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(args.device)
    logger.info("Loaded model from %s (epoch %d, val_loss=%.4f)",
                args.model_path, checkpoint["epoch"], checkpoint["val_loss"])

    scores, coords = predict_tile_scores(
        model, args.h5_path, k=args.k, batch_size=args.batch_size, device=args.device,
    )
    logger.info("Predicted %d tile scores, range [%.3f, %.3f]", len(scores), scores.min(), scores.max())

    stem = Path(args.h5_path).stem

    # Read tile geometry from H5 metadata rather than hardcoding
    with h5py.File(args.h5_path, "r") as f:
        tile_size = int(f.attrs.get("tile_size", 224))
        stride = int(f.attrs.get("stride", tile_size))

    x_grid = coords[:, 0] // stride
    y_grid = coords[:, 1] // stride
    nx = x_grid.max() + 1
    ny = y_grid.max() + 1

    heatmap = np.full((ny, nx), np.nan, dtype=np.float32)
    for i in range(len(scores)):
        heatmap[y_grid[i], x_grid[i]] = scores[i]

    fig, ax = plt.subplots(figsize=(max(6, nx / 10), max(4, ny / 10)))
    im = ax.imshow(heatmap, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Predicted purity")
    ax.set_title(f"Purity heatmap: {stem}")
    ax.axis("off")
    fig.tight_layout()

    out_path = args.out_dir / f"{stem}_heatmap.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_path)

    np.savez_compressed(
        args.out_dir / f"{stem}_scores.npz",
        scores=scores, coords=coords,
    )


if __name__ == "__main__":
    main()
