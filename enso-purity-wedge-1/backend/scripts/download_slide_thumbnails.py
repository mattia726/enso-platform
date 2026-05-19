"""Download low-resolution thumbnails for TS/BS pairs from GDC.

Uses HTTP range requests via fsspec+tifffile to avoid downloading
entire SVS files (hundreds of MB each).
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import fsspec
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from PIL import Image

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

GDC_DATA_URL = "https://api.gdc.cancer.gov/data"


def download_thumbnail(file_id: str, barcode: str, out_dir: Path) -> Path | None:
    """Extract the lowest-resolution page from an SVS file on GDC."""
    url = f"{GDC_DATA_URL}/{file_id}"
    out_path = out_dir / f"{barcode}.png"
    if out_path.exists():
        logger.info("Already exists: %s", out_path)
        return out_path

    try:
        with fsspec.open(url, mode="rb") as f:
            with tifffile.TiffFile(f) as tif:
                if len(tif.pages) < 2:
                    logger.warning("Only 1 page for %s — using page 0", barcode)
                    page = tif.pages[0]
                else:
                    page = tif.pages[1]

                arr = page.asarray()
                img = Image.fromarray(arr)
                img.save(out_path)
                logger.info("Saved %s  (%dx%d)", out_path, img.width, img.height)
                return out_path
    except Exception:
        logger.exception("Failed to download %s (%s)", barcode, file_id)
        return None


def make_comparison_grid(
    pairs: pd.DataFrame,
    thumb_dir: Path,
    out_path: Path,
) -> None:
    """Create a side-by-side comparison grid of TS vs BS thumbnails."""
    n = len(pairs)
    fig, axes = plt.subplots(n, 2, figsize=(10, 4 * n))
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (_, row) in enumerate(pairs.iterrows()):
        ts_path = thumb_dir / f"{row['ts_barcode']}.png"
        bs_path = thumb_dir / f"{row['bs_barcode']}.png"

        for j, (path, label) in enumerate([
            (ts_path, f"TS: {row['ts_barcode']}\nportion={row['ts_portion']}"),
            (bs_path, f"BS: {row['bs_barcode']}\nportion={row['bs_portion']}"),
        ]):
            ax = axes[i, j]
            if path.exists():
                img = Image.open(path)
                ax.imshow(np.array(img))
            else:
                ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=14)
            ax.set_title(label, fontsize=8)
            ax.axis("off")

        axes[i, 0].set_ylabel(row["sample_vial"], fontsize=9, rotation=0, labelpad=90, va="center")

    fig.suptitle("TS vs BS pairs — same sample_vial, different barcode portions", fontsize=12, y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved comparison grid: %s", out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Download thumbnails for TS/BS pairs")
    ap.add_argument("--pairs-csv", type=Path, default=Path("/tmp/ts_bs_pairs.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/reports/slide_thumbnails"))
    ap.add_argument("--grid-out", type=Path, default=Path("data/reports/ts_vs_bs_comparison.png"))
    ap.add_argument("--max-pairs", type=int, default=10)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)

    pairs = pd.read_csv(args.pairs_csv).head(args.max_pairs)
    logger.info("Processing %d TS/BS pairs", len(pairs))

    for _, row in pairs.iterrows():
        logger.info("=== %s ===", row["sample_vial"])
        download_thumbnail(row["ts_file_id"], row["ts_barcode"], args.out_dir)
        download_thumbnail(row["bs_file_id"], row["bs_barcode"], args.out_dir)

    make_comparison_grid(pairs, args.out_dir, args.grid_out)


if __name__ == "__main__":
    main()
