"""CLI that turns rendered PNG masks into per-tile prediction artifacts.

The build pipeline writes two files per case under ``frontend/public/cases/``:

* ``case_{N}_tiles.json`` — schema-versioned metadata (case identity, tile
  geometry, model versions, default thresholds).
* ``case_{N}_grid.bin`` — Float32Array packed ``(grid_ny, grid_nx, 6)`` in
  row-major order with channels in the order described by
  :data:`enso_purity.macrodissection.cases.CASE_TILES_BIN_CHANNELS`:
  ``[purity, purity_sd, nuclei, nuclei_sd, tumor_nuclei, tissue_fraction]``.

Two input paths are supported:

1. **Live path** — invoked from the same VM that runs the ML inference,
   feeding raw scalar grids directly. *Not exercised here* (production-only).
2. **Offline path** — invoked on the demo CI VM, where only the PNG masks
   are checked into the repo. We recover the scalar values via
   nearest-neighbour LUT lookup against the known forward palettes.

The offline path is deterministic and idempotent.

Usage::

    python -m enso_purity.macrodissection.build_artifacts \
        --cases-dir frontend/public/cases \
        --gallery frontend/gallery/gallery_summary.csv

The script never fails the whole pipeline because of one bad case; it logs a
warning and continues. Re-running is cheap.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image

from .cases import CASE_TILES_BIN_CHANNELS
from .grid_detect import GridSpec, detect_grid, per_tile_modal_color
from .inverse_cmap import (
    CELLULARITY_SPEC,
    PURITY_SPEC,
    decode_rgb,
)


LOG = logging.getLogger(__name__)

ARTIFACT_SCHEMA_VERSION = 1

# Default known geometry: each tile is 224×224 px at 0.5 µm/px on the WSI.
TILE_SIZE_UM_DEFAULT = 0.5 * 224.0
TILE_AREA_MM2_DEFAULT = (TILE_SIZE_UM_DEFAULT / 1000.0) ** 2  # 12544 µm² = 0.012544 mm²

# Default per-tile uncertainty when the model does not supply one.
DEFAULT_PURITY_SD = 0.07  # ~7 percentage-points, calibrated for MIL outputs
DEFAULT_NUCLEI_SD = 5.0   # ±5 nuclei per tile, matches the user spec


def _iter_gallery_rows(gallery_csv: Optional[Path]) -> Iterable[dict[str, str]]:
    if gallery_csv is None or not gallery_csv.exists():
        return ()
    with gallery_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _gallery_index(rows: Iterable[dict[str, str]]) -> dict[int, dict[str, str]]:
    return {i + 1: row for i, row in enumerate(rows)}


def _detect_alpha_only(rgba: np.ndarray) -> np.ndarray:
    """Return a binary tissue mask from the alpha channel of a heatmap."""

    return rgba[..., 3] > 8


def _tile_tissue_fraction(
    alpha_mask: np.ndarray, grid: GridSpec
) -> np.ndarray:
    """Fraction of base-image pixels in each tile that carry data."""

    ny, nx = grid.grid_ny, grid.grid_nx
    out = np.zeros((ny, nx), dtype=np.float32)
    h, w = alpha_mask.shape
    for iy in range(ny):
        for ix in range(nx):
            x0, y0, x1, y1 = grid.tile_rect(ix, iy)
            x1 = min(x1, w)
            y1 = min(y1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            window = alpha_mask[y0:y1, x0:x1]
            if window.size == 0:
                continue
            out[iy, ix] = float(window.mean())
    return out


def _decode_grid_from_tiles(
    rgba_tiles: np.ndarray, spec_kind: str
) -> np.ndarray:
    """Decode the per-tile color array into a scalar grid.

    ``rgba_tiles`` has shape ``(ny, nx, 4)``. Tiles whose alpha channel is
    zero are treated as missing and return NaN.
    """

    if spec_kind == "purity":
        spec = PURITY_SPEC
    elif spec_kind == "cellularity":
        spec = CELLULARITY_SPEC
    else:
        raise ValueError(f"Unknown spec_kind {spec_kind!r}")

    rgb = rgba_tiles[..., :3]
    alpha = rgba_tiles[..., 3]
    values, _dist = decode_rgb(rgb, spec)
    return np.where(alpha > 8, values, np.float32(np.nan))


def _load_image_to_rgba(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.array(im.convert("RGBA"))


def _build_one_case(
    case_id: int,
    cases_dir: Path,
    gallery_row: dict[str, str] | None,
    *,
    purity_model_version: str,
    cellularity_model_version: str,
    tile_encoder_version: str,
    purity_sd_default: float,
    nuclei_sd_default: float,
    log: logging.Logger,
) -> dict | None:
    base_path = cases_dir / f"case_{case_id}_base.jpg"
    purity_path = cases_dir / f"case_{case_id}_mask.png"
    cell_path = cases_dir / f"case_{case_id}_cell_count_mask.png"
    if not base_path.exists():
        log.warning("case_%d: missing base image %s; skipping", case_id, base_path.name)
        return None
    if not purity_path.exists() and not cell_path.exists():
        log.warning("case_%d: no mask PNG available; skipping", case_id)
        return None

    # Determine the geometric grid from whichever PNG is present.
    geom_source = purity_path if purity_path.exists() else cell_path
    rgba_geom = _load_image_to_rgba(geom_source)
    base_h, base_w = rgba_geom.shape[:2]
    grid = detect_grid(rgba_geom)

    log.info(
        "case_%d: base=%dx%d  grid=%dx%d  stride=%dx%d  offset=(%d,%d)",
        case_id,
        base_w,
        base_h,
        grid.grid_nx,
        grid.grid_ny,
        grid.stride_x,
        grid.stride_y,
        grid.offset_x,
        grid.offset_y,
    )

    # Decode per-tile colors → scalar grids.
    if purity_path.exists():
        purity_rgba = rgba_geom if geom_source == purity_path else _load_image_to_rgba(purity_path)
        purity_tiles_rgba = per_tile_modal_color(purity_rgba, grid)
        purity_grid = _decode_grid_from_tiles(purity_tiles_rgba, "purity")
    else:
        purity_grid = np.full((grid.grid_ny, grid.grid_nx), np.nan, dtype=np.float32)

    if cell_path.exists():
        cell_rgba = rgba_geom if geom_source == cell_path else _load_image_to_rgba(cell_path)
        cell_tiles_rgba = per_tile_modal_color(cell_rgba, grid)
        cell_grid = _decode_grid_from_tiles(cell_tiles_rgba, "cellularity")
    else:
        cell_grid = np.full((grid.grid_ny, grid.grid_nx), np.nan, dtype=np.float32)

    # Tissue mask is the alpha plane of the geometry source.
    alpha_mask = _detect_alpha_only(rgba_geom)
    tissue_fraction = _tile_tissue_fraction(alpha_mask, grid)

    # Synthetic per-tile uncertainties — small constants by default. Future
    # versions can plug in actual model-predicted variances.
    purity_sd = np.where(
        np.isnan(purity_grid),
        np.float32(np.nan),
        np.float32(purity_sd_default),
    )
    nuclei_sd = np.where(
        np.isnan(cell_grid),
        np.float32(np.nan),
        np.float32(nuclei_sd_default),
    )
    # Tumor nuclei is the point estimate p * n; NaN-safe.
    tumor_nuclei = purity_grid * cell_grid
    tumor_nuclei = np.where(np.isnan(tumor_nuclei), 0.0, tumor_nuclei).astype(np.float32)

    # Pack into the 6-channel float32 binary.
    packed = np.stack(
        [
            purity_grid.astype(np.float32),
            purity_sd.astype(np.float32),
            cell_grid.astype(np.float32),
            nuclei_sd.astype(np.float32),
            tumor_nuclei.astype(np.float32),
            tissue_fraction.astype(np.float32),
        ],
        axis=-1,
    )

    bin_name = f"case_{case_id}_grid.bin"
    bin_path = cases_dir / bin_name
    packed.tofile(bin_path)

    # Physical scales — every tile occupies 224 px at 0.5 mpp on the WSI;
    # the thumbnail is mask-resolution, so the mpp per thumbnail pixel is
    # TILE_SIZE_UM / stride_px.
    mpp_thumb_x = TILE_SIZE_UM_DEFAULT / max(grid.stride_x, 1)
    mpp_thumb_y = TILE_SIZE_UM_DEFAULT / max(grid.stride_y, 1)

    n_tiles_tissue = int((tissue_fraction > 0).sum())

    meta = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "case_id": case_id,
        "barcode": gallery_row.get("barcode", "") if gallery_row else "",
        "project_id": gallery_row.get("project_id", "") if gallery_row else "",
        "file_uuid": gallery_row.get("file_uuid_original", "") if gallery_row else "",
        "base_width": int(base_w),
        "base_height": int(base_h),
        "tile_pix_w": int(grid.stride_x),
        "tile_pix_h": int(grid.stride_y),
        "offset_x": int(grid.offset_x),
        "offset_y": int(grid.offset_y),
        "tile_size_um": TILE_SIZE_UM_DEFAULT,
        "tile_area_mm2": TILE_AREA_MM2_DEFAULT,
        "mpp_thumb_x": mpp_thumb_x,
        "mpp_thumb_y": mpp_thumb_y,
        "grid_nx": int(grid.grid_nx),
        "grid_ny": int(grid.grid_ny),
        "n_tiles_tissue": n_tiles_tissue,
        "purity_model_version": purity_model_version,
        "cellularity_model_version": cellularity_model_version,
        "tile_encoder_version": tile_encoder_version,
        "thresholds_default": {
            "purity_min": 0.20,
            "tumor_cells_min": 1000,
            "borderline_purity_band": 0.05,
            "borderline_tumor_cells_band": 200,
        },
        "tiles_bin": bin_name,
        "tiles_bin_layout": list(CASE_TILES_BIN_CHANNELS),
        "uncertainty_defaults": {
            "purity_sd": purity_sd_default,
            "nuclei_sd": nuclei_sd_default,
        },
        "source": "inverse_colormap",
    }

    meta_path = cases_dir / f"case_{case_id}_tiles.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
        f.write("\n")

    log.info(
        "case_%d: wrote %s (%d tiles, %d with tissue)",
        case_id,
        bin_path.name,
        grid.grid_nx * grid.grid_ny,
        n_tiles_tissue,
    )
    return meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--cases-dir",
        type=Path,
        default=Path("frontend/public/cases"),
        help="Directory containing case_N_*.jpg/png artifacts (in/out).",
    )
    parser.add_argument(
        "--gallery",
        type=Path,
        default=Path("frontend/gallery/gallery_summary.csv"),
        help="Optional gallery CSV with barcode/project_id/file_uuid info.",
    )
    parser.add_argument(
        "--cases",
        type=int,
        nargs="*",
        default=None,
        help="Restrict the build to the given case numbers.",
    )
    parser.add_argument(
        "--purity-model-version",
        default="v3_fold0",
        help="Stamp this version string into the artifact JSON.",
    )
    parser.add_argument(
        "--cellularity-model-version",
        default="cellularity_ssd_fold1",
    )
    parser.add_argument("--tile-encoder-version", default="virchow_v1")
    parser.add_argument("--purity-sd", type=float, default=DEFAULT_PURITY_SD)
    parser.add_argument("--nuclei-sd", type=float, default=DEFAULT_NUCLEI_SD)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    cases_dir: Path = args.cases_dir
    if not cases_dir.exists():
        LOG.error("cases dir %s does not exist", cases_dir)
        return 2

    gallery_rows = _iter_gallery_rows(args.gallery)
    gallery_idx = _gallery_index(gallery_rows)

    if args.cases:
        case_ids = sorted(set(args.cases))
    else:
        case_ids = sorted(
            {
                int(name.split("_")[1])
                for name in (p.name for p in cases_dir.glob("case_*_base.jpg"))
            }
        )
    LOG.info("building %d cases under %s", len(case_ids), cases_dir)

    built: list[dict] = []
    for case_id in case_ids:
        try:
            meta = _build_one_case(
                case_id,
                cases_dir,
                gallery_idx.get(case_id),
                purity_model_version=args.purity_model_version,
                cellularity_model_version=args.cellularity_model_version,
                tile_encoder_version=args.tile_encoder_version,
                purity_sd_default=args.purity_sd,
                nuclei_sd_default=args.nuclei_sd,
                log=LOG,
            )
        except Exception:  # pragma: no cover - production-side robustness
            LOG.exception("case_%d build failed; continuing", case_id)
            continue
        if meta is not None:
            built.append(meta)

    summary = cases_dir / "macrodissection_build_summary.json"
    with summary.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "schema_version": ARTIFACT_SCHEMA_VERSION,
                "n_cases_built": len(built),
                "cases": [
                    {
                        "case_id": m["case_id"],
                        "grid_nx": m["grid_nx"],
                        "grid_ny": m["grid_ny"],
                        "n_tiles_tissue": m["n_tiles_tissue"],
                        "barcode": m["barcode"],
                        "project_id": m["project_id"],
                    }
                    for m in built
                ],
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")
    LOG.info("wrote %s (%d cases)", summary, len(built))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
