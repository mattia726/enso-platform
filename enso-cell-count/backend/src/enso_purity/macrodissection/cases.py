"""Case discovery and tile-grid loader.

A *case* in the macrodissection workbench is a slide with three artifacts
under ``frontend/public/cases/``:

* ``case_{N}_base.jpg`` — H&E thumbnail.
* ``case_{N}_mask.png`` — pre-rendered purity heatmap (RdYlBu_r).
* ``case_{N}_cell_count_mask.png`` — pre-rendered cellularity heatmap.

Plus the structured artifact emitted by :mod:`build_artifacts`:

* ``case_{N}_tiles.json`` — schema-versioned metadata.
* ``case_{N}_grid.bin`` — Float32Array packed ``(grid_ny, grid_nx, 6)`` with
  channels ``[purity, purity_sd, nuclei, nuclei_sd, tumor_nuclei,
  tissue_fraction]``.

This module knows how to enumerate cases and lazily load the tile arrays
for one case.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from .roi import TileArrays, TileGrid


CASE_TILES_BIN_CHANNELS = (
    "purity",
    "purity_sd",
    "nuclei",
    "nuclei_sd",
    "tumor_nuclei",
    "tissue_fraction",
)


@dataclass(frozen=True)
class CaseMeta:
    """Lightweight metadata for one case (for the case-list endpoint)."""

    case_id: int
    barcode: str
    project_id: str
    file_uuid: str
    base_width: int
    base_height: int
    grid_nx: int
    grid_ny: int
    n_tiles_tissue: int
    has_purity: bool
    has_cellularity: bool
    base_image: str  # public URL relative to /cases/
    purity_mask: str
    cellularity_mask: str
    tiles_meta: str  # URL to the JSON
    tiles_bin: str  # URL to the binary grid
    tile_size_um: float
    mpp_thumb_x: float
    mpp_thumb_y: float
    purity_model_version: str
    cellularity_model_version: str
    tile_encoder_version: str

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _case_id_from_filename(name: str) -> int | None:
    m = re.match(r"case_(\d+)_tiles\.json$", name)
    return int(m.group(1)) if m else None


def discover_cases(cases_dir: Path) -> list[CaseMeta]:
    """Return the list of cases with structured artifacts under ``cases_dir``."""

    out: list[CaseMeta] = []
    if not cases_dir.exists():
        return out
    for path in sorted(cases_dir.glob("case_*_tiles.json")):
        case_id = _case_id_from_filename(path.name)
        if case_id is None:
            continue
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        out.append(_meta_from_json(case_id, meta, cases_dir))
    return out


def _public_url(cases_dir: Path, name: str) -> str:
    """Return a URL relative to the Next.js public dir."""

    return f"/cases/{name}" if (cases_dir / name).exists() else ""


def _meta_from_json(case_id: int, meta: dict, cases_dir: Path) -> CaseMeta:
    base_name = f"case_{case_id}_base.jpg"
    purity_name = f"case_{case_id}_mask.png"
    cell_name = f"case_{case_id}_cell_count_mask.png"
    tiles_meta = f"case_{case_id}_tiles.json"
    tiles_bin = meta.get("tiles_bin", f"case_{case_id}_grid.bin")
    return CaseMeta(
        case_id=case_id,
        barcode=str(meta.get("barcode", "")),
        project_id=str(meta.get("project_id", "")),
        file_uuid=str(meta.get("file_uuid", "")),
        base_width=int(meta.get("base_width", 0)),
        base_height=int(meta.get("base_height", 0)),
        grid_nx=int(meta.get("grid_nx", 0)),
        grid_ny=int(meta.get("grid_ny", 0)),
        n_tiles_tissue=int(meta.get("n_tiles_tissue", 0)),
        has_purity=(cases_dir / purity_name).exists(),
        has_cellularity=(cases_dir / cell_name).exists(),
        base_image=_public_url(cases_dir, base_name),
        purity_mask=_public_url(cases_dir, purity_name),
        cellularity_mask=_public_url(cases_dir, cell_name),
        tiles_meta=_public_url(cases_dir, tiles_meta),
        tiles_bin=_public_url(cases_dir, tiles_bin),
        tile_size_um=float(meta.get("tile_size_um", 0.0)),
        mpp_thumb_x=float(meta.get("mpp_thumb_x", 0.0)),
        mpp_thumb_y=float(meta.get("mpp_thumb_y", 0.0)),
        purity_model_version=str(meta.get("purity_model_version", "unknown")),
        cellularity_model_version=str(meta.get("cellularity_model_version", "unknown")),
        tile_encoder_version=str(meta.get("tile_encoder_version", "unknown")),
    )


@lru_cache(maxsize=64)
def load_tile_arrays(cases_dir: str, case_id: int) -> TileArrays:
    """Load the per-tile prediction grid for one case (LRU-cached)."""

    cases_path = Path(cases_dir)
    meta_path = cases_path / f"case_{case_id}_tiles.json"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    grid = TileGrid(
        stride_x=float(meta["tile_pix_w"]),
        stride_y=float(meta["tile_pix_h"]),
        offset_x=float(meta.get("offset_x", 0.0)),
        offset_y=float(meta.get("offset_y", 0.0)),
        grid_nx=int(meta["grid_nx"]),
        grid_ny=int(meta["grid_ny"]),
        tile_area_mm2=float(meta["tile_area_mm2"]),
    )
    bin_path = cases_path / meta.get("tiles_bin", f"case_{case_id}_grid.bin")
    raw = np.fromfile(bin_path, dtype=np.float32)
    expected = grid.grid_nx * grid.grid_ny * len(CASE_TILES_BIN_CHANNELS)
    if raw.size != expected:
        raise ValueError(
            f"case {case_id}: expected {expected} float32 values in "
            f"{bin_path.name}, got {raw.size}"
        )
    arr = raw.reshape(grid.grid_ny, grid.grid_nx, len(CASE_TILES_BIN_CHANNELS))
    return TileArrays(
        grid=grid,
        purity=arr[..., 0].copy(),
        purity_sd=arr[..., 1].copy(),
        nuclei=arr[..., 2].copy(),
        nuclei_sd=arr[..., 3].copy(),
        tissue_fraction=arr[..., 5].copy(),
    )


def clear_case_cache() -> None:
    """Drop the LRU cache (used by tests when artifacts change on disk)."""

    load_tile_arrays.cache_clear()
