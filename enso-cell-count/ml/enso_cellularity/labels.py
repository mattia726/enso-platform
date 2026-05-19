"""Tile-level label helpers for EnsoCellularity."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_COUNT_BIN_EDGES = (0.0, 10.0, 50.0, 150.0, 300.0)


@dataclass(frozen=True)
class TileGridSpec:
    """Geometry needed to map base-level centroids to embedding tile rows."""

    grid_nx: int
    grid_ny: int
    stride_level0: int
    tile_size_level0: int
    pad_left_level0: int
    pad_top_level0: int
    mpp_x: float
    mpp_y: float
    target_mpp: float
    tile_size: int

    @property
    def tile_area_mm2(self) -> float:
        width_um = self.tile_size_level0 * self.mpp_x
        height_um = self.tile_size_level0 * self.mpp_y
        return (width_um * height_um) / 1_000_000.0


def count_bin_indices(
    counts: np.ndarray | pd.Series,
    *,
    edges: tuple[float, ...] = DEFAULT_COUNT_BIN_EDGES,
) -> np.ndarray:
    """Map counts to ordered UI bins.

    Default bins are:
    ``0``, ``1-10``, ``10-50``, ``50-150``, ``150-300``, ``>300``.
    """

    values = np.asarray(counts, dtype=np.float32)
    return np.digitize(values, np.asarray(edges, dtype=np.float32), right=True).astype(np.int16)


def ordinal_targets_from_counts(
    counts: np.ndarray,
    *,
    edges: tuple[float, ...] = DEFAULT_COUNT_BIN_EDGES,
) -> np.ndarray:
    """Return cumulative ordinal BCE targets for count thresholds."""

    values = np.asarray(counts, dtype=np.float32).reshape(-1, 1)
    thresholds = np.asarray(edges, dtype=np.float32).reshape(1, -1)
    return (values > thresholds).astype(np.float32)


def tile_grid_spec_from_h5_attrs(attrs: Any) -> TileGridSpec:
    """Build a ``TileGridSpec`` from embedding H5 attributes."""

    mpp = float(attrs.get("mpp", attrs.get("target_mpp", 0.5)))
    mpp_x = float(attrs.get("mpp_x", attrs.get("base_mpp_x", mpp)))
    mpp_y = float(attrs.get("mpp_y", attrs.get("base_mpp_y", mpp)))
    tile_size = int(attrs.get("tile_size", 224))
    target_mpp = float(attrs.get("target_mpp", 0.5))
    tile_size_level0 = int(
        attrs.get(
            "extracted_level0_size",
            round(tile_size * target_mpp / max(mpp_x, 1e-6)),
        )
    )
    stride_level0 = int(attrs.get("stride_level0", tile_size_level0))
    return TileGridSpec(
        grid_nx=int(attrs.get("grid_nx", 0)),
        grid_ny=int(attrs.get("grid_ny", 0)),
        stride_level0=stride_level0,
        tile_size_level0=tile_size_level0,
        pad_left_level0=int(attrs.get("pad_left_L0", attrs.get("pad_left_level0", 0))),
        pad_top_level0=int(attrs.get("pad_top_L0", attrs.get("pad_top_level0", 0))),
        mpp_x=mpp_x,
        mpp_y=mpp_y,
        target_mpp=target_mpp,
        tile_size=tile_size,
    )


def tile_keys_from_level0_coords(coords_level0: np.ndarray, spec: TileGridSpec) -> np.ndarray:
    """Return stable row-major grid keys for H5 ``coords_level0`` rows.

    The Virchow embedder stores ``coords_level0`` as ``[x, y]`` tile top-left
    coordinates in base-level pixels.
    """

    coords = np.asarray(coords_level0, dtype=np.int64)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords_level0 shaped (N, 2), got {coords.shape}")
    col = np.rint((coords[:, 0] + spec.pad_left_level0) / spec.stride_level0).astype(np.int64)
    row = np.rint((coords[:, 1] + spec.pad_top_level0) / spec.stride_level0).astype(np.int64)
    return row * int(spec.grid_nx) + col


def count_centroids_in_tiles(
    centroid_x: np.ndarray,
    centroid_y: np.ndarray,
    coords_level0: np.ndarray,
    spec: TileGridSpec,
) -> np.ndarray:
    """Count centroid points inside each kept embedding tile.

    This assumes the embedding grid uses non-overlapping tiles, which is true for
    the current Virchow H5 files where ``stride_level0 == extracted_level0_size``.
    Points falling in filtered-out/non-kept tiles are ignored.
    """

    if spec.grid_nx <= 0 or spec.grid_ny <= 0:
        raise ValueError("grid_nx and grid_ny must be positive.")
    if spec.stride_level0 <= 0:
        raise ValueError("stride_level0 must be positive.")
    if spec.tile_size_level0 != spec.stride_level0:
        raise ValueError(
            "Only non-overlapping tile grids are supported for fast counting: "
            f"tile_size_level0={spec.tile_size_level0}, stride_level0={spec.stride_level0}"
        )

    kept_keys = tile_keys_from_level0_coords(coords_level0, spec)
    order = np.argsort(kept_keys)
    sorted_keys = kept_keys[order]

    x = np.asarray(centroid_x, dtype=np.float64)
    y = np.asarray(centroid_y, dtype=np.float64)
    col = np.floor((x + spec.pad_left_level0) / spec.stride_level0).astype(np.int64)
    row = np.floor((y + spec.pad_top_level0) / spec.stride_level0).astype(np.int64)
    valid = (col >= 0) & (col < spec.grid_nx) & (row >= 0) & (row < spec.grid_ny)
    point_keys = row[valid] * int(spec.grid_nx) + col[valid]

    pos = np.searchsorted(sorted_keys, point_keys)
    matched = pos < len(sorted_keys)
    matched[matched] &= sorted_keys[pos[matched]] == point_keys[matched]
    tile_positions = order[pos[matched]]

    return np.bincount(tile_positions, minlength=len(coords_level0)).astype(np.int32)


def make_tile_label_frame(
    *,
    file_id: str,
    slide_barcode: str,
    project_id: str,
    case_id: str,
    coords: np.ndarray,
    coords_level0: np.ndarray,
    counts: np.ndarray,
    spec: TileGridSpec,
    source: str = "pan_cancer_nuclei_seg",
    teacher_confidence: float = 1.0,
    teacher_disagreement: float = 0.0,
) -> pd.DataFrame:
    """Create the canonical tile-cellularity label table for one slide."""

    counts = np.asarray(counts, dtype=np.int32)
    coords = np.asarray(coords, dtype=np.int32)
    coords_level0 = np.asarray(coords_level0, dtype=np.int32)
    tile_area = float(spec.tile_area_mm2)
    exposure = tile_area
    out = pd.DataFrame(
        {
            "file_uuid_original": file_id,
            "barcode": slide_barcode,
            "project_id": project_id,
            "case_id": case_id,
            "embedding_index": np.arange(len(counts), dtype=np.int32),
            "tile_y": coords[:, 0],
            "tile_x": coords[:, 1],
            "tile_x_level0": coords_level0[:, 0],
            "tile_y_level0": coords_level0[:, 1],
            "tile_w_level0": np.full(len(counts), spec.tile_size_level0, dtype=np.int32),
            "tile_h_level0": np.full(len(counts), spec.tile_size_level0, dtype=np.int32),
            "mpp_x": np.full(len(counts), spec.mpp_x, dtype=np.float32),
            "mpp_y": np.full(len(counts), spec.mpp_y, dtype=np.float32),
            "target_mpp": np.full(len(counts), spec.target_mpp, dtype=np.float32),
            "tile_area_mm2": np.full(len(counts), tile_area, dtype=np.float32),
            "tissue_fraction": np.ones(len(counts), dtype=np.float32),
            "exposure_mm2": np.full(len(counts), exposure, dtype=np.float32),
            "teacher_total_nuclei": counts,
            "teacher_confidence": np.full(len(counts), teacher_confidence, dtype=np.float32),
            "teacher_disagreement": np.full(len(counts), teacher_disagreement, dtype=np.float32),
            "quality_flags": "pan_cancer_ann",
            "source": source,
        }
    )
    out["nuclei_density_per_mm2"] = out["teacher_total_nuclei"] / out["exposure_mm2"].clip(
        lower=1e-8
    )
    out["count_bin"] = count_bin_indices(out["teacher_total_nuclei"])
    return out
