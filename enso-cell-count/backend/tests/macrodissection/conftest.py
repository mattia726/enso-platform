"""Shared fixtures for macrodissection tests."""

from __future__ import annotations

import numpy as np
import pytest

from enso_purity.macrodissection.roi import TileArrays, TileGrid


@pytest.fixture()
def synthetic_grid() -> TileGrid:
    """A small synthetic tile grid (10x6 tiles, 20px stride)."""

    return TileGrid(
        stride_x=20.0,
        stride_y=20.0,
        offset_x=0.0,
        offset_y=0.0,
        grid_nx=10,
        grid_ny=6,
        tile_area_mm2=0.012544,
    )


def _make_tile_arrays(
    grid: TileGrid,
    *,
    purity_value: float,
    nuclei_value: float,
    purity_sd: float = 0.0,
    nuclei_sd: float = 0.0,
    tissue_fraction: float = 1.0,
    nan_mask: np.ndarray | None = None,
) -> TileArrays:
    shape = (grid.grid_ny, grid.grid_nx)
    purity = np.full(shape, purity_value, dtype=np.float32)
    purity_sd_a = np.full(shape, purity_sd, dtype=np.float32)
    nuclei = np.full(shape, nuclei_value, dtype=np.float32)
    nuclei_sd_a = np.full(shape, nuclei_sd, dtype=np.float32)
    tissue = np.full(shape, tissue_fraction, dtype=np.float32)
    if nan_mask is not None:
        purity[nan_mask] = np.float32(np.nan)
        nuclei[nan_mask] = np.float32(np.nan)
        tissue[nan_mask] = 0.0
    return TileArrays(
        grid=grid,
        purity=purity,
        purity_sd=purity_sd_a,
        nuclei=nuclei,
        nuclei_sd=nuclei_sd_a,
        tissue_fraction=tissue,
    )


@pytest.fixture()
def constant_tiles(synthetic_grid: TileGrid) -> TileArrays:
    """Tile grid with constant purity=0.4 and 100 nuclei/tile."""

    return _make_tile_arrays(
        synthetic_grid,
        purity_value=0.4,
        nuclei_value=100.0,
        purity_sd=0.05,
        nuclei_sd=5.0,
    )


@pytest.fixture()
def make_tiles():
    """Factory fixture for arbitrary tile arrays."""

    return _make_tile_arrays
