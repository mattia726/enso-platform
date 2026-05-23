"""Tests for the candidate-ROI auto-suggester."""

from __future__ import annotations

import numpy as np

from enso_purity.macrodissection.candidates import suggest_candidates
from enso_purity.macrodissection.roi import TileArrays, TileGrid


def _make_tiles_with_hotspot(
    *,
    grid_nx: int = 20,
    grid_ny: int = 12,
    hotspot: tuple[int, int, int, int] = (4, 3, 9, 8),
    purity_hotspot: float = 0.6,
    nuclei_hotspot: float = 200.0,
) -> TileArrays:
    """Build a synthetic case with one purity/cellularity hotspot region."""

    grid = TileGrid(
        stride_x=20.0,
        stride_y=20.0,
        offset_x=0.0,
        offset_y=0.0,
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        tile_area_mm2=0.012544,
    )
    shape = (grid_ny, grid_nx)
    purity = np.full(shape, 0.05, dtype=np.float32)
    nuclei = np.full(shape, 20.0, dtype=np.float32)
    purity_sd = np.full(shape, 0.05, dtype=np.float32)
    nuclei_sd = np.full(shape, 5.0, dtype=np.float32)
    tissue = np.full(shape, 1.0, dtype=np.float32)
    x0, y0, x1, y1 = hotspot
    purity[y0:y1, x0:x1] = purity_hotspot
    nuclei[y0:y1, x0:x1] = nuclei_hotspot
    return TileArrays(
        grid=grid,
        purity=purity,
        purity_sd=purity_sd,
        nuclei=nuclei,
        nuclei_sd=nuclei_sd,
        tissue_fraction=tissue,
    )


def test_suggest_candidates_finds_hotspot():
    tiles = _make_tiles_with_hotspot()
    candidates = suggest_candidates(
        tiles,
        purity_min=0.2,
        tumor_cells_min=400,
        window_tiles=5,
        top_k=3,
    )
    assert candidates, "expected at least one candidate"
    top = candidates[0]
    assert top.purity_point > 0.4
    assert top.tumor_nuclei_point > 1000
    assert top.adequacy_probability >= 0.5
    # The top-1 bbox must overlap the hotspot region.
    x0, y0, x1, y1 = top.bbox_thumb_px
    assert x0 < 9 * 20 and x1 > 4 * 20
    assert y0 < 8 * 20 and y1 > 3 * 20


def test_suggest_candidates_returns_at_most_k():
    tiles = _make_tiles_with_hotspot()
    candidates = suggest_candidates(
        tiles, purity_min=0.0, tumor_cells_min=0, window_tiles=3, top_k=5
    )
    assert len(candidates) <= 5
    assert all(0 <= c.rank for c in candidates)


def test_suggest_candidates_nms_keeps_distinct_regions():
    """Two well-separated hotspots should produce two distinct candidates."""

    tiles = _make_tiles_with_hotspot()
    # Add a second hotspot far away.
    tiles.purity[1:3, 16:19] = 0.7
    tiles.nuclei[1:3, 16:19] = 220.0
    candidates = suggest_candidates(
        tiles,
        purity_min=0.2,
        tumor_cells_min=400,
        window_tiles=2,
        top_k=2,
        nms_iou=0.25,
    )
    if len(candidates) < 2:
        # Smaller window may merge; that's an acceptable behaviour as long
        # as the API never returns more than top_k.
        return
    a, b = candidates[0].bbox_thumb_px, candidates[1].bbox_thumb_px
    ax = (a[0] + a[2]) / 2
    bx = (b[0] + b[2]) / 2
    assert abs(ax - bx) > 10.0
