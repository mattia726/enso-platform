"""Unit tests for polygon-tile weighting and Monte-Carlo ROI metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely.geometry as sg
import shapely.affinity as sa

from enso_purity.macrodissection.roi import (
    TileArrays,
    TileGrid,
    clip_polygon_to_rect,
    compute_roi_metrics,
    monte_carlo,
    point_estimates,
    polygon_area,
    polygon_hash_seed,
    tile_weights,
)


# ---------- polygon × tile weighting ---------------------------------------


def test_polygon_area_simple_square():
    poly = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert polygon_area(poly) == 100.0


def test_polygon_area_signed_orientation_agnostic():
    cw = [(0, 0), (0, 10), (10, 10), (10, 0)]
    ccw = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert polygon_area(cw) == polygon_area(ccw) == 100.0


def test_clip_polygon_to_rect_full_inside():
    poly = [(2, 2), (8, 2), (8, 8), (2, 8)]
    clipped = clip_polygon_to_rect(poly, 0, 0, 10, 10)
    assert polygon_area(clipped) == pytest.approx(36.0)


def test_clip_polygon_to_rect_half():
    poly = [(0, 0), (20, 0), (20, 10), (0, 10)]
    clipped = clip_polygon_to_rect(poly, 0, 0, 10, 10)
    assert polygon_area(clipped) == pytest.approx(100.0)


def test_clip_polygon_outside_rect():
    poly = [(20, 20), (30, 20), (30, 30), (20, 30)]
    clipped = clip_polygon_to_rect(poly, 0, 0, 10, 10)
    assert polygon_area(clipped) == 0.0


def test_tile_weights_full_tile(synthetic_grid: TileGrid):
    poly = [(0, 0), (20, 0), (20, 20), (0, 20)]
    weights = tile_weights(poly, synthetic_grid)
    assert len(weights) == 1
    assert weights[0].ix == 0 and weights[0].iy == 0
    assert weights[0].weight == pytest.approx(1.0, abs=1e-9)


def test_tile_weights_half_tile(synthetic_grid: TileGrid):
    # Polygon covers exactly half of tile (0, 0) along x.
    poly = [(0, 0), (10, 0), (10, 20), (0, 20)]
    weights = tile_weights(poly, synthetic_grid)
    assert len(weights) == 1
    assert weights[0].weight == pytest.approx(0.5, abs=1e-9)


def test_tile_weights_multiple_tiles(synthetic_grid: TileGrid):
    poly = [(0, 0), (40, 0), (40, 20), (0, 20)]
    weights = tile_weights(poly, synthetic_grid)
    assert {(w.ix, w.iy) for w in weights} == {(0, 0), (1, 0)}
    assert sum(w.weight for w in weights) == pytest.approx(2.0, abs=1e-9)


def test_tile_weights_match_shapely_oracle(synthetic_grid: TileGrid):
    """Cross-check against shapely for a randomly-rotated complex polygon."""

    rng = np.random.default_rng(0)
    base = sg.Polygon([(20, 20), (160, 20), (160, 100), (50, 110), (20, 80)])
    poly = sa.rotate(base, 23.0, origin=(80, 60), use_radians=False)
    poly_list = list(poly.exterior.coords)[:-1]
    weights = tile_weights(poly_list, synthetic_grid)
    # Recompute via shapely as ground truth.
    total_via_ours = sum(w.inter_area_thumbpx2 for w in weights)
    total_via_shapely = poly.intersection(
        sg.box(
            synthetic_grid.offset_x,
            synthetic_grid.offset_y,
            synthetic_grid.offset_x + synthetic_grid.stride_x * synthetic_grid.grid_nx,
            synthetic_grid.offset_y + synthetic_grid.stride_y * synthetic_grid.grid_ny,
        )
    ).area
    assert total_via_ours == pytest.approx(total_via_shapely, rel=1e-6, abs=1e-6)


# ---------- point estimates and Monte-Carlo --------------------------------


def test_point_estimates_constant_grid(constant_tiles: TileArrays):
    grid = constant_tiles.grid
    poly = [(0, 0), (grid.stride_x * grid.grid_nx, 0),
            (grid.stride_x * grid.grid_nx, grid.stride_y * grid.grid_ny),
            (0, grid.stride_y * grid.grid_ny)]
    weights = tile_weights(poly, grid)
    est = point_estimates(weights, constant_tiles)
    assert est.purity == pytest.approx(0.4)
    n_tiles = grid.grid_nx * grid.grid_ny
    assert est.total_nuclei == pytest.approx(100.0 * n_tiles)
    assert est.tumor_nuclei == pytest.approx(40.0 * n_tiles)


def test_monte_carlo_deterministic(constant_tiles: TileArrays):
    poly = [(0, 0), (60, 0), (60, 40), (0, 40)]
    weights = tile_weights(poly, constant_tiles.grid)
    m1 = monte_carlo(
        weights,
        constant_tiles,
        thresholds_purity_min=0.2,
        thresholds_tumor_cells_min=200,
        n_samples=200,
        seed=12345,
    )
    m2 = monte_carlo(
        weights,
        constant_tiles,
        thresholds_purity_min=0.2,
        thresholds_tumor_cells_min=200,
        n_samples=200,
        seed=12345,
    )
    assert m1.purity.median == m2.purity.median
    assert m1.total_nuclei.median == m2.total_nuclei.median
    assert m1.adequacy_probability == m2.adequacy_probability


def test_monte_carlo_low_variance_collapses_to_point(make_tiles, synthetic_grid):
    tiles = make_tiles(
        synthetic_grid,
        purity_value=0.5,
        nuclei_value=120.0,
        purity_sd=0.0,
        nuclei_sd=0.0,
    )
    poly = [(0, 0), (60, 0), (60, 40), (0, 40)]
    weights = tile_weights(poly, tiles.grid)
    metrics = monte_carlo(
        weights,
        tiles,
        thresholds_purity_min=0.2,
        thresholds_tumor_cells_min=200,
        n_samples=200,
        seed=42,
    )
    assert metrics.purity.median == pytest.approx(0.5, abs=1e-6)
    assert metrics.purity.low == pytest.approx(metrics.purity.high, abs=1e-6)
    assert metrics.adequacy_probability == 1.0


def test_compute_roi_metrics_seed_stability(constant_tiles: TileArrays):
    poly = [(2.0, 3.0), (47.0, 4.0), (45.0, 38.0), (1.0, 35.0)]
    a = compute_roi_metrics(
        poly, constant_tiles, thresholds_purity_min=0.2, thresholds_tumor_cells_min=200
    )
    b = compute_roi_metrics(
        poly, constant_tiles, thresholds_purity_min=0.2, thresholds_tumor_cells_min=200
    )
    assert a.purity.median == b.purity.median


def test_polygon_hash_seed_unique_for_different_polygons():
    a = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)]
    b = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.000001)]
    assert polygon_hash_seed(a) != polygon_hash_seed(b)


def test_monte_carlo_empty_polygon_returns_zero(constant_tiles: TileArrays):
    metrics = compute_roi_metrics(
        [(1000.0, 1000.0), (1001.0, 1000.0), (1001.0, 1001.0)],
        constant_tiles,
        thresholds_purity_min=0.2,
        thresholds_tumor_cells_min=200,
    )
    assert metrics.n_tiles == 0
    assert metrics.adequacy_probability == 0.0


def test_monte_carlo_handles_nan_tiles(make_tiles, synthetic_grid):
    nan_mask = np.zeros((synthetic_grid.grid_ny, synthetic_grid.grid_nx), dtype=bool)
    nan_mask[0, 0] = True
    tiles = make_tiles(
        synthetic_grid,
        purity_value=0.5,
        nuclei_value=100.0,
        purity_sd=0.0,
        nuclei_sd=0.0,
        nan_mask=nan_mask,
    )
    poly = [(0, 0), (40, 0), (40, 40), (0, 40)]
    weights = tile_weights(poly, synthetic_grid)
    est = point_estimates(weights, tiles)
    # tile (0, 0) is NaN; remaining three tiles contribute 100 nuclei each.
    assert est.total_nuclei == pytest.approx(300.0)
    assert est.tumor_nuclei == pytest.approx(150.0)
