"""Unit tests for tile-grid stride detection."""

from __future__ import annotations

import numpy as np

from enso_purity.macrodissection.grid_detect import (
    detect_grid,
    per_tile_modal_color,
)


def _make_synthetic_mask(
    grid_nx: int, grid_ny: int, stride: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Render a piecewise-constant RGBA image with a known tile grid.

    Returns the rendered image and the per-tile RGBA palette used.
    """

    rng = np.random.default_rng(seed)
    palette = rng.integers(20, 235, size=(grid_ny, grid_nx, 3), dtype=np.uint8)
    alpha = (rng.integers(0, 2, size=(grid_ny, grid_nx)) * 255).astype(np.uint8)
    rgba_grid = np.concatenate([palette, alpha[..., None]], axis=-1)
    # Resize via repeat (== NEAREST upscale).
    big = np.repeat(np.repeat(rgba_grid, stride, axis=0), stride, axis=1)
    return big, rgba_grid


def test_detect_grid_recovers_exact_stride():
    """A synthetic 17×9 grid at 12 px stride is recovered exactly."""

    img, _ = _make_synthetic_mask(17, 9, 12, seed=42)
    spec = detect_grid(img)
    assert spec.stride_x == 12
    assert spec.stride_y == 12
    assert spec.grid_nx == 17
    assert spec.grid_ny == 9


def test_detect_grid_handles_various_strides():
    for stride in (8, 14, 20, 28, 36):
        img, _ = _make_synthetic_mask(11, 7, stride, seed=stride)
        spec = detect_grid(img, min_stride=4)
        assert spec.stride_x == stride, f"stride_x mismatch at stride={stride}"
        assert spec.stride_y == stride, f"stride_y mismatch at stride={stride}"


def test_per_tile_modal_color_roundtrip():
    """Sampling the centre of each tile recovers the source palette."""

    img, palette = _make_synthetic_mask(13, 7, 16, seed=7)
    spec = detect_grid(img)
    sampled = per_tile_modal_color(img, spec, sample_box=3)
    # Palette and detected geometry should match shapes.
    assert sampled.shape == palette.shape
    np.testing.assert_array_equal(sampled[..., :3], palette[..., :3])
    np.testing.assert_array_equal(sampled[..., 3], palette[..., 3])


def test_detect_grid_on_real_asset_if_present(tmp_path):
    """If the demo case_1 mask is checked in, the detector recovers a
    sensible grid (square stride, dimensions that fit the base image)."""

    import numpy as np
    from pathlib import Path
    from PIL import Image

    asset = (
        Path(__file__).resolve().parents[3]
        / "frontend"
        / "public"
        / "cases"
        / "case_1_mask.png"
    )
    if not asset.exists():
        # Repo without the demo PNGs; nothing to assert.
        return
    rgba = np.array(Image.open(asset).convert("RGBA"))
    h, w = rgba.shape[:2]
    spec = detect_grid(rgba)
    # Stride must be square (or near-square) and reasonably small.
    assert spec.stride_x == spec.stride_y, (spec.stride_x, spec.stride_y)
    assert 4 <= spec.stride_x <= min(w, h) // 8
    assert 0 <= spec.offset_x < spec.stride_x
    assert 0 <= spec.offset_y < spec.stride_y
    # Grid must cover most of the image.
    assert spec.grid_nx * spec.stride_x <= w
    assert spec.grid_ny * spec.stride_y <= h
    assert spec.grid_nx * spec.stride_x >= w - spec.stride_x
    assert spec.grid_ny * spec.stride_y >= h - spec.stride_y
