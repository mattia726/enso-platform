"""Tile-grid stride detection for pre-rendered heatmap PNGs.

The ML export pipeline rasterizes the per-tile predictions to a
``(grid_ny, grid_nx)`` RGBA buffer and then resizes it with
``PIL.Image.NEAREST`` to the base image dimensions. The result is a PNG where
each *original tile* occupies a rectangle of approximately ``S`` base pixels
on a side. Because the resize is integer-rounded, some adjacent tiles
receive blocks one pixel wider/taller than others.

The recovery algorithm works in two passes:

1. Locate every position where adjacent base pixels differ along each axis;
   these positions are tile boundaries.
2. Compute the **mode** of the gaps between consecutive change positions.
   For a NEAREST-resized rendered grid that mode is the tile stride (modulo
   a ±1 px wobble from integer rounding). The mode is robust to "missing"
   boundaries that disappear when two adjacent tiles happen to render the
   same colour.

A brute-force search with a ±1 px tolerance per boundary handles the rare
cases where the mode comes out unreasonable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """Recovered tile-grid geometry on the base image."""

    grid_nx: int
    grid_ny: int
    stride_x: int  # base pixels per tile, X axis
    stride_y: int  # base pixels per tile, Y axis
    offset_x: int = 0  # base pixel of the first tile (top-left corner)
    offset_y: int = 0

    def tile_rect(self, ix: int, iy: int) -> tuple[int, int, int, int]:
        """Return (x0, y0, x1, y1) base-pixel bounds of tile (ix, iy)."""

        x0 = self.offset_x + ix * self.stride_x
        y0 = self.offset_y + iy * self.stride_y
        return x0, y0, x0 + self.stride_x, y0 + self.stride_y


def _column_change_positions(arr: np.ndarray) -> np.ndarray:
    """Return the sorted positions where ``arr`` changes along axis=1.

    Always includes position 0 so the left edge is part of the lattice.
    """

    if arr.ndim != 3:
        raise ValueError("Expected (H, W, C) image")
    width = arr.shape[1]
    if width <= 1:
        return np.array([0], dtype=np.int64)
    diff = arr[:, 1:, :].astype(np.int32) - arr[:, :-1, :].astype(np.int32)
    any_change = np.any(diff != 0, axis=(0, 2))
    pos = np.flatnonzero(any_change) + 1
    return np.concatenate(([0], pos.astype(np.int64)))


def _detect_axis_stride(
    positions: np.ndarray,
    min_stride: int,
    max_stride: int,
    *,
    length: int,
) -> tuple[int, int]:
    """Return ``(stride, offset)`` for one axis."""

    if positions.size < 2:
        return length + 1, 0

    gaps = np.diff(positions).astype(np.int64)
    median_gap = float(np.median(gaps))
    keep = gaps[gaps <= max(1, int(round(3 * median_gap)))]
    if keep.size:
        values, counts = np.unique(keep, return_counts=True)
        mode_gap = int(values[np.argmax(counts)])
    else:
        mode_gap = int(median_gap)

    if min_stride <= mode_gap <= max_stride:
        offset = int(positions[0]) % max(mode_gap, 1)
        return mode_gap, offset

    best_score = -1
    best_stride = max(min_stride, 1)
    best_offset = int(positions[0])
    pos_arr = positions.astype(np.int64)
    for stride in range(min_stride, min(max_stride, length) + 1):
        offset = int(pos_arr[0]) % stride
        rem = (pos_arr - offset) % stride
        score = int(np.sum((rem == 0) | (rem == 1) | (rem == stride - 1)))
        if score > best_score or (score == best_score and stride > best_stride):
            best_score = score
            best_stride = stride
            best_offset = offset
    return best_stride, best_offset


def detect_grid(
    rgba: np.ndarray,
    *,
    min_stride: int = 4,
    max_stride: int | None = None,
) -> GridSpec:
    """Recover the tile grid that explains the rendered heatmap.

    Parameters
    ----------
    rgba
        ``(H, W, 4)`` uint8 RGBA image of the rendered heatmap.
    min_stride
        Minimum search stride. Defaults to 4 base-pixels.
    max_stride
        Maximum search stride. Defaults to ``min(W, H) // 2``.
    """

    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError(f"Expected RGBA, got shape {rgba.shape}")
    h, w = rgba.shape[:2]
    if max_stride is None:
        max_stride = max(min_stride + 1, min(w, h) // 2)

    col_pos = _column_change_positions(rgba)
    row_pos = _column_change_positions(np.transpose(rgba, (1, 0, 2)))

    stride_x, offset_x = _detect_axis_stride(col_pos, min_stride, max_stride, length=w)
    stride_y, offset_y = _detect_axis_stride(row_pos, min_stride, max_stride, length=h)

    # Tiles are square in the live pipeline. When the two axes disagree by
    # more than 10%, harmonise to whichever stride divides its image axis
    # most cleanly.
    if abs(stride_x - stride_y) / max(stride_x, stride_y, 1) > 0.1:
        candidates = sorted(
            {stride_x, stride_y},
            key=lambda s: (
                min((w - offset_x) % s, s - (w - offset_x) % s)
                + min((h - offset_y) % s, s - (h - offset_y) % s)
            ),
        )
        chosen = candidates[0]
        stride_x = stride_y = chosen
        offset_x = offset_x % chosen
        offset_y = offset_y % chosen

    grid_nx = max(int((w - offset_x) // max(stride_x, 1)), 1)
    grid_ny = max(int((h - offset_y) // max(stride_y, 1)), 1)

    return GridSpec(
        grid_nx=grid_nx,
        grid_ny=grid_ny,
        stride_x=int(stride_x),
        stride_y=int(stride_y),
        offset_x=int(offset_x),
        offset_y=int(offset_y),
    )


def per_tile_modal_color(
    rgba: np.ndarray,
    spec: GridSpec,
    *,
    sample_box: int = 3,
) -> np.ndarray:
    """Return the per-tile representative RGBA color as a ``(ny, nx, 4)`` array.

    Each tile is summarized by averaging a small central window. This is
    robust against the one-pixel staircase artefact that occasionally
    appears when the NEAREST resize lands on a fractional position.
    """

    ny, nx = spec.grid_ny, spec.grid_nx
    out = np.zeros((ny, nx, 4), dtype=np.float64)
    h, w = rgba.shape[:2]
    for iy in range(ny):
        for ix in range(nx):
            x0, y0, x1, y1 = spec.tile_rect(ix, iy)
            x1 = min(x1, w)
            y1 = min(y1, h)
            if x1 <= x0 or y1 <= y0:
                continue
            cx0 = (x0 + x1) // 2 - sample_box // 2
            cy0 = (y0 + y1) // 2 - sample_box // 2
            cx1 = cx0 + sample_box
            cy1 = cy0 + sample_box
            cx0, cy0 = max(cx0, x0), max(cy0, y0)
            cx1, cy1 = min(cx1, x1), min(cy1, y1)
            window = rgba[cy0:cy1, cx0:cx1, :].astype(np.float64)
            if window.size == 0:
                continue
            out[iy, ix] = window.mean(axis=(0, 1))
    return np.clip(out, 0.0, 255.0).astype(np.uint8)
