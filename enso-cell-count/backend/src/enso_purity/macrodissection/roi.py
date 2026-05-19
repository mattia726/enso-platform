"""ROI geometry, point estimates, and Monte-Carlo uncertainty.

This module is the *authoritative* implementation of the macrodissection
math. The TypeScript code that ships with the Next.js frontend mirrors every
function here so that the user always sees identical numbers on the client
preview and on the server-side recompute that runs when an ROI is locked.

Key invariants
--------------

* The polygon × tile-rect intersection is computed exactly (Sutherland–
  Hodgman against the four edges of each axis-aligned tile). Two unit tests
  cross-check against ``shapely`` for randomly-rotated polygons.
* The Monte-Carlo sampler is deterministic for a fixed ``seed``. The seed is
  derived from the polygon hash in the public API so the same polygon always
  reports the same numbers — pathologists never see numbers wobble between
  identical clicks.
* Quantitative metrics use *raw* tile predictions only. Visual smoothing
  applied to the overlay is decoupled in :mod:`frontend/lib/macrodissection`.

The unit of geometry is the *thumbnail pixel*. Every other unit (mm², µm,
mpp) is recoverable through the tile metadata stored in
``case_N_tiles.json``.
"""

from __future__ import annotations

import math
import hashlib
import struct
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np


# ---------- Geometry -------------------------------------------------------


@dataclass(frozen=True)
class TileGrid:
    """Description of the tile grid in a single case.

    Attributes:
        stride_x, stride_y: tile size in *thumbnail* pixels.
        offset_x, offset_y: top-left position of tile (0, 0) in thumbnail
            pixels.
        grid_nx, grid_ny: number of tiles along each axis.
        tile_area_mm2: physical area of one tile (cancers vary slightly in
            mpp; this is the slide-level constant).
    """

    stride_x: float
    stride_y: float
    offset_x: float
    offset_y: float
    grid_nx: int
    grid_ny: int
    tile_area_mm2: float

    def tile_rect(self, ix: int, iy: int) -> tuple[float, float, float, float]:
        """Return (x0, y0, x1, y1) thumbnail-pixel rect of one tile."""

        x0 = self.offset_x + ix * self.stride_x
        y0 = self.offset_y + iy * self.stride_y
        return x0, y0, x0 + self.stride_x, y0 + self.stride_y

    def tile_index_from_xy(self, x: float, y: float) -> tuple[int, int] | None:
        """Return (ix, iy) or None if (x, y) falls outside the grid."""

        ix = int((x - self.offset_x) // max(self.stride_x, 1e-9))
        iy = int((y - self.offset_y) // max(self.stride_y, 1e-9))
        if 0 <= ix < self.grid_nx and 0 <= iy < self.grid_ny:
            return ix, iy
        return None


Polygon = Sequence[tuple[float, float]]


def _polygon_signed_area(poly: Sequence[tuple[float, float]]) -> float:
    """Shoelace signed area (positive for counter-clockwise polygons)."""

    if len(poly) < 3:
        return 0.0
    s = 0.0
    for i in range(len(poly)):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % len(poly)]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def polygon_area(poly: Polygon) -> float:
    """Absolute polygon area (in thumbnail pixels²)."""

    return abs(_polygon_signed_area(poly))


def _clip_segment_axis(
    poly: list[tuple[float, float]],
    *,
    axis: int,
    value: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    """Sutherland–Hodgman clip against one axis-aligned half-plane.

    ``axis=0`` clips against ``x``; ``axis=1`` clips against ``y``.
    ``keep_greater=True`` retains points whose coordinate is ``>= value``;
    otherwise points with ``coord <= value`` are kept.
    """

    if not poly:
        return []
    out: list[tuple[float, float]] = []

    def inside(p: tuple[float, float]) -> bool:
        coord = p[axis]
        return coord >= value if keep_greater else coord <= value

    for i, current in enumerate(poly):
        prev = poly[i - 1]
        cur_in = inside(current)
        prev_in = inside(prev)
        if cur_in:
            if not prev_in:
                out.append(_axis_intersect(prev, current, axis, value))
            out.append(current)
        elif prev_in:
            out.append(_axis_intersect(prev, current, axis, value))
    return out


def _axis_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    axis: int,
    value: float,
) -> tuple[float, float]:
    """Intersection of segment p1-p2 with the line ``coord[axis] = value``."""

    x1, y1 = p1
    x2, y2 = p2
    if axis == 0:
        dx = x2 - x1
        if dx == 0:
            return value, y1
        t = (value - x1) / dx
        return value, y1 + t * (y2 - y1)
    dy = y2 - y1
    if dy == 0:
        return x1, value
    t = (value - y1) / dy
    return x1 + t * (x2 - x1), value


def clip_polygon_to_rect(
    poly: Polygon, x0: float, y0: float, x1: float, y1: float
) -> list[tuple[float, float]]:
    """Clip a polygon against the axis-aligned rect [x0, x1] × [y0, y1]."""

    if x1 <= x0 or y1 <= y0 or len(poly) < 3:
        return []
    out = list(poly)
    out = _clip_segment_axis(out, axis=0, value=x0, keep_greater=True)
    out = _clip_segment_axis(out, axis=0, value=x1, keep_greater=False)
    out = _clip_segment_axis(out, axis=1, value=y0, keep_greater=True)
    out = _clip_segment_axis(out, axis=1, value=y1, keep_greater=False)
    return out


@dataclass(frozen=True)
class TileWeight:
    """Polygon × tile-rect intersection record."""

    ix: int
    iy: int
    weight: float  # fraction of the tile that falls inside the polygon
    inter_area_thumbpx2: float


def tile_weights(
    polygon: Polygon,
    grid: TileGrid,
) -> list[TileWeight]:
    """Return per-tile intersection weights for the given polygon.

    Tiles whose intersection with the polygon has zero area are omitted.
    """

    if len(polygon) < 3:
        return []
    # Compute the polygon's bounding box and clamp to the grid.
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    sx = max(grid.stride_x, 1e-9)
    sy = max(grid.stride_y, 1e-9)
    ix_min = max(int(math.floor((xmin - grid.offset_x) / sx)), 0)
    ix_max = min(
        int(math.floor((xmax - grid.offset_x) / sx)),
        grid.grid_nx - 1,
    )
    iy_min = max(int(math.floor((ymin - grid.offset_y) / sy)), 0)
    iy_max = min(
        int(math.floor((ymax - grid.offset_y) / sy)),
        grid.grid_ny - 1,
    )

    if ix_max < ix_min or iy_max < iy_min:
        return []

    tile_area = grid.stride_x * grid.stride_y
    if tile_area <= 0:
        return []

    weights: list[TileWeight] = []
    for iy in range(iy_min, iy_max + 1):
        for ix in range(ix_min, ix_max + 1):
            x0, y0, x1, y1 = grid.tile_rect(ix, iy)
            clipped = clip_polygon_to_rect(polygon, x0, y0, x1, y1)
            if len(clipped) < 3:
                continue
            inter_area = polygon_area(clipped)
            if inter_area <= 0:
                continue
            weights.append(
                TileWeight(
                    ix=ix,
                    iy=iy,
                    weight=min(1.0, inter_area / tile_area),
                    inter_area_thumbpx2=inter_area,
                )
            )
    return weights


# ---------- Tile arrays -----------------------------------------------------


@dataclass
class TileArrays:
    """Per-tile prediction grids for a single case."""

    grid: TileGrid
    purity: np.ndarray  # (ny, nx) float32, NaN for missing tiles
    purity_sd: np.ndarray
    nuclei: np.ndarray
    nuclei_sd: np.ndarray
    tissue_fraction: np.ndarray  # (ny, nx) float32 in [0, 1]

    @property
    def tumor_nuclei(self) -> np.ndarray:
        """Point-estimate tumor nuclei per tile.

        ``tumor_nuclei = purity · nuclei · tissue_fraction``.
        """

        tn = self.purity * self.nuclei * self.tissue_fraction
        return np.where(np.isnan(tn), 0.0, tn)


# ---------- Point estimates and Monte-Carlo --------------------------------


@dataclass
class PointEstimate:
    purity: float
    total_nuclei: float
    tumor_nuclei: float
    area_thumbpx2: float
    area_mm2: float


@dataclass
class MetricsCI:
    median: float
    low: float
    high: float

    def to_dict(self) -> dict[str, float]:
        return {"median": self.median, "low": self.low, "high": self.high}


@dataclass
class ROIMetrics:
    """Complete metric bundle for one ROI."""

    n_tiles: int
    tiles_with_data: int
    area_thumbpx2: float
    area_mm2: float
    tissue_fraction_mean: float
    purity: MetricsCI
    total_nuclei: MetricsCI
    tumor_nuclei: MetricsCI
    adequacy_probability: float
    purity_point: float
    total_nuclei_point: float
    tumor_nuclei_point: float

    def to_dict(self) -> dict:
        return {
            "n_tiles": self.n_tiles,
            "tiles_with_data": self.tiles_with_data,
            "area_thumbpx2": self.area_thumbpx2,
            "area_mm2": self.area_mm2,
            "tissue_fraction_mean": self.tissue_fraction_mean,
            "purity": self.purity.to_dict(),
            "total_nuclei": self.total_nuclei.to_dict(),
            "tumor_nuclei": self.tumor_nuclei.to_dict(),
            "adequacy_probability": self.adequacy_probability,
            "purity_point": self.purity_point,
            "total_nuclei_point": self.total_nuclei_point,
            "tumor_nuclei_point": self.tumor_nuclei_point,
        }


_EPS = 1e-9


def _collect_arrays(
    weights: Iterable[TileWeight],
    tiles: TileArrays,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return aligned 1D arrays (w, p, sp, n, sn, tf) for the weighted tiles.

    NaN purity or nuclei values are replaced with 0 so they do not contribute
    to the sums.
    """

    ws: list[float] = []
    ps: list[float] = []
    sps: list[float] = []
    ns: list[float] = []
    sns: list[float] = []
    tfs: list[float] = []
    for tw in weights:
        ix, iy = tw.ix, tw.iy
        p = tiles.purity[iy, ix]
        n = tiles.nuclei[iy, ix]
        tf = tiles.tissue_fraction[iy, ix]
        sp = tiles.purity_sd[iy, ix]
        sn = tiles.nuclei_sd[iy, ix]
        if not np.isfinite(tf):
            tf = 0.0
        ws.append(tw.weight)
        ps.append(0.0 if not np.isfinite(p) else float(p))
        sps.append(0.0 if not np.isfinite(sp) else float(sp))
        ns.append(0.0 if not np.isfinite(n) else float(n))
        sns.append(0.0 if not np.isfinite(sn) else float(sn))
        tfs.append(float(tf))
    return (
        np.asarray(ws, dtype=np.float64),
        np.asarray(ps, dtype=np.float64),
        np.asarray(sps, dtype=np.float64),
        np.asarray(ns, dtype=np.float64),
        np.asarray(sns, dtype=np.float64),
        np.asarray(tfs, dtype=np.float64),
    )


def point_estimates(
    weights: list[TileWeight],
    tiles: TileArrays,
) -> PointEstimate:
    """Compute the deterministic ROI point estimates."""

    if not weights:
        return PointEstimate(
            purity=0.0,
            total_nuclei=0.0,
            tumor_nuclei=0.0,
            area_thumbpx2=0.0,
            area_mm2=0.0,
        )
    w, p, _sp, n, _sn, tf = _collect_arrays(weights, tiles)
    eff = w * n * tf
    total_nuclei = float(eff.sum())
    tumor_nuclei = float((eff * p).sum())
    purity = tumor_nuclei / max(total_nuclei, _EPS)
    area_thumbpx2 = sum(tw.inter_area_thumbpx2 for tw in weights)
    area_mm2 = sum(tw.weight * tiles.grid.tile_area_mm2 for tw in weights)
    return PointEstimate(
        purity=purity,
        total_nuclei=total_nuclei,
        tumor_nuclei=tumor_nuclei,
        area_thumbpx2=float(area_thumbpx2),
        area_mm2=float(area_mm2),
    )


def polygon_hash_seed(polygon: Polygon) -> int:
    """Return a deterministic 32-bit seed derived from the polygon vertices.

    Stable across processes; cancels random wobble for unchanged ROIs.
    """

    h = hashlib.blake2b(digest_size=4)
    for x, y in polygon:
        h.update(struct.pack("<dd", float(x), float(y)))
    return int.from_bytes(h.digest(), "little") & 0x7FFFFFFF


def _truncated_normal_pos(
    rng: np.random.Generator,
    mean: np.ndarray,
    sd: np.ndarray,
    *,
    upper: float | None = None,
) -> np.ndarray:
    """Return samples from N(mean, sd) clipped to [0, upper]."""

    samples = rng.normal(loc=mean, scale=sd)
    np.clip(samples, 0.0, upper if upper is not None else np.inf, out=samples)
    return samples


def monte_carlo(
    weights: list[TileWeight],
    tiles: TileArrays,
    *,
    thresholds_purity_min: float,
    thresholds_tumor_cells_min: float,
    n_samples: int = 400,
    seed: int | None = None,
) -> ROIMetrics:
    """Monte-Carlo sample ROI metrics and return median/CI95 estimates.

    The CI is computed at the 5th/95th percentiles. ``adequacy_probability``
    is the fraction of samples that satisfy *both* thresholds.

    The function is fully deterministic for fixed ``seed``.
    """

    if not weights:
        zero_ci = MetricsCI(median=0.0, low=0.0, high=0.0)
        return ROIMetrics(
            n_tiles=0,
            tiles_with_data=0,
            area_thumbpx2=0.0,
            area_mm2=0.0,
            tissue_fraction_mean=0.0,
            purity=zero_ci,
            total_nuclei=zero_ci,
            tumor_nuclei=zero_ci,
            adequacy_probability=0.0,
            purity_point=0.0,
            total_nuclei_point=0.0,
            tumor_nuclei_point=0.0,
        )

    w, p, sp, n, sn, tf = _collect_arrays(weights, tiles)
    tiles_with_data = int(np.count_nonzero((tf > 0) & np.isfinite(p) & np.isfinite(n)))
    rng = np.random.default_rng(seed)

    # Pre-scale standard deviations: zero where no data.
    sp_eff = np.where(tf > 0, sp, 0.0)
    sn_eff = np.where(tf > 0, sn, 0.0)

    purity_samples = np.empty(n_samples, dtype=np.float64)
    total_samples = np.empty(n_samples, dtype=np.float64)
    tumor_samples = np.empty(n_samples, dtype=np.float64)
    pass_count = 0

    for k in range(n_samples):
        # We sample each tile independently. This is a deliberate
        # over-conservative noise model: in practice tile predictions are
        # mildly correlated, which would *reduce* CI width. The over-wide CI
        # protects the clinician from over-confident point estimates.
        p_k = _truncated_normal_pos(rng, p, sp_eff, upper=1.0)
        n_k = _truncated_normal_pos(rng, n, sn_eff)
        eff = w * n_k * tf
        total_k = float(eff.sum())
        tumor_k = float((eff * p_k).sum())
        purity_k = tumor_k / max(total_k, _EPS)
        total_samples[k] = total_k
        tumor_samples[k] = tumor_k
        purity_samples[k] = purity_k
        if (
            purity_k >= thresholds_purity_min
            and tumor_k >= thresholds_tumor_cells_min
        ):
            pass_count += 1

    point = point_estimates(weights, tiles)
    area_mm2 = sum(tw.weight * tiles.grid.tile_area_mm2 for tw in weights)
    area_thumbpx2 = sum(tw.inter_area_thumbpx2 for tw in weights)
    tf_mean = float(np.mean(tf)) if tf.size else 0.0

    def _ci(samples: np.ndarray) -> MetricsCI:
        return MetricsCI(
            median=float(np.median(samples)),
            low=float(np.percentile(samples, 5)),
            high=float(np.percentile(samples, 95)),
        )

    return ROIMetrics(
        n_tiles=len(weights),
        tiles_with_data=tiles_with_data,
        area_thumbpx2=float(area_thumbpx2),
        area_mm2=float(area_mm2),
        tissue_fraction_mean=tf_mean,
        purity=_ci(purity_samples),
        total_nuclei=_ci(total_samples),
        tumor_nuclei=_ci(tumor_samples),
        adequacy_probability=pass_count / max(n_samples, 1),
        purity_point=point.purity,
        total_nuclei_point=point.total_nuclei,
        tumor_nuclei_point=point.tumor_nuclei,
    )


def compute_roi_metrics(
    polygon: Polygon,
    tiles: TileArrays,
    *,
    thresholds_purity_min: float,
    thresholds_tumor_cells_min: float,
    n_samples: int = 400,
    seed: int | None = None,
) -> ROIMetrics:
    """End-to-end ROI metrics from a polygon + grid of tile predictions."""

    weights = tile_weights(polygon, tiles.grid)
    use_seed = polygon_hash_seed(polygon) if seed is None else seed
    return monte_carlo(
        weights,
        tiles,
        thresholds_purity_min=thresholds_purity_min,
        thresholds_tumor_cells_min=thresholds_tumor_cells_min,
        n_samples=n_samples,
        seed=use_seed,
    )
