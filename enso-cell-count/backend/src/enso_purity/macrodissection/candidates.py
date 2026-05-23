"""Candidate macrodissection ROI auto-suggestion.

The frontend exposes a "candidate areas" panel that lists, for each slide,
the top few rectangular regions that are most promising for macrodissection.
Pathologists can click a candidate to load it as an editable polygon and
then refine the boundary.

The candidate scoring is intentionally simple to keep behaviour predictable:

* Build a sliding window of size ``window_tiles × window_tiles`` over the
  tile grid.
* For each window with non-zero tissue, compute the deterministic point
  estimates (``purity``, ``total_nuclei``, ``tumor_nuclei``).
* Compute a quick adequacy probability proxy from the per-tile σ values
  using a closed-form Gaussian approximation (Monte-Carlo would be too slow
  to score thousands of windows). The result is later confirmed by the full
  MC engine when the user clicks the candidate.
* Rank windows by ``adequacy_probability``, breaking ties on
  ``tumor_nuclei`` (more material is better).
* Apply rectangle non-max-suppression so the top K do not all hug the same
  hotspot.

The function always returns *axis-aligned rectangles* as polygons in
thumbnail-pixel coordinates; the frontend reshapes them into editable
polygons.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .roi import Polygon, TileArrays


@dataclass
class CandidateROI:
    rank: int
    score: float
    bbox_thumb_px: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    polygon: list[tuple[float, float]]
    purity_point: float
    total_nuclei_point: float
    tumor_nuclei_point: float
    adequacy_probability: float


def _proxy_adequacy_probability(
    purity_point: float,
    purity_sd: float,
    tumor_nuclei_point: float,
    tumor_nuclei_sd: float,
    *,
    purity_min: float,
    tumor_cells_min: float,
) -> float:
    """Closed-form Gaussian-tail approximation of the adequacy probability.

    Assumes independent purity and tumor-nuclei estimates and a Gaussian noise
    model. The result is conservative when the two are positively correlated
    (which they typically are), so callers can use it as a lower bound.
    """

    if purity_sd <= 0:
        p_purity = 1.0 if purity_point >= purity_min else 0.0
    else:
        z_purity = (purity_point - purity_min) / max(purity_sd, 1e-6)
        p_purity = 0.5 * (1.0 + math.erf(z_purity / math.sqrt(2)))

    if tumor_nuclei_sd <= 0:
        p_cells = 1.0 if tumor_nuclei_point >= tumor_cells_min else 0.0
    else:
        z_cells = (tumor_nuclei_point - tumor_cells_min) / max(tumor_nuclei_sd, 1e-6)
        p_cells = 0.5 * (1.0 + math.erf(z_cells / math.sqrt(2)))

    return float(max(0.0, min(1.0, p_purity * p_cells)))


def _rect_iou(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 0.0)
    area_b = max((bx1 - bx0) * (by1 - by0), 0.0)
    union = area_a + area_b - inter
    return inter / max(union, 1e-9)


def suggest_candidates(
    tiles: TileArrays,
    *,
    purity_min: float,
    tumor_cells_min: float,
    window_tiles: int = 5,
    top_k: int = 5,
    nms_iou: float = 0.30,
) -> list[CandidateROI]:
    """Return up to ``top_k`` candidate rectangular ROIs."""

    grid = tiles.grid
    nx, ny = grid.grid_nx, grid.grid_ny
    if nx < 1 or ny < 1:
        return []
    wt = min(window_tiles, nx, ny)
    if wt < 1:
        return []

    # Pre-fill NaN-safe arrays.
    p = np.where(np.isnan(tiles.purity), 0.0, tiles.purity).astype(np.float64)
    sp = np.where(np.isnan(tiles.purity_sd), 0.0, tiles.purity_sd).astype(np.float64)
    n = np.where(np.isnan(tiles.nuclei), 0.0, tiles.nuclei).astype(np.float64)
    sn = np.where(np.isnan(tiles.nuclei_sd), 0.0, tiles.nuclei_sd).astype(np.float64)
    tf = np.where(np.isnan(tiles.tissue_fraction), 0.0, tiles.tissue_fraction).astype(
        np.float64
    )

    eff = n * tf
    eff_purity = eff * p

    # Cumulative sum trick for fast window sums.
    def _window_sum(arr: np.ndarray) -> np.ndarray:
        # 2D cumulative sum with zero-padding on top/left.
        cs = np.pad(arr, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
        # Window sums over wt × wt blocks at the top-left corner (iy, ix).
        out_h = ny - wt + 1
        out_w = nx - wt + 1
        if out_h <= 0 or out_w <= 0:
            return np.zeros((0, 0), dtype=np.float64)
        a = cs[wt : wt + out_h, wt : wt + out_w]
        b = cs[:out_h, wt : wt + out_w]
        c = cs[wt : wt + out_h, :out_w]
        d = cs[:out_h, :out_w]
        return a - b - c + d

    total_w = _window_sum(eff)
    tumor_w = _window_sum(eff_purity)
    # For σ propagation we sum variances (independent tile assumption).
    sn_var = (sn * tf) ** 2  # variance of nuclei per tile
    sp_var = (sp * p) ** 2  # rough propagation
    sn_w = _window_sum(sn_var)
    sp_w = _window_sum(sp_var)
    if total_w.size == 0:
        return []

    # Vectorized scoring per window.
    flat_idx = np.arange(total_w.size)
    iy_idx = flat_idx // total_w.shape[1]
    ix_idx = flat_idx % total_w.shape[1]
    candidates_raw: list[CandidateROI] = []
    for k in range(total_w.size):
        iy = int(iy_idx[k])
        ix = int(ix_idx[k])
        total = float(total_w[iy, ix])
        tumor = float(tumor_w[iy, ix])
        if total <= 0:
            continue
        purity_point = tumor / max(total, 1e-9)
        tumor_sd = math.sqrt(max(float(sn_w[iy, ix]) + float(sp_w[iy, ix]), 0.0))
        purity_sd = math.sqrt(max(float(sp_w[iy, ix]), 0.0) / max(total**2, 1e-9))
        prob = _proxy_adequacy_probability(
            purity_point=purity_point,
            purity_sd=purity_sd,
            tumor_nuclei_point=tumor,
            tumor_nuclei_sd=tumor_sd,
            purity_min=purity_min,
            tumor_cells_min=tumor_cells_min,
        )
        # Composite score: probability is primary, tumor nuclei breaks ties.
        score = prob + 1e-6 * tumor
        x0 = grid.offset_x + ix * grid.stride_x
        y0 = grid.offset_y + iy * grid.stride_y
        x1 = x0 + wt * grid.stride_x
        y1 = y0 + wt * grid.stride_y
        candidates_raw.append(
            CandidateROI(
                rank=0,
                score=score,
                bbox_thumb_px=(x0, y0, x1, y1),
                polygon=[(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                purity_point=purity_point,
                total_nuclei_point=total,
                tumor_nuclei_point=tumor,
                adequacy_probability=prob,
            )
        )

    candidates_raw.sort(key=lambda c: c.score, reverse=True)
    selected: list[CandidateROI] = []
    for cand in candidates_raw:
        if all(_rect_iou(cand.bbox_thumb_px, s.bbox_thumb_px) < nms_iou for s in selected):
            cand.rank = len(selected) + 1
            selected.append(cand)
        if len(selected) >= top_k:
            break
    return selected


def candidate_polygon_to_polygon(cand: CandidateROI) -> Polygon:
    return cand.polygon
