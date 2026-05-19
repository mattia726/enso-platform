// Client-side candidate ROI auto-suggester.
//
// Mirrors ``enso_purity.macrodissection.candidates`` so the static demo
// (no backend) can still surface candidate macrodissection regions.

import type { Point2D, TileArrays, ThresholdProfile } from "./types";

export interface ClientCandidate {
  rank: number;
  score: number;
  bbox_thumb_px: readonly [number, number, number, number];
  polygon: readonly Point2D[];
  purity_point: number;
  total_nuclei_point: number;
  tumor_nuclei_point: number;
  adequacy_probability: number;
}

function erfApprox(x: number): number {
  // Abramowitz & Stegun 7.1.26 — max error 1.5e-7.
  const sign = x < 0 ? -1 : 1;
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) *
      t +
      0.254829592) *
      t *
      Math.exp(-ax * ax);
  return sign * y;
}

function proxyAdequacyProb(
  purityPoint: number,
  puritySd: number,
  tumorPoint: number,
  tumorSd: number,
  purityMin: number,
  tumorCellsMin: number,
): number {
  const zPurity =
    puritySd <= 0
      ? purityPoint >= purityMin
        ? 8
        : -8
      : (purityPoint - purityMin) / puritySd;
  const zCells =
    tumorSd <= 0
      ? tumorPoint >= tumorCellsMin
        ? 8
        : -8
      : (tumorPoint - tumorCellsMin) / tumorSd;
  const pPurity = 0.5 * (1 + erfApprox(zPurity / Math.SQRT2));
  const pCells = 0.5 * (1 + erfApprox(zCells / Math.SQRT2));
  return Math.max(0, Math.min(1, pPurity * pCells));
}

function rectIou(
  a: readonly [number, number, number, number],
  b: readonly [number, number, number, number],
): number {
  const ix0 = Math.max(a[0], b[0]);
  const iy0 = Math.max(a[1], b[1]);
  const ix1 = Math.min(a[2], b[2]);
  const iy1 = Math.min(a[3], b[3]);
  if (ix0 >= ix1 || iy0 >= iy1) return 0;
  const inter = (ix1 - ix0) * (iy1 - iy0);
  const areaA = Math.max((a[2] - a[0]) * (a[3] - a[1]), 0);
  const areaB = Math.max((b[2] - b[0]) * (b[3] - b[1]), 0);
  const union = areaA + areaB - inter;
  return inter / Math.max(union, 1e-9);
}

export function suggestCandidates(
  tiles: TileArrays,
  threshold: ThresholdProfile,
  opts: { windowTiles?: number; topK?: number; nmsIou?: number } = {},
): ClientCandidate[] {
  const grid = tiles.grid;
  const nx = grid.grid_nx;
  const ny = grid.grid_ny;
  const wt = Math.min(opts.windowTiles ?? 5, nx, ny);
  if (wt < 1) return [];
  const topK = opts.topK ?? 5;
  const nmsIou = opts.nmsIou ?? 0.3;

  const sanitize = (a: Float32Array) => {
    const out = new Float32Array(a.length);
    for (let i = 0; i < a.length; i++) {
      out[i] = Number.isFinite(a[i]) ? a[i] : 0;
    }
    return out;
  };
  const p = sanitize(tiles.purity);
  const sp = sanitize(tiles.purity_sd);
  const n = sanitize(tiles.nuclei);
  const sn = sanitize(tiles.nuclei_sd);
  const tf = sanitize(tiles.tissue_fraction);

  const totalGrid = new Float32Array(nx * ny);
  const tumorGrid = new Float32Array(nx * ny);
  const snVarGrid = new Float32Array(nx * ny);
  const spVarGrid = new Float32Array(nx * ny);
  for (let i = 0; i < nx * ny; i++) {
    const eff = n[i] * tf[i];
    totalGrid[i] = eff;
    tumorGrid[i] = eff * p[i];
    snVarGrid[i] = (sn[i] * tf[i]) ** 2;
    spVarGrid[i] = (sp[i] * p[i]) ** 2;
  }

  // Build integral images for fast window sums.
  function integral(src: Float32Array): Float64Array {
    const out = new Float64Array((nx + 1) * (ny + 1));
    for (let y = 1; y <= ny; y++) {
      let rowSum = 0;
      for (let x = 1; x <= nx; x++) {
        rowSum += src[(y - 1) * nx + (x - 1)];
        out[y * (nx + 1) + x] =
          rowSum + out[(y - 1) * (nx + 1) + x];
      }
    }
    return out;
  }
  const I_total = integral(totalGrid);
  const I_tumor = integral(tumorGrid);
  const I_snVar = integral(snVarGrid);
  const I_spVar = integral(spVarGrid);

  function windowSum(I: Float64Array, ix: number, iy: number): number {
    const x0 = ix;
    const y0 = iy;
    const x1 = ix + wt;
    const y1 = iy + wt;
    const s =
      I[y1 * (nx + 1) + x1] -
      I[y0 * (nx + 1) + x1] -
      I[y1 * (nx + 1) + x0] +
      I[y0 * (nx + 1) + x0];
    return s;
  }

  type Raw = ClientCandidate & { rawScore: number };
  const raw: Raw[] = [];
  for (let iy = 0; iy + wt <= ny; iy++) {
    for (let ix = 0; ix + wt <= nx; ix++) {
      const total = windowSum(I_total, ix, iy);
      if (total <= 0) continue;
      const tumor = windowSum(I_tumor, ix, iy);
      const tumorSd = Math.sqrt(
        Math.max(windowSum(I_snVar, ix, iy) + windowSum(I_spVar, ix, iy), 0),
      );
      const puritySd = Math.sqrt(
        Math.max(windowSum(I_spVar, ix, iy) / Math.max(total * total, 1e-9), 0),
      );
      const purity = tumor / Math.max(total, 1e-9);
      const prob = proxyAdequacyProb(
        purity,
        puritySd,
        tumor,
        tumorSd,
        threshold.purity_min,
        threshold.tumor_cells_min,
      );
      const score = prob + 1e-6 * tumor;
      const x0 = grid.offset_x + ix * grid.stride_x;
      const y0 = grid.offset_y + iy * grid.stride_y;
      const x1 = x0 + wt * grid.stride_x;
      const y1 = y0 + wt * grid.stride_y;
      raw.push({
        rank: 0,
        score,
        rawScore: score,
        bbox_thumb_px: [x0, y0, x1, y1] as const,
        polygon: [
          [x0, y0],
          [x1, y0],
          [x1, y1],
          [x0, y1],
        ],
        purity_point: purity,
        total_nuclei_point: total,
        tumor_nuclei_point: tumor,
        adequacy_probability: prob,
      });
    }
  }
  raw.sort((a, b) => b.score - a.score);
  const selected: ClientCandidate[] = [];
  for (const cand of raw) {
    if (selected.every((s) => rectIou(cand.bbox_thumb_px, s.bbox_thumb_px) < nmsIou)) {
      selected.push({ ...cand, rank: selected.length + 1 });
    }
    if (selected.length >= topK) break;
  }
  return selected;
}
