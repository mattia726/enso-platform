// Point estimates and Monte-Carlo uncertainty propagation for ROI metrics.
//
// Algorithms mirror ``enso_purity.macrodissection.roi`` exactly so the
// numbers shown in the client preview match the server-side recompute.

import type {
  MetricsCI,
  Point2D,
  ROIMetrics,
  TileArrays,
  TileGridMeta,
} from "./types";
import {
  tileWeights,
  type TileWeight,
} from "./polygon";
import {
  mulberry32,
  polygonHashSeed,
  randNormalFromUniform,
} from "./rng";

const EPS = 1e-9;

interface CollectedArrays {
  readonly w: Float64Array;
  readonly p: Float64Array;
  readonly sp: Float64Array;
  readonly n: Float64Array;
  readonly sn: Float64Array;
  readonly tf: Float64Array;
  readonly tilesWithData: number;
}

function readTile(
  arr: Float32Array,
  ix: number,
  iy: number,
  grid: TileGridMeta,
): number {
  return arr[iy * grid.grid_nx + ix];
}

function safeFinite(v: number): number {
  return Number.isFinite(v) ? v : 0;
}

function collect(
  weights: readonly TileWeight[],
  tiles: TileArrays,
): CollectedArrays {
  const n = weights.length;
  const w = new Float64Array(n);
  const p = new Float64Array(n);
  const sp = new Float64Array(n);
  const nu = new Float64Array(n);
  const sn = new Float64Array(n);
  const tf = new Float64Array(n);
  let tilesWithData = 0;
  for (let i = 0; i < n; i++) {
    const tw = weights[i];
    const tfRaw = readTile(tiles.tissue_fraction, tw.ix, tw.iy, tiles.grid);
    const pRaw = readTile(tiles.purity, tw.ix, tw.iy, tiles.grid);
    const nRaw = readTile(tiles.nuclei, tw.ix, tw.iy, tiles.grid);
    const spRaw = readTile(tiles.purity_sd, tw.ix, tw.iy, tiles.grid);
    const snRaw = readTile(tiles.nuclei_sd, tw.ix, tw.iy, tiles.grid);
    w[i] = tw.weight;
    tf[i] = safeFinite(tfRaw);
    p[i] = safeFinite(pRaw);
    nu[i] = safeFinite(nRaw);
    sp[i] = safeFinite(spRaw);
    sn[i] = safeFinite(snRaw);
    if (tf[i] > 0 && Number.isFinite(pRaw) && Number.isFinite(nRaw)) {
      tilesWithData += 1;
    }
  }
  return { w, p, sp, n: nu, sn, tf, tilesWithData };
}

export interface PointEstimate {
  readonly purity: number;
  readonly total_nuclei: number;
  readonly tumor_nuclei: number;
  readonly area_thumbpx2: number;
  readonly area_mm2: number;
}

export function pointEstimates(
  weights: readonly TileWeight[],
  tiles: TileArrays,
): PointEstimate {
  if (weights.length === 0) {
    return {
      purity: 0,
      total_nuclei: 0,
      tumor_nuclei: 0,
      area_thumbpx2: 0,
      area_mm2: 0,
    };
  }
  const c = collect(weights, tiles);
  let total = 0;
  let tumor = 0;
  let area = 0;
  let areaMm2 = 0;
  for (let i = 0; i < c.w.length; i++) {
    const eff = c.w[i] * c.n[i] * c.tf[i];
    total += eff;
    tumor += eff * c.p[i];
    area += weights[i].interAreaThumbPx2;
    areaMm2 += c.w[i] * tiles.grid.tile_area_mm2;
  }
  return {
    purity: tumor / Math.max(total, EPS),
    total_nuclei: total,
    tumor_nuclei: tumor,
    area_thumbpx2: area,
    area_mm2: areaMm2,
  };
}

function percentile(sorted: Float64Array, q: number): number {
  const n = sorted.length;
  if (n === 0) return 0;
  const pos = (n - 1) * q;
  const i = Math.floor(pos);
  const f = pos - i;
  if (i + 1 >= n) return sorted[n - 1];
  return sorted[i] + (sorted[i + 1] - sorted[i]) * f;
}

function ci(samples: Float64Array): MetricsCI {
  // Avoid mutating the original buffer for the next computation.
  const copy = samples.slice();
  copy.sort();
  return {
    median: percentile(copy, 0.5),
    low: percentile(copy, 0.05),
    high: percentile(copy, 0.95),
  };
}

export interface MonteCarloOptions {
  readonly thresholdsPurityMin: number;
  readonly thresholdsTumorCellsMin: number;
  readonly nSamples?: number;
  readonly seed?: number;
}

export function monteCarlo(
  weights: readonly TileWeight[],
  tiles: TileArrays,
  opts: MonteCarloOptions,
): ROIMetrics {
  const nSamples = opts.nSamples ?? 400;

  if (weights.length === 0) {
    const zero: MetricsCI = { median: 0, low: 0, high: 0 };
    return {
      n_tiles: 0,
      tiles_with_data: 0,
      area_thumbpx2: 0,
      area_mm2: 0,
      tissue_fraction_mean: 0,
      purity: zero,
      total_nuclei: zero,
      tumor_nuclei: zero,
      adequacy_probability: 0,
      purity_point: 0,
      total_nuclei_point: 0,
      tumor_nuclei_point: 0,
    };
  }

  const c = collect(weights, tiles);
  const rng = mulberry32(opts.seed ?? 0);
  const purityCs = new Float64Array(nSamples);
  const totalCs = new Float64Array(nSamples);
  const tumorCs = new Float64Array(nSamples);
  let pass = 0;
  const m = c.w.length;
  // Pre-zero SD where tissue is empty (mirrors the Python sampler).
  const spEff = new Float64Array(m);
  const snEff = new Float64Array(m);
  for (let i = 0; i < m; i++) {
    spEff[i] = c.tf[i] > 0 ? c.sp[i] : 0;
    snEff[i] = c.tf[i] > 0 ? c.sn[i] : 0;
  }

  for (let k = 0; k < nSamples; k++) {
    let total = 0;
    let tumor = 0;
    for (let i = 0; i < m; i++) {
      const zP = randNormalFromUniform(rng);
      const zN = randNormalFromUniform(rng);
      let pi = c.p[i] + zP * spEff[i];
      let ni = c.n[i] + zN * snEff[i];
      if (pi < 0) pi = 0;
      else if (pi > 1) pi = 1;
      if (ni < 0) ni = 0;
      const eff = c.w[i] * ni * c.tf[i];
      total += eff;
      tumor += eff * pi;
    }
    const purityK = tumor / Math.max(total, EPS);
    totalCs[k] = total;
    tumorCs[k] = tumor;
    purityCs[k] = purityK;
    if (purityK >= opts.thresholdsPurityMin && tumor >= opts.thresholdsTumorCellsMin) {
      pass += 1;
    }
  }

  const point = pointEstimates(weights, tiles);
  let tfSum = 0;
  for (let i = 0; i < m; i++) tfSum += c.tf[i];
  const tfMean = m > 0 ? tfSum / m : 0;

  return {
    n_tiles: weights.length,
    tiles_with_data: c.tilesWithData,
    area_thumbpx2: point.area_thumbpx2,
    area_mm2: point.area_mm2,
    tissue_fraction_mean: tfMean,
    purity: ci(purityCs),
    total_nuclei: ci(totalCs),
    tumor_nuclei: ci(tumorCs),
    adequacy_probability: pass / Math.max(nSamples, 1),
    purity_point: point.purity,
    total_nuclei_point: point.total_nuclei,
    tumor_nuclei_point: point.tumor_nuclei,
  };
}

export function computeROIMetrics(
  polygon: readonly Point2D[],
  tiles: TileArrays,
  opts: MonteCarloOptions,
): ROIMetrics {
  const weights = tileWeights(polygon, tiles.grid);
  const seed = opts.seed ?? polygonHashSeed(polygon);
  return monteCarlo(weights, tiles, { ...opts, seed });
}
