import { describe, expect, it } from "vitest";
import {
  computeROIMetrics,
  monteCarlo,
  pointEstimates,
} from "@/lib/macrodissection/metrics";
import { tileWeights } from "@/lib/macrodissection/polygon";
import type { TileArrays, TileGridMeta } from "@/lib/macrodissection/types";

interface TileTemplate {
  purity: number;
  nuclei: number;
  purity_sd?: number;
  nuclei_sd?: number;
  tissue?: number;
}

function makeTiles(
  grid_nx: number,
  grid_ny: number,
  t: TileTemplate,
): TileArrays {
  const n = grid_nx * grid_ny;
  const grid: TileGridMeta = {
    grid_nx,
    grid_ny,
    stride_x: 20,
    stride_y: 20,
    offset_x: 0,
    offset_y: 0,
    tile_area_mm2: 0.012544,
    tile_size_um: 112,
    mpp_thumb_x: 5.6,
    mpp_thumb_y: 5.6,
  };
  return {
    grid,
    purity: new Float32Array(n).fill(t.purity),
    purity_sd: new Float32Array(n).fill(t.purity_sd ?? 0),
    nuclei: new Float32Array(n).fill(t.nuclei),
    nuclei_sd: new Float32Array(n).fill(t.nuclei_sd ?? 0),
    tumor_nuclei: new Float32Array(n).fill(t.purity * t.nuclei),
    tissue_fraction: new Float32Array(n).fill(t.tissue ?? 1),
  };
}

describe("ROI metrics", () => {
  it("point estimate for a uniform constant grid", () => {
    const tiles = makeTiles(10, 6, { purity: 0.4, nuclei: 100 });
    const ws = tileWeights(
      [[0, 0], [200, 0], [200, 120], [0, 120]],
      tiles.grid,
    );
    const est = pointEstimates(ws, tiles);
    expect(est.purity).toBeCloseTo(0.4, 5);
    expect(est.total_nuclei).toBeCloseTo(100 * 60, 2);
    expect(est.tumor_nuclei).toBeCloseTo(40 * 60, 2);
  });

  it("Monte-Carlo is deterministic for a fixed seed", () => {
    const tiles = makeTiles(10, 6, {
      purity: 0.4,
      nuclei: 100,
      purity_sd: 0.05,
      nuclei_sd: 5,
    });
    const ws = tileWeights(
      [[0, 0], [60, 0], [60, 40], [0, 40]],
      tiles.grid,
    );
    const a = monteCarlo(ws, tiles, {
      thresholdsPurityMin: 0.2,
      thresholdsTumorCellsMin: 200,
      nSamples: 200,
      seed: 12345,
    });
    const b = monteCarlo(ws, tiles, {
      thresholdsPurityMin: 0.2,
      thresholdsTumorCellsMin: 200,
      nSamples: 200,
      seed: 12345,
    });
    expect(a.purity.median).toBe(b.purity.median);
    expect(a.adequacy_probability).toBe(b.adequacy_probability);
  });

  it("Monte-Carlo collapses to point estimate when σ → 0", () => {
    const tiles = makeTiles(10, 6, { purity: 0.5, nuclei: 120 });
    const ws = tileWeights(
      [[0, 0], [60, 0], [60, 40], [0, 40]],
      tiles.grid,
    );
    const m = monteCarlo(ws, tiles, {
      thresholdsPurityMin: 0.2,
      thresholdsTumorCellsMin: 200,
      nSamples: 100,
      seed: 7,
    });
    expect(m.purity.median).toBeCloseTo(0.5, 6);
    expect(m.purity.low).toBeCloseTo(m.purity.high, 6);
    expect(m.adequacy_probability).toBe(1);
  });

  it("compute ROI metrics seeds from polygon hash", () => {
    const tiles = makeTiles(10, 6, { purity: 0.5, nuclei: 100 });
    const a = computeROIMetrics(
      [[1, 1], [40, 2], [38, 30], [0, 28]],
      tiles,
      { thresholdsPurityMin: 0.2, thresholdsTumorCellsMin: 200 },
    );
    const b = computeROIMetrics(
      [[1, 1], [40, 2], [38, 30], [0, 28]],
      tiles,
      { thresholdsPurityMin: 0.2, thresholdsTumorCellsMin: 200 },
    );
    expect(a.purity.median).toBe(b.purity.median);
  });

  it("empty polygon returns all zeros", () => {
    const tiles = makeTiles(10, 6, { purity: 0.5, nuclei: 100 });
    const m = computeROIMetrics(
      [[400, 400], [401, 400], [401, 401]],
      tiles,
      { thresholdsPurityMin: 0.2, thresholdsTumorCellsMin: 200 },
    );
    expect(m.n_tiles).toBe(0);
    expect(m.adequacy_probability).toBe(0);
  });
});
