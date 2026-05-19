import { describe, expect, it } from "vitest";
import {
  clipPolygonToRect,
  polygonArea,
  polygonSignedArea,
  tileWeights,
} from "@/lib/macrodissection/polygon";
import type { TileGridMeta } from "@/lib/macrodissection/types";

const grid: TileGridMeta = {
  grid_nx: 10,
  grid_ny: 6,
  stride_x: 20,
  stride_y: 20,
  offset_x: 0,
  offset_y: 0,
  tile_area_mm2: 0.012544,
  tile_size_um: 112,
  mpp_thumb_x: 5.6,
  mpp_thumb_y: 5.6,
};

describe("polygon math", () => {
  it("computes simple square area", () => {
    expect(polygonArea([[0, 0], [10, 0], [10, 10], [0, 10]])).toBe(100);
  });

  it("ignores winding direction", () => {
    expect(
      polygonArea([[0, 0], [0, 10], [10, 10], [10, 0]]),
    ).toBe(polygonArea([[0, 0], [10, 0], [10, 10], [0, 10]]));
  });

  it("signed area is positive for CCW", () => {
    const signed = polygonSignedArea([[0, 0], [10, 0], [10, 10], [0, 10]]);
    expect(signed).toBeGreaterThan(0);
  });

  it("clipping leaves a polygon fully inside intact", () => {
    const out = clipPolygonToRect([[2, 2], [8, 2], [8, 8], [2, 8]], 0, 0, 10, 10);
    expect(polygonArea(out)).toBeCloseTo(36, 6);
  });

  it("clipping cuts a polygon spanning the rect", () => {
    const out = clipPolygonToRect([[0, 0], [20, 0], [20, 10], [0, 10]], 0, 0, 10, 10);
    expect(polygonArea(out)).toBeCloseTo(100, 6);
  });

  it("returns zero area when polygon is outside the rect", () => {
    const out = clipPolygonToRect(
      [[20, 20], [30, 20], [30, 30], [20, 30]],
      0, 0, 10, 10,
    );
    expect(polygonArea(out)).toBe(0);
  });

  it("tileWeights covers exactly one tile when polygon equals tile", () => {
    const weights = tileWeights(
      [[0, 0], [20, 0], [20, 20], [0, 20]],
      grid,
    );
    expect(weights).toHaveLength(1);
    expect(weights[0].weight).toBeCloseTo(1, 9);
  });

  it("tileWeights gives 0.5 weight on a half-tile polygon", () => {
    const weights = tileWeights(
      [[0, 0], [10, 0], [10, 20], [0, 20]],
      grid,
    );
    expect(weights).toHaveLength(1);
    expect(weights[0].weight).toBeCloseTo(0.5, 9);
  });

  it("tileWeights spans multiple tiles", () => {
    const weights = tileWeights(
      [[0, 0], [40, 0], [40, 20], [0, 20]],
      grid,
    );
    const keys = weights.map((w) => `${w.ix}:${w.iy}`).sort();
    expect(keys).toEqual(["0:0", "1:0"]);
    const total = weights.reduce((s, w) => s + w.weight, 0);
    expect(total).toBeCloseTo(2, 9);
  });

  it("tileWeights returns empty array for polygon outside the grid", () => {
    const weights = tileWeights(
      [[400, 400], [410, 400], [410, 410]],
      grid,
    );
    expect(weights).toEqual([]);
  });
});
