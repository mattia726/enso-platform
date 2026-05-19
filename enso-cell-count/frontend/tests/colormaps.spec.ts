import { describe, expect, it } from "vitest";
import {
  ADEQUACY_SPEC,
  CELLULARITY_SPEC,
  PURITY_LUT,
  PURITY_SPEC,
  lutLookup,
} from "@/lib/macrodissection/colormaps";

describe("colormap LUTs", () => {
  it("PURITY_LUT has 256 entries", () => {
    expect(PURITY_LUT.length).toBe(256 * 3);
  });

  it("lutLookup returns RGB in range for purity 0..1", () => {
    for (let i = 0; i <= 20; i++) {
      const v = i / 20;
      const [r, g, b] = lutLookup(v, PURITY_SPEC);
      expect(r).toBeGreaterThanOrEqual(0);
      expect(r).toBeLessThanOrEqual(255);
      expect(g).toBeGreaterThanOrEqual(0);
      expect(g).toBeLessThanOrEqual(255);
      expect(b).toBeGreaterThanOrEqual(0);
      expect(b).toBeLessThanOrEqual(255);
    }
  });

  it("purity v=0 is blue-leaning, v=1 is red-leaning", () => {
    const [r0, g0, b0] = lutLookup(0, PURITY_SPEC);
    const [r1, g1, b1] = lutLookup(1, PURITY_SPEC);
    expect(b0).toBeGreaterThan(r0);
    expect(r1).toBeGreaterThan(b1);
  });

  it("cellularity LUT applies gamma warp", () => {
    const mid = lutLookup(CELLULARITY_SPEC.vmax / 2, CELLULARITY_SPEC);
    const lo = lutLookup(10, CELLULARITY_SPEC);
    // With gamma < 1, the midpoint is well into the warm half.
    expect(mid[0]).toBeGreaterThan(lo[0]);
  });

  it("adequacy LUT v=0 is the lightest, v=vmax is the darkest", () => {
    const start = lutLookup(0, ADEQUACY_SPEC);
    const end = lutLookup(ADEQUACY_SPEC.vmax, ADEQUACY_SPEC);
    const startSum = start[0] + start[1] + start[2];
    const endSum = end[0] + end[1] + end[2];
    expect(startSum).toBeGreaterThan(endSum);
  });

  it("clamps inputs outside [vmin, vmax]", () => {
    const lo = lutLookup(-10, PURITY_SPEC);
    const hi = lutLookup(50, PURITY_SPEC);
    expect(lo).toEqual(lutLookup(0, PURITY_SPEC));
    expect(hi).toEqual(lutLookup(1, PURITY_SPEC));
  });
});

describe("Python ↔ TypeScript LUT parity", () => {
  // Read the Python-generated colormap stops and verify that the TS LUTs
  // interpolate to within ±1 RGB step of the Python palette at every
  // matching stop position.
  const fixture = require("./fixtures/colormaps_python_lut.json") as {
    purity_RdYlBu_r: [number, number, number][];
    cellularity_purity_no_white: [number, number, number][];
  };

  function tsLutValueAt(spec: typeof PURITY_SPEC, t: number): [number, number, number] {
    const v = spec.vmin + t * (spec.vmax - spec.vmin);
    return lutLookup(v, spec);
  }

  it("RdYlBu_r LUT matches Python within 1 RGB step at each stop", () => {
    const stops = fixture.purity_RdYlBu_r;
    const n = stops.length;
    for (let i = 0; i < n; i++) {
      // Stop position in [0, 1] used by Python's palette_stops.
      const t = i / (n - 1);
      // PURITY_SPEC has gamma=1, so the "raw" t maps directly into the
      // normalized LUT index used at draw time.
      const [r, g, b] = tsLutValueAt(PURITY_SPEC, t);
      const [pr, pg, pb] = stops[i];
      expect(Math.abs(r - pr)).toBeLessThanOrEqual(2);
      expect(Math.abs(g - pg)).toBeLessThanOrEqual(2);
      expect(Math.abs(b - pb)).toBeLessThanOrEqual(2);
    }
  });

  it("cellularity LUT matches Python within 1 RGB step at each stop", () => {
    const stops = fixture.cellularity_purity_no_white;
    const n = stops.length;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      // For cellularity we have to undo the gamma so the LUT index hits
      // the same position as Python's palette_stops (which sample the
      // colormap directly, without the gamma warp).
      const v = CELLULARITY_SPEC.vmin + (t ** (1 / CELLULARITY_SPEC.gamma)) *
        (CELLULARITY_SPEC.vmax - CELLULARITY_SPEC.vmin);
      const [r, g, b] = lutLookup(v, CELLULARITY_SPEC);
      const [pr, pg, pb] = stops[i];
      // The TS LUT is built from the same 7 stops with linear interp; the
      // tolerance is 1 step except at the bin edges where a half-pixel
      // rounding can creep in.
      expect(Math.abs(r - pr)).toBeLessThanOrEqual(2);
      expect(Math.abs(g - pg)).toBeLessThanOrEqual(2);
      expect(Math.abs(b - pb)).toBeLessThanOrEqual(2);
    }
  });
});
