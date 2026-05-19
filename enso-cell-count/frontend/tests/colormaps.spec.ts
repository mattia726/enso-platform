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
