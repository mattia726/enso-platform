import { describe, expect, it } from "vitest";
import {
  fnv1a,
  mulberry32,
  polygonHashSeed,
  randNormalFromUniform,
} from "@/lib/macrodissection/rng";

describe("rng", () => {
  it("fnv1a is deterministic", () => {
    expect(fnv1a(new Uint8Array([1, 2, 3]))).toBe(
      fnv1a(new Uint8Array([1, 2, 3])),
    );
  });

  it("polygonHashSeed differs for distinct polygons", () => {
    const a = polygonHashSeed([[0, 0], [1, 0], [1, 1]]);
    const b = polygonHashSeed([[0, 0], [1, 0], [1, 1.000001]]);
    expect(a).not.toBe(b);
  });

  it("polygonHashSeed is stable across calls", () => {
    const poly = [[0, 0], [10, 0], [10, 10], [0, 10]] as const;
    expect(polygonHashSeed(poly as any)).toBe(polygonHashSeed(poly as any));
  });

  it("mulberry32 produces identical sequences for the same seed", () => {
    const a = mulberry32(7);
    const b = mulberry32(7);
    for (let i = 0; i < 10; i++) {
      expect(a()).toBe(b());
    }
  });

  it("randNormalFromUniform has unit variance over many samples", () => {
    const rng = mulberry32(99);
    let sum = 0;
    let sq = 0;
    const n = 5000;
    for (let i = 0; i < n; i++) {
      const v = randNormalFromUniform(rng);
      sum += v;
      sq += v * v;
    }
    const mean = sum / n;
    const variance = sq / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.1);
    expect(Math.abs(variance - 1)).toBeLessThan(0.1);
  });
});
