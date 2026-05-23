import { describe, expect, it } from "vitest";
import {
  gaussianBlur2D,
  inpaintNaN,
  sigmaForZoom,
} from "@/lib/macrodissection/smoothing";

describe("sigmaForZoom", () => {
  it("returns max σ at zoom=0", () => {
    expect(sigmaForZoom(0, 10, "balanced")).toBeGreaterThan(1.0);
  });
  it("returns 0 at zoom=maxZoom", () => {
    expect(sigmaForZoom(10, 10, "balanced")).toBe(0);
  });
  it("overview mode gives larger σ than balanced", () => {
    expect(sigmaForZoom(2, 10, "overview")).toBeGreaterThan(
      sigmaForZoom(2, 10, "balanced"),
    );
  });
  it("detail mode gives smaller σ than balanced", () => {
    expect(sigmaForZoom(2, 10, "detail")).toBeLessThan(
      sigmaForZoom(2, 10, "balanced"),
    );
  });
});

describe("inpaintNaN", () => {
  it("fills isolated NaN from neighbours", () => {
    const data = Float32Array.from([
      1, 1, 1,
      1, NaN, 1,
      1, 1, 1,
    ]);
    const filled = inpaintNaN(data, 3, 3);
    expect(filled[4]).toBeCloseTo(1, 6);
  });

  it("leaves finite values untouched", () => {
    const data = Float32Array.from([1, 2, 3, 4]);
    const filled = inpaintNaN(data, 2, 2);
    expect(Array.from(filled)).toEqual([1, 2, 3, 4]);
  });

  it("handles a fully-NaN buffer without crashing", () => {
    const data = new Float32Array(4);
    data.fill(NaN);
    const filled = inpaintNaN(data, 2, 2);
    // Cells remain NaN because no finite neighbour exists; just verify
    // the call returns and shape is preserved.
    expect(filled.length).toBe(4);
  });
});

describe("gaussianBlur2D", () => {
  it("identity when sigma=0", () => {
    const data = Float32Array.from([1, 2, 3, 4]);
    const blurred = gaussianBlur2D(data, 2, 2, 0);
    expect(Array.from(blurred)).toEqual([1, 2, 3, 4]);
  });

  it("preserves total energy for an interior pulse", () => {
    const data = new Float32Array(81);
    data[40] = 1.0;
    const blurred = gaussianBlur2D(data, 9, 9, 0.8);
    const sum = Array.from(blurred).reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 1);
  });

  it("blurs an edge into a soft gradient", () => {
    const w = 9;
    const h = 1;
    const data = new Float32Array(w * h);
    for (let i = 0; i < w; i++) data[i] = i < 4 ? 0 : 1;
    const blurred = gaussianBlur2D(data, w, h, 1.0);
    // Values around the edge should fall strictly between 0 and 1.
    expect(blurred[3]).toBeGreaterThan(0);
    expect(blurred[3]).toBeLessThan(1);
    expect(blurred[4]).toBeGreaterThan(blurred[3]);
  });
});
