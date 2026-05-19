import { describe, expect, it } from "vitest";
import { labelAdequacy } from "@/lib/macrodissection/adequacy";
import { THRESHOLD_PROFILES } from "@/lib/macrodissection/thresholds";
import type { ROIMetrics } from "@/lib/macrodissection/types";

function makeMetrics(
  args: {
    purityMedian: number;
    tumorMedian: number;
    adequacyProb: number;
    n_tiles?: number;
    tiles_with_data?: number;
    totalMedian?: number;
  },
): ROIMetrics {
  const tot = args.totalMedian ?? 1000;
  return {
    n_tiles: args.n_tiles ?? 12,
    tiles_with_data: args.tiles_with_data ?? 10,
    area_thumbpx2: 2400,
    area_mm2: 0.15,
    tissue_fraction_mean: 1.0,
    purity: {
      median: args.purityMedian,
      low: args.purityMedian - 0.05,
      high: args.purityMedian + 0.05,
    },
    total_nuclei: { median: tot, low: tot * 0.9, high: tot * 1.1 },
    tumor_nuclei: {
      median: args.tumorMedian,
      low: args.tumorMedian * 0.9,
      high: args.tumorMedian * 1.1,
    },
    adequacy_probability: args.adequacyProb,
    purity_point: args.purityMedian,
    total_nuclei_point: tot,
    tumor_nuclei_point: args.tumorMedian,
  };
}

describe("labelAdequacy", () => {
  it("returns pass well above thresholds", () => {
    const v = labelAdequacy(
      makeMetrics({ purityMedian: 0.5, tumorMedian: 3000, adequacyProb: 0.97 }),
      THRESHOLD_PROFILES.humanitas_ngs,
    );
    expect(v.label).toBe("pass");
  });

  it("returns fail well below thresholds", () => {
    const v = labelAdequacy(
      makeMetrics({ purityMedian: 0.05, tumorMedian: 100, adequacyProb: 0.04 }),
      THRESHOLD_PROFILES.humanitas_ngs,
    );
    expect(v.label).toBe("fail");
  });

  it("returns borderline inside the purity band", () => {
    const v = labelAdequacy(
      makeMetrics({ purityMedian: 0.22, tumorMedian: 1500, adequacyProb: 0.93 }),
      THRESHOLD_PROFILES.humanitas_ngs,
    );
    expect(v.label).toBe("borderline");
  });

  it("returns not_quantifiable when ROI has no tissue", () => {
    const v = labelAdequacy(
      makeMetrics({
        purityMedian: 0.5,
        tumorMedian: 1000,
        adequacyProb: 0.6,
        n_tiles: 0,
        tiles_with_data: 0,
      }),
      THRESHOLD_PROFILES.humanitas_ngs,
    );
    expect(v.label).toBe("not_quantifiable");
  });
});
