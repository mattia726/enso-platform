// Component test: ROI metrics card renders the right numbers and the
// right pass/borderline/fail pill when supplied with known inputs.

import { describe, expect, it } from "vitest";
import { render, screen, within } from "@testing-library/react";

import RoiMetricsCard from "@/app/macrodissection/components/RoiMetricsCard";
import { THRESHOLD_PROFILES } from "@/lib/macrodissection/thresholds";
import { labelAdequacy } from "@/lib/macrodissection/adequacy";
import type { ROIMetrics } from "@/lib/macrodissection/types";

function makeMetrics(args: {
  purity: number;
  totalNuc: number;
  tumorNuc: number;
  prob: number;
}): ROIMetrics {
  return {
    n_tiles: 12,
    tiles_with_data: 11,
    area_thumbpx2: 2400,
    area_mm2: 0.15,
    tissue_fraction_mean: 0.9,
    purity: { median: args.purity, low: args.purity - 0.05, high: args.purity + 0.05 },
    total_nuclei: {
      median: args.totalNuc,
      low: args.totalNuc * 0.9,
      high: args.totalNuc * 1.1,
    },
    tumor_nuclei: {
      median: args.tumorNuc,
      low: args.tumorNuc * 0.9,
      high: args.tumorNuc * 1.1,
    },
    adequacy_probability: args.prob,
    purity_point: args.purity,
    total_nuclei_point: args.totalNuc,
    tumor_nuclei_point: args.tumorNuc,
  };
}

describe("<RoiMetricsCard />", () => {
  it("renders the empty state when no ROI has been drawn", () => {
    render(<RoiMetricsCard metrics={null} verdict={null} status="no_roi" />);
    expect(screen.getByText(/Draw a polygon/i)).toBeDefined();
    expect(
      screen.getByText(/Awaiting ROI/i),
    ).toBeDefined();
  });

  it("renders a PASS verdict with the right numbers", () => {
    const metrics = makeMetrics({
      purity: 0.5,
      totalNuc: 5000,
      tumorNuc: 3000,
      prob: 0.97,
    });
    const verdict = labelAdequacy(metrics, THRESHOLD_PROFILES.humanitas_ngs);
    render(<RoiMetricsCard metrics={metrics} verdict={verdict} status="ready" />);
    const pill = screen.getByText(/PASS/i, { selector: "[data-verdict-label]" });
    expect(pill.getAttribute("data-verdict-label")).toBe("pass");

    // Numbers
    const card = screen.getByText("ROI adequacy").closest("aside")!;
    expect(within(card).getByText("50%")).toBeDefined();           // purity median
    expect(within(card).getByText("5,000")).toBeDefined();         // total nuclei
    expect(within(card).getByText("3,000")).toBeDefined();         // tumor nuclei
    expect(within(card).getByText("97%")).toBeDefined();           // adequacy confidence
  });

  it("renders a FAIL verdict when below thresholds", () => {
    const metrics = makeMetrics({
      purity: 0.05,
      totalNuc: 600,
      tumorNuc: 30,
      prob: 0.02,
    });
    const verdict = labelAdequacy(metrics, THRESHOLD_PROFILES.humanitas_ngs);
    render(<RoiMetricsCard metrics={metrics} verdict={verdict} status="ready" />);
    const pill = screen.getByText(/fail/i, { selector: "[data-verdict-label]" });
    expect(pill.getAttribute("data-verdict-label")).toBe("fail");
  });

  it("renders BORDERLINE inside the purity band", () => {
    const metrics = makeMetrics({
      purity: 0.22,
      totalNuc: 5000,
      tumorNuc: 1100,
      prob: 0.85,
    });
    const verdict = labelAdequacy(metrics, THRESHOLD_PROFILES.humanitas_ngs);
    render(<RoiMetricsCard metrics={metrics} verdict={verdict} status="ready" />);
    const pill = screen.getByText(/borderline/i, { selector: "[data-verdict-label]" });
    expect(pill.getAttribute("data-verdict-label")).toBe("borderline");
  });

  it("renders the NOT QUANTIFIABLE state for a tissueless ROI", () => {
    const metrics: ROIMetrics = {
      n_tiles: 0,
      tiles_with_data: 0,
      area_thumbpx2: 0,
      area_mm2: 0,
      tissue_fraction_mean: 0,
      purity: { median: 0, low: 0, high: 0 },
      total_nuclei: { median: 0, low: 0, high: 0 },
      tumor_nuclei: { median: 0, low: 0, high: 0 },
      adequacy_probability: 0,
      purity_point: 0,
      total_nuclei_point: 0,
      tumor_nuclei_point: 0,
    };
    const verdict = labelAdequacy(metrics, THRESHOLD_PROFILES.humanitas_ngs);
    render(<RoiMetricsCard metrics={metrics} verdict={verdict} status="ready" />);
    expect(screen.getByText(/Not quantifiable/i)).toBeDefined();
  });
});
