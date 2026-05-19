// Adequacy labelling — TypeScript twin of ``enso_purity.macrodissection.adequacy``.

import type {
  AdequacyLabel,
  AdequacyVerdict,
  ROIMetrics,
  ThresholdProfile,
} from "./types";

function fmtPct(p: number): string {
  return `${Math.round(p * 100)}%`;
}

function fmtNum(n: number): string {
  return n >= 1000 ? n.toLocaleString("en-US", { maximumFractionDigits: 0 }) : n.toFixed(0);
}

export function labelAdequacy(
  metrics: ROIMetrics,
  threshold: ThresholdProfile,
): AdequacyVerdict {
  if (metrics.n_tiles === 0 || metrics.tiles_with_data === 0) {
    return {
      label: "not_quantifiable",
      confidence: 0,
      reasons: [
        "ROI does not overlap any tissue tile; no quantitative estimate possible.",
      ],
      thresholds: threshold,
      metrics_snapshot: metrics,
    };
  }
  const purityOk = metrics.purity.median >= threshold.purity_min;
  const cellsOk = metrics.tumor_nuclei.median >= threshold.tumor_cells_min;
  const inPurityBand =
    purityOk &&
    metrics.purity.median - threshold.purity_min <=
      threshold.borderline_purity_band;
  const inCellsBand =
    cellsOk &&
    metrics.tumor_nuclei.median - threshold.tumor_cells_min <=
      threshold.borderline_tumor_cells_band;
  const prob = metrics.adequacy_probability;
  let label: AdequacyLabel;
  if (
    prob >= threshold.pass_probability &&
    purityOk &&
    cellsOk &&
    !inPurityBand &&
    !inCellsBand
  ) {
    label = "pass";
  } else if (prob >= threshold.borderline_probability && purityOk && cellsOk) {
    label = "borderline";
  } else if (prob >= threshold.borderline_probability) {
    label = "borderline";
  } else {
    label = "fail";
  }

  const reasons: string[] = [];
  if (purityOk) {
    const margin = metrics.purity.median - threshold.purity_min;
    if (margin <= threshold.borderline_purity_band) {
      reasons.push(
        `Purity ${fmtPct(metrics.purity.median)} just above the ${fmtPct(threshold.purity_min)} threshold (within the ${fmtPct(threshold.borderline_purity_band)} borderline band).`,
      );
    } else {
      reasons.push(
        `Purity ${fmtPct(metrics.purity.median)} ≥ threshold ${fmtPct(threshold.purity_min)}.`,
      );
    }
  } else {
    reasons.push(
      `Purity ${fmtPct(metrics.purity.median)} below threshold ${fmtPct(threshold.purity_min)}.`,
    );
  }
  if (cellsOk) {
    const margin = metrics.tumor_nuclei.median - threshold.tumor_cells_min;
    if (margin <= threshold.borderline_tumor_cells_band) {
      reasons.push(
        `Tumor nuclei ${fmtNum(metrics.tumor_nuclei.median)} just above the ${fmtNum(threshold.tumor_cells_min)} threshold (within the ${fmtNum(threshold.borderline_tumor_cells_band)}-cell borderline band).`,
      );
    } else {
      reasons.push(
        `Tumor nuclei ${fmtNum(metrics.tumor_nuclei.median)} ≥ threshold ${fmtNum(threshold.tumor_cells_min)}.`,
      );
    }
  } else {
    reasons.push(
      `Tumor nuclei ${fmtNum(metrics.tumor_nuclei.median)} below threshold ${fmtNum(threshold.tumor_cells_min)}.`,
    );
  }
  reasons.push(`Adequacy confidence ${fmtPct(metrics.adequacy_probability)}.`);

  return {
    label,
    confidence: metrics.adequacy_probability,
    reasons,
    thresholds: threshold,
    metrics_snapshot: metrics,
  };
}
