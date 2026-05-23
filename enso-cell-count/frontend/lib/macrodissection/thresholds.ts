// Threshold profiles — TS mirror of the Python ``PROFILES`` dict.

import type { ThresholdProfile } from "./types";

export const THRESHOLD_PROFILES: Record<string, ThresholdProfile> = {
  humanitas_ngs: {
    name: "humanitas_ngs",
    display_name: "Humanitas NGS pilot",
    purity_min: 0.20,
    tumor_cells_min: 1000,
    borderline_purity_band: 0.05,
    borderline_tumor_cells_band: 200,
    pass_probability: 0.90,
    borderline_probability: 0.50,
    notes:
      "Pilot profile based on the Humanitas macrodissection workflow: " +
      "tumor area must be ≥20% pure and contain at least one thousand " +
      "tumor nuclei for downstream NGS to be reliable.",
  },
  research: {
    name: "research",
    display_name: "Research / exploratory",
    purity_min: 0.10,
    tumor_cells_min: 200,
    borderline_purity_band: 0.05,
    borderline_tumor_cells_band: 100,
    pass_probability: 0.85,
    borderline_probability: 0.40,
    notes:
      "Relaxed thresholds intended for translational research where " +
      "lower-yield specimens may still be informative.",
  },
  strict_solid_tumor: {
    name: "strict_solid_tumor",
    display_name: "Strict solid tumor",
    purity_min: 0.30,
    tumor_cells_min: 2000,
    borderline_purity_band: 0.05,
    borderline_tumor_cells_band: 300,
    pass_probability: 0.95,
    borderline_probability: 0.70,
    notes:
      "Conservative profile suited to deeply-sequenced solid tumor " +
      "assays where false positives from contaminating normal tissue " +
      "are particularly costly.",
  },
};

export const DEFAULT_PROFILE = THRESHOLD_PROFILES.humanitas_ngs;

export function listProfiles(): ThresholdProfile[] {
  return Object.values(THRESHOLD_PROFILES);
}

export function resolveProfile(
  name: string,
  override?: Partial<Omit<ThresholdProfile, "name" | "display_name" | "notes">>,
): ThresholdProfile {
  const base = THRESHOLD_PROFILES[name] ?? DEFAULT_PROFILE;
  if (!override) return base;
  return { ...base, ...override };
}
