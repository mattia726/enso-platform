"use client";

import type { AdequacyVerdict, ROIMetrics } from "@/lib/macrodissection/types";

export interface RoiMetricsCardProps {
  metrics: ROIMetrics | null;
  verdict: AdequacyVerdict | null;
  status?: "idle" | "computing" | "ready" | "no_roi";
}

function fmtPct(v: number): string {
  if (!Number.isFinite(v)) return "—";
  return `${Math.round(v * 100)}%`;
}

function fmtNumber(v: number): string {
  if (!Number.isFinite(v)) return "—";
  if (v >= 1000) {
    return v.toLocaleString("en-US", { maximumFractionDigits: 0 });
  }
  return v.toFixed(0);
}

function VerdictPill({ verdict }: { verdict: AdequacyVerdict | null }) {
  if (!verdict) {
    return (
      <span className="inline-flex items-center gap-2 rounded-full bg-[var(--surface)] border border-[var(--border)] px-3 py-1 text-xs uppercase tracking-widest text-[var(--muted)]">
        Awaiting ROI
      </span>
    );
  }
  const palette =
    verdict.label === "pass"
      ? "bg-green-500/15 border-green-500/60 text-green-300"
      : verdict.label === "borderline"
        ? "bg-amber-500/20 border-amber-500/60 text-amber-200"
        : verdict.label === "fail"
          ? "bg-red-500/20 border-red-500/60 text-red-200"
          : "bg-slate-500/20 border-slate-500/60 text-slate-200";
  const label =
    verdict.label === "not_quantifiable" ? "Not quantifiable" : verdict.label;
  return (
    <span
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs uppercase tracking-widest font-bold ${palette}`}
      data-verdict-label={verdict.label}
    >
      <span
        className={`inline-block h-2 w-2 rounded-full ${
          verdict.label === "pass"
            ? "bg-green-400"
            : verdict.label === "borderline"
              ? "bg-amber-400"
              : verdict.label === "fail"
                ? "bg-red-400"
                : "bg-slate-300"
        }`}
        aria-hidden
      />
      {label}
    </span>
  );
}

export default function RoiMetricsCard({
  metrics,
  verdict,
  status = "idle",
}: RoiMetricsCardProps) {
  const empty = status === "no_roi" || (metrics === null && status === "idle");

  return (
    <aside
      className="flex flex-col gap-3 rounded-2xl border border-[var(--border)] bg-[var(--surface)]/80 backdrop-blur p-4 shadow-sm"
      data-roi-metrics-card
    >
      <header className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-[var(--text)]">
          ROI adequacy
        </h3>
        <VerdictPill verdict={verdict} />
      </header>
      {empty ? (
        <p className="text-sm text-[var(--muted)] leading-relaxed">
          Draw a polygon on the slide to see estimated purity, total nuclei,
          tumor nuclei and adequacy probability for that region. Numbers
          update live while you edit.
        </p>
      ) : metrics ? (
        <div className="grid gap-2 text-sm">
          <div className="grid grid-cols-3 gap-3 text-center">
            <Metric label="Purity" value={fmtPct(metrics.purity.median)} ci={`${fmtPct(metrics.purity.low)} – ${fmtPct(metrics.purity.high)}`} />
            <Metric
              label="Total nuclei"
              value={fmtNumber(metrics.total_nuclei.median)}
              ci={`${fmtNumber(metrics.total_nuclei.low)} – ${fmtNumber(metrics.total_nuclei.high)}`}
            />
            <Metric
              label="Tumor nuclei"
              value={fmtNumber(metrics.tumor_nuclei.median)}
              ci={`${fmtNumber(metrics.tumor_nuclei.low)} – ${fmtNumber(metrics.tumor_nuclei.high)}`}
            />
          </div>
          <div className="grid grid-cols-2 gap-3 text-center mt-1">
            <Metric
              label="Area"
              value={`${metrics.area_mm2.toFixed(2)} mm²`}
              ci={`${metrics.n_tiles} tiles · ${metrics.tiles_with_data} with tissue`}
            />
            <Metric
              label="Adequacy confidence"
              value={fmtPct(metrics.adequacy_probability)}
              ci={
                verdict?.thresholds
                  ? `≥${fmtPct(verdict.thresholds.purity_min)} · ≥${fmtNumber(
                      verdict.thresholds.tumor_cells_min,
                    )} tumor nuclei`
                  : ""
              }
            />
          </div>
          {verdict && verdict.reasons.length > 0 && (
            <ul className="mt-2 list-disc pl-5 text-[12px] text-[var(--muted)] leading-relaxed">
              {verdict.reasons.map((reason, i) => (
                <li key={i}>{reason}</li>
              ))}
            </ul>
          )}
        </div>
      ) : (
        <p className="text-sm text-[var(--muted)]">Computing…</p>
      )}
      <p className="text-[10px] uppercase tracking-widest text-[var(--muted)] mt-1">
        Visual smoothing is decorative; numbers above are derived from the raw
        per-tile predictions.
      </p>
    </aside>
  );
}

function Metric({
  label,
  value,
  ci,
}: {
  label: string;
  value: string;
  ci?: string;
}) {
  return (
    <div className="rounded-lg bg-[var(--bg)]/40 border border-[var(--border)]/60 px-3 py-2">
      <div className="text-[10px] uppercase tracking-widest text-[var(--muted)] font-bold">
        {label}
      </div>
      <div className="font-mono text-lg font-bold text-[var(--text)] mt-0.5" data-metric-value={label.toLowerCase()}>
        {value}
      </div>
      {ci && (
        <div className="text-[10px] text-[var(--muted)] mt-0.5">{ci}</div>
      )}
    </div>
  );
}
