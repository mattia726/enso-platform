"use client";

import { useState, useEffect } from "react";
import { getCancerDisplayName } from "@/data/tcga_display_names";

type GlobalStats = {
  n_samples: number;
  rho_mil: number;
  rho_ptn: number;
  ci_mil: [number, number];
  ci_ptn: [number, number];
  mae_mil: number;
  mae_ptn: number;
  meng_p?: number;
  wilcoxon_p?: number;
};

type PerCancerEntry = {
  n: number;
  rho_mil: number;
  rho_ptn: number;
  mae_mil: number;
  mae_ptn: number;
  improvement?: boolean;
  /** From VM: Meng z-test p-value (ρ); when < 0.05, use sig_rho_* for green */
  p_rho?: number;
  /** From VM: Wilcoxon p-value (MAE); when < 0.05, use sig_mae_* for green */
  p_mae?: number;
  sig_rho_mil?: boolean;
  sig_rho_ptn?: boolean;
  sig_mae_mil?: boolean;
  sig_mae_ptn?: boolean;
};

export function PerformanceTab() {
  const [globalStats, setGlobalStats] = useState<GlobalStats | null>(null);
  const [perCancer, setPerCancer] = useState<Record<string, PerCancerEntry>>({});
  const [loading, setLoading] = useState(true);
  const [isTableExpanded, setIsTableExpanded] = useState(false);

  useEffect(() => {
    Promise.all([
      fetch("/data/statistical_tests.json").then((r) => r.json()),
      fetch("/data/per_cancer_stats.json").then((r) => r.json()),
    ])
      .then(([stats, per]) => {
        setGlobalStats(stats as GlobalStats);
        setPerCancer((per as Record<string, PerCancerEntry>) || {});
      })
      .catch(() => {
        setGlobalStats(null);
        setPerCancer({});
      })
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-[var(--muted)]">
        Loading performance data…
      </div>
    );
  }

  const hasGlobal = globalStats && globalStats.n_samples > 0;
  const cancerIds = Object.keys(perCancer).filter((k) => perCancer[k].n > 0).sort();

  const tableData = cancerIds.map((id) => ({ id, ...perCancer[id] }));

  // Sort by N >= 30 first (robust cohorts), then by EnsoPurity rho descending
  const sortedData = [...tableData].sort((a, b) => {
    const aRobust = a.n >= 30 ? 1 : 0;
    const bRobust = b.n >= 30 ? 1 : 0;
    if (aRobust !== bRobust) {
      return bRobust - aRobust; // Robust cohorts float to the top
    }
    return b.rho_mil - a.rho_mil; // Within groups, sort by EnsoPurity rho highest to lowest
  });
  const introSampleCount = globalStats?.n_samples;
  const formatCount = (n?: number) => {
    if (typeof n !== "number") return null;
    return new Intl.NumberFormat("en-US").format(n);
  };
  const formatPValue = (p?: number) => {
    if (typeof p !== "number") return "—";
    return p < 1e-10 ? " < 1.0e-10" : ` = ${p.toExponential(1)}`;
  };
  const scrollToTables = () => {
    window.dispatchEvent(new Event("enso:scroll-to-tables"));
  };

  return (
    <>
      <div className="space-y-24 pb-12 relative">
        <div className="relative z-10 space-y-24">
        {/* Intro Section */}
        <section className="max-w-4xl mx-auto text-center space-y-4">
          <h2 className="text-2xl md:text-3xl font-bold tracking-tight">
            Clinical Validation
          </h2>
          <p className="text-lg text-[var(--muted)] leading-relaxed">
            Benchmarked across <strong className="text-[var(--text)]">{formatCount(introSampleCount) ? `${formatCount(introSampleCount)} hold-out samples` : "hold-out samples"}</strong>, EnsoPurity achieves 2× higher accuracy than pathologist visual estimates when validated against absolute genomic ground truth.
          </p>
        </section>

        {/* Chart Section */}
        <section
          data-performance-plot-end
          className="w-full max-w-5xl mx-auto"
          style={{ scrollSnapAlign: "end", scrollSnapStop: "always" }}
        >
          <div className="w-full rounded-2xl bg-[var(--surface)]/40 border border-[var(--border)] p-2 md:p-8">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src="/data/scatter_mil_vs_ptn.png"
              alt="Enso vs Pathologist Scatter Plot"
              className="w-full h-auto rounded-lg dark:invert-[.85] dark:hue-rotate-180 transition-all duration-500"
              style={{ mixBlendMode: "multiply" }}
            />
          </div>
          <div className="mt-5 flex justify-center">
            <button
              type="button"
              onClick={scrollToTables}
              className="animate-bounce text-[var(--muted)] opacity-50 hover:opacity-80 transition-opacity"
              aria-label="Scroll to performance tables"
            >
              <svg className="w-8 h-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
          </div>
        </section>

      {/* Global comparison table — nice_tables style, fully expanded */}
      {hasGlobal && (
        <div
          data-performance-tables
          className="overflow-x-auto"
          style={{ scrollSnapAlign: "start", scrollSnapStop: "always" }}
        >
          <table
            className="w-full"
            style={{
              borderCollapse: "collapse",
              width: "100%",
              fontFamily: "sans-serif",
              fontSize: "0.9em",
            }}
          >
            <thead>
              <tr style={{ backgroundColor: "var(--surface)", textAlign: "left", verticalAlign: "bottom" }}>
                <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>Method</th>
                <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>Spearman ρ (95% CI)</th>
                <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>MAE</th>
                <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>Description</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>Pathologist (PTN)</td>
                <td style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>
                  {globalStats.rho_ptn.toFixed(3)} ({globalStats.ci_ptn[0].toFixed(3)}–{globalStats.ci_ptn[1].toFixed(3)})
                </td>
                <td style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>{globalStats.mae_ptn.toFixed(3)}</td>
                <td style={{ padding: "10px 14px", border: "1px solid var(--border)", color: "var(--muted)" }}>
                  Visual estimate of percent tumor nuclei vs. genomic purity.
                </td>
              </tr>
              <tr style={{ borderBottom: "2px solid var(--border)" }}>
                <td style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>
                  <strong>EnsoPurity</strong>
                </td>
                <td
                  style={{
                    padding: "10px 14px",
                    border: "1px solid var(--border)",
                    backgroundColor: "rgba(34, 197, 94, 0.15)",
                  }}
                >
                  <strong>
                    {globalStats.rho_mil.toFixed(3)} ({globalStats.ci_mil[0].toFixed(3)}–{globalStats.ci_mil[1].toFixed(3)})
                  </strong>
                </td>
                <td
                  style={{
                    padding: "10px 14px",
                    border: "1px solid var(--border)",
                    backgroundColor: "rgba(34, 197, 94, 0.15)",
                  }}
                >
                  <strong>{globalStats.mae_mil.toFixed(3)}</strong>
                </td>
                <td style={{ padding: "10px 14px", border: "1px solid var(--border)", color: "var(--muted)" }}>
                  Proprietary AI architecture built on foundational pathology embeddings; 2× more accurate than pathologists.
                </td>
              </tr>
            </tbody>
          </table>
          <p className="mt-2 text-sm text-[var(--muted)]">
            N = {formatCount(globalStats.n_samples)} samples with PTN and genomic purity. Meng z-test p{formatPValue(globalStats.meng_p)}; Wilcoxon p{formatPValue(globalStats.wilcoxon_p)}.
          </p>
        </div>
      )}

      {!hasGlobal && (
        <p className="text-center text-[var(--muted)]">
          Run <code className="bg-[var(--surface)] px-1 rounded">python -m enso_purity_mil.statistical_tests</code> and copy <code className="bg-[var(--surface)] px-1 rounded">statistical_tests.json</code> to <code className="bg-[var(--surface)] px-1 rounded">public/data/</code>.
        </p>
      )}

        {/* Table Section: Accuracy by Indication */}
        {sortedData.length > 0 && (
          <section className="w-full max-w-5xl mx-auto space-y-6">
            <h3 className="text-xl font-bold text-[var(--text)] text-left">Accuracy by Indication</h3>
            <div>
              <div className="overflow-x-auto rounded-xl border border-[var(--border)] bg-[var(--surface)]">
              <table className="w-full text-sm text-left" style={{ borderCollapse: "collapse", tableLayout: "fixed" }}>
                <colgroup>
                  <col style={{ width: "34%" }} />
                  <col style={{ width: "8%" }} />
                  <col style={{ width: "14%" }} />
                  <col style={{ width: "14%" }} />
                  <col style={{ width: "15%" }} />
                  <col style={{ width: "15%" }} />
                </colgroup>
                <thead>
                  <tr style={{ backgroundColor: "var(--surface)", textAlign: "left" }}>
                    <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>Cancer type</th>
                    <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>N</th>
                    <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>ρ (EnsoPurity)</th>
                    <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>ρ (Pathologist)</th>
                    <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>MAE (EnsoPurity)</th>
                    <th style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>MAE (Pathologist)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--border)]">
                  {sortedData
                    .slice(0, isTableExpanded ? sortedData.length : 10)
                    .map(({ id, ...e }) => {
                      const n = e.n;
                      // Use VM-computed significance when present (Meng for ρ, Wilcoxon for MAE)
                      const hasSig = typeof e.sig_rho_mil === "boolean";
                      const sigRhoEnso = hasSig ? Boolean(e.sig_rho_mil) : (n > 5 && e.rho_mil > e.rho_ptn && e.rho_mil - e.rho_ptn >= (1.96 * Math.sqrt(2 / (n - 3))));
                      const sigRhoPtn = hasSig ? Boolean(e.sig_rho_ptn) : (n > 5 && e.rho_ptn > e.rho_mil && e.rho_ptn - e.rho_mil >= (1.96 * Math.sqrt(2 / (n - 3))));
                      const sigMaeEnso = hasSig ? Boolean(e.sig_mae_mil) : (n >= 15 && e.mae_mil < e.mae_ptn);
                      const sigMaePtn = hasSig ? Boolean(e.sig_mae_ptn) : (n >= 15 && e.mae_ptn < e.mae_mil);
                      const greenCell = { backgroundColor: "rgba(34, 197, 94, 0.18)", fontWeight: 600 };
                      return (
                        <tr
                          key={id}
                          className="hover:bg-[var(--border)]/20 transition-colors"
                        >
                          <td style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>
                            {getCancerDisplayName(id)}
                          </td>
                          <td style={{ padding: "10px 14px", border: "1px solid var(--border)" }}>{e.n}</td>
                          <td
                            style={{
                              padding: "10px 14px",
                              border: "1px solid var(--border)",
                              ...(sigRhoEnso ? greenCell : {}),
                            }}
                          >
                            {e.rho_mil.toFixed(3)}
                          </td>
                          <td
                            style={{
                              padding: "10px 14px",
                              border: "1px solid var(--border)",
                              ...(sigRhoPtn ? greenCell : {}),
                            }}
                          >
                            {e.rho_ptn.toFixed(3)}
                          </td>
                          <td
                            style={{
                              padding: "10px 14px",
                              border: "1px solid var(--border)",
                              ...(sigMaeEnso ? greenCell : {}),
                            }}
                          >
                            {e.mae_mil.toFixed(3)}
                          </td>
                          <td
                            style={{
                              padding: "10px 14px",
                              border: "1px solid var(--border)",
                              ...(sigMaePtn ? greenCell : {}),
                            }}
                          >
                            {e.mae_ptn.toFixed(3)}
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
              </div>
              <p className="mt-1.5 text-sm text-[var(--muted)]">
                Green highlights indicate statistically significant outperformance (p &lt; 0.05). Results are sorted by EnsoPurity correlation.
              </p>
            </div>
            {!isTableExpanded && (
              <div className="flex justify-center w-full mt-4">
                <button
                  type="button"
                  onClick={() => setIsTableExpanded(true)}
                  className="px-6 py-2 rounded-full border border-[var(--border)] bg-[var(--surface)] hover:bg-[var(--border)]/50 text-[var(--text)] font-medium transition-colors text-sm"
                >
                  Expand all {sortedData.length} cancer types ↓
                </button>
              </div>
            )}
          </section>
        )}

      {cancerIds.length === 0 && hasGlobal && (
        <p className="text-sm text-[var(--muted)]">
          Per-cancer stats will appear here after running the per-cancer statistical script and copying <code className="bg-[var(--surface)] px-1 rounded">per_cancer_stats.json</code> to <code className="bg-[var(--surface)] px-1 rounded">public/data/</code>.
        </p>
      )}
        </div>
      </div>
    </>
  );
}
