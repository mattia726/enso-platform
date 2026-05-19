"use client";

export interface CandidateView {
  rank: number;
  score: number;
  purity_point: number;
  total_nuclei_point: number;
  tumor_nuclei_point: number;
  adequacy_probability: number;
}

export interface CandidateListProps<C extends CandidateView> {
  candidates: readonly C[] | null;
  onSelect: (cand: C) => void;
  loading?: boolean;
}

export default function CandidateList<C extends CandidateView>({
  candidates,
  onSelect,
  loading,
}: CandidateListProps<C>) {
  return (
    <div className="flex flex-col gap-2" data-candidate-list>
      <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--muted)]">
        Candidate areas
      </h3>
      {loading && (
        <p className="text-xs text-[var(--muted)]">Searching for tumor-rich regions…</p>
      )}
      {!loading && (!candidates || candidates.length === 0) && (
        <p className="text-xs text-[var(--muted)] leading-snug">
          No candidates yet. Use the button below to surface the most
          adequate macrodissection regions; the pathologist always reviews
          and edits the final boundary.
        </p>
      )}
      {candidates && candidates.length > 0 && (
        <ul className="grid grid-cols-1 gap-2">
          {candidates.map((c) => (
            <li key={c.rank}>
              <button
                type="button"
                onClick={() => onSelect(c)}
                className="w-full text-left rounded-lg border border-[var(--border)] bg-[var(--surface)] hover:border-orange-500/60 hover:bg-orange-500/10 px-3 py-2 transition-colors"
                data-candidate-rank={c.rank}
              >
                <div className="flex items-baseline justify-between">
                  <span className="text-sm font-bold text-[var(--text)]">
                    Candidate {String.fromCharCode(64 + c.rank)}
                  </span>
                  <span className="text-[11px] text-[var(--muted)]">
                    {Math.round(c.adequacy_probability * 100)}% adequacy
                  </span>
                </div>
                <div className="text-[11px] text-[var(--muted)] mt-1 grid grid-cols-3 gap-2">
                  <span>
                    Purity{" "}
                    <span className="text-[var(--text)] font-mono">
                      {Math.round(c.purity_point * 100)}%
                    </span>
                  </span>
                  <span>
                    Tumor nuclei{" "}
                    <span className="text-[var(--text)] font-mono">
                      {Math.round(c.tumor_nuclei_point).toLocaleString("en-US")}
                    </span>
                  </span>
                  <span>
                    Total nuclei{" "}
                    <span className="text-[var(--text)] font-mono">
                      {Math.round(c.total_nuclei_point).toLocaleString("en-US")}
                    </span>
                  </span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
