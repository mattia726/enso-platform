"use client";

import type { CaseMeta } from "@/lib/macrodissection/types";

export interface CaseSidebarProps {
  cases: readonly CaseMeta[];
  selectedCaseId: number | null;
  onSelect: (caseId: number) => void;
}

export default function CaseSidebar({
  cases,
  selectedCaseId,
  onSelect,
}: CaseSidebarProps) {
  return (
    <aside
      className="flex flex-col h-full overflow-hidden bg-[var(--surface)]/60 border-r border-[var(--border)]"
      data-case-sidebar
    >
      <div className="flex items-center justify-between px-3 py-3 border-b border-[var(--border)]">
        <h2 className="text-xs font-bold uppercase tracking-widest text-[var(--muted)]">
          Cases
        </h2>
        <span className="text-xs text-[var(--muted)]">{cases.length}</span>
      </div>
      <div className="overflow-y-auto flex-1">
        <ul>
          {cases.map((c) => (
            <li key={c.case_id}>
              <button
                type="button"
                onClick={() => onSelect(c.case_id)}
                className={`w-full text-left px-3 py-2 border-b border-[var(--border)]/60 flex gap-3 items-center transition-colors ${
                  selectedCaseId === c.case_id
                    ? "bg-orange-500/15"
                    : "hover:bg-[var(--bg)]/40"
                }`}
                data-case-id={c.case_id}
                aria-pressed={selectedCaseId === c.case_id}
              >
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={c.base_image}
                  alt=""
                  loading="lazy"
                  className="w-14 h-10 object-cover rounded-sm bg-black"
                />
                <div className="flex flex-col">
                  <span className="text-sm font-semibold text-[var(--text)] leading-tight">
                    Case {c.case_id}
                  </span>
                  <span className="text-[11px] text-[var(--muted)]">
                    {c.project_id || "—"}
                  </span>
                  <span className="text-[10px] text-[var(--muted)]/80">
                    {c.grid_nx}×{c.grid_ny} tiles · {c.n_tiles_tissue} tissue
                  </span>
                </div>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
