"use client";

import type { SavedROI } from "../MacrodissectionClient";

export interface RoiHistoryListProps {
  rois: readonly SavedROI[];
  onSelect: (id: string) => void;
  selectedId: string | null;
  onLock: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function RoiHistoryList({
  rois,
  onSelect,
  selectedId,
  onLock,
  onDelete,
}: RoiHistoryListProps) {
  if (rois.length === 0) {
    return (
      <p className="text-xs text-[var(--muted)] leading-snug">
        Saved ROIs appear here. Drafts can be edited; locked ROIs become
        permanent records with an audit trail.
      </p>
    );
  }
  return (
    <ul className="flex flex-col gap-2" data-roi-history>
      {rois.map((r) => (
        <li key={r.id}>
          <div
            className={`rounded-lg border px-3 py-2 transition-colors ${
              selectedId === r.id
                ? "border-orange-500 bg-orange-500/10"
                : "border-[var(--border)] bg-[var(--surface)] hover:border-[var(--muted)]"
            }`}
          >
            <button
              type="button"
              onClick={() => onSelect(r.id)}
              className="w-full text-left"
              data-roi-history-item={r.id}
            >
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-[var(--text)]">
                  {r.label}
                </span>
                <span
                  className={`text-[10px] uppercase tracking-widest font-bold ${
                    r.locked ? "text-green-400" : "text-amber-300"
                  }`}
                >
                  {r.locked ? "Locked" : "Draft"}
                </span>
              </div>
              <div className="text-[11px] text-[var(--muted)] mt-1">
                {r.metrics
                  ? `${Math.round(r.metrics.purity.median * 100)}% purity · ${Math.round(r.metrics.tumor_nuclei.median).toLocaleString("en-US")} tumor nuclei`
                  : "—"}
              </div>
            </button>
            <div className="mt-2 flex gap-2 text-[11px]">
              {!r.locked && (
                <button
                  type="button"
                  onClick={() => onLock(r.id)}
                  className="rounded px-2 py-1 bg-green-500/15 text-green-300 border border-green-500/40 hover:bg-green-500/25"
                  data-roi-lock={r.id}
                >
                  Lock & sign
                </button>
              )}
              {!r.locked && (
                <button
                  type="button"
                  onClick={() => onDelete(r.id)}
                  className="rounded px-2 py-1 bg-red-500/10 text-red-300 border border-red-500/30 hover:bg-red-500/20"
                  data-roi-delete={r.id}
                >
                  Delete
                </button>
              )}
            </div>
          </div>
        </li>
      ))}
    </ul>
  );
}
