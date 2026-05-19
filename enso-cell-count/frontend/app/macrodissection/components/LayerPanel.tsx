"use client";

import type { LayerName, SmoothingMode } from "@/lib/macrodissection/types";
import { THRESHOLD_PROFILES } from "@/lib/macrodissection/thresholds";

export interface LayerPanelState {
  layer: LayerName;
  opacity: number;
  smoothing: SmoothingMode;
  showRawTiles: boolean;
  profileName: string;
  roiToolEnabled: boolean;
}

export interface LayerPanelProps {
  state: LayerPanelState;
  onChange: (next: LayerPanelState) => void;
}

const LAYER_OPTIONS: { value: LayerName; label: string; description: string }[] = [
  {
    value: "adequacy",
    label: "Adequacy",
    description: "Estimated tumor nuclei per tile — primary macrodissection guide.",
  },
  {
    value: "purity",
    label: "Purity",
    description: "Tumor fraction estimated per tile (EnsoPurity).",
  },
  {
    value: "cellularity",
    label: "Cellularity",
    description: "Total nuclei estimated per tile (EnsoCellularity).",
  },
  {
    value: "uncertainty",
    label: "Uncertainty",
    description: "Per-tile model uncertainty band.",
  },
];

const SMOOTHING_OPTIONS: { value: SmoothingMode; label: string }[] = [
  { value: "overview", label: "Overview" },
  { value: "balanced", label: "Balanced" },
  { value: "detail", label: "Tile detail" },
];

export default function LayerPanel({ state, onChange }: LayerPanelProps) {
  return (
    <div className="flex flex-col gap-5 text-sm">
      <section>
        <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--muted)] mb-3">
          AI overlay
        </h3>
        <div className="grid grid-cols-2 gap-2">
          {LAYER_OPTIONS.map((opt) => (
            <button
              type="button"
              key={opt.value}
              onClick={() => onChange({ ...state, layer: opt.value })}
              className={`text-left rounded-lg border px-3 py-2 transition-colors ${
                state.layer === opt.value
                  ? "border-orange-500/60 bg-orange-500/10 text-[var(--text)]"
                  : "border-[var(--border)] bg-[var(--surface)] hover:border-[var(--muted)]/70"
              }`}
              data-layer-button={opt.value}
              aria-pressed={state.layer === opt.value}
            >
              <span className="block font-semibold">{opt.label}</span>
              <span className="block text-[11px] text-[var(--muted)] leading-snug mt-0.5">
                {opt.description}
              </span>
            </button>
          ))}
        </div>
      </section>

      <section className="grid grid-cols-1 gap-3">
        <label className="flex flex-col gap-1">
          <span className="text-[11px] uppercase tracking-wider text-[var(--muted)] font-bold">
            Overlay opacity — {Math.round(state.opacity * 100)}%
          </span>
          <input
            type="range"
            min={0}
            max={1}
            step={0.01}
            value={state.opacity}
            onChange={(e) =>
              onChange({ ...state, opacity: parseFloat(e.target.value) })
            }
            className="accent-orange-500"
            data-opacity-slider
          />
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-[11px] uppercase tracking-wider text-[var(--muted)] font-bold">
            Visual smoothing
          </span>
          <div className="flex gap-2">
            {SMOOTHING_OPTIONS.map((opt) => (
              <button
                type="button"
                key={opt.value}
                onClick={() => onChange({ ...state, smoothing: opt.value })}
                className={`flex-1 rounded-md px-2 py-1.5 text-xs border ${
                  state.smoothing === opt.value
                    ? "border-orange-500 bg-orange-500/15 text-[var(--text)]"
                    : "border-[var(--border)] bg-[var(--surface)] text-[var(--muted)]"
                }`}
                data-smoothing-button={opt.value}
              >
                {opt.label}
              </button>
            ))}
          </div>
          <span className="text-[11px] text-[var(--muted)]">
            Visual only — ROI metrics always use the raw tile predictions.
          </span>
        </label>
      </section>

      <section className="flex flex-col gap-2">
        <label className="text-[11px] uppercase tracking-wider text-[var(--muted)] font-bold">
          Threshold profile
        </label>
        <select
          value={state.profileName}
          onChange={(e) =>
            onChange({ ...state, profileName: e.target.value })
          }
          className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-sm"
          data-profile-select
        >
          {Object.values(THRESHOLD_PROFILES).map((p) => (
            <option key={p.name} value={p.name}>
              {p.display_name}
            </option>
          ))}
        </select>
        <span className="text-[11px] text-[var(--muted)] leading-snug">
          {THRESHOLD_PROFILES[state.profileName]?.notes}
        </span>
      </section>

      <section className="flex flex-col gap-2">
        <label className="text-[11px] uppercase tracking-wider text-[var(--muted)] font-bold">
          ROI tool
        </label>
        <button
          type="button"
          onClick={() =>
            onChange({ ...state, roiToolEnabled: !state.roiToolEnabled })
          }
          className={`rounded-md px-3 py-2 text-sm border ${
            state.roiToolEnabled
              ? "border-red-500 bg-red-500/15 text-[var(--text)]"
              : "border-[var(--border)] bg-[var(--surface)] text-[var(--text)] hover:border-[var(--muted)]"
          }`}
          data-roi-tool-toggle
          aria-pressed={state.roiToolEnabled}
        >
          {state.roiToolEnabled
            ? "Stop drawing (pan / zoom mode)"
            : "Draw macrodissection ROI"}
        </button>
        <span className="text-[11px] text-[var(--muted)] leading-snug">
          Click to add vertices; double-click or close to finish; drag white
          handles to edit. Toggle off to pan / zoom the slide.
        </span>
      </section>
    </div>
  );
}
