"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type OpenSeadragon from "openseadragon";

import WSIViewer from "./components/WSIViewer";
import HeatmapOverlay from "./components/HeatmapOverlay";
import ROILayer, { type ROIDraft } from "./components/ROILayer";
import LayerPanel, { type LayerPanelState } from "./components/LayerPanel";
import RoiMetricsCard from "./components/RoiMetricsCard";
import CaseSidebar from "./components/CaseSidebar";
import RoiHistoryList from "./components/RoiHistoryList";
import CandidateList from "./components/CandidateList";
import ReportSheet from "./components/ReportSheet";

import { computeROIMetrics } from "@/lib/macrodissection/metrics";
import { labelAdequacy } from "@/lib/macrodissection/adequacy";
import { THRESHOLD_PROFILES } from "@/lib/macrodissection/thresholds";
import { loadStaticCaseList } from "@/lib/macrodissection/static-cases";
import { loadTileArraysForCase } from "@/lib/macrodissection/tiles";
import { suggestCandidates, type ClientCandidate } from "@/lib/macrodissection/candidates";
import type {
  AdequacyVerdict,
  CaseMeta,
  ROIMetrics,
  TileArrays,
} from "@/lib/macrodissection/types";

export interface SavedROI {
  id: string;
  label: string;
  caseId: number;
  points: readonly [number, number][];
  locked: boolean;
  userId: string;
  metrics: ROIMetrics | null;
  verdict: AdequacyVerdict | null;
  thresholds: typeof THRESHOLD_PROFILES.humanitas_ngs;
  createdAt: string;
  updatedAt: string;
  revision: number;
}

const DEFAULT_LAYER_STATE: LayerPanelState = {
  layer: "adequacy",
  opacity: 0.65,
  smoothing: "balanced",
  showRawTiles: false,
  profileName: "humanitas_ngs",
  roiToolEnabled: true,
};

function nowIso(): string {
  return new Date().toISOString();
}

function uid(): string {
  return `roi_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 8)}`;
}

export default function MacrodissectionClient() {
  const [cases, setCases] = useState<CaseMeta[]>([]);
  const [selectedCaseId, setSelectedCaseId] = useState<number | null>(null);
  const [tiles, setTiles] = useState<TileArrays | null>(null);
  const [tilesError, setTilesError] = useState<string | null>(null);
  const [layerState, setLayerState] = useState<LayerPanelState>(DEFAULT_LAYER_STATE);
  const [draft, setDraft] = useState<ROIDraft | null>(null);
  const [savedROIs, setSavedROIs] = useState<SavedROI[]>([]);
  const [selectedROIId, setSelectedROIId] = useState<string | null>(null);
  const [previewMetrics, setPreviewMetrics] = useState<ROIMetrics | null>(null);
  const [previewVerdict, setPreviewVerdict] = useState<AdequacyVerdict | null>(null);
  const [previewStatus, setPreviewStatus] = useState<"idle" | "computing" | "ready" | "no_roi">("no_roi");
  const [candidates, setCandidates] = useState<ClientCandidate[] | null>(null);
  const [reportOpen, setReportOpen] = useState(false);
  const [viewer, setViewer] = useState<OpenSeadragon.Viewer | null>(null);
  const debounceRef = useRef<number | null>(null);

  // The macrodissection workbench is always shown with the clinical dark
  // theme, regardless of what the rest of the site is using. We restore
  // the original theme class on unmount.
  useEffect(() => {
    const html = document.documentElement;
    const had = html.classList.contains("dark");
    html.classList.add("dark");
    return () => {
      if (!had) html.classList.remove("dark");
    };
  }, []);

  const selectedCase = useMemo(
    () => cases.find((c) => c.case_id === selectedCaseId) ?? null,
    [cases, selectedCaseId],
  );

  const profile = useMemo(
    () => THRESHOLD_PROFILES[layerState.profileName] ?? THRESHOLD_PROFILES.humanitas_ngs,
    [layerState.profileName],
  );

  // Load case list on mount.
  useEffect(() => {
    loadStaticCaseList()
      .then((cs) => {
        setCases(cs);
        if (cs.length > 0) {
          // Default to case 1 (TCGA-THYM, high-purity tumor) for the demo
          // because it shows the brightest adequacy overlay; fall back to
          // any case if case_1 is missing.
          const preferred = cs.find((c) => c.case_id === 1) ?? cs[0];
          setSelectedCaseId(preferred.case_id);
        }
      })
      .catch((err) => {
        // eslint-disable-next-line no-console
        console.error("failed to load case list", err);
      });
  }, []);

  // Load tile arrays whenever the selected case changes.
  useEffect(() => {
    if (!selectedCase) return;
    setTilesError(null);
    setTiles(null);
    setDraft(null);
    setSavedROIs([]);
    setSelectedROIId(null);
    setPreviewMetrics(null);
    setPreviewVerdict(null);
    setPreviewStatus("no_roi");
    setCandidates(null);
    loadTileArraysForCase(selectedCase)
      .then(setTiles)
      .catch((err) => {
        setTilesError(err instanceof Error ? err.message : String(err));
      });
  }, [selectedCase]);

  // Recompute preview metrics whenever the draft polygon or profile changes.
  useEffect(() => {
    if (!tiles || !draft || !draft.closed || draft.points.length < 3) {
      setPreviewMetrics(null);
      setPreviewVerdict(null);
      setPreviewStatus("no_roi");
      return;
    }
    setPreviewStatus("computing");
    if (debounceRef.current) {
      window.clearTimeout(debounceRef.current);
    }
    const polygon = draft.points;
    debounceRef.current = window.setTimeout(() => {
      const metrics = computeROIMetrics(polygon, tiles, {
        thresholdsPurityMin: profile.purity_min,
        thresholdsTumorCellsMin: profile.tumor_cells_min,
        nSamples: 250,
      });
      const verdict = labelAdequacy(metrics, profile);
      setPreviewMetrics(metrics);
      setPreviewVerdict(verdict);
      setPreviewStatus("ready");
    }, 50);
    return () => {
      if (debounceRef.current) window.clearTimeout(debounceRef.current);
    };
  }, [draft, tiles, profile]);

  const handleSaveDraft = useCallback(() => {
    if (!draft || !draft.closed || !tiles || !selectedCase) return;
    if (!previewMetrics || !previewVerdict) return;
    const id = uid();
    const rec: SavedROI = {
      id,
      label: `ROI ${savedROIs.length + 1}`,
      caseId: selectedCase.case_id,
      points: draft.points.map(([x, y]) => [x, y] as [number, number]),
      locked: false,
      userId: "demo-pathologist",
      metrics: previewMetrics,
      verdict: previewVerdict,
      thresholds: profile,
      createdAt: nowIso(),
      updatedAt: nowIso(),
      revision: 1,
    };
    setSavedROIs((prev) => [...prev, rec]);
    setSelectedROIId(id);
  }, [draft, tiles, selectedCase, previewMetrics, previewVerdict, savedROIs.length, profile]);

  const handleLock = useCallback(
    (id: string) => {
      setSavedROIs((prev) =>
        prev.map((r) =>
          r.id === id
            ? {
                ...r,
                locked: true,
                updatedAt: nowIso(),
                revision: r.revision + 1,
              }
            : r,
        ),
      );
    },
    [],
  );

  const handleDelete = useCallback(
    (id: string) => {
      setSavedROIs((prev) => prev.filter((r) => r.id !== id || r.locked));
      setSelectedROIId((cur) => (cur === id ? null : cur));
    },
    [],
  );

  const handleSelectROI = useCallback(
    (id: string) => {
      const rec = savedROIs.find((r) => r.id === id);
      if (!rec) return;
      setSelectedROIId(id);
      setDraft({ points: rec.points.map(([x, y]) => [x, y]), closed: true });
    },
    [savedROIs],
  );

  const handleAutoSuggest = useCallback(() => {
    if (!tiles) return;
    const cands = suggestCandidates(tiles, profile, { windowTiles: 6, topK: 5 });
    setCandidates(cands);
  }, [tiles, profile]);

  const handleCandidatePick = useCallback((cand: ClientCandidate) => {
    setDraft({
      points: cand.polygon.map(([x, y]) => [x, y] as [number, number]),
      closed: true,
    });
  }, []);

  const selectedSavedROI = savedROIs.find((r) => r.id === selectedROIId) ?? null;
  const reportROI = selectedSavedROI && {
    id: selectedSavedROI.id,
    label: selectedSavedROI.label,
    points: selectedSavedROI.points,
    locked: selectedSavedROI.locked,
    userId: selectedSavedROI.userId,
    revision: selectedSavedROI.revision,
    createdAt: selectedSavedROI.createdAt,
    updatedAt: selectedSavedROI.updatedAt,
  };

  const lockedPolygons = useMemo(
    () =>
      savedROIs
        .filter((r) => r.locked || (selectedROIId !== null && r.id !== selectedROIId))
        .map((r) => ({
          id: r.id,
          points: r.points,
          label: r.locked ? `${r.label} • locked` : r.label,
        })),
    [savedROIs, selectedROIId],
  );

  return (
    <div className="h-screen w-screen flex flex-col bg-[var(--bg)] text-[var(--text)] overflow-hidden">
      <header className="flex items-center justify-between gap-4 border-b border-[var(--border)] px-4 py-2.5">
        <div className="flex items-center gap-3 min-w-0">
          <a href="/" className="text-sm font-bold text-[#007ba7] hover:underline">
            Enso · Macrodissection Workbench
          </a>
          <span className="text-[11px] uppercase tracking-widest text-[var(--muted)] hidden md:inline">
            Pathologist-in-the-loop ROI adequacy guide
          </span>
        </div>
        <div className="flex items-center gap-3 text-xs text-[var(--muted)]">
          {selectedCase ? (
            <>
              <span>
                Case <strong className="text-[var(--text)]">{selectedCase.case_id}</strong>
              </span>
              <span className="hidden md:inline">{selectedCase.barcode || ""}</span>
              <span className="hidden md:inline">
                {selectedCase.grid_nx}×{selectedCase.grid_ny} tiles
              </span>
              <span className="hidden md:inline">
                models {selectedCase.purity_model_version} ·{" "}
                {selectedCase.cellularity_model_version}
              </span>
            </>
          ) : (
            <span>Loading…</span>
          )}
          <button
            type="button"
            disabled={!selectedSavedROI}
            onClick={() => setReportOpen(true)}
            className="rounded border border-[var(--border)] bg-[var(--surface)] px-3 py-1.5 text-xs font-semibold text-[var(--text)] disabled:opacity-40 hover:border-orange-500/70"
            data-open-report
          >
            Export macrodissection sheet
          </button>
        </div>
      </header>
      <div className="flex flex-1 min-h-0">
        <div className="w-56 shrink-0">
          <CaseSidebar
            cases={cases}
            selectedCaseId={selectedCaseId}
            onSelect={(id) => setSelectedCaseId(id)}
          />
        </div>
        <main className="flex-1 min-w-0 flex flex-col bg-black">
          <div className="flex-1 min-h-0 relative">
            {!selectedCase && (
              <div className="absolute inset-0 flex items-center justify-center text-[var(--muted)]">
                Select a case to begin
              </div>
            )}
            {selectedCase && tilesError && (
              <div className="absolute inset-0 flex items-center justify-center text-red-300 text-sm px-4">
                {tilesError}
              </div>
            )}
            {selectedCase && !tilesError && (
              <WSIViewer
                key={`viewer-${selectedCase.case_id}`}
                baseImageUrl={selectedCase.base_image}
                width={selectedCase.base_width || 800}
                height={selectedCase.base_height || 600}
                onReady={(v) => setViewer(v)}
              >
                {tiles && viewer && (
                  <HeatmapOverlay
                    viewer={viewer}
                    tiles={tiles}
                    layer={layerState.layer}
                    opacity={layerState.opacity}
                    smoothing={layerState.smoothing}
                    baseWidth={selectedCase.base_width}
                    baseHeight={selectedCase.base_height}
                  />
                )}
                {viewer && (
                  <ROILayer
                    viewer={viewer}
                    baseWidth={selectedCase.base_width}
                    baseHeight={selectedCase.base_height}
                    draft={draft}
                    onDraftChange={setDraft}
                    enabled={layerState.roiToolEnabled}
                    lockedPolygons={lockedPolygons}
                  />
                )}
              </WSIViewer>
            )}
          </div>
        </main>
        <aside className="w-80 shrink-0 border-l border-[var(--border)] flex flex-col overflow-hidden bg-[var(--bg)]">
          <div className="flex-1 overflow-y-auto px-4 py-4 flex flex-col gap-5">
            <LayerPanel state={layerState} onChange={setLayerState} />
            <RoiMetricsCard
              metrics={previewMetrics}
              verdict={previewVerdict}
              status={previewStatus}
            />
            <div className="flex flex-wrap gap-2">
              <button
                type="button"
                onClick={handleSaveDraft}
                disabled={!draft?.closed || !previewMetrics}
                className="rounded-md bg-orange-500 text-black px-3 py-2 text-xs font-semibold disabled:opacity-40 hover:bg-orange-400"
                data-save-roi
              >
                Save ROI
              </button>
              <button
                type="button"
                onClick={() => setDraft(null)}
                disabled={!draft}
                className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs font-semibold disabled:opacity-40"
                data-clear-draft
              >
                Clear draft
              </button>
              <button
                type="button"
                onClick={handleAutoSuggest}
                disabled={!tiles}
                className="rounded-md border border-[var(--border)] bg-[var(--surface)] px-3 py-2 text-xs font-semibold disabled:opacity-40 ml-auto"
                data-auto-suggest
              >
                Auto-suggest candidates
              </button>
            </div>
            <CandidateList
              candidates={candidates}
              onSelect={handleCandidatePick}
            />
            <div>
              <h3 className="text-xs font-bold uppercase tracking-widest text-[var(--muted)] mb-2">
                Saved ROIs
              </h3>
              <RoiHistoryList
                rois={savedROIs}
                onSelect={handleSelectROI}
                selectedId={selectedROIId}
                onLock={handleLock}
                onDelete={handleDelete}
              />
            </div>
          </div>
        </aside>
      </div>
      <ReportSheet
        open={reportOpen}
        onClose={() => setReportOpen(false)}
        caseMeta={selectedCase}
        roi={reportROI}
        metrics={selectedSavedROI?.metrics ?? null}
        verdict={selectedSavedROI?.verdict ?? null}
        threshold={selectedSavedROI?.thresholds ?? null}
      />
    </div>
  );
}
