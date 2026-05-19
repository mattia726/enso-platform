"use client";

import { useState, useEffect, useRef, useCallback, useMemo, type ReactNode, type WheelEvent } from "react";
import { getCancerDisplayName } from "@/data/tcga_display_names";
import { HeroTypewriter } from "@/components/HeroTypewriter";

type GalleryRow = {
  file_uuid_original: string;
  file_uuid_new?: string;
  barcode: string;
  project_id: string;
  aliquot_barcode: string;
  expected: number;
  predicted: number;
  ptn: number;
  err_mil: number;
  err_ptn: number;
  originalFileIndex?: number;
};

type CaseExplorerProps = {
  onScroll?: (scrollTop: number) => void;
  onActiveViewChange?: (view: "explore" | "performance" | null) => void;
  performanceSlot?: ReactNode;
  scrollRequest?: { target: "explore" | "performance"; id: number } | null;
};

type SnapTarget = "hero" | "viewer" | "performance" | "plotEnd" | "tables";
type SnapPoint = {
  target: SnapTarget;
  top: number;
};

const SNAP_TARGET_TOLERANCE = 20;
const SNAP_SPEED_PX_PER_MS = 3.2;
const SNAP_MIN_DURATION_MS = 280;
const DEFAULT_SCROLL_SNAP_TYPE = "y proximity";
const DEFAULT_SCROLL_BEHAVIOR = "smooth";

const easeOutSine = (t: number) => {
  return Math.sin((t * Math.PI) / 2);
};

export function CaseExplorer({
  onScroll,
  onActiveViewChange,
  performanceSlot,
  scrollRequest,
}: CaseExplorerProps) {
  const [gallery, setGallery] = useState<GalleryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [index, setIndex] = useState(0);
  const [viewerUrl, setViewerUrl] = useState<string | null>(null);
  const [activeSection, setActiveSection] = useState<0 | 1>(0);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [staticAssetsError, setStaticAssetsError] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const section2Ref = useRef<HTMLElement>(null);
  const performanceRef = useRef<HTMLElement>(null);
  const activeViewRef = useRef<"explore" | "performance" | null>(null);
  const snapReleaseTimerRef = useRef<number | null>(null);
  const snapAnimationFrameRef = useRef<number | null>(null);
  const snapStyleRestoreRef = useRef<string | null>(null);
  const scrollBehaviorRestoreRef = useRef<string | null>(null);
  const snapTargetRef = useRef<SnapTarget | null>(null);

  useEffect(() => {
    fetch("/gallery/gallery_summary.csv")
      .then((r) => r.text())
      .then((text) => {
        const lines = text.trim().split("\n");
        if (lines.length < 2) {
          setGallery([]);
          return;
        }
        const header = lines[0].split(",").map((h) => h.trim());
        const rows: GalleryRow[] = [];
        for (let i = 1; i < lines.length; i++) {
          const values = lines[i].split(",");
          const row: Record<string, string | number | undefined> = {};
          header.forEach((h, j) => {
            const v = values[j]?.trim() ?? "";
            if (h === "expected" || h === "predicted" || h === "ptn" || h === "err_mil" || h === "err_ptn") row[h] = parseFloat(v) || 0;
            else if (h === "file_uuid_new") row[h] = v || undefined;
            else row[h] = v;
          });
          rows.push(row as unknown as GalleryRow);
        }
        setGallery(rows);
      })
      .catch(() => setGallery([]))
      .finally(() => setLoading(false));
  }, []);

  const displayGallery = useMemo(() => {
    const mapped = gallery.map((r, i) => ({
      ...r,
      originalFileIndex: i + 1,
    }));
    const swapped = [...mapped];

    const swapByName = (targetIndex: number, targetName: string) => {
      const currentIndex = swapped.findIndex(
        (r) => getCancerDisplayName(r.project_id) === targetName
      );
      if (currentIndex !== -1) {
        const temp = swapped[targetIndex];
        swapped[targetIndex] = swapped[currentIndex];
        swapped[currentIndex] = temp;
      }
    };

    swapByName(0, "Bladder Cancer");
    swapByName(1, "Lymphoid Neoplasm");
    swapByName(2, "Rectal Adenocarcinoma");

    return swapped;
  }, [gallery]);

  const row = displayGallery[index];
  useEffect(() => {
    if (!row) {
      setViewerUrl(null);
      return;
    }
    setViewerUrl(null);
    let cancelled = false;
    const candidates = [row.file_uuid_original];
    if (row.file_uuid_new) candidates.push(row.file_uuid_new);
    (async () => {
      for (const uuid of candidates) {
        if (cancelled) return;
        const res = await fetch(`/gallery/interactive_${uuid}.html`, { method: "HEAD" });
        if (res.ok) {
          setViewerUrl(`/gallery/interactive_${uuid}.html`);
          return;
        }
      }
      setViewerUrl(null);
    })();
    return () => { cancelled = true; };
  }, [row?.file_uuid_original, row?.file_uuid_new]);

  useEffect(() => {
    setStaticAssetsError(false);
  }, [index]);

  const getTopInContainer = useCallback((element: HTMLElement) => {
    const container = scrollRef.current;
    if (!container) return 0;
    const containerRect = container.getBoundingClientRect();
    const elementRect = element.getBoundingClientRect();
    return elementRect.top - containerRect.top + container.scrollTop;
  }, []);

  const computeActiveSection = useCallback((): 0 | 1 => {
    const container = scrollRef.current;
    const section2 = section2Ref.current;
    if (!container || !section2) return 0;
    const section2TopInContainer = getTopInContainer(section2);
    const viewportCenter = container.scrollTop + container.clientHeight / 2;
    return viewportCenter >= section2TopInContainer ? 1 : 0;
  }, [getTopInContainer]);

  const computeActiveView = useCallback((): "explore" | "performance" | null => {
    const container = scrollRef.current;
    const viewer = section2Ref.current;
    const performance = performanceRef.current;
    if (!container || !viewer || !performance) return null;

    const viewerTopInContainer = getTopInContainer(viewer);
    const performanceTopInContainer = getTopInContainer(performance);
    const activationPoint = container.scrollTop + container.clientHeight * 0.45;
    if (activationPoint >= performanceTopInContainer) return "performance";
    if (activationPoint >= viewerTopInContainer) return "explore";
    return null;
  }, [getTopInContainer]);

  const getViewerTargetTop = useCallback(() => {
    return section2Ref.current ? getTopInContainer(section2Ref.current) : 0;
  }, [getTopInContainer]);

  const getPerformanceTargetTop = useCallback(() => {
    return performanceRef.current ? getTopInContainer(performanceRef.current) : getViewerTargetTop();
  }, [getTopInContainer, getViewerTargetTop]);

  const getTablesTargetTop = useCallback(() => {
    const container = scrollRef.current;
    const tables = container?.querySelector<HTMLElement>("[data-performance-tables]");
    if (!container || !tables) return getPerformanceTargetTop();
    return Math.max(0, getTopInContainer(tables) - container.clientHeight * 0.05);
  }, [getPerformanceTargetTop, getTopInContainer]);

  const getPlotEndTargetTop = useCallback(() => {
    const container = scrollRef.current;
    const plotEnd = container?.querySelector<HTMLElement>("[data-performance-plot-end]");
    if (!container || !plotEnd) return getPerformanceTargetTop();
    return Math.max(0, getTopInContainer(plotEnd) + plotEnd.offsetHeight - container.clientHeight);
  }, [getPerformanceTargetTop, getTopInContainer]);

  const getSnapPoints = useCallback((): SnapPoint[] => {
    const container = scrollRef.current;
    const maxScrollTop = container ? Math.max(0, container.scrollHeight - container.clientHeight) : 0;
    const clampTop = (top: number) => Math.max(0, Math.min(maxScrollTop, top));
    const performanceTop = getPerformanceTargetTop();
    const plotEndTop = getPlotEndTargetTop();
    const tablesTop = getTablesTargetTop();
    const rawPoints: SnapPoint[] = [
      { target: "hero", top: 0 },
      { target: "viewer", top: getViewerTargetTop() },
    ];

    if (performanceSlot) {
      rawPoints.push({ target: "performance", top: performanceTop });
      if (plotEndTop > performanceTop + SNAP_TARGET_TOLERANCE && plotEndTop < tablesTop - SNAP_TARGET_TOLERANCE) {
        rawPoints.push({ target: "plotEnd", top: plotEndTop });
      }
      rawPoints.push({ target: "tables", top: tablesTop });
    }

    const points: SnapPoint[] = [];
    for (const point of rawPoints) {
      const normalized = { ...point, top: clampTop(point.top) };
      const previous = points[points.length - 1];
      if (!previous || normalized.top > previous.top + SNAP_TARGET_TOLERANCE) {
        points.push(normalized);
      }
    }
    return points;
  }, [getPerformanceTargetTop, getPlotEndTargetTop, getTablesTargetTop, getViewerTargetTop, performanceSlot]);

  const releaseSnapLock = useCallback((delay = 680) => {
    if (snapReleaseTimerRef.current) {
      window.clearTimeout(snapReleaseTimerRef.current);
    }
    snapReleaseTimerRef.current = window.setTimeout(() => {
      snapTargetRef.current = null;
      snapReleaseTimerRef.current = null;
    }, delay);
  }, []);

  const animateScrollTo = useCallback((top: number, keepSnapDisabled = false) => {
    const container = scrollRef.current;
    if (!container) return;

    if (snapAnimationFrameRef.current) {
      window.cancelAnimationFrame(snapAnimationFrameRef.current);
      snapAnimationFrameRef.current = null;
    }
    if (snapStyleRestoreRef.current !== null) {
      container.style.scrollSnapType = snapStyleRestoreRef.current;
      snapStyleRestoreRef.current = null;
    }
    if (scrollBehaviorRestoreRef.current !== null) {
      container.style.scrollBehavior = scrollBehaviorRestoreRef.current;
      scrollBehaviorRestoreRef.current = null;
    }

    const maxScrollTop = Math.max(0, container.scrollHeight - container.clientHeight);
    const targetTop = Math.max(0, Math.min(maxScrollTop, top));
    const startTop = container.scrollTop;
    const distance = targetTop - startTop;
    const duration = Math.max(SNAP_MIN_DURATION_MS, Math.abs(distance) / SNAP_SPEED_PX_PER_MS);

    if (Math.abs(distance) < 1) {
      container.scrollTop = targetTop;
      if (keepSnapDisabled) {
        container.style.scrollSnapType = "none";
      }
      releaseSnapLock(120);
      return;
    }

    snapStyleRestoreRef.current = DEFAULT_SCROLL_SNAP_TYPE;
    scrollBehaviorRestoreRef.current = container.style.scrollBehavior || DEFAULT_SCROLL_BEHAVIOR;
    container.style.scrollSnapType = "none";
    container.style.scrollBehavior = "auto";
    const startTime = performance.now();

    const step = (now: number) => {
      const elapsed = now - startTime;
      const progress = Math.min(1, elapsed / duration);
      container.scrollTop = startTop + distance * easeOutSine(progress);

      if (progress < 1) {
        snapAnimationFrameRef.current = window.requestAnimationFrame(step);
        return;
      }

      container.scrollTop = targetTop;
      if (keepSnapDisabled) {
        container.style.scrollSnapType = "none";
        snapStyleRestoreRef.current = null;
      } else if (snapStyleRestoreRef.current !== null) {
        container.style.scrollSnapType = snapStyleRestoreRef.current;
        snapStyleRestoreRef.current = null;
      }
      if (scrollBehaviorRestoreRef.current !== null) {
        container.style.scrollBehavior = scrollBehaviorRestoreRef.current;
        scrollBehaviorRestoreRef.current = null;
      }
      snapAnimationFrameRef.current = null;
      releaseSnapLock(120);
    };

    snapAnimationFrameRef.current = window.requestAnimationFrame(step);
  }, [releaseSnapLock]);

  const snapTo = useCallback((target: SnapTarget) => {
    const container = scrollRef.current;
    if (!container) return;

    const top =
      target === "hero"
        ? 0
        : target === "viewer"
          ? getViewerTargetTop()
          : target === "performance"
            ? getPerformanceTargetTop()
            : target === "plotEnd"
              ? getPlotEndTargetTop()
              : getTablesTargetTop();

    snapTargetRef.current = target;
    animateScrollTo(top, target === "tables");
  }, [animateScrollTo, getPerformanceTargetTop, getPlotEndTargetTop, getTablesTargetTop, getViewerTargetTop]);

  const handleScroll = useCallback(() => {
    const container = scrollRef.current;
    if (!container) return;
    onScroll?.(container.scrollTop);
    const nextActive = computeActiveSection();
    setActiveSection(nextActive);
    const nextView = computeActiveView();
    if (activeViewRef.current !== nextView) {
      activeViewRef.current = nextView;
      onActiveViewChange?.(nextView);
    }
  }, [computeActiveSection, computeActiveView, onActiveViewChange, onScroll]);

  useEffect(() => {
    handleScroll();
  }, [handleScroll]);

  useEffect(() => {
    return () => {
      const container = scrollRef.current;
      if (snapAnimationFrameRef.current) {
        window.cancelAnimationFrame(snapAnimationFrameRef.current);
      }
      if (container && snapStyleRestoreRef.current !== null) {
        container.style.scrollSnapType = snapStyleRestoreRef.current;
      }
      if (container && scrollBehaviorRestoreRef.current !== null) {
        container.style.scrollBehavior = scrollBehaviorRestoreRef.current;
      }
      if (snapReleaseTimerRef.current) {
        window.clearTimeout(snapReleaseTimerRef.current);
      }
    };
  }, []);

  useEffect(() => {
    const container = scrollRef.current;
    if (loading || gallery.length === 0 || !container || !scrollRequest) return;

    snapTo(scrollRequest.target === "performance" ? "performance" : "viewer");
  }, [gallery.length, loading, scrollRequest, snapTo]);

  const handleWheel = useCallback((event: WheelEvent<HTMLDivElement>) => {
    const container = scrollRef.current;
    if (!container || Math.abs(event.deltaY) < 4) return;

    const target = event.target;
    if (target instanceof HTMLElement && target.closest("a, button, input, select, textarea, [role='button']")) {
      return;
    }

    if (snapTargetRef.current) {
      event.preventDefault();
      return;
    }

    const scrollTop = container.scrollTop;
    const points = getSnapPoints();
    const tablesPoint = points.find((point) => point.target === "tables");
    if (tablesPoint && scrollTop >= tablesPoint.top - 1) {
      container.style.scrollSnapType = "none";
      return;
    }
    if (container.style.scrollSnapType === "none") {
      container.style.scrollSnapType = DEFAULT_SCROLL_SNAP_TYPE;
    }

    const nextPoint = event.deltaY > 0
      ? points.find((point) => point.top > scrollTop + SNAP_TARGET_TOLERANCE)
      : [...points].reverse().find((point) => point.top < scrollTop - SNAP_TARGET_TOLERANCE);

    if (!nextPoint) return;

    event.preventDefault();
    snapTo(nextPoint.target);
  }, [getSnapPoints, snapTo]);

  const scrollToPerformance = useCallback(() => {
    snapTo("performance");
  }, [snapTo]);

  useEffect(() => {
    const handleTableScrollRequest = () => snapTo("tables");
    window.addEventListener("enso:scroll-to-tables", handleTableScrollRequest);
    return () => window.removeEventListener("enso:scroll-to-tables", handleTableScrollRequest);
  }, [snapTo]);

  const section2Opacity = activeSection === 1 ? 1 : 0.65;

  if (loading) {
    return (
      <div className="flex items-center justify-center py-24 text-[var(--muted)]">
        Loading gallery…
      </div>
    );
  }

  if (gallery.length === 0) {
    return (
      <div className="py-12 text-center text-[var(--muted)]">
        Gallery not built yet. Run <code className="bg-[var(--surface)] px-1 rounded">build_demo_gallery.py</code> and copy output to <code className="bg-[var(--surface)] px-1 rounded">public/gallery/</code>.
      </div>
    );
  }

  const ensoCloser = row ? row.err_mil < row.err_ptn : false;

  return (
    <div className="h-full flex flex-col min-h-0 flex-1">
      <div
        ref={scrollRef}
        data-case-scroll
        onScroll={handleScroll}
        onWheel={handleWheel}
        className="flex-1 min-h-0 overflow-y-auto overflow-x-hidden"
        style={{
          scrollSnapType: DEFAULT_SCROLL_SNAP_TYPE,
          scrollBehavior: "smooth",
          WebkitOverflowScrolling: "touch",
        }}
      >
        {/* Section 1: EnsoPurity + subtitle + hero + impact — height so a portion of section 2 peeks */}
        <section
          className="min-h-[72svh] flex flex-col items-center justify-center py-12 px-4 box-border"
          style={{ scrollSnapAlign: "center" }}
        >
          <div className="w-full max-w-3xl mx-auto space-y-5 text-center">
            <h1
              className="text-4xl md:text-6xl font-bold tracking-tight pb-2"
              style={{
                background: "linear-gradient(to right, #2563eb 0%, #60a5fa 45%, #93c5fd 75%, rgba(255,255,255,0.92) 100%)",
                WebkitBackgroundClip: "text",
                backgroundClip: "text",
                color: "transparent",
              }}
            >
              EnsoPurity
            </h1>
            <p className="text-lg md:text-xl font-normal tracking-tight text-[var(--text)]">
              Predicting <strong>tumor %</strong> (<strong>purity</strong>) from digital pathology.
            </p>
            <HeroTypewriter />
            {/* Subtitle: Investor & Clinical Impact */}
            <div className="text-center max-w-3xl mx-auto space-y-3 mt-6">
              <p className="text-sm md:text-base text-[var(--muted)] font-medium leading-relaxed">
                Inaccurate tumor purity leads to failed NGS runs, missed actionable mutations, and wasted resources.
              </p>
              <p className="text-sm md:text-base text-[var(--text)] font-semibold leading-relaxed">
                Enso delivers precise, H&E-based purity mapping to optimize clinical workflows and reduce avoidable sequencing costs.
              </p>
            </div>
            {/* Scroll Arrow (Pushed further down) */}
            <div className="mt-24 animate-bounce text-[var(--muted)] opacity-50" aria-hidden>
              <svg className="w-8 h-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </div>
          </div>
        </section>

        {/* Small gap so section 2 peeks */}
        <div className="min-h-[6svh] shrink-0" aria-hidden />

        {/* Section 2: Command center + viewer */}
        <section
          ref={section2Ref}
          className="relative h-full w-full flex flex-col py-6 px-6 md:px-12 box-border overflow-hidden"
          style={{
            opacity: section2Opacity,
            transition: "opacity 500ms ease",
            scrollSnapAlign: "start",
          }}
        >
          {/* TOP COMMAND ROW */}
          <div className="flex flex-col xl:flex-row justify-between items-start xl:items-center w-full mb-5 shrink-0 gap-6">
            {/* Left: Viewer Title, Tumor Name, View in GDC */}
            <div className="flex-1 flex flex-col gap-1">
              <p className="text-xs font-bold uppercase tracking-widest text-[var(--muted)]">
                Interactive Purity Viewer
              </p>
              <h2 className="text-3xl md:text-4xl font-bold tracking-tight text-[var(--text)]">
                {row && getCancerDisplayName(row.project_id)}
              </h2>
              {row && (
                <a
                  href={`https://portal.gdc.cancer.gov/files/${row.file_uuid_original}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-medium text-blue-500 hover:text-blue-600 hover:underline mt-0.5 w-fit transition-colors"
                >
                  View in GDC Portal →
                </a>
              )}
            </div>

            {/* Center Group: Scorecard + Navigation Arrows */}
            <div className="flex flex-col md:flex-row items-stretch gap-3 shrink-0">
              {/* The Balanced Scorecard */}
              <div className="flex items-stretch bg-[var(--surface)]/80 backdrop-blur border border-[var(--border)] rounded-2xl shadow-sm shrink-0 overflow-hidden min-w-[380px]">
                {/* Actual Tumor % */}
                <div className="flex flex-col items-center justify-center bg-[var(--border)]/30 px-6 py-4 border-r border-[var(--border)]/50">
                  <span className="text-[11px] uppercase tracking-wider text-[var(--muted)] font-bold mb-1.5 whitespace-nowrap">
                    Actual Tumor %
                  </span>
                  <span className="font-mono text-3xl font-black text-[var(--text)]">
                    {row?.expected.toFixed(2) ?? "—"}
                  </span>
                </div>

                {/* Predictions - Right Side */}
                <div className="flex flex-col justify-center px-6 py-4 flex-1 gap-3">
                  <div className="flex items-center justify-between gap-6">
                    <span className="text-[11px] uppercase tracking-wider text-[var(--text)] font-bold flex items-center gap-1.5 whitespace-nowrap">
                      <span className={`w-2 h-2 rounded-full ${ensoCloser ? "bg-green-500 animate-pulse" : "bg-[var(--muted)]"}`} />
                      Enso AI Prediction
                    </span>
                    <span className="font-mono text-xl text-green-600 dark:text-green-400 font-bold leading-none">
                      {row?.predicted.toFixed(2) ?? "—"}
                    </span>
                  </div>
                  <div className="w-full h-px bg-[var(--border)]/60" />
                  <div className="flex items-center justify-between gap-6">
                    <span className="text-[11px] uppercase tracking-wider text-[var(--muted)] font-bold whitespace-nowrap">
                      Pathologist Estimate
                    </span>
                    <span className="font-mono text-lg text-red-500 dark:text-red-400 font-medium leading-none opacity-80">
                      {row?.ptn.toFixed(2) ?? "—"}
                    </span>
                  </div>
                </div>
              </div>

              {/* Case Navigation Arrows - anchored next to the scorecard */}
              <div className="flex items-center justify-center gap-4 px-2 py-3 shrink-0">
                <button
                  type="button"
                  onClick={() => setIndex((i) => Math.max(0, i - 1))}
                  disabled={index === 0}
                  className="p-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[var(--border)] transition-colors"
                  aria-label="Previous slide"
                >
                  <span className="text-lg">←</span>
                </button>
                <span className="text-sm font-medium text-[var(--muted)] min-w-[4rem] text-center">
                  {index + 1} / {displayGallery.length}
                </span>
                <button
                  type="button"
                  onClick={() => setIndex((i) => Math.min(displayGallery.length - 1, i + 1))}
                  disabled={index === displayGallery.length - 1}
                  className="p-2 rounded-lg border border-[var(--border)] bg-[var(--surface)] disabled:opacity-40 disabled:cursor-not-allowed hover:bg-[var(--border)] transition-colors"
                  aria-label="Next slide"
                >
                  <span className="text-lg">→</span>
                </button>
              </div>
            </div>

            {/* Right Spacer: keeps the center group centered on xl */}
            <div className="hidden xl:block flex-1" aria-hidden />
          </div>

          {/* NATIVE CONTROLS: Slider + Legend grouped on the right (slider left of legend) */}
          <div className="flex items-center justify-end gap-10 w-full mb-4 px-3 shrink-0">
            <div className="max-w-[200px] flex flex-col gap-2 shrink-0">
              <div className="flex justify-between text-[10px] font-bold text-[var(--muted)] uppercase tracking-wider">
                <span>Overlay Opacity</span>
                <span className="text-[var(--text)]">{Math.round(overlayOpacity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.01"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
                className="w-full h-1.5 bg-[var(--border)] rounded-full appearance-none cursor-pointer accent-blue-500 hover:accent-blue-400 transition-all"
              />
            </div>
            <div className="flex flex-col items-center gap-2 shrink-0">
              <span className="text-[10px] font-bold uppercase tracking-widest text-[var(--muted)]">
                Tumor purity
              </span>
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-mono font-medium text-[var(--muted)]">Low</span>
                <div
                  className="w-28 h-3.5 rounded-full border border-black/10 dark:border-white/10 shadow-inner"
                  style={{
                    background: "linear-gradient(to right, #4575b4 0%, #ffffbf 50%, #d73027 100%)",
                  }}
                />
                <span className="text-[10px] font-mono font-medium text-[var(--muted)]">High</span>
              </div>
            </div>
          </div>

          {/* NATIVE VIEWER: Base image + heatmap overlay (static, Cloudflare-ready) */}
          <div className="flex-1 w-full min-h-0 rounded-2xl border border-[var(--border)] bg-[#050505] shadow-inner relative overflow-hidden flex items-center justify-center">
            {staticAssetsError ? (
              <div className="w-full h-full flex flex-col items-center justify-center text-[var(--muted)] text-center px-4">
                <p className="text-sm font-medium">Static assets not found.</p>
                <p className="text-xs mt-1">Add case_{row?.originalFileIndex ?? "?"}_base.jpg and case_{row?.originalFileIndex ?? "?"}_mask.png to public/cases/</p>
              </div>
            ) : (
              <>
                <img
                  src={`/cases/case_${row?.originalFileIndex ?? 1}_base.jpg`}
                  alt="Base H&E"
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none select-none"
                  onError={() => setStaticAssetsError(true)}
                />
                <img
                  src={`/cases/case_${row?.originalFileIndex ?? 1}_mask.png`}
                  alt="Purity Heatmap"
                  style={{ opacity: overlayOpacity }}
                  className="absolute inset-0 w-full h-full object-contain pointer-events-none select-none transition-opacity duration-75"
                  onError={() => setStaticAssetsError(true)}
                />
              </>
            )}
          </div>

          {/* BOTTOM FOOTER: Concise Savings Copy */}
          <div className="shrink-0 mt-4 text-center">
            <p className="text-sm font-medium text-[var(--muted)]">
              <span className="text-green-600 dark:text-green-400 font-semibold">Prevents costly NGS reruns.</span> Saves up to $2,500 per case through reliable tumor triage.
            </p>
            <div className="relative h-0">
              <button
                type="button"
                onClick={scrollToPerformance}
                className="absolute left-1/2 top-2 -translate-x-1/2 animate-bounce text-[var(--muted)] opacity-50 hover:opacity-80 transition-opacity"
                aria-label="Scroll to clinical validation"
              >
                <svg className="w-8 h-8 mx-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>
            </div>
          </div>
        </section>
        {performanceSlot && (
          <section
            ref={performanceRef}
            className="min-h-full w-full flex flex-col bg-[var(--bg)]"
            style={{
              scrollSnapAlign: "start",
            }}
          >
            <div className="max-w-6xl mx-auto px-4 pt-[8svh] pb-16 md:pt-[9svh] md:pb-20 w-full flex-1">
              {performanceSlot}
            </div>
            <div className="w-full text-right pb-4 pr-6 text-[10px] md:text-xs font-medium text-[var(--muted)] opacity-50 pointer-events-none shrink-0">
              &copy; 2026 Enso Biosciences
            </div>
          </section>
        )}
      </div>
    </div>
  );
}
