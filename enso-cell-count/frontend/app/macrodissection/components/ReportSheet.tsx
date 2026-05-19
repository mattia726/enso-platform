"use client";

import { useEffect, useMemo, useRef } from "react";

import type {
  AdequacyVerdict,
  CaseMeta,
  Point2D,
  ROIMetrics,
  ThresholdProfile,
} from "@/lib/macrodissection/types";

export interface ReportSheetProps {
  open: boolean;
  onClose: () => void;
  caseMeta: CaseMeta | null;
  roi: {
    id: string;
    label: string;
    points: readonly Point2D[];
    locked: boolean;
    userId: string;
    revision: number;
    createdAt: string;
    updatedAt: string;
  } | null;
  metrics: ROIMetrics | null;
  verdict: AdequacyVerdict | null;
  threshold: ThresholdProfile | null;
}

function fmtPct(v: number): string {
  return Number.isFinite(v) ? `${Math.round(v * 100)}%` : "—";
}
function fmtNum(v: number): string {
  return Number.isFinite(v)
    ? v >= 1000
      ? v.toLocaleString("en-US", { maximumFractionDigits: 0 })
      : v.toFixed(0)
    : "—";
}

export default function ReportSheet({
  open,
  onClose,
  caseMeta,
  roi,
  metrics,
  verdict,
  threshold,
}: ReportSheetProps) {
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const cropRef = useRef<HTMLCanvasElement | null>(null);

  // Draw two reference views: the whole slide with the ROI outlined, and
  // a 1.2x bbox crop that zooms into the ROI for a closer look. Both
  // images live in <canvas> elements so the printed sheet can include
  // them as raster artwork without external network dependencies.
  useEffect(() => {
    if (!open || !overlayRef.current || !caseMeta || !roi) return;
    const overlay = overlayRef.current;
    const crop = cropRef.current;
    const ctxOverlay = overlay.getContext("2d");
    if (!ctxOverlay) return;
    const img = new window.Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      // --- Whole-slide thumbnail with ROI outline --------------------
      const aspect = caseMeta.base_height / Math.max(caseMeta.base_width, 1);
      overlay.width = 720;
      overlay.height = Math.max(80, Math.round(720 * aspect));
      ctxOverlay.fillStyle = "#0b0b0b";
      ctxOverlay.fillRect(0, 0, overlay.width, overlay.height);
      ctxOverlay.drawImage(img, 0, 0, overlay.width, overlay.height);
      ctxOverlay.strokeStyle = "#ef4444";
      ctxOverlay.lineWidth = 3;
      ctxOverlay.beginPath();
      roi.points.forEach((p, i) => {
        const x = (p[0] / caseMeta.base_width) * overlay.width;
        const y = (p[1] / caseMeta.base_height) * overlay.height;
        if (i === 0) ctxOverlay.moveTo(x, y);
        else ctxOverlay.lineTo(x, y);
      });
      ctxOverlay.closePath();
      ctxOverlay.fillStyle = "rgba(239, 68, 68, 0.15)";
      ctxOverlay.fill();
      ctxOverlay.stroke();

      // --- 1.2x bbox crop --------------------------------------------
      if (!crop) return;
      const ctxCrop = crop.getContext("2d");
      if (!ctxCrop) return;
      let xmin = Infinity, ymin = Infinity, xmax = -Infinity, ymax = -Infinity;
      for (const [x, y] of roi.points) {
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
      }
      const bw = Math.max(xmax - xmin, 1);
      const bh = Math.max(ymax - ymin, 1);
      const cx = (xmin + xmax) / 2;
      const cy = (ymin + ymax) / 2;
      const halfW = (bw / 2) * 1.2;
      const halfH = (bh / 2) * 1.2;
      const sx = Math.max(0, cx - halfW);
      const sy = Math.max(0, cy - halfH);
      const sw = Math.min(caseMeta.base_width - sx, 2 * halfW);
      const sh = Math.min(caseMeta.base_height - sy, 2 * halfH);
      const cropAspect = sh / Math.max(sw, 1);
      crop.width = 480;
      crop.height = Math.max(80, Math.round(480 * cropAspect));
      ctxCrop.fillStyle = "#0b0b0b";
      ctxCrop.fillRect(0, 0, crop.width, crop.height);
      ctxCrop.drawImage(img, sx, sy, sw, sh, 0, 0, crop.width, crop.height);
      ctxCrop.strokeStyle = "#ef4444";
      ctxCrop.lineWidth = 2;
      ctxCrop.beginPath();
      roi.points.forEach((p, i) => {
        const xN = ((p[0] - sx) / sw) * crop.width;
        const yN = ((p[1] - sy) / sh) * crop.height;
        if (i === 0) ctxCrop.moveTo(xN, yN);
        else ctxCrop.lineTo(xN, yN);
      });
      ctxCrop.closePath();
      ctxCrop.fillStyle = "rgba(239, 68, 68, 0.15)";
      ctxCrop.fill();
      ctxCrop.stroke();
    };
    img.src = caseMeta.base_image;
  }, [open, caseMeta, roi]);

  const verdictColor = useMemo(() => {
    if (!verdict) return "#94a3b8";
    return verdict.label === "pass"
      ? "#22c55e"
      : verdict.label === "borderline"
        ? "#f59e0b"
        : verdict.label === "fail"
          ? "#ef4444"
          : "#94a3b8";
  }, [verdict]);

  if (!open) return null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 print:bg-white print:p-0"
      data-report-sheet
    >
      <div className="bg-white text-black w-full max-w-4xl rounded-lg shadow-2xl overflow-hidden flex flex-col print:rounded-none print:max-w-none print:shadow-none">
        <div className="flex items-center justify-between bg-[#0b1d33] text-white px-6 py-4 print:bg-white print:text-black print:border-b print:border-black/30">
          <div>
            <p className="text-[11px] uppercase tracking-widest opacity-70">
              Enso Macrodissection sheet
            </p>
            <h2 className="text-xl font-bold">
              {caseMeta ? `${caseMeta.project_id || "Case"} · ${caseMeta.barcode || "—"}` : "Case"}
            </h2>
          </div>
          <div className="flex gap-2 print:hidden">
            <button
              type="button"
              onClick={() => window.print()}
              className="rounded bg-white text-[#0b1d33] px-3 py-1.5 text-sm font-semibold"
            >
              Print / Save PDF
            </button>
            <button
              type="button"
              onClick={onClose}
              className="rounded border border-white/40 px-3 py-1.5 text-sm"
              data-report-close
            >
              Close
            </button>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 p-6">
          <div className="md:col-span-2 flex flex-col gap-3">
            <div className="rounded border border-black/10 overflow-hidden">
              <canvas ref={overlayRef} className="w-full block" data-report-canvas />
            </div>
            <p className="text-[12px] text-black/60 leading-snug">
              ROI outline overlaid on the H&E thumbnail (low-resolution
              rendering for review; the live workbench shows the same
              region at full slide resolution).
            </p>
            <div className="grid grid-cols-2 gap-3 mt-1">
              <div className="rounded border border-black/10 overflow-hidden">
                <canvas ref={cropRef} className="w-full block" data-report-crop />
              </div>
              <div className="text-[12px] text-black/60 leading-snug">
                <p className="font-semibold text-black/80">ROI close-up</p>
                <p>
                  The crop on the left zooms to 1.2× of the ROI bounding box
                  so the lab technician can identify the region on glass.
                </p>
              </div>
            </div>
          </div>
          <div className="flex flex-col gap-4 text-sm">
            <section>
              <h3 className="text-xs uppercase tracking-widest font-bold text-black/60">
                Adequacy verdict
              </h3>
              <p
                className="text-2xl font-extrabold mt-1"
                style={{ color: verdictColor }}
              >
                {verdict?.label?.toUpperCase() ?? "—"}
              </p>
              <p className="text-[12px] text-black/60">
                Confidence{" "}
                {verdict ? `${Math.round(verdict.confidence * 100)}%` : "—"}
              </p>
            </section>
            <section>
              <h3 className="text-xs uppercase tracking-widest font-bold text-black/60">
                ROI metrics
              </h3>
              <table className="mt-2 w-full text-[13px] border border-black/10">
                <tbody>
                  <tr className="border-b border-black/10">
                    <td className="py-1.5 pr-2 text-black/60">Purity</td>
                    <td className="py-1.5 font-mono">
                      {metrics ? `${fmtPct(metrics.purity.median)} [${fmtPct(metrics.purity.low)} – ${fmtPct(metrics.purity.high)}]` : "—"}
                    </td>
                  </tr>
                  <tr className="border-b border-black/10">
                    <td className="py-1.5 pr-2 text-black/60">Total nuclei</td>
                    <td className="py-1.5 font-mono">
                      {metrics ? `${fmtNum(metrics.total_nuclei.median)} [${fmtNum(metrics.total_nuclei.low)} – ${fmtNum(metrics.total_nuclei.high)}]` : "—"}
                    </td>
                  </tr>
                  <tr className="border-b border-black/10">
                    <td className="py-1.5 pr-2 text-black/60">Tumor nuclei</td>
                    <td className="py-1.5 font-mono">
                      {metrics ? `${fmtNum(metrics.tumor_nuclei.median)} [${fmtNum(metrics.tumor_nuclei.low)} – ${fmtNum(metrics.tumor_nuclei.high)}]` : "—"}
                    </td>
                  </tr>
                  <tr>
                    <td className="py-1.5 pr-2 text-black/60">Area</td>
                    <td className="py-1.5 font-mono">
                      {metrics ? `${metrics.area_mm2.toFixed(2)} mm² · ${metrics.n_tiles} tiles` : "—"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </section>
            <section>
              <h3 className="text-xs uppercase tracking-widest font-bold text-black/60">
                Thresholds applied
              </h3>
              <p className="text-[12px] mt-1">
                {threshold?.display_name ?? "—"} · purity ≥{" "}
                {threshold ? fmtPct(threshold.purity_min) : "—"} · tumor
                nuclei ≥ {threshold ? fmtNum(threshold.tumor_cells_min) : "—"}
              </p>
              <p className="text-[11px] text-black/50 mt-1 leading-snug">
                {threshold?.notes ?? ""}
              </p>
            </section>
            <section>
              <h3 className="text-xs uppercase tracking-widest font-bold text-black/60">
                Audit
              </h3>
              <ul className="text-[12px] mt-1 leading-snug">
                <li>Pathologist: {roi?.userId ?? "—"}</li>
                <li>Created: {roi?.createdAt ?? "—"}</li>
                <li>Updated: {roi?.updatedAt ?? "—"}</li>
                <li>Revision: {roi?.revision ?? "—"}</li>
                <li>Locked: {roi?.locked ? "yes" : "no"}</li>
                <li>
                  Purity model: {caseMeta?.purity_model_version ?? "—"}
                </li>
                <li>
                  Cellularity model: {caseMeta?.cellularity_model_version ?? "—"}
                </li>
                <li>
                  Tile encoder: {caseMeta?.tile_encoder_version ?? "—"}
                </li>
              </ul>
            </section>
          </div>
        </div>
        <div className="bg-black/[0.04] border-t border-black/10 px-6 py-3 text-[11px] text-black/60 print:bg-white">
          AI-assisted estimate. The final macrodissection ROI must be selected
          and signed off by the reviewing pathologist; the EnsoPurity and
          EnsoCellularity outputs are decision support, not autonomous
          decision making.
        </div>
      </div>
    </div>
  );
}
