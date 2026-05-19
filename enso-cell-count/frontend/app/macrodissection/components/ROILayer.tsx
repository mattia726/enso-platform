"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import type OpenSeadragon from "openseadragon";

import type { Point2D } from "@/lib/macrodissection/types";

export interface ROIDraft {
  points: Point2D[];
  closed: boolean;
}

export interface ROILayerProps {
  viewer: OpenSeadragon.Viewer | null;
  baseWidth: number;
  baseHeight: number;
  draft: ROIDraft | null;
  onDraftChange: (draft: ROIDraft | null) => void;
  enabled: boolean;
  lockedPolygons?: readonly { id: string; points: readonly Point2D[]; label: string }[];
}

/**
 * SVG overlay that lets the pathologist draw and edit a polygon ROI on top
 * of the OpenSeadragon viewer. Coordinates are stored in *image* pixels so
 * they survive zoom/pan and match the units used by the ROI math library.
 */
export default function ROILayer({
  viewer,
  baseWidth,
  baseHeight,
  draft,
  onDraftChange,
  enabled,
  lockedPolygons = [],
}: ROILayerProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const dragVertexRef = useRef<number | null>(null);

  // Mount the SVG as an OSD overlay so it tracks the H&E perfectly.
  useEffect(() => {
    if (!viewer || !svgRef.current) return;
    const el = svgRef.current;
    const placement = new (window as any).OpenSeadragon.Rect(
      0,
      0,
      1,
      baseHeight / baseWidth,
    );
    el.style.position = "absolute";
    el.style.left = "0";
    el.style.top = "0";
    el.style.width = "100%";
    el.style.height = "100%";
    try {
      viewer.removeOverlay(el);
    } catch {
      /* ignore */
    }
    viewer.addOverlay({ element: el as unknown as HTMLElement, location: placement });
    return () => {
      try {
        viewer.removeOverlay(el as unknown as HTMLElement);
      } catch {
        /* ignore */
      }
    };
  }, [viewer, baseWidth, baseHeight]);

  // Translate a pointer event to image-pixel coordinates.
  const eventToImagePoint = (e: { clientX: number; clientY: number }): Point2D | null => {
    if (!svgRef.current) return null;
    const rect = svgRef.current.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return null;
    const x = ((e.clientX - rect.left) / rect.width) * baseWidth;
    const y = ((e.clientY - rect.top) / rect.height) * baseHeight;
    return [x, y];
  };

  // Pointer handlers — gated by ``enabled`` so the layer can be turned off
  // for pan/zoom-only interaction.
  const onPointerDown = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!enabled) return;
    const p = eventToImagePoint(e);
    if (!p) return;
    e.preventDefault();
    e.stopPropagation();
    if (!draft) {
      onDraftChange({ points: [p], closed: false });
      return;
    }
    if (draft.closed) {
      // Click on a vertex to start dragging it; clicks elsewhere reset the
      // selection.
      const vIdx = closestVertex(p, draft.points, hitRadiusImage());
      if (vIdx !== null) {
        dragVertexRef.current = vIdx;
        return;
      }
      onDraftChange({ points: [p], closed: false });
      return;
    }
    onDraftChange({
      points: [...draft.points, p],
      closed: false,
    });
  };

  const onPointerMove = (e: React.PointerEvent<SVGSVGElement>) => {
    if (!enabled || !draft) return;
    if (draft.closed) {
      if (dragVertexRef.current === null) return;
      const p = eventToImagePoint(e);
      if (!p) return;
      const newPoints = draft.points.slice();
      newPoints[dragVertexRef.current] = p;
      onDraftChange({ points: newPoints, closed: true });
    }
  };

  const onPointerUp = () => {
    if (dragVertexRef.current !== null) {
      dragVertexRef.current = null;
    }
  };

  const onDoubleClick = (e: React.MouseEvent<SVGSVGElement>) => {
    if (!enabled || !draft || draft.closed || draft.points.length < 3) return;
    e.preventDefault();
    onDraftChange({ points: draft.points, closed: true });
  };

  function hitRadiusImage(): number {
    // Generous hit radius so vertex handles are easy to grab.
    return Math.max(baseWidth, baseHeight) * 0.012;
  }

  const draftPath = useMemo(() => {
    if (!draft || draft.points.length === 0) return "";
    const cmds = draft.points
      .map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`)
      .join(" ");
    return draft.closed ? `${cmds} Z` : cmds;
  }, [draft]);

  return (
    <svg
      ref={svgRef}
      viewBox={`0 0 ${baseWidth} ${baseHeight}`}
      preserveAspectRatio="none"
      style={{ position: "absolute", inset: 0, pointerEvents: enabled ? "auto" : "none" }}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      onDoubleClick={onDoubleClick}
      data-roi-layer
      data-roi-enabled={enabled ? "true" : "false"}
    >
      {/* Locked polygons (read-only) */}
      {lockedPolygons.map((p) => (
        <g key={p.id} data-roi-locked={p.id}>
          <path
            d={pointsToPath(p.points, true)}
            fill="rgba(34,197,94,0.10)"
            stroke="#22c55e"
            strokeWidth={Math.max(baseWidth, baseHeight) / 280}
            strokeLinejoin="round"
          />
          <text
            x={p.points[0]?.[0] ?? 0}
            y={(p.points[0]?.[1] ?? 0) - 8}
            fontFamily="ui-sans-serif, system-ui"
            fontWeight={600}
            fontSize={Math.max(baseWidth, baseHeight) / 60}
            fill="#22c55e"
            stroke="black"
            strokeWidth={Math.max(baseWidth, baseHeight) / 400}
            paintOrder="stroke"
          >
            {p.label}
          </text>
        </g>
      ))}
      {/* Current draft */}
      {draftPath && (
        <path
          d={draftPath}
          fill={draft?.closed ? "rgba(239,68,68,0.16)" : "rgba(239,68,68,0.08)"}
          stroke="#ef4444"
          strokeWidth={Math.max(baseWidth, baseHeight) / 260}
          strokeLinejoin="round"
          strokeDasharray={draft?.closed ? "" : `${Math.max(baseWidth, baseHeight) / 160} ${Math.max(baseWidth, baseHeight) / 320}`}
        />
      )}
      {draft && draft.closed &&
        draft.points.map((p, i) => (
          <circle
            key={i}
            cx={p[0]}
            cy={p[1]}
            r={Math.max(baseWidth, baseHeight) / 220}
            fill="#fff"
            stroke="#ef4444"
            strokeWidth={Math.max(baseWidth, baseHeight) / 450}
            data-roi-vertex={i}
          />
        ))}
    </svg>
  );
}

function closestVertex(
  p: Point2D,
  points: readonly Point2D[],
  maxDist: number,
): number | null {
  let bestIdx: number | null = null;
  let bestDist = Infinity;
  for (let i = 0; i < points.length; i++) {
    const dx = points[i][0] - p[0];
    const dy = points[i][1] - p[1];
    const d = Math.hypot(dx, dy);
    if (d < bestDist) {
      bestDist = d;
      bestIdx = i;
    }
  }
  if (bestIdx === null || bestDist > maxDist) return null;
  return bestIdx;
}

function pointsToPath(points: readonly Point2D[], closed: boolean): string {
  if (points.length === 0) return "";
  const cmds = points
    .map((p, i) => `${i === 0 ? "M" : "L"} ${p[0]} ${p[1]}`)
    .join(" ");
  return closed ? `${cmds} Z` : cmds;
}
