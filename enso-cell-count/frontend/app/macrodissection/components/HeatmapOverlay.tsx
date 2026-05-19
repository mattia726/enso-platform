"use client";

import { useEffect, useMemo, useRef } from "react";
import type OpenSeadragon from "openseadragon";

import type {
  LayerName,
  SmoothingMode,
  TileArrays,
} from "@/lib/macrodissection/types";
import {
  ADEQUACY_SPEC,
  CELLULARITY_SPEC,
  PURITY_SPEC,
  UNCERTAINTY_SPEC,
} from "@/lib/macrodissection/colormaps";
import { renderHeatmap } from "@/lib/macrodissection/overlay";
import { sigmaForZoom } from "@/lib/macrodissection/smoothing";

export interface HeatmapOverlayProps {
  viewer: OpenSeadragon.Viewer | null;
  tiles: TileArrays;
  layer: LayerName;
  opacity: number;
  smoothing: SmoothingMode;
  baseWidth: number;
  baseHeight: number;
}

const PIXI_CONTAINER_ID = "enso-heatmap-overlay";

function valuesFor(layer: LayerName, tiles: TileArrays): Float32Array {
  switch (layer) {
    case "purity":
      return tiles.purity;
    case "cellularity":
      return tiles.nuclei;
    case "adequacy":
      return tiles.tumor_nuclei;
    case "uncertainty":
      return tiles.purity_sd;
  }
}

function specFor(layer: LayerName) {
  switch (layer) {
    case "purity":
      return PURITY_SPEC;
    case "cellularity":
      return CELLULARITY_SPEC;
    case "adequacy":
      return ADEQUACY_SPEC;
    case "uncertainty":
      return UNCERTAINTY_SPEC;
  }
}

/**
 * Renders one heatmap layer as a canvas overlay positioned in viewport
 * coordinates by OpenSeadragon. The canvas always lives at tile-grid
 * resolution; CSS / browser scaling stretches it to fit the H&E image.
 */
export default function HeatmapOverlay({
  viewer,
  tiles,
  layer,
  opacity,
  smoothing,
  baseWidth,
  baseHeight,
}: HeatmapOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // Build / refresh the canvas pixel data when the layer settings change.
  useEffect(() => {
    if (!canvasRef.current || !viewer) return;
    const c = canvasRef.current;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    const { grid_nx: nx, grid_ny: ny } = tiles.grid;
    c.width = nx;
    c.height = ny;
    const zoom = viewer.viewport.getZoom(true);
    const maxZoom = viewer.viewport.getMaxZoom();
    const sigma = sigmaForZoom(zoom, maxZoom, smoothing);
    const img = renderHeatmap({
      width: nx,
      height: ny,
      values: valuesFor(layer, tiles),
      tissueFraction: tiles.tissue_fraction,
      spec: specFor(layer),
      opacity,
      sigma,
    });
    ctx.putImageData(img, 0, 0);
  }, [layer, opacity, smoothing, tiles, viewer]);

  // Re-blur on zoom (purely visual; metrics stay computed against raw tiles).
  useEffect(() => {
    if (!viewer) return;
    const refresh = () => {
      if (!canvasRef.current) return;
      const ctx = canvasRef.current.getContext("2d");
      if (!ctx) return;
      const { grid_nx: nx, grid_ny: ny } = tiles.grid;
      canvasRef.current.width = nx;
      canvasRef.current.height = ny;
      const sigma = sigmaForZoom(
        viewer.viewport.getZoom(true),
        viewer.viewport.getMaxZoom(),
        smoothing,
      );
      const img = renderHeatmap({
        width: nx,
        height: ny,
        values: valuesFor(layer, tiles),
        tissueFraction: tiles.tissue_fraction,
        spec: specFor(layer),
        opacity,
        sigma,
      });
      ctx.putImageData(img, 0, 0);
    };
    viewer.addHandler("zoom", refresh);
    return () => {
      try {
        viewer.removeHandler("zoom", refresh);
      } catch {
        /* ignore */
      }
    };
  }, [viewer, layer, opacity, smoothing, tiles]);

  // Mount as an OSD overlay aligned to image bounds.
  useEffect(() => {
    if (!viewer || !canvasRef.current) return;
    const c = canvasRef.current;
    c.style.width = "100%";
    c.style.height = "100%";
    c.style.position = "absolute";
    c.style.left = "0";
    c.style.top = "0";
    c.style.imageRendering = "pixelated";
    c.style.pointerEvents = "none";
    const placement = new (window as any).OpenSeadragon.Rect(
      0,
      0,
      1,
      baseHeight / baseWidth,
    );
    // OpenSeadragon image-coordinate placement is (x, y, w, h) in normalized
    // viewport units where width=1 == image width. The placement above
    // anchors the overlay to fit the entire H&E.
    try {
      viewer.removeOverlay(c);
    } catch {
      /* not yet attached */
    }
    viewer.addOverlay({ element: c, location: placement });
    return () => {
      try {
        viewer.removeOverlay(c);
      } catch {
        /* ignore */
      }
    };
  }, [viewer, baseWidth, baseHeight]);

  return (
    <canvas
      ref={canvasRef}
      id={PIXI_CONTAINER_ID}
      data-layer={layer}
      aria-label={`AI heatmap overlay: ${layer}`}
    />
  );
}
