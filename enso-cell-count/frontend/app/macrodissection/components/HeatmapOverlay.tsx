"use client";

import { useEffect, useRef } from "react";
import OpenSeadragon from "openseadragon";

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

function drawHeatmap(
  canvas: HTMLCanvasElement,
  tiles: TileArrays,
  layer: LayerName,
  opacity: number,
  sigma: number,
) {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { grid_nx: nx, grid_ny: ny } = tiles.grid;
  canvas.width = nx;
  canvas.height = ny;
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
}

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

  // Mount as an OSD overlay aligned to the image bounds. We re-mount when
  // the case (and therefore baseWidth/baseHeight) changes.
  useEffect(() => {
    if (!viewer || !canvasRef.current) return;
    const c = canvasRef.current;
    c.style.width = "100%";
    c.style.height = "100%";
    c.style.imageRendering = "pixelated";
    c.style.pointerEvents = "none";
    // OSD accepts a plain {x, y, width, height} object as a location and
    // normalises it into an OpenSeadragon.Rect internally. We anchor (0,0)
    // to the image top-left and 1 to its full width; the height is in the
    // same units (so we divide by aspect ratio).
    try {
      viewer.removeOverlay(c);
    } catch {
      /* not yet attached */
    }
    viewer.addOverlay({
      element: c,
      location: new OpenSeadragon.Rect(0, 0, 1, baseHeight / baseWidth),
    });
    return () => {
      try {
        viewer.removeOverlay(c);
      } catch {
        /* ignore */
      }
    };
  }, [viewer, baseWidth, baseHeight]);

  // Redraw whenever the layer settings or tiles change.
  useEffect(() => {
    if (!canvasRef.current || !viewer) return;
    const sigma = sigmaForZoom(
      viewer.viewport.getZoom(true),
      viewer.viewport.getMaxZoom(),
      smoothing,
    );
    drawHeatmap(canvasRef.current, tiles, layer, opacity, sigma);
  }, [layer, opacity, smoothing, tiles, viewer]);

  // Recompute sigma on zoom (purely visual; metric numbers stay raw).
  useEffect(() => {
    if (!viewer) return;
    const handler = () => {
      if (!canvasRef.current) return;
      const sigma = sigmaForZoom(
        viewer.viewport.getZoom(true),
        viewer.viewport.getMaxZoom(),
        smoothing,
      );
      drawHeatmap(canvasRef.current, tiles, layer, opacity, sigma);
    };
    viewer.addHandler("zoom", handler);
    return () => {
      try {
        viewer.removeHandler("zoom", handler);
      } catch {
        /* ignore */
      }
    };
  }, [viewer, layer, opacity, smoothing, tiles]);

  return (
    <canvas
      ref={canvasRef}
      id={PIXI_CONTAINER_ID}
      data-layer={layer}
      aria-label={`AI heatmap overlay: ${layer}`}
    />
  );
}
