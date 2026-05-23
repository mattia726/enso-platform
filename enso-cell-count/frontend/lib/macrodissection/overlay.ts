// Render a tile-grid scalar field onto an RGBA canvas buffer.
//
// Used by the WSI viewer overlay component. We optionally:
//   * inpaint NaN cells from their finite neighbours (keeps colour bleeds
//     out of holes),
//   * apply a zoom-adaptive Gaussian blur to the *colour* channels only,
//     leaving the alpha channel intact,
//   * map values through a LutSpec colormap.
//
// The resulting ImageData has the same width/height as the tile grid; the
// HTML5 canvas itself is then scaled up to the slide base size via CSS or
// the OpenSeadragon overlay matrix.

import type { LutSpec } from "./colormaps";
import { lutLookup } from "./colormaps";
import { gaussianBlur2D, inpaintNaN, sigmaForZoom, type SmoothingMode } from "./smoothing";

export interface RenderHeatmapOptions {
  readonly width: number;
  readonly height: number;
  readonly values: Float32Array;     // length width*height; NaN = missing
  readonly tissueFraction: Float32Array;
  readonly spec: LutSpec;
  readonly opacity: number;          // 0..1
  readonly sigma: number;            // tiles; 0 disables blur
  readonly minAlpha?: number;        // floor applied to non-empty tiles
}

export function renderHeatmap(opts: RenderHeatmapOptions): ImageData {
  const { width, height, values, tissueFraction, spec, opacity, sigma } = opts;
  const minAlpha = opts.minAlpha ?? 0;
  const tileCount = width * height;
  if (values.length !== tileCount || tissueFraction.length !== tileCount) {
    throw new Error(
      `renderHeatmap: array length mismatch (${values.length}, ${tissueFraction.length}) vs ${tileCount}`,
    );
  }

  let working = inpaintNaN(values, width, height);
  if (sigma > 0) {
    working = gaussianBlur2D(working, width, height, sigma);
  }

  const img = new ImageData(width, height);
  for (let i = 0; i < tileCount; i++) {
    const tf = tissueFraction[i];
    if (!Number.isFinite(tf) || tf <= 0) {
      // Tissue absent — fully transparent.
      img.data[i * 4 + 0] = 0;
      img.data[i * 4 + 1] = 0;
      img.data[i * 4 + 2] = 0;
      img.data[i * 4 + 3] = 0;
      continue;
    }
    const v = working[i];
    const [r, g, b] = lutLookup(v, spec);
    const baseAlpha = 255 * opacity;
    const alpha = Math.round(Math.max(minAlpha, baseAlpha) * Math.min(1, tf));
    img.data[i * 4 + 0] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = alpha;
  }
  return img;
}

export function smoothingSigmaForZoom(
  zoom: number,
  maxZoom: number,
  mode: SmoothingMode,
): number {
  return sigmaForZoom(zoom, maxZoom, mode);
}
