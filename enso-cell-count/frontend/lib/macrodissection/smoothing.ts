// Visual smoothing for the heatmap overlay canvas.
//
// The smoothing is *purely* visual — it never feeds back into the metrics
// computation. Numbers in the ROI adequacy card are always derived from the
// raw tile arrays via ``metrics.ts``.
//
// Design (matches §5 of the user spec):
//
//   σ_tiles(zoom) = max(0, 1.5 · (1 - t)^1.7)     where t = currentZoom / maxZoom
//
// At maximum zoom the overlay is identical to the raw tile grid (σ ≈ 0).
// At minimum zoom σ ≈ 1.5 tiles, which gives the soft anatomical look
// suitable for getting an overview.
//
// We render the alpha channel and the RGB channels separately and only
// apply the blur to the colour, leaving the alpha mask intact. That way
// tissue holes are not bled into and outside-of-tissue regions stay
// transparent.

export type SmoothingMode = "overview" | "balanced" | "detail";

export function sigmaForZoom(
  zoom: number,
  maxZoom: number,
  mode: SmoothingMode,
): number {
  const t = Math.max(0, Math.min(1, maxZoom > 0 ? zoom / maxZoom : 0));
  const base = 1.5 * Math.pow(1 - t, 1.7);
  const factor = mode === "overview" ? 1.6 : mode === "balanced" ? 1.0 : 0.4;
  return Math.max(0, base * factor);
}

// Replace NaN cells with the nearest finite neighbour (small inpaint).
export function inpaintNaN(
  values: Float32Array,
  width: number,
  height: number,
): Float32Array {
  const out = values.slice();
  const todo: number[] = [];
  for (let i = 0; i < out.length; i++) {
    if (!Number.isFinite(out[i])) todo.push(i);
  }
  // Iterate a few passes; each pass replaces every NaN cell with the mean
  // of its finite 4-neighbours.
  for (let pass = 0; pass < 4; pass++) {
    let changed = false;
    for (const idx of todo) {
      if (Number.isFinite(out[idx])) continue;
      const y = Math.floor(idx / width);
      const x = idx - y * width;
      let sum = 0;
      let count = 0;
      const neigh = [
        x > 0 ? idx - 1 : -1,
        x + 1 < width ? idx + 1 : -1,
        y > 0 ? idx - width : -1,
        y + 1 < height ? idx + width : -1,
      ];
      for (const n of neigh) {
        if (n >= 0 && Number.isFinite(out[n])) {
          sum += out[n];
          count += 1;
        }
      }
      if (count > 0) {
        out[idx] = sum / count;
        changed = true;
      }
    }
    if (!changed) break;
  }
  return out;
}

/**
 * 2D separable Gaussian blur on a Float32 buffer. Pads with replicate so
 * the borders don't darken.
 */
export function gaussianBlur2D(
  data: Float32Array,
  width: number,
  height: number,
  sigma: number,
): Float32Array {
  if (sigma <= 0) return data;
  const radius = Math.max(1, Math.ceil(sigma * 3));
  const kernel = new Float32Array(radius * 2 + 1);
  const denom = 2 * sigma * sigma;
  let sum = 0;
  for (let i = -radius; i <= radius; i++) {
    const v = Math.exp(-(i * i) / denom);
    kernel[i + radius] = v;
    sum += v;
  }
  for (let i = 0; i < kernel.length; i++) kernel[i] /= sum;

  const tmp = new Float32Array(data.length);
  // Horizontal pass.
  for (let y = 0; y < height; y++) {
    const row = y * width;
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let xx = x + k;
        if (xx < 0) xx = 0;
        else if (xx >= width) xx = width - 1;
        acc += data[row + xx] * kernel[k + radius];
      }
      tmp[row + x] = acc;
    }
  }
  // Vertical pass.
  const out = new Float32Array(data.length);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let acc = 0;
      for (let k = -radius; k <= radius; k++) {
        let yy = y + k;
        if (yy < 0) yy = 0;
        else if (yy >= height) yy = height - 1;
        acc += tmp[yy * width + x] * kernel[k + radius];
      }
      out[y * width + x] = acc;
    }
  }
  return out;
}
