// Colormap LUTs for the workbench overlays.
//
// All four LUTs are 256-entry uint8 RGB tables. The purity LUT matches
// matplotlib's RdYlBu_r so the client-rendered overlay is identical to the
// pre-rendered server PNG. Cellularity uses the bespoke "purity_no_white"
// palette (no white midpoint). Adequacy and uncertainty use new
// purpose-built palettes designed for the workbench.

export type Lut = Readonly<Uint8Array>; // length 256*3, packed [R,G,B,R,G,B,...]

function buildLut(stops: readonly (readonly [number, number, number])[]): Lut {
  // Build a 256-entry LUT by linearly interpolating between the stop colors.
  // ``stops`` must be sorted by ascending position; positions are normalized
  // into [0, 1] by the indices of the stop list.
  const n = 256;
  const out = new Uint8Array(n * 3);
  const segs = stops.length - 1;
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const sf = t * segs;
    const seg = Math.min(Math.floor(sf), segs - 1);
    const localT = sf - seg;
    const a = stops[seg];
    const b = stops[seg + 1];
    out[i * 3 + 0] = Math.round(a[0] + (b[0] - a[0]) * localT);
    out[i * 3 + 1] = Math.round(a[1] + (b[1] - a[1]) * localT);
    out[i * 3 + 2] = Math.round(a[2] + (b[2] - a[2]) * localT);
  }
  return out;
}

// RdYlBu_r (reversed RdYlBu): blue → yellow → red. 11-point matplotlib stops.
export const PURITY_LUT: Lut = buildLut([
  [49, 54, 149],
  [69, 117, 180],
  [116, 173, 209],
  [171, 217, 233],
  [224, 243, 248],
  [255, 255, 191],
  [254, 224, 144],
  [253, 174, 97],
  [244, 109, 67],
  [215, 48, 39],
  [165, 0, 38],
]);

// purity_no_white (seven-stop linear segmented map, no white midpoint).
export const CELLULARITY_LUT: Lut = buildLut([
  [49, 54, 149],
  [69, 117, 180],
  [116, 173, 209],
  [254, 224, 139],
  [253, 174, 97],
  [244, 109, 67],
  [165, 0, 38],
]);

// Adequacy: transparent → soft orange → red. Used for the *primary* layer
// the pathologist sees by default. Reads naturally as "warmer means better
// candidate" and contrasts with H&E pink/purple.
export const ADEQUACY_LUT: Lut = buildLut([
  [255, 248, 232],
  [254, 217, 142],
  [254, 153, 41],
  [217, 95, 14],
  [153, 52, 4],
]);

// Uncertainty: cool gray → warm gray. Used as a low-saturation overlay so
// it can coexist with one of the primary layers.
export const UNCERTAINTY_LUT: Lut = buildLut([
  [240, 240, 240],
  [180, 180, 200],
  [120, 120, 140],
  [80, 80, 100],
  [60, 50, 70],
]);

export interface LutSpec {
  readonly lut: Lut;
  readonly vmin: number;
  readonly vmax: number;
  readonly gamma: number;
}

export const PURITY_SPEC: LutSpec = {
  lut: PURITY_LUT,
  vmin: 0,
  vmax: 1,
  gamma: 1,
};

export const CELLULARITY_SPEC: LutSpec = {
  lut: CELLULARITY_LUT,
  vmin: 0,
  vmax: 180,
  gamma: 0.65,
};

export const ADEQUACY_SPEC: LutSpec = {
  lut: ADEQUACY_LUT,
  vmin: 0,
  // Adequacy density unit = estimated tumor nuclei per tile. 200 is a
  // sensible upper bound that places typical adequate ROIs in the warm
  // upper third of the palette.
  vmax: 200,
  gamma: 0.65,
};

export const UNCERTAINTY_SPEC: LutSpec = {
  lut: UNCERTAINTY_LUT,
  vmin: 0,
  vmax: 1,
  gamma: 1,
};

/** Map one scalar through a LUT spec into an RGB triple. */
export function lutLookup(value: number, spec: LutSpec): [number, number, number] {
  if (!Number.isFinite(value)) return [0, 0, 0];
  const span = Math.max(spec.vmax - spec.vmin, 1e-12);
  let t = (value - spec.vmin) / span;
  if (t < 0) t = 0;
  else if (t > 1) t = 1;
  if (spec.gamma !== 1) t = Math.pow(t, spec.gamma);
  const idx = Math.round(t * 255);
  const off = idx * 3;
  return [spec.lut[off], spec.lut[off + 1], spec.lut[off + 2]];
}
