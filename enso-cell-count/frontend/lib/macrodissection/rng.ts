// Deterministic random-number generators.
//
// We use a 32-bit hash of the polygon vertices as the Monte-Carlo seed so a
// given polygon always reports the same metrics. The hash is computed via
// FNV-1a over the little-endian Float64 representation of every vertex.

export function fnv1a(bytes: Uint8Array): number {
  let h = 0x811c9dc5 >>> 0;
  for (let i = 0; i < bytes.length; i++) {
    h ^= bytes[i];
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function polygonHashSeed(
  points: readonly (readonly [number, number])[],
): number {
  // Pack vertices as little-endian Float64 — same layout as struct.pack("<dd")
  // in the Python implementation so the seed agrees byte-for-byte.
  const buf = new ArrayBuffer(points.length * 16);
  const view = new DataView(buf);
  for (let i = 0; i < points.length; i++) {
    view.setFloat64(i * 16 + 0, points[i][0], true);
    view.setFloat64(i * 16 + 8, points[i][1], true);
  }
  return fnv1a(new Uint8Array(buf)) & 0x7fffffff;
}

/**
 * A small deterministic PRNG (Mulberry32). Identical sequence across
 * platforms for the same seed; good enough quality for ROI MC sampling.
 */
export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function rand(): number {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1) >>> 0;
    t ^= (t + Math.imul(t ^ (t >>> 7), t | 61)) >>> 0;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller transform: returns one N(0, 1) sample per call. */
export function randNormalFromUniform(
  next: () => number,
): number {
  let u1 = next();
  if (u1 === 0) u1 = 1e-12;
  const u2 = next();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
