// Per-case tile-prediction loader.
//
// Fetches ``case_N_tiles.json`` (metadata) + ``case_N_grid.bin`` (packed
// Float32Array, channels in the order
// ``[purity, purity_sd, nuclei, nuclei_sd, tumor_nuclei, tissue_fraction]``)
// and assembles a :type:`TileArrays` object. The binary download is cached
// in the browser's HTTP cache; the parsed arrays are returned uncached so
// callers can decide their own memory lifecycle.

import type { CaseMeta, TileArrays, TileGridMeta } from "./types";

interface TilesJson {
  readonly schema_version: number;
  readonly case_id: number;
  readonly grid_nx: number;
  readonly grid_ny: number;
  readonly tile_pix_w: number;
  readonly tile_pix_h: number;
  readonly offset_x?: number;
  readonly offset_y?: number;
  readonly tile_area_mm2: number;
  readonly tile_size_um: number;
  readonly mpp_thumb_x: number;
  readonly mpp_thumb_y: number;
  readonly tiles_bin: string;
  readonly tiles_bin_layout: readonly string[];
}

const EXPECTED_LAYOUT = [
  "purity",
  "purity_sd",
  "nuclei",
  "nuclei_sd",
  "tumor_nuclei",
  "tissue_fraction",
] as const;

function gridMetaFromJson(json: TilesJson): TileGridMeta {
  return {
    grid_nx: json.grid_nx,
    grid_ny: json.grid_ny,
    stride_x: json.tile_pix_w,
    stride_y: json.tile_pix_h,
    offset_x: json.offset_x ?? 0,
    offset_y: json.offset_y ?? 0,
    tile_area_mm2: json.tile_area_mm2,
    tile_size_um: json.tile_size_um,
    mpp_thumb_x: json.mpp_thumb_x,
    mpp_thumb_y: json.mpp_thumb_y,
  };
}

export async function loadTileArrays(
  tilesMetaUrl: string,
  tilesBinUrl: string,
  fetcher: typeof fetch = fetch,
): Promise<TileArrays> {
  const [metaRes, binRes] = await Promise.all([
    fetcher(tilesMetaUrl),
    fetcher(tilesBinUrl),
  ]);
  if (!metaRes.ok) {
    throw new Error(`Failed to load ${tilesMetaUrl}: ${metaRes.status}`);
  }
  if (!binRes.ok) {
    throw new Error(`Failed to load ${tilesBinUrl}: ${binRes.status}`);
  }
  const json: TilesJson = await metaRes.json();
  const buf = await binRes.arrayBuffer();
  if (!arraysEqual(json.tiles_bin_layout, EXPECTED_LAYOUT)) {
    throw new Error(
      `Unexpected tile bin layout in ${tilesMetaUrl}: ${json.tiles_bin_layout}`,
    );
  }
  const channels = EXPECTED_LAYOUT.length;
  const tileCount = json.grid_nx * json.grid_ny;
  const expectedBytes = tileCount * channels * 4;
  if (buf.byteLength !== expectedBytes) {
    throw new Error(
      `Bin size mismatch in ${tilesBinUrl}: expected ${expectedBytes} bytes, got ${buf.byteLength}`,
    );
  }
  const all = new Float32Array(buf);
  // The packed layout is (ny, nx, C) row-major. We want per-channel
  // Float32Arrays in (ny * nx) order.
  const channelArrays: Float32Array[] = [];
  for (let c = 0; c < channels; c++) {
    const arr = new Float32Array(tileCount);
    for (let i = 0; i < tileCount; i++) {
      arr[i] = all[i * channels + c];
    }
    channelArrays.push(arr);
  }
  return {
    grid: gridMetaFromJson(json),
    purity: channelArrays[0],
    purity_sd: channelArrays[1],
    nuclei: channelArrays[2],
    nuclei_sd: channelArrays[3],
    tumor_nuclei: channelArrays[4],
    tissue_fraction: channelArrays[5],
  };
}

function arraysEqual<T>(a: readonly T[], b: readonly T[]): boolean {
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
  return true;
}

/** Convenience wrapper that loads tile arrays for one case. */
export async function loadTileArraysForCase(
  caseMeta: CaseMeta,
  fetcher: typeof fetch = fetch,
): Promise<TileArrays> {
  return loadTileArrays(caseMeta.tiles_meta, caseMeta.tiles_bin, fetcher);
}
