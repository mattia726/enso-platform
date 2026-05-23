// Sutherland-Hodgman polygon clipping and exact polygon × tile intersection.
//
// This TypeScript module mirrors the Python implementation in
// ``enso_purity.macrodissection.roi``. Both implementations share the same
// algorithms, the same edge handling, and the same unit-test invariants so
// the user always sees the same numbers in the client-side preview and in
// the authoritative server recompute.

import type { Point2D, TileGridMeta } from "./types";

export interface TileWeight {
  readonly ix: number;
  readonly iy: number;
  readonly weight: number;          // fraction of the tile inside the polygon
  readonly interAreaThumbPx2: number;
}

/** Shoelace signed area (positive for counter-clockwise polygons). */
export function polygonSignedArea(poly: readonly Point2D[]): number {
  if (poly.length < 3) return 0;
  let s = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x1, y1] = poly[i];
    const [x2, y2] = poly[(i + 1) % poly.length];
    s += x1 * y2 - x2 * y1;
  }
  return s * 0.5;
}

export function polygonArea(poly: readonly Point2D[]): number {
  return Math.abs(polygonSignedArea(poly));
}

function axisIntersect(
  p1: Point2D,
  p2: Point2D,
  axis: 0 | 1,
  value: number,
): Point2D {
  const [x1, y1] = p1;
  const [x2, y2] = p2;
  if (axis === 0) {
    const dx = x2 - x1;
    if (dx === 0) return [value, y1];
    const t = (value - x1) / dx;
    return [value, y1 + t * (y2 - y1)];
  }
  const dy = y2 - y1;
  if (dy === 0) return [x1, value];
  const t = (value - y1) / dy;
  return [x1 + t * (x2 - x1), value];
}

function clipAxis(
  poly: readonly Point2D[],
  axis: 0 | 1,
  value: number,
  keepGreater: boolean,
): Point2D[] {
  if (poly.length === 0) return [];
  const out: Point2D[] = [];
  const inside = (p: Point2D) => {
    const c = p[axis];
    return keepGreater ? c >= value : c <= value;
  };
  for (let i = 0; i < poly.length; i++) {
    const current = poly[i];
    const prev = poly[i === 0 ? poly.length - 1 : i - 1];
    const curIn = inside(current);
    const prevIn = inside(prev);
    if (curIn) {
      if (!prevIn) out.push(axisIntersect(prev, current, axis, value));
      out.push(current);
    } else if (prevIn) {
      out.push(axisIntersect(prev, current, axis, value));
    }
  }
  return out;
}

export function clipPolygonToRect(
  poly: readonly Point2D[],
  x0: number,
  y0: number,
  x1: number,
  y1: number,
): Point2D[] {
  if (x1 <= x0 || y1 <= y0 || poly.length < 3) return [];
  let out = clipAxis(poly, 0, x0, true);
  out = clipAxis(out, 0, x1, false);
  out = clipAxis(out, 1, y0, true);
  out = clipAxis(out, 1, y1, false);
  return out;
}

export function tileRect(
  grid: TileGridMeta,
  ix: number,
  iy: number,
): readonly [number, number, number, number] {
  const x0 = grid.offset_x + ix * grid.stride_x;
  const y0 = grid.offset_y + iy * grid.stride_y;
  return [x0, y0, x0 + grid.stride_x, y0 + grid.stride_y] as const;
}

export function tileWeights(
  polygon: readonly Point2D[],
  grid: TileGridMeta,
): TileWeight[] {
  if (polygon.length < 3) return [];
  let xmin = Infinity;
  let ymin = Infinity;
  let xmax = -Infinity;
  let ymax = -Infinity;
  for (const [x, y] of polygon) {
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  const sx = Math.max(grid.stride_x, 1e-9);
  const sy = Math.max(grid.stride_y, 1e-9);
  const ixMin = Math.max(Math.floor((xmin - grid.offset_x) / sx), 0);
  const ixMax = Math.min(
    Math.floor((xmax - grid.offset_x) / sx),
    grid.grid_nx - 1,
  );
  const iyMin = Math.max(Math.floor((ymin - grid.offset_y) / sy), 0);
  const iyMax = Math.min(
    Math.floor((ymax - grid.offset_y) / sy),
    grid.grid_ny - 1,
  );
  if (ixMax < ixMin || iyMax < iyMin) return [];
  const tileArea = grid.stride_x * grid.stride_y;
  if (tileArea <= 0) return [];

  const out: TileWeight[] = [];
  for (let iy = iyMin; iy <= iyMax; iy++) {
    for (let ix = ixMin; ix <= ixMax; ix++) {
      const [x0, y0, x1, y1] = tileRect(grid, ix, iy);
      const clipped = clipPolygonToRect(polygon, x0, y0, x1, y1);
      if (clipped.length < 3) continue;
      const inter = polygonArea(clipped);
      if (inter <= 0) continue;
      out.push({
        ix,
        iy,
        weight: Math.min(1.0, inter / tileArea),
        interAreaThumbPx2: inter,
      });
    }
  }
  return out;
}

/** Convert a GeoJSON polygon (outer ring) into a plain point list. */
export function geoJsonPolygonToPoints(
  geom: { readonly coordinates: readonly (readonly Point2D[])[] },
): Point2D[] {
  const ring = geom.coordinates[0] ?? [];
  return ring.map(([x, y]) => [x, y] as Point2D);
}
