// Thin HTTP client for the backend macrodissection API.
//
// The frontend can run entirely client-side for the demo (no backend
// required to compute preview metrics), so the helpers in this file are
// optional and only used for the *authoritative* save/lock/report flows.

import type {
  AdequacyVerdict,
  CaseMeta,
  GeoJSONPolygon,
  ROIMetrics,
  ROIRecord,
  ThresholdProfile,
} from "./types";

const DEFAULT_BASE =
  (typeof process !== "undefined" && process.env?.NEXT_PUBLIC_BACKEND_URL) ||
  "http://localhost:8000";

export interface MacroApiOptions {
  readonly baseUrl?: string;
  readonly fetcher?: typeof fetch;
}

function url(base: string, path: string): string {
  return base.replace(/\/$/, "") + path;
}

async function jsonRequest<T>(
  method: string,
  endpoint: string,
  body: unknown | undefined,
  opts: MacroApiOptions = {},
): Promise<T> {
  const base = opts.baseUrl ?? DEFAULT_BASE;
  const fetcher = opts.fetcher ?? fetch;
  const res = await fetcher(url(base, endpoint), {
    method,
    headers: body !== undefined ? { "Content-Type": "application/json" } : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    throw new Error(`${method} ${endpoint} → ${res.status}: ${await res.text()}`);
  }
  return (await res.json()) as T;
}

export async function listCases(opts?: MacroApiOptions): Promise<CaseMeta[]> {
  return jsonRequest("GET", "/api/macrodissection/cases", undefined, opts);
}

export async function getThresholdProfiles(
  opts?: MacroApiOptions,
): Promise<ThresholdProfile[]> {
  return jsonRequest(
    "GET",
    "/api/macrodissection/threshold-profiles",
    undefined,
    opts,
  );
}

export interface PreviewPayload {
  polygon: GeoJSONPolygon;
  thresholds?: { profile: string };
  n_samples?: number;
  seed?: number;
}

export async function previewROI(
  caseId: number,
  payload: PreviewPayload,
  opts?: MacroApiOptions,
): Promise<{ metrics: ROIMetrics; verdict: AdequacyVerdict }> {
  return jsonRequest(
    "POST",
    `/api/macrodissection/cases/${caseId}/rois/preview`,
    payload,
    opts,
  );
}

export interface SavePayload {
  polygon: GeoJSONPolygon;
  label?: string;
  user_id?: string;
  notes?: string;
  thresholds?: { profile: string };
}

export async function saveROI(
  caseId: number,
  payload: SavePayload,
  opts?: MacroApiOptions,
): Promise<ROIRecord> {
  return jsonRequest(
    "POST",
    `/api/macrodissection/cases/${caseId}/rois`,
    payload,
    opts,
  );
}

export async function lockROI(
  caseId: number,
  roiId: string,
  opts?: MacroApiOptions,
): Promise<ROIRecord> {
  return jsonRequest(
    "POST",
    `/api/macrodissection/cases/${caseId}/rois/${roiId}/lock`,
    {},
    opts,
  );
}

export async function listROIs(
  caseId: number,
  opts?: MacroApiOptions,
): Promise<ROIRecord[]> {
  return jsonRequest(
    "GET",
    `/api/macrodissection/cases/${caseId}/rois`,
    undefined,
    opts,
  );
}

export async function deleteROI(
  caseId: number,
  roiId: string,
  opts?: MacroApiOptions,
): Promise<{ status: string }> {
  return jsonRequest(
    "DELETE",
    `/api/macrodissection/cases/${caseId}/rois/${roiId}`,
    undefined,
    opts,
  );
}

export interface CandidatePayload {
  rank: number;
  score: number;
  bbox_thumb_px: readonly [number, number, number, number];
  polygon: GeoJSONPolygon;
  purity_point: number;
  total_nuclei_point: number;
  tumor_nuclei_point: number;
  adequacy_probability: number;
}

export async function getCandidates(
  caseId: number,
  params: { k?: number; profile?: string; windowTiles?: number; nmsIou?: number } = {},
  opts?: MacroApiOptions,
): Promise<CandidatePayload[]> {
  const search = new URLSearchParams();
  if (params.k !== undefined) search.set("k", String(params.k));
  if (params.profile !== undefined) search.set("profile", params.profile);
  if (params.windowTiles !== undefined)
    search.set("window_tiles", String(params.windowTiles));
  if (params.nmsIou !== undefined) search.set("nms_iou", String(params.nmsIou));
  const q = search.toString();
  return jsonRequest(
    "GET",
    `/api/macrodissection/cases/${caseId}/candidates${q ? `?${q}` : ""}`,
    undefined,
    opts,
  );
}

export interface ReportPayload {
  generated_at: string;
  case: CaseMeta;
  roi: {
    roi_id: string;
    label: string;
    geometry_thumb_px: GeoJSONPolygon;
    locked: boolean;
    created_at: string;
    updated_at: string;
    user_id: string;
    revision: number;
    notes: string;
  };
  verdict: AdequacyVerdict;
  threshold_profile: ThresholdProfile;
  models: {
    purity_model_version: string;
    cellularity_model_version: string;
    tile_encoder_version: string;
  };
  disclaimer: string;
}

export async function getReport(
  caseId: number,
  roiId: string,
  opts?: MacroApiOptions,
): Promise<ReportPayload> {
  return jsonRequest(
    "GET",
    `/api/macrodissection/cases/${caseId}/rois/${roiId}/report`,
    undefined,
    opts,
  );
}
