// Static case discovery for the Next.js build.
//
// When the backend is *not* running (Cloudflare-style static deploy) we
// still want the workbench to work — the build_artifacts CLI writes
// ``macrodissection_build_summary.json`` next to the case files and we use
// that as the source of truth. The per-case full metadata can then be
// fetched lazily from ``case_N_tiles.json``.

import type { CaseMeta } from "./types";

interface BuildSummary {
  readonly schema_version: number;
  readonly n_cases_built: number;
  readonly cases: ReadonlyArray<{
    readonly case_id: number;
    readonly grid_nx: number;
    readonly grid_ny: number;
    readonly n_tiles_tissue: number;
    readonly barcode: string;
    readonly project_id: string;
  }>;
}

interface PerCaseMeta {
  readonly schema_version: number;
  readonly case_id: number;
  readonly barcode: string;
  readonly project_id: string;
  readonly file_uuid: string;
  readonly base_width: number;
  readonly base_height: number;
  readonly grid_nx: number;
  readonly grid_ny: number;
  readonly n_tiles_tissue: number;
  readonly tile_size_um: number;
  readonly mpp_thumb_x: number;
  readonly mpp_thumb_y: number;
  readonly purity_model_version: string;
  readonly cellularity_model_version: string;
  readonly tile_encoder_version: string;
}

const CASES_PUBLIC_ROOT = "/cases";

function publicUrl(name: string): string {
  return `${CASES_PUBLIC_ROOT}/${name}`;
}

export async function loadStaticCaseList(
  fetcher: typeof fetch = fetch,
): Promise<CaseMeta[]> {
  const summaryRes = await fetcher(`${CASES_PUBLIC_ROOT}/macrodissection_build_summary.json`);
  if (!summaryRes.ok) {
    throw new Error(
      `Could not load macrodissection_build_summary.json: ${summaryRes.status}`,
    );
  }
  const summary = (await summaryRes.json()) as BuildSummary;
  // Fetch per-case JSONs in parallel; failures get filtered out.
  const metas = await Promise.all(
    summary.cases.map((c) =>
      fetcher(publicUrl(`case_${c.case_id}_tiles.json`))
        .then((r) => (r.ok ? (r.json() as Promise<PerCaseMeta>) : null))
        .catch(() => null),
    ),
  );
  const out: CaseMeta[] = [];
  for (let i = 0; i < summary.cases.length; i++) {
    const fallback = summary.cases[i];
    const meta = metas[i];
    const case_id = fallback.case_id;
    out.push({
      case_id,
      barcode: meta?.barcode ?? fallback.barcode,
      project_id: meta?.project_id ?? fallback.project_id,
      file_uuid: meta?.file_uuid ?? "",
      base_width: meta?.base_width ?? 0,
      base_height: meta?.base_height ?? 0,
      grid_nx: meta?.grid_nx ?? fallback.grid_nx,
      grid_ny: meta?.grid_ny ?? fallback.grid_ny,
      n_tiles_tissue: meta?.n_tiles_tissue ?? fallback.n_tiles_tissue,
      has_purity: true,
      has_cellularity: true,
      base_image: publicUrl(`case_${case_id}_base.jpg`),
      purity_mask: publicUrl(`case_${case_id}_mask.png`),
      cellularity_mask: publicUrl(`case_${case_id}_cell_count_mask.png`),
      tiles_meta: publicUrl(`case_${case_id}_tiles.json`),
      tiles_bin: publicUrl(`case_${case_id}_grid.bin`),
      tile_size_um: meta?.tile_size_um ?? 112.0,
      mpp_thumb_x: meta?.mpp_thumb_x ?? 0,
      mpp_thumb_y: meta?.mpp_thumb_y ?? 0,
      purity_model_version: meta?.purity_model_version ?? "unknown",
      cellularity_model_version: meta?.cellularity_model_version ?? "unknown",
      tile_encoder_version: meta?.tile_encoder_version ?? "unknown",
    });
  }
  return out;
}
