// Shared types for the macrodissection workbench frontend.
//
// These shapes mirror the Python module
// ``enso_purity.macrodissection`` so the client preview and the
// authoritative server recompute always agree on the contract.

export type Point2D = readonly [number, number];

export interface GeoJSONPolygon {
  readonly type: "Polygon";
  // GeoJSON-style: list of rings, first ring is the outer boundary.
  readonly coordinates: readonly (readonly Point2D[])[];
}

export interface ThresholdProfile {
  readonly name: string;
  readonly display_name: string;
  readonly purity_min: number;
  readonly tumor_cells_min: number;
  readonly borderline_purity_band: number;
  readonly borderline_tumor_cells_band: number;
  readonly pass_probability: number;
  readonly borderline_probability: number;
  readonly notes: string;
}

export interface TileGridMeta {
  readonly grid_nx: number;
  readonly grid_ny: number;
  readonly stride_x: number;  // tile width in thumbnail pixels
  readonly stride_y: number;
  readonly offset_x: number;
  readonly offset_y: number;
  readonly tile_area_mm2: number;
  readonly tile_size_um: number;
  readonly mpp_thumb_x: number;
  readonly mpp_thumb_y: number;
}

export interface CaseMeta {
  readonly case_id: number;
  readonly barcode: string;
  readonly project_id: string;
  readonly file_uuid: string;
  readonly base_width: number;
  readonly base_height: number;
  readonly grid_nx: number;
  readonly grid_ny: number;
  readonly n_tiles_tissue: number;
  readonly has_purity: boolean;
  readonly has_cellularity: boolean;
  readonly base_image: string;
  readonly purity_mask: string;
  readonly cellularity_mask: string;
  readonly tiles_meta: string;
  readonly tiles_bin: string;
  readonly tile_size_um: number;
  readonly mpp_thumb_x: number;
  readonly mpp_thumb_y: number;
  readonly purity_model_version: string;
  readonly cellularity_model_version: string;
  readonly tile_encoder_version: string;
}

// Tile-prediction arrays loaded from `<case>_grid.bin`. Each channel is a
// Float32 array of length grid_nx * grid_ny in row-major order.
export interface TileArrays {
  readonly grid: TileGridMeta;
  readonly purity: Float32Array;
  readonly purity_sd: Float32Array;
  readonly nuclei: Float32Array;
  readonly nuclei_sd: Float32Array;
  readonly tumor_nuclei: Float32Array;
  readonly tissue_fraction: Float32Array;
}

export interface MetricsCI {
  readonly median: number;
  readonly low: number;
  readonly high: number;
}

export interface ROIMetrics {
  readonly n_tiles: number;
  readonly tiles_with_data: number;
  readonly area_thumbpx2: number;
  readonly area_mm2: number;
  readonly tissue_fraction_mean: number;
  readonly purity: MetricsCI;
  readonly total_nuclei: MetricsCI;
  readonly tumor_nuclei: MetricsCI;
  readonly adequacy_probability: number;
  readonly purity_point: number;
  readonly total_nuclei_point: number;
  readonly tumor_nuclei_point: number;
}

export type AdequacyLabel = "pass" | "borderline" | "fail" | "not_quantifiable";

export interface AdequacyVerdict {
  readonly label: AdequacyLabel;
  readonly confidence: number;
  readonly reasons: readonly string[];
  readonly thresholds: ThresholdProfile;
  readonly metrics_snapshot: ROIMetrics;
}

export type LayerName = "purity" | "cellularity" | "adequacy" | "uncertainty";
export type SmoothingMode = "overview" | "balanced" | "detail";

export interface LayerVisibility {
  readonly layer: LayerName;
  readonly opacity: number;     // 0..1
  readonly enabled: boolean;
  readonly smoothing: SmoothingMode;
}

export interface ROIRecord {
  readonly roi_id: string;
  readonly case_id: number;
  readonly user_id: string;
  readonly label: string;
  readonly geometry_thumb_px: GeoJSONPolygon;
  readonly thresholds: ThresholdProfile;
  readonly created_at: string;
  readonly updated_at: string;
  readonly locked: boolean;
  readonly model_run: {
    readonly purity_model_version: string;
    readonly cellularity_model_version: string;
    readonly tile_encoder_version: string;
  };
  readonly metrics_snapshot:
    | { metrics: ROIMetrics; verdict: AdequacyVerdict }
    | null;
  readonly notes: string;
  readonly revision: number;
}
