# Macrodissection Workbench

The macrodissection workbench is an AI-assisted decision-support tool that
mirrors the pathologist's existing glass-slide workflow: inspect the H&E,
identify tumor-rich tissue, circle the region, estimate tumor cellularity,
and decide whether that region is adequate for downstream molecular
testing. The workbench preserves that workflow — the pathologist always
selects, edits, and locks the final ROI. EnsoPurity and EnsoCellularity
quantify whether the selected region is likely *adequate* for the assay.

## Intended use

> EnsoPurity + EnsoCellularity **assist** a pathologist in selecting and
> documenting a macrodissection ROI for downstream molecular testing by
> overlaying AI-estimated tumor purity, total cellularity, and adequacy
> metrics on a digital H&E slide. The pathologist always selects, edits,
> and locks the final ROI. The system records the user, ROI geometry,
> threshold profile, model versions, and timestamps for audit.

## Architecture overview

```
┌─────────────────────────────────────────────────────────────────────┐
│ EnsoPurity / EnsoCellularity ML inference                            │
│   (run once per slide; emits rendered RGBA masks + tile predictions) │
└────────────────────────────────────┬────────────────────────────────┘
                                     │
                                     ▼
              backend/src/enso_purity/macrodissection/
              ┌─────────────────────────────────────────┐
              │  build_artifacts → case_N_tiles.json    │
              │                  → case_N_grid.bin      │
              └────────────────────────────────────┬────┘
                                                   │
                                                   ▼
                  ┌────────────────────────────────────────────────────┐
                  │                FastAPI router                       │
                  │  /api/macrodissection/cases                         │
                  │  /api/macrodissection/cases/{id}/rois/preview       │
                  │  /api/macrodissection/cases/{id}/rois  (save)       │
                  │  /api/macrodissection/cases/{id}/rois/{rid}/lock    │
                  │  /api/macrodissection/cases/{id}/candidates         │
                  │  /api/macrodissection/cases/{id}/rois/{rid}/report  │
                  └─────────────────┬─────────────────┬────────────────┘
                                    │                 │
                            (preview / save)        (Cloudflare-deploy
                                    │                 frontend reads
                                    ▼                 the same artifacts
              ┌─────────────────────────────────────┐ directly)
              │       Next.js workbench page         │◄───────────────┐
              │       /macrodissection               │                │
              │                                      │                │
              │  ┌──────────────┐  ┌──────────────┐  │                │
              │  │  WSIViewer   │  │ LayerPanel   │  │                │
              │  │ (OpenSea-    │  │   purity     │  │                │
              │  │  dragon)     │  │   cellularity│  │                │
              │  └──────┬───────┘  │   adequacy   │  │                │
              │         │          │   uncertainty│  │                │
              │  ┌──────▼───────┐  └──────────────┘  │                │
              │  │HeatmapOverlay│  ┌──────────────┐  │                │
              │  │ (zoom-       │  │  ROILayer    │  │                │
              │  │  adaptive    │  │ (SVG draw +  │  │                │
              │  │  smoothing)  │  │  vertex edit)│  │                │
              │  └──────────────┘  └──────────────┘  │                │
              │                                      │                │
              │  ┌─────────────────────────────────┐ │                │
              │  │  RoiMetricsCard (live preview)  │ │                │
              │  └─────────────────────────────────┘ │                │
              │  ┌─────────────────────────────────┐ │                │
              │  │  CandidateList / ReportSheet    │ │                │
              │  └─────────────────────────────────┘ │                │
              └─────────────────────────────────────┘                 │
                              ▲                                       │
                              └────── frontend/public/cases/ ─────────┘
```

## ROI math (Python + TypeScript twins)

| File (Python)                                          | File (TypeScript)                            |
| ------------------------------------------------------ | -------------------------------------------- |
| `enso_purity.macrodissection.roi.tile_weights`         | `frontend/lib/macrodissection/polygon.ts`    |
| `enso_purity.macrodissection.roi.point_estimates`      | `frontend/lib/macrodissection/metrics.ts`    |
| `enso_purity.macrodissection.roi.monte_carlo`          | `frontend/lib/macrodissection/metrics.ts`    |
| `enso_purity.macrodissection.adequacy.label_adequacy`  | `frontend/lib/macrodissection/adequacy.ts`   |
| `enso_purity.macrodissection.candidates.suggest`       | `frontend/lib/macrodissection/candidates.ts` |
| `enso_purity.macrodissection.thresholds`               | `frontend/lib/macrodissection/thresholds.ts` |

For an ROI polygon `P` and tile rectangle `T_i`:

```
w_i        = area(P ∩ T_i) / area(T_i)            (Sutherland–Hodgman)
N_total    = Σ w_i · n_i · tissue_fraction_i
N_tumor    = Σ w_i · n_i · tissue_fraction_i · p_i
Purity_ROI = N_tumor / max(N_total, ε)
```

Purity is **cellularity-weighted**, never a simple mean of tile purity —
this matches the clinically meaningful question (“is the ROI sufficiently
tumor-rich *and* does it contain enough cells?”).

Monte-Carlo with `N_samples = 400` propagates per-tile σ to ROI-level
percentiles. The PRNG is deterministic given the polygon hash, so the same
ROI always reports the same numbers.

The adequacy label is

```
pass        : adequacy_prob ≥ 0.90 AND median_purity ≥ purity_min
              AND median_tumor_nuclei ≥ tumor_cells_min
borderline  : adequacy_prob ≥ 0.50    (or above-threshold but in band)
fail        : adequacy_prob < 0.50
not_quantif : ROI overlaps no tissue
```

## Layer / overlay strategy

| Layer       | Palette                                        | Use                                                             |
| ----------- | ---------------------------------------------- | --------------------------------------------------------------- |
| Adequacy    | white → orange → red (gamma 0.65, vmax = 200)  | **Default** layer — the per-tile tumor-nuclei density.          |
| Purity      | RdYlBu_r (blue → yellow → red)                 | Tile-level tumor fraction.                                      |
| Cellularity | purity_no_white (blue → warm; gamma 0.65)      | Per-tile nuclei count.                                          |
| Uncertainty | gray → warm-gray                               | Per-tile model uncertainty band.                                |

Smoothing is **purely visual** — the metrics card always uses the raw tile
predictions. The visual `σ` follows `σ = 1.5 · (1 − zoom/maxZoom)^1.7` and
collapses to zero at maximum zoom so high-zoom users see the raw tile
lattice exactly as the model produced it.

## Threshold profiles

Three built-in profiles ship with the workbench:

| Profile               | purity ≥ | tumor nuclei ≥ | pass prob |
| --------------------- | -------- | --------------- | ---------- |
| `humanitas_ngs`       | 20%      | 1000            | 90%        |
| `research`            | 10%      | 200             | 85%        |
| `strict_solid_tumor`  | 30%      | 2000            | 95%        |

Per-call overrides are accepted at the `preview` and `save` endpoints; the
locked snapshot stores the resolved profile so the report sheet can show
exactly which numbers gated the verdict.

## Persistence

ROIs persist as JSON-lines under `backend/.runtime/rois/case_{N}.jsonl`.
Every revision (geometry edit, threshold change, lock) appends a new line;
the file is append-only so an auditor can replay the entire ROI history.

## Local development

```sh
make backend-install
make backend-test                # 57 backend pytest passing
make macrodissection-artifacts   # rebuild per-case JSON + grid from PNGs
make macrodissection-serve       # uvicorn on :8000

make frontend-install
make frontend-test               # 40 vitest passing
make frontend-build              # next build (static export)
make visual-check                # Playwright → docs/screenshots/*.png
```

## Visual reference

The eight reference screenshots are produced by the Playwright visual run
and live under [`docs/screenshots/`](screenshots/):

| Name                                                | Captures                                              |
| --------------------------------------------------- | ----------------------------------------------------- |
| `01_workbench_overview_adequacy_low_zoom.png`       | Default state: adequacy overlay, low zoom, smoothed.  |
| `02_purity_overlay_low_zoom.png`                    | Purity (RdYlBu_r) toggled on.                         |
| `03_cellularity_overlay_low_zoom.png`               | Cellularity overlay toggled on.                       |
| `04_adequacy_detail_smoothing.png`                  | Adequacy at the detail smoothing tier — raw lattice.  |
| `05_candidate_areas_list.png`                       | Auto-suggested macrodissection candidates panel.      |
| `06_roi_drawn_with_metrics_card.png`                | ROI polygon drawn; adequacy card shows median + CI.   |
| `07_locked_roi_in_history.png`                      | ROI saved and locked; appears in the sidebar history. |
| `08_macrodissection_sheet.png`                      | Printable macrodissection sheet (export view).        |

## Disclaimer (rendered into the sheet)

> AI-assisted estimate. The final macrodissection ROI must be selected
> and signed off by the reviewing pathologist; the EnsoPurity and
> EnsoCellularity outputs are decision support, not autonomous decision
> making.
