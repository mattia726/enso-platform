PLAN — AI-Assisted Macrodissection Workbench
============================================

0\. Executive summary
---------------------

Transform the existing single-page purity demo into a decision-grade macrodissection workbench in the `enso-cell-count` repo (it already carries both purity and cellularity masks for 32 cases on the same H&E thumbnails).

The new page `/macrodissection` lets a pathologist:

1.  open an H&E slide,
2.  pan and zoom it like a real digital-pathology viewer,
3.  toggle AI overlays — purity, cellularity, adequacy, uncertainty — with zoom-adaptive smoothing and alpha-only blur,
4.  draw a polygon/freehand macrodissection ROI,
5.  see in real time the ROI adequacy card:
    *   estimated tumor purity ± CI,
    *   estimated total nuclei ± CI,
    *   estimated tumor nuclei ± CI,
    *   adequacy probability vs. configurable assay thresholds,
    *   pass / borderline / fail label with reasons,
6.  accept auto-suggested candidate ROIs,
7.  lock the ROI (authoritative server-side recompute with audit metadata),
8.  export a printable macrodissection sheet for the lab.

All ROI math is implemented twice (Python on the server, TypeScript on the client) with the same algorithms and unit-tested for parity. Visual smoothing is decoupled from metric computation: smoothing only affects the overlay canvas, never the numbers in the adequacy card.

The plan ships in seven phases, each with concrete files, tests, and visual checks. Final deliverables include a green test suite, screenshots at each zoom level, and a self-contained printable sheet.

* * *

1\. Target repo and conventions
-------------------------------

*   Repo: `/workspace/enso-cell-count/`
*   Backend module: `backend/src/enso_purity/` (Python; new `macrodissection` sub-package added).
*   Frontend route: `/macrodissection` (new), implemented under `frontend/app/macrodissection/`. Existing `/` (Case Explorer) is preserved for the investor narrative.
*   Static assets path: `frontend/public/cases/` (existing) — we add `case_N_tiles.json` and `case_N_grid.bin` artifacts.
*   No new infra deps: no DB, no Docker. JSON-on-disk for ROI persistence, matching the existing AGENTS.md guidance.

* * *

2\. Intended use (locked Phase 0 statement)
-------------------------------------------

> EnsoPurity + EnsoCellularity assist a pathologist in selecting and documenting a macrodissection ROI for downstream molecular testing by overlaying AI-estimated tumor purity, total cellularity, and adequacy metrics on a digital H&E slide. The pathologist always selects, edits, and locks the final ROI. The system records the user, ROI geometry, threshold profile, model versions, and timestamps for audit.

This statement is the header of `frontend/app/macrodissection/page.tsx`, the home of the printed sheet, and the README of the macrodissection module.

* * *

3\. Data model
--------------

### 3.1 Per-tile prediction artifact (per case)

`frontend/public/cases/case_{N}_tiles.json`:

json

    {  "schema_version": 1,  "case_id": 12,  "barcode": "TCGA-DX-A23T-01A-01-TSA",  "project_id": "TCGA-SARC",  "file_uuid": "4f4548cf-870a-4962-9e04-413da8c0725e",  "base_width": 3302,  "base_height": 1480,  "tile_pix_w": 56,                // tile width in *thumbnail* pixels  "tile_pix_h": 56,  "tile_size_um": 112.0,           // tile physical side (mpp 0.5 * 224)  "tile_area_mm2": 0.012544,  "mpp_thumb_x": 6.598,            // µm per thumbnail pixel  "mpp_thumb_y": 6.598,  "grid_nx": 59,  "grid_ny": 27,  "purity_model_version": "v3_fold0",  "cellularity_model_version": "ssd_fold1",  "tile_encoder_version": "virchow_v1",  "thresholds_default": {    "purity_min": 0.20,    "tumor_cells_min": 1000,    "borderline_band": 0.10  },  "tiles_bin": "case_12_grid.bin",  "tiles_bin_layout": ["purity", "purity_sd", "nuclei", "nuclei_sd",                        "tumor_nuclei", "tissue_fraction"],  "n_tiles_tissue": 1342}

`case_{N}_grid.bin` is a `Float32Array` packed as `(grid_ny, grid_nx, 6)` row-major (purity, purity\_sd, nuclei, nuclei\_sd, tumor\_nuclei, tissue\_fraction). NaN signals "no tissue / no prediction". This is loaded once with `fetch()` and kept in memory.

### 3.2 ROI annotation (server-side, JSON-lines)

`backend/.runtime/rois/case_{N}.jsonl`:

json

    {  "roi_id": "roi_2026-05-19T07-30-12_a4b1",  "case_id": 12,  "user_id": "demo-pathologist",  "label": "ROI 1",  "geometry_thumb_px": {    "type": "Polygon",    "coordinates": [[[120, 60], [180, 80], [200, 120], [140, 130]]]  },  "created_at": "2026-05-19T07:30:12Z",  "updated_at": "2026-05-19T07:31:48Z",  "locked": true,  "thresholds": { "purity_min": 0.2, "tumor_cells_min": 1000 },  "metrics_snapshot": { "...": "see §5" },  "model_run": {    "purity_model_version": "v3_fold0",    "cellularity_model_version": "ssd_fold1",    "tile_encoder_version": "virchow_v1"  }}

### 3.3 Threshold profiles

`backend/src/enso_purity/macrodissection/thresholds.py`:

python

    PROFILES = {  "humanitas_ngs": {    "purity_min": 0.20,    "tumor_cells_min": 1000,    "borderline_purity_band": 0.05,    "borderline_tumor_cells_band": 200,  },  "research": {    "purity_min": 0.10,    "tumor_cells_min": 200,    "borderline_purity_band": 0.05,    "borderline_tumor_cells_band": 100,  },  "custom": {"...": "user-defined"},}

Mirror in TypeScript at `frontend/lib/thresholds.ts`.

* * *

4\. Adequacy math (shared algorithm)
------------------------------------

For every tile _i_ we have raw scalars `p_i, σ_p_i, n_i, σ_n_i, tissue_fraction_i` plus its axis-aligned thumbnail-pixel rectangle `(x0_i, y0_i, w_i, h_i)`.

### 4.1 Polygon × tile weight

For a polygon `P` (thumbnail-pixel coordinates):

    w_i = area(P ∩ tile_rect_i) / area(tile_rect_i)        # in [0, 1]

Implemented via Sutherland–Hodgman clipping (axis-aligned clip lines per tile edge). Pure functions in both languages, unit-tested for boundary cases.

### 4.2 Point estimates

    N_total      = Σ w_i · n_i · tissue_fraction_iN_tumor      = Σ w_i · n_i · tissue_fraction_i · p_iPurity_ROI   = N_tumor / max(N_total, ε)Area_mm2     = Σ w_i · tile_area_mm2 · tissue_fraction_i

Purity is cellularity-weighted, exactly as the user spec demands.

### 4.3 Monte-Carlo uncertainty

    For k in 1..K:  for each tile i:    n_i_k = truncated_normal(mean=n_i, sd=σ_n_i, lower=0)    p_i_k = clipped_normal(mean=p_i, sd=σ_p_i, lower=0, upper=1)  N_total_k = Σ w_i · n_i_k · tissue_fraction_i  N_tumor_k = Σ w_i · n_i_k · tissue_fraction_i · p_i_k  Purity_k  = N_tumor_k / max(N_total_k, ε)P_total, P_tumor, P_purity = percentiles (5, 50, 95)adequacy_prob = mean(    (Purity_k ≥ purity_min) & (N_tumor_k ≥ tumor_cells_min))

K=400 samples is the default; sampler uses a deterministic seed derived from the polygon hash so the same ROI always reports the same number.

### 4.4 Adequacy label

    pass        : adequacy_prob ≥ 0.90 AND median_purity ≥ purity_min              AND median_tumor_nuclei ≥ tumor_cells_minborderline  : adequacy_prob ≥ 0.50 (otherwise)fail        : adequacy_prob < 0.50not_quantif : Σ w_i == 0 OR all tiles NaN OR mpp missing

Reason strings are constructed from the same threshold values for the UI card.

* * *

5\. Phased delivery
-------------------

### Phase 1 — Tile-artifact pipeline (backend + scripts)

New files:

*   `backend/src/enso_purity/macrodissection/__init__.py`
*   `backend/src/enso_purity/macrodissection/inverse_cmap.py`
    *   Builds 256-entry LUTs for `RdYlBu_r` and the cellularity `purity_no_white` cmap.
    *   `decode_rgb(rgb_array, lut, vmin, vmax, gamma) -> values, validity`.
*   `backend/src/enso_purity/macrodissection/grid_detect.py`
    *   `detect_tile_stride(rgba) -> (stride_x, stride_y, nx, ny)` — finds the spacing at which the mask is piecewise constant (min within-block variance over \[1..K\] candidates).
*   `backend/src/enso_purity/macrodissection/build_artifacts.py`
    *   CLI: `python -m enso_purity.macrodissection.build_artifacts --cases-dir frontend/public/cases --gallery frontend/gallery/gallery_summary.csv`.
    *   For each case\_N:
        1.  Open `case_N_base.jpg`, `case_N_mask.png`, `case_N_cell_count_mask.png`.
        2.  Detect grid via `grid_detect`.
        3.  Decode purity and cellularity to scalar grids.
        4.  Compute tumor\_nuclei = purity · nuclei, tissue\_fraction from alpha.
        5.  Write `case_N_tiles.json` + `case_N_grid.bin`.
    *   Idempotent; emits a console summary table.
*   `backend/src/enso_purity/macrodissection/roi.py`
    *   `tile_weights(polygon, grid_meta)` (Sutherland–Hodgman).
    *   `point_estimates(weighted_tiles)`.
    *   `monte_carlo(weighted_tiles, K=400, seed=...)`.
    *   `adequacy_label(metrics, thresholds)`.
*   `backend/src/enso_purity/macrodissection/candidates.py`
    *   Sliding-window scoring over the tile grid; non-max-suppression on overlapping rectangles; returns top-K candidate polygons.
*   `backend/src/enso_purity/macrodissection/storage.py`
    *   Append-only JSONL store for ROIs under `backend/.runtime/rois/`.
*   `backend/src/enso_purity/macrodissection/report.py`
    *   Builds report data (slide info + ROI metrics + threshold profile + model versions + thumbnail path); rendering is done in the frontend.

Backend tests (`backend/tests/macrodissection/test_*.py`):

*   `test_lut_roundtrip` — for every value in \[0, 1\], encoding then decoding recovers within ±1 LUT step.
*   `test_grid_detect_synthetic` — synthetic 60×30 grid with random colors, resized 4× via NEAREST → detector recovers stride exactly.
*   `test_grid_detect_real_assets` — running on `case_1_mask.png` returns a non-trivial grid that fits within the base dimensions.
*   `test_tile_weights_full_tile` — square ROI covering tile fully → w=1.0.
*   `test_tile_weights_half_tile` — ROI bisecting tile → w==0.5 ± 1e-6.
*   `test_tile_weights_rotated` — 45° rotated ROI vs reference shapely impl (uses shapely only as oracle in tests, not in product).
*   `test_point_estimates` — known grid + known polygon → known numbers.
*   `test_monte_carlo_seed_determinism` — same seed → identical output.
*   `test_monte_carlo_low_variance` — σ→0 ⇒ percentiles collapse to point est.
*   `test_adequacy_pass_fail` — bracket cases around thresholds.
*   `test_storage_append_lock_roundtrip` — write + reload + lock + reload.
*   `test_candidates_returns_top_k` — synthetic grid with planted hotspot is ranked first.

Acceptance criteria for Phase 1:

*   `python -m enso_purity.macrodissection.build_artifacts ...` emits one `_tiles.json` + one `_grid.bin` per case under `frontend/public/cases/`.
*   All new pytest tests pass.
*   `ruff check backend/src/enso_purity/macrodissection` clean.

### Phase 2 — Backend ROI API

Edit: `backend/src/enso_purity/api/main.py` — add a router for the macrodissection endpoints (keeps existing `/health`, `/predict_purity` intact):

    GET    /api/macrodissection/casesGET    /api/macrodissection/cases/{case_id}GET    /api/macrodissection/cases/{case_id}/tiles          # streams JSON+binGET    /api/macrodissection/cases/{case_id}/roisPOST   /api/macrodissection/cases/{case_id}/rois/preview   # returns metricsPOST   /api/macrodissection/cases/{case_id}/rois           # save (draft)POST   /api/macrodissection/cases/{case_id}/rois/{roi_id}/lockPATCH  /api/macrodissection/cases/{case_id}/rois/{roi_id}  # edit geometryDELETE /api/macrodissection/cases/{case_id}/rois/{roi_id}GET    /api/macrodissection/cases/{case_id}/rois/{roi_id}/reportGET    /api/macrodissection/cases/{case_id}/candidates?k=5GET    /api/macrodissection/threshold-profiles

`preview` is fast (no disk write, just MC + adequacy on the in-memory grid). `save` and `lock` both append a new JSONL line with the recomputed metrics snapshot; `locked: true` means subsequent edits must duplicate-then-lock a new record. CORS is enabled for the Next dev server origin.

Backend tests: add `backend/tests/macrodissection/test_api.py`:

*   preview returns valid metrics for a synthetic case + ROI,
*   save/list/lock/get-report round-trips correctly,
*   preview is idempotent for the same polygon,
*   candidates endpoint returns ≤K polygons with non-overlapping bounding boxes,
*   DELETE removes drafts only and refuses to delete a locked ROI.

Acceptance criteria for Phase 2:

*   New endpoints visible in Swagger UI and reachable from the frontend.
*   `make backend-test` (or `python -m pytest -q`) green.

### Phase 3 — Frontend: viewer scaffolding

Add deps (`frontend/package.json`):

    "dependencies": {  "openseadragon": "^4.1.0",  "@types/openseadragon": "^3.0.10"  // devDeps},"devDependencies": {  "vitest": "^1.6.0",  "@vitest/coverage-v8": "^1.6.0",  "playwright": "^1.45.0"}

New files (`frontend/app/macrodissection/`):

*   `page.tsx` — Next page entrypoint (server component that fetches case list).
*   `MacrodissectionClient.tsx` — root client component (state, layout, wiring).
*   `components/WSIViewer.tsx` — wraps OpenSeadragon (`tileSources: { type: "image", url: "/cases/case_N_base.jpg" }`), exposes `useViewer()` hook with current viewport bounds and zoom factor.
*   `components/HeatmapOverlay.tsx` — `<canvas>` overlay positioned via the OSD overlay API; redraws on viewport / layer change / sigma change.
*   `components/ROILayer.tsx` — overlay for polygon drawing & vertex edit.
*   `components/LayerPanel.tsx` — layer toggles (purity, cellularity, adequacy, uncertainty), opacity sliders, smoothing select (Overview / Balanced / Detail), threshold-profile picker.
*   `components/RoiMetricsCard.tsx` — adequacy card (median + CI95 for purity, total\_nuclei, tumor\_nuclei; adequacy probability; pass/borderline/fail pill; reason list).
*   `components/RoiHistoryList.tsx` — saved + locked ROIs sidebar.
*   `components/CandidateList.tsx` — top-K candidate ROIs with previews.
*   `components/CaseSidebar.tsx` — case selector with thumbnails.
*   `components/ReportSheet.tsx` — printable report (window.print()-friendly).

Shared TS math (`frontend/lib/macrodissection/`):

*   `tiles.ts` — load & decode `case_N_tiles.json` + `case_N_grid.bin`.
*   `polygon.ts` — Sutherland–Hodgman tile clipping (mirror of Python).
*   `metrics.ts` — point estimates + Monte-Carlo + adequacy label.
*   `adequacy.ts` — pass/borderline/fail mapping + reason strings.
*   `colormaps.ts` — JS LUTs identical to the Python ones (purity → 0..1, cellularity → 0..vmax, adequacy → 0..vmax\_tumor\_density, uncertainty → 0..1).
*   `smoothing.ts` — alpha-only blur + zoom-dependent σ + NaN inpainting, exactly mirroring the user's pasted prototype.
*   `report.ts` — build report payload for ReportSheet.

Frontend tests (`frontend/tests/`):

*   `polygon.spec.ts` — full/half/rotated tile clipping vs analytic ground truth.
*   `metrics.spec.ts` — known grid + polygon → known numbers; MC determinism.
*   `adequacy.spec.ts` — bracket cases around thresholds.
*   `colormaps.spec.ts` — TS LUT matches Python LUT to ≤1/256 RGB.
*   `smoothing.spec.ts` — alpha mask preserves holes; sigma=0 ⇒ identity.

Acceptance criteria for Phase 3:

*   `npm run --workspace frontend test` (vitest) green.
*   `next build` succeeds.
*   Loading `/macrodissection` shows case sidebar + OSD viewer with H&E.

### Phase 4 — Frontend: overlays, ROI, metrics

*   `HeatmapOverlay` honours `LayerPanel` state and current OSD zoom; default layer = Adequacy.
*   Smoothing select drives σ\_tiles per the formula `σ = max(0, 1.5 · (1 - t)^1.7)` where `t = currentZoom / maxZoom`. At max zoom the overlay is identical to the raw tile grid (no blur).
*   `ROILayer`: click-and-drag freehand, click polygon (with auto-close), vertex handles, drag-to-edit, delete + duplicate + split + label.
*   Metrics card runs the TS MC engine for every preview (debounced 80 ms while dragging). On `Save & Lock`, the frontend POSTs to the backend and uses the server-recomputed snapshot as the source of truth, displaying any delta vs the client estimate (should be zero by construction).

Frontend tests:

*   React Testing Library snapshot of the empty workbench.
*   Component test: drawing a square ROI on a synthetic case fixture updates the metrics card to known values.
*   Server-vs-client parity test (Playwright): for 3 ROIs on 3 cases, the numbers shown by the card match what the backend returns to ≤1e-3.

Acceptance criteria for Phase 4:

*   Drawing an ROI updates metrics in < 100 ms.
*   Smoothing slider visibly softens the overlay at low zoom and disappears at high zoom (verified by Playwright pixel diff).
*   Server/client metric parity holds for the smoke fixtures.

### Phase 5 — Candidate ROIs and report sheet

*   Candidate list backed by `GET /candidates?k=5`; clicking a row inserts a fresh editable polygon, scrolls/zooms the viewer to fit, and updates the metrics card. Cards display thumbnail, purity, tumor\_nuclei, adequacy prob.
*   ReportSheet: a clean, A4-landscape print layout with
    *   slide thumbnail (case\_N\_base.jpg, downscaled);
    *   H&E with locked-ROI outline (rendered into a hidden canvas, exported as PNG and embedded);
    *   ROI crop (zoomed to 1.2× bbox);
    *   metrics table with CIs and thresholds used;
    *   model versions + signature line + disclaimer.
*   A "Print / Save PDF" button calls `window.print()` with the right CSS.

Frontend tests: Playwright captures the printable layout via `page.emulateMedia({ media: "print" })`.

### Phase 6 — Tests + visual assessment

*   Run the full pytest suite for the backend.
    
*   Run vitest + Playwright for the frontend.
    
*   Playwright visual assessment script (`frontend/tests/visual/macrodissection.spec.ts`) captures:
    
    1.  Empty workbench (case 1 loaded).
    2.  H&E + purity overlay (low zoom, smoothed).
    3.  H&E + cellularity overlay (low zoom, smoothed).
    4.  H&E + adequacy overlay (low zoom, smoothed) — default state.
    5.  Adequacy overlay at high zoom (raw tiles visible).
    6.  ROI drawn + adequacy card visible.
    7.  Candidate ROI list opened.
    8.  Locked ROI + report sheet preview.
    
    Screenshots written to `/opt/cursor/artifacts/assets/screenshots/macrodissection_<i>.png` (unique per run; immutable).
    
*   Manual inspection checklist embedded in the PR body.
    

### Phase 7 — Hardening, polish, docs

*   README in `enso-cell-count/docs/macrodissection.md` with overview, intended use, architecture diagram, screenshots, threshold profiles, audit data shape, and developer how-to.
*   Update `enso-cell-count/AGENTS.md` with the new workbench commands: `make build-artifacts`, `make macrodissection-dev`, `make macrodissection-test`.
*   Add `Makefile` targets (`build-artifacts`, `macrodissection-dev`, `macrodissection-test`, `visual-check`).
*   Add a `requirements.txt` row for any new backend dep (none expected beyond numpy/Pillow already present).
*   Confirm `next build` passes; commit + push.
*   Open a draft PR with the visual assessment screenshots embedded.

* * *

6\. File-by-file change list (summary)
--------------------------------------

enso-cell-count/backend (new):

*   `src/enso_purity/macrodissection/__init__.py`
*   `src/enso_purity/macrodissection/inverse_cmap.py`
*   `src/enso_purity/macrodissection/grid_detect.py`
*   `src/enso_purity/macrodissection/build_artifacts.py`
*   `src/enso_purity/macrodissection/roi.py`
*   `src/enso_purity/macrodissection/candidates.py`
*   `src/enso_purity/macrodissection/storage.py`
*   `src/enso_purity/macrodissection/thresholds.py`
*   `src/enso_purity/macrodissection/report.py`
*   `src/enso_purity/macrodissection/router.py`
*   `tests/macrodissection/test_*.py`

enso-cell-count/backend (edit):

*   `src/enso_purity/api/main.py` — mount the new router + enable CORS for `http://localhost:3000`.

enso-cell-count/frontend (new):

*   `app/macrodissection/page.tsx`
*   `app/macrodissection/MacrodissectionClient.tsx`
*   `app/macrodissection/components/*.tsx` (10 files, see §5 Phase 3)
*   `lib/macrodissection/*.ts` (7 files, see §5 Phase 3)
*   `tests/*.spec.ts` (vitest)
*   `tests/visual/macrodissection.spec.ts` (Playwright)
*   `vitest.config.ts`, `playwright.config.ts`
*   `public/cases/case_*_tiles.json` and `case_*_grid.bin` (generated, but committed for static deploy).

enso-cell-count/frontend (edit):

*   `package.json` — add `openseadragon`, `vitest`, `playwright`, scripts.
*   `app/layout.tsx` — add a top-nav link to `/macrodissection`.
*   `next.config.mjs` — no changes expected (static files already served).

enso-cell-count (root):

*   `Makefile` — new targets.
*   `docs/macrodissection.md` — new docs.
*   `AGENTS.md` — add workbench section.

* * *

7\. Testing strategy (consolidated)
-----------------------------------

Layer

Tool

Coverage

Backend math

pytest

LUT roundtrip, grid detection, Sutherland–Hodgman parity vs shapely (oracle), MC determinism, adequacy boundary cases, candidates ranking.

Backend API

pytest+TestClient

All endpoints: preview, save, lock, list, candidates, report, threshold-profiles.

Backend lint

ruff

`ruff check backend/src/enso_purity/macrodissection backend/tests/macrodissection`.

Frontend math

vitest

TS mirrors of polygon, metrics, MC, adequacy, smoothing, colormaps.

Frontend UI

RTL+vitest

LayerPanel toggles, ROI drawing reducer, RoiMetricsCard rendering.

End-to-end

Playwright

Full workflow on case 1 → draw ROI → metrics card update → save → lock → report; server/client metric parity.

Visual

Playwright

8 named screenshots saved to `/opt/cursor/artifacts/assets/screenshots/`.

Build

next build

Production build green.

Acceptance: all of pytest, vitest, Playwright suites green; `next build` green; 8 visual screenshots present.

* * *

8\. Risks and mitigations
-------------------------

Risk

Mitigation

Inverse-cmap decoding misreads a tile

LUT roundtrip test + tolerance band; tiles flagged as low-confidence when LUT distance > δ.

Grid detection picks wrong stride

Search over multiple candidates and pick min within-block variance ratio; cap to base/2.

OpenSeadragon overlay drift while zooming

Use OSD's native `addOverlay` API for HTML overlays; redraw canvas on every viewport event.

Smoothing implies false precision

Metrics card always uses RAW tile grid; smoothing toggle is labelled "visual smoothing".

Frontend MC too slow

K=400 + Float32Array workloads; debounce 80 ms; web worker if profiling shows hot path.

Playwright unavailable in CI VM

Install `playwright install --with-deps chromium` in update script; document fallback.

Static-only deploy breaks `/api/*` endpoints

Keep backend optional: client-side MC is the source of preview metrics; backend save only.

Server/client metric drift

Parity unit test in pytest + a Playwright smoke test asserting ≤1e-3 difference.

* * *

9\. Definition of done
----------------------

1.  `make build-artifacts` regenerates per-case JSON + binary grids.
2.  `make backend-test` and `make frontend-test` are both green.
3.  `make visual-check` runs Playwright and emits 8 screenshots into `/opt/cursor/artifacts/assets/screenshots/`, each named, of correct size, and recognisably showing each described UI state.
4.  `next build` succeeds in `frontend/`.
5.  A draft PR is opened with title "Macrodissection workbench: ROI adequacy + uncertainty + report", description containing
    *   intended-use box,
    *   architecture diagram (mermaid),
    *   8 visual screenshots (embedded),
    *   test-suite summary,
    *   migration / rollout notes.
6.  `frontend/app/page.tsx` Case Explorer remains unaffected.
7.  `AGENTS.md` updated with how to run the new workbench locally and in CI.

* * *

10\. Out of scope (explicitly)
------------------------------

*   Real WSI tile-pyramid serving (OpenSlide / DZI) — the base JPGs are sufficient for the demo; the API contract for tiles is designed so a real DZI source can be plugged in later without code changes.
*   Multi-user auth, RBAC, encrypted storage — Phase 6 of the user spec; flagged but not built in this iteration.
*   LIS integration.
*   Anything beyond pathology image analysis (no PHI ingestion).