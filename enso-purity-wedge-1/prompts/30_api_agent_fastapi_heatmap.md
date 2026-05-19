# Prompt: API agent — FastAPI inference + heatmap (TDD)

Goal:
Build a FastAPI app serving purity prediction and a heatmap over ROIs.

Endpoints:
- GET `/health`
- POST `/predict_purity`
  - accepts either:
    (A) uploaded `.h5` file, or
    (B) JSON with `embedding_path` / `file_id`
  - returns:
    - `purity_wsi` (median over ROI preds)
    - `purity_ta` (p95 over ROI preds)
    - `heatmap_png_base64`

Heatmap method:
- Use tile coordinates from `coords` (and optionally `coords_level0`)
- Slide a window over tissue; for each ROI, sample e.g. 16 tiles and run model
- Render heatmap as PNG

TDD:
- tests using FastAPI TestClient
- deterministic fixture `.h5` included in tests (tiny)

Deliverables:
- `backend/src/enso_purity/api/main.py`
- `backend/tests/api/*`
