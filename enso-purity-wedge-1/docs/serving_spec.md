# Serving + demo spec

## Backend

FastAPI app that:

- loads trained MIL weights at startup
- provides `/health` and `/v1/predict`

`/v1/predict` should support:

- input: slide ID OR uploaded embedding file (H5)
- output:
  - `purity_wsi` (median over all ROIs / tiles)
  - `purity_ta` (tumor-area proxy, e.g., top 95th percentile)
  - optional heatmap image

## Frontend

Next.js dashboard that:

- drag-and-drop file upload
- clear loading indicator
- results view with metrics + heatmap overlay

## Deployment (MVP)

- Backend on a GCP VM with GPU
- Frontend on Vercel
