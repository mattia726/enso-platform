# Prompt: FRONTEND agent — Next.js dashboard

Goal:
Create a clean, modern dark-mode dashboard for Enso Purity MVP.

Requirements:
- Next.js + TypeScript + Tailwind
- Drag-and-drop file upload for `.h5`
- Loading state: “Running Causal MIL Engine…”
- Results view:
  - heatmap image
  - Global Purity + Tumor Area Purity numbers
- Robust error display
- Minimal dependencies

API contract:
- call backend POST `/predict_purity` (multipart upload)
- response: `{purity_wsi, purity_ta, heatmap_png_base64}`

Deliverables:
- `frontend/` app
- Instructions in README
