# Frontend UI spec (dashboard)

**Layout**
- Dark-mode medical dashboard
- Left: upload / input controls
- Right: results panel (metrics + heatmap)

**Inputs**
- Upload `.h5` file (embedding)
- Or paste `file_id` / `slide_id` (optional)

**States**
- Idle
- Uploading / Running inference
- Success (show results)
- Error (display actionable message)

**Outputs**
- Global Purity (median over ROIs)
- Tumor Area Purity (p95 over ROIs)
- Heatmap image

