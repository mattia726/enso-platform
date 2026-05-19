# Product spec (MVP)

**User story**
A user can provide a slide embedding file (`.h5`) or a slide ID and receive:
- Global predicted tumor purity (0–1)
- Tumor-area purity proxy (e.g., 95th percentile over ROI predictions)
- Heatmap image of predicted purity across the slide

**Constraints**
- Must run on a single GPU VM (GCP L4) and be demoable via a web UI.
- Must be reproducible enough for an exportable technical transcript: tests, logs, clear code.

**Non-goals for MVP**
- Production auth, multi-tenant storage, complex WSI viewer
- Perfect calibration across all cancer types

