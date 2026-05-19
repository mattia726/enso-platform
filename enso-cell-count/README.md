# Enso Wedge MVP — Purity from WSI Embeddings

This repo is a **monorepo MVP** that builds a production-grade “wedge” demo:
- **Dataset linkage**: TCGA slide → portion → sequencing aliquot (ABSOLUTE purity label), with tests + mismatch report vs naive barcode matching.
- **MIL regressor**: precomputed WSI tile embeddings → distribution pooling → scalar purity in [0,1].
- **FastAPI** backend: `/predict_purity` returns global purity + ROI heatmap.
- **Next.js** frontend (generated separately): upload `.h5` embeddings / choose slide ID → see metrics + heatmap.

## Repository layout

- `backend/`: FastAPI inference service and dataset-linkage scripts.
- `ml/`: MIL training, evaluation, fold handling, cache building, and retrain utilities.
- `data/processed/`: tracked canonical manifests used by the training pipeline, including TCGA frozen-section linkage and CPTAC processed tumour/normal inputs.
- `docs/`: specs, decisions, and lightweight run registries for locked model builds.

## Quickstart (backend)

Venv and runs use the **project disk** (e.g. D:) so Python on C: is only the base interpreter; packages and cache live on the project drive.

```bash
cd backend
python -m venv .venv
# Windows:  .\.venv\Scripts\Activate.ps1
# Linux/macOS:  source .venv/bin/activate
pip install -r requirements.txt
pytest -q
```

## Repo conventions

- All code must be **test-driven**: add/adjust tests first, then implement.
- All scripts must be runnable with `python -m ...` and accept `--help`.
- No “live” API calls inside unit tests. Integration tests are opt-in.

See:
- `docs/` for specs and decision log
- `docs/model_runs/` for concise locked-run summaries such as the TCGA+CPTAC retrain
- `.cursor/rules/` and `.cursor/commands/` for Cursor agent behavior
- `prompts/` for copy-paste prompts (Cursor)
