# MIL regressor training spec

## Goal

Train a Multiple Instance Learning (MIL) model that predicts tumor purity (scalar in **[0, 1]**) from **precomputed Virchow v1 tile embeddings**.

## Inputs

- Precomputed embeddings stored as H5 files per slide (produced by your Virchow embedder).
- A dataset manifest produced by the linkage step:
  - slide → purity label
  - patient/case ID for leak-free splitting

## Architecture (high level)

- Input: bag of tile embeddings (D-dim)
- Projection head: MLP D → 128
- Distribution pooling over instances (mean/std or learned histogram pooling)
- Regressor head: MLP → scalar, output constrained to [0, 1] (e.g., sigmoid)

## Evaluation

- Patient-level split (no leakage)
- Metrics: MAE primary, plus correlation
- Optional: k-fold cross-validation if time permits

## Deliverables

- Training entrypoint (CLI)
- Configurable paths (GCS/local)
- Smoke test run
- Saved weights + a minimal model card (metrics + data split)
