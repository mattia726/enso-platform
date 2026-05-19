# Prompt: MODEL agent — distribution pooling MIL regressor (TDD)

Goal:
Adapt the MIL architecture (distribution pooling) to **precomputed Virchow embeddings** stored in `.h5`.

Requirements:
- Input: tile embeddings `features` shape [N, D] from `.h5`
- Sample a bag of N=200 tiles per step (configurable)
- Projection head: MLP D -> 128 (with dropout)
- Distribution pooling layer (KDE bins) to bag representation
- Regressor MLP -> scalar purity in [0,1] (sigmoid output)
- Loss: MAE (L1) or Huber (configurable)
- Patient-level splitting
- Save checkpoints + metrics JSON

TDD:
- tests for pooling output shape and numerical stability
- tests for model forward pass output range [0,1]
- a smoke training test on tiny synthetic data (CPU) that runs fast

Deliverables:
- `enso_purity/models/*`
- `enso_purity/train/train.py`
- `scripts/train_smoke.py`
