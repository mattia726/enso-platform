# Enso Purity Wedge Repository Guide

This repository is a monorepo for predicting tumor purity from precomputed
whole-slide-image (WSI) tile embeddings. The codebase is organized around four
main concerns:

1. Dataset linkage and API serving in `backend/`
2. Model training, evaluation, and artifact generation in `ml/`
3. Demo presentation in `frontend/`
4. Raw/processed assets, reports, specs, and reference material in `data/`,
   `docs/`, and `third_party/`

The tree below is intentionally schematic rather than exhaustive. It covers
every top-level folder and the principal files that define behavior. Large
generated collections are collapsed with patterns such as `interactive_<uuid>.html`
or `case_<n>_*.png`.

## End-to-End Flow

1. `data/raw/` holds TCGA metadata inputs such as the ABSOLUTE purity table and
   slide metadata export.
2. `backend/scripts/build_wedge_dataset.py` cleans those inputs, resolves GDC
   UUID drift, links slides to purity labels, and writes the wedge manifest in
   `data/processed/`.
3. `ml/enso_purity_mil/dataset.py` loads H5 embedding bags from precomputed
   Virchow features, grouped at aliquot level.
4. `ml/enso_purity_mil/model.py` defines the main Enso MIL model used for
   training and richer inference.
5. `ml/enso_purity_mil/train_cli.py` trains fold-based checkpoints into
   `ml/runs*/`.
6. `backend/src/enso_purity/api/inference.py` can load a trained checkpoint and
   expose prediction and heatmap endpoints.
7. `frontend/` renders the demo UI from generated gallery/statistics artifacts.

## Source of Truth by Concern

- Dataset build: `backend/scripts/build_wedge_dataset.py`
- Dataset linkage rules: `backend/src/enso_purity/data/`
- Main training model: `ml/enso_purity_mil/model.py`
- Training loop and losses: `ml/enso_purity_mil/training.py`
- H5 bag loading and cache strategy: `ml/enso_purity_mil/dataset.py`
- Heatmap scoring: `ml/enso_purity_mil/heatmap.py`
- Inference API: `backend/src/enso_purity/api/inference.py`
- Frontend app shell: `frontend/app/page.tsx`

## Model Architecture Map

There are two related model stacks in the repository:

### 1. Main training and inference architecture (`ml/enso_purity_mil/`)

The current training pipeline and richer inference path are centered on
`ml/enso_purity_mil/model.py`. That file defines the full Enso MIL model:

- `EnsoModelConfig`
  - Central hyperparameter object for feature dimensions, KDE bins, prototype
    count, dropout, uncertainty, and regularization behavior.
- `VirchowAdapter`
  - Converts 2560-dimensional Virchow tile embeddings into a bounded,
    lower-dimensional feature space.
  - Structure: `LayerNorm -> Linear -> GELU -> Dropout -> Linear -> LayerNorm -> sigmoid`
  - Purpose: adapt foundation-model embeddings into stable features suitable for
    distribution pooling.
- `MultiScaleDistributionPooling`
  - Applies KDE-style pooling across instances (tiles) for multiple sigma values.
  - Purpose: summarize the distribution of tile features, not only their mean.
- `PrototypeHistogram`
  - Learns soft feature-space prototypes and produces a histogram of prototype
    occupancy.
  - Purpose: capture global composition and co-occurrence structure in each bag.
- Optional moment features
  - Adds per-feature mean and standard deviation to the pooled representation.
- `EnsoMILModel`
  - Combines adapter, KDE pooling, prototype histogram, optional moments, a
    trunk MLP, and three output heads:
  - `mu_head`: purity mean prediction
  - `kappa_head`: uncertainty / concentration for beta-distribution outputs
  - `tumor_head`: auxiliary tumor-vs-normal prediction

Supporting files around that architecture:

- `ml/enso_purity_mil/dataset.py`
  - Defines what a bag is: tumor bags are grouped by aliquot, normals by slide.
  - Handles direct H5 loading, deterministic sampling, and local `.pt` caches.
  - This file is the core I/O boundary between raw embeddings and the model.
- `ml/enso_purity_mil/training.py`
  - Runs epochs and composes the actual optimization objective.
  - Primary tracked loss is L1 on purity, with optional auxiliary BCE, beta-NLL,
    prototype-entropy regularization, and consistency loss.
- `ml/enso_purity_mil/train_cli.py`
  - Orchestrates fold creation, loaders, optimizer/scheduler setup, checkpoint
    saving, and early stopping.
- `ml/enso_purity_mil/heatmap.py`
  - Reuses the trained model for tile-level scoring by building local
    neighborhoods around each tile and running batched inference.

### 2. Compact backend-local model (`backend/src/enso_purity/models/`)

The backend package also contains a simpler MIL stack:

- `backend/src/enso_purity/models/distribution_pooling.py`
  - Single-scale differentiable KDE pooling.
- `backend/src/enso_purity/models/mil_regressor.py`
  - Simpler projection MLP, single KDE pooling stage, and scalar sigmoid head.

This backend-local model is useful as a compact reference or lightweight
deployment target. However, the more complete inference path in
`backend/src/enso_purity/api/inference.py` currently imports the trained
`EnsoMILModel` from the `ml/` package, not the simpler backend-only regressor.

## EnsoMIL From a Machine Learning Perspective

This section focuses on the main model in `ml/enso_purity_mil/model.py`, not the
smaller backend-only regressor. From an ML perspective, `EnsoMILModel` is a
weakly supervised, bag-level multiple-instance learning (MIL) regressor over
precomputed foundation-model tile embeddings.

### 1. Learning Problem

The supervised signal is a scalar tumor purity target in `[0, 1]`, derived from
genomic ABSOLUTE labels and attached to a bag of tiles rather than to
individual tiles.

Formally, the model receives:

- a bag `X = {x_1, ..., x_N}` of tile embeddings for one slide-group / aliquot
- each tile embedding `x_i` has dimension `2560`
- a bag-level target `y in [0, 1]` representing tumor purity

This is a classic MIL setting because:

- supervision exists at bag level, not tile level
- the number of tiles per case is variable
- the tile order is meaningless, so the model must be permutation-invariant

The job of the network is therefore not "classify this tile", but:

1. transform each tile embedding into a more task-specific latent space
2. summarize the distribution of latent tile features across the whole bag
3. regress purity from that bag summary
4. optionally estimate uncertainty and an auxiliary tumor-vs-normal signal

### 2. Conceptual Version Lineage

The code currently checked into this repository corresponds to the more advanced
Enso MIL family. Conceptually, the progression is:

| Aspect | v1 Baseline | v2 Major Upgrade | Current checked-in stack |
|---|---|---|---|
| Architecture | Simpler DPF-style MIL regressor: linear adapter, single KDE, single purity head | Adapter upgraded to low-rank MLP, multi-scale KDE, prototype histogram, moments, multi-head outputs | Same model class as v2 in `model.py` |
| Main loss | Pure L1 / MAE purity regression | Composite objective with auxiliary regularization terms | Same composite objective shape, but with more conservative default weights |
| Optimizer | Standard AdamW style | Similar overall regime | AdamW with decay exclusions for norms/biases/1D params |
| Batch regime | No accumulation logic | No accumulation logic | Gradient-accumulation support added in `train_cli.py` |
| Validation | Sampled bags only | Sampled bags only | Deterministic and all-tiles validation supported; all-tiles is the default |
| Logging | Basic fixed logging | Similar | Log caps per epoch to avoid very noisy runs |

The most important practical reading is:

- `model.py` is architecturally "v2-style"
- `train_cli.py` and `training.py` behave like a stronger "v3-style" training protocol

In other words, the network itself is already the upgraded model, and the
current training defaults mostly improve optimization, regularization, and
validation behavior rather than changing the network topology.

### 3. Input Representation and MIL Setup

The data pipeline in `ml/enso_purity_mil/dataset.py` defines what a bag is:

- tumor bags are grouped by `aliquot_barcode`
- normal bags are single-slide bags
- training usually samples `4096` tile embeddings per bag
- validation can either sub-sample or use all available tiles

This matters for ML behavior:

- grouping at aliquot level lets the model learn a purity estimate that is less
  tied to a single slide view and more tied to the underlying biospecimen
- stochastic tile sampling acts as data augmentation across epochs
- all-tiles validation reduces variance in model selection by making evaluation
  closer to the full latent evidence available for each case

### 4. Architectural Decomposition

At a high level, the model computes:

1. instance adaptation
2. distributional bag pooling
3. bag-level regression and uncertainty estimation

With default dimensions, the flow is:

- input bag: `(B, N, 2560)`
- adapted features: `(B, N, 128)`
- multi-scale KDE tensor: `(B, 128 * 3, 21)`
- flattened KDE features: `(B, 8064)`
- prototype histogram: `(B, 64)`
- optional moments: `(B, 256)` from mean and std
- concatenated bag representation: `(B, 8384)`
- trunk embedding: `(B, 128)`
- outputs: purity mean, concentration, tumor probability

### 5. Adapter: Why the Model Starts With a Learnable Bottleneck

The `VirchowAdapter` is:

```text
LayerNorm(2560)
-> Linear(2560, 256)
-> GELU
-> Dropout(p_adapter)
-> Linear(256, 128)
-> LayerNorm(128)
-> Sigmoid(z / tau)
```

From an ML perspective, this module serves several purposes:

- domain adaptation:
  Virchow embeddings are general pathology features, not purity-specific
  features; the adapter learns a task-focused subspace
- compression:
  reducing `2560 -> 128` lowers the cost of downstream distribution pooling
- stabilization:
  LayerNorm before and after the MLP reduces scale sensitivity across slides and
  across embedding channels
- bounded support:
  the final sigmoid constrains features to `[0, 1]`, which makes KDE pooling on
  fixed bins mathematically natural
- temperature control:
  `tau` controls how sharply or softly features are pushed into that bounded
  interval; because it can be learnable, the model can tune feature saturation

This is a meaningful upgrade over a simple linear adapter because the MLP can
learn nonlinear reparameterizations of the original foundation-model space.

### 6. Multi-Scale KDE Pooling: The Core MIL Inductive Bias

The most distinctive part of the model is `MultiScaleDistributionPooling`.
Instead of averaging or attention-weighting tile embeddings into one vector, the
model estimates a per-feature distribution over the bag.

Mechanically:

- for each latent feature dimension `j`
- for each sigma in `(0.02, 0.05, 0.10)` by default
- for each of `21` equally spaced bins in `[0, 1]`
- compute Gaussian-kernel responses of all tiles to that bin
- average across tiles

This yields a smooth histogram for each feature channel at multiple bandwidths.

ML intuition:

- mean pooling only captures the first moment
- attention pooling emphasizes salient instances but can discard the global
  shape of the bag distribution
- KDE pooling captures whether a feature is unimodal, bimodal, broad, sharp,
  concentrated near zero, concentrated near one, and so on
- using multiple sigmas lets the model observe bag structure at fine, medium,
  and coarse resolution

Why this is useful for tumor purity:

- purity is not purely "does one highly malignant tile exist?"
- it is closer to "what fraction and distribution of tissue phenotypes exist in
  the specimen?"
- distributional pooling matches that target because it summarizes prevalence
  and composition, not only presence

### 7. Prototype Histogram: Recovering Joint Structure

One limitation of marginal KDE pooling is that it treats each feature dimension
independently. That can miss interactions between features.

The `PrototypeHistogram` branch compensates for this:

- learn `K = 64` prototypes in the adapted feature space
- normalize both tiles and prototypes
- compute soft assignments of each tile to prototypes
- average assignments across tiles

Mathematically, let:

- `z_i in R^J` be the adapted feature vector for tile `i`, with `J = 128`
- `p_k in R^J` be prototype `k`, with `k = 1, ..., K`
- `N` be the number of tiles in the bag
- `K` be the number of prototypes, default `64`
- `T` be the learned temperature

The implementation first L2-normalizes both the tile features and the
prototypes:

```text
z_i_hat = z_i / ||z_i||
p_k_hat = p_k / ||p_k||
```

That turns their dot product into a cosine similarity:

```text
s_{ik} = z_i_hat^T p_k_hat
```

These similarities are then divided by the temperature and pushed through a
softmax across prototypes:

```text
a_{ik} = exp(s_{ik} / T) / sum_{l=1..K} exp(s_{il} / T)
```

So for each tile `i`, the vector `(a_{i1}, ..., a_{iK})` is a probability
distribution over prototypes:

- if a tile is very close to one prototype, its assignment becomes peaked
- if it is ambiguous, probability mass is spread across multiple prototypes

The final bag representation from this branch is the mean assignment over all
tiles:

```text
h_k = (1 / N) * sum_{i=1..N} a_{ik}
```

for each prototype `k`. The resulting vector

```text
h = (h_1, ..., h_K)
```

is a soft histogram over learned tissue archetypes. Because each tile assignment
vector sums to `1`, the averaged histogram also sums to `1`:

```text
sum_{k=1..K} h_k = 1
```

This is why it is useful to think of the branch as a mixture estimator. It does
not merely say whether a prototype appears; it estimates the proportion of bag
mass explained by each prototype.

The role of the temperature is important:

- small `T` makes assignments sharper, closer to hard clustering
- larger `T` makes assignments softer and more distributed

In code, the temperature is learnable and clamped to a stable range, so the
model can decide how discrete or diffuse prototype membership should be during
training.

This branch is different from classical clustering in two important ways:

- the prototypes are learned end-to-end for the prediction task, not fit in a
  separate preprocessing step
- assignments are soft and differentiable, so gradients from the purity loss can
  reshape the prototype geometry

The result is a bag-level histogram over learned tissue archetypes.

ML interpretation:

- KDE pooling says "what is the marginal distribution of each latent feature?"
- the prototype branch says "what mixture of latent tissue phenotypes is in the bag?"
- together they approximate both marginal and compositional structure

In the actual model assembly, this histogram is concatenated with the flattened
multi-scale KDE representation, and then the optional mean/std moments are added
before the combined vector is passed into the bag-level trunk MLP. So the
prototype branch is a parallel bag-summary branch, not a separate prediction
head.

This is especially useful in histopathology, where purity is influenced by the
mixture of tumor, stroma, inflammation, necrosis, and other morphologic states.

### 8. Optional Moments: Cheap First- and Second-Order Statistics

When `use_moments=True`, the model concatenates:

- the per-feature mean across tiles
- the per-feature standard deviation across tiles

These moment features are redundant in the strict mathematical sense because
they are related to the information already present in the KDE histograms, but
in practice they provide:

- a low-friction shortcut for optimization
- easier access to coarse global statistics
- complementary signals when finite-bin KDE resolution is imperfect

### 9. Bag-Level Trunk and Output Heads

After concatenating KDE features, prototype histogram, and moments, the model
uses a trunk MLP:

```text
Dropout(p_head)
-> Linear(8384, 512)
-> GELU
-> Dropout(p_head)
-> Linear(512, 128)
-> GELU
```

This trunk converts the large engineered bag representation into a compact
128-dimensional bag embedding. Three heads are then applied:

- `mu_head`
  - predicts the expected purity mean in `[0, 1]` after sigmoid
- `kappa_head`
  - predicts a positive concentration parameter via softplus
- `tumor_head`
  - predicts whether the bag is tumor or normal

The beta-distribution parameters are then formed as:

- `alpha = mu * kappa`
- `beta = (1 - mu) * kappa`

Interpretation:

- `mu` is the point estimate of purity
- `kappa` controls confidence / sharpness
- high `kappa` means a narrow beta distribution around the mean
- low `kappa` means a broader, more uncertain distribution

This is more expressive than a single regression head because it allows the
model to represent both estimate and confidence, even if the uncertainty term is
not always emphasized during training.

### 10. Training Objective

The training logic lives in `ml/enso_purity_mil/training.py`. The main tracked
objective is still L1 on purity, but the full optimization target can include
multiple terms:

```text
L =
  L1(mu, y)
  + lambda_bce * BCE(tumor_prob, is_tumor)
  + lambda_beta * BetaNLL(y ; alpha, beta)
  + lambda_proto * (-entropy(proto_hist))
  + lambda_cons * L1(mu_1, stopgrad(mu_2))
```

Where each term means:

- `L1(mu, y)`
  - primary regression objective; robust and directly aligned with purity error
- `BCE(tumor_prob, is_tumor)`
  - auxiliary classification loss helping the representation distinguish tumor
    from normal tissue
- `BetaNLL`
  - trains the uncertainty head to match the target as a beta-distributed random
    variable rather than only a point estimate
- prototype entropy penalty
  - discourages degenerate prototype collapse by pushing assignments away from
    low-entropy solutions
- consistency term
  - compares two stochastic forward passes and encourages prediction stability
    under internal stochasticity

Current default weights in `train_cli.py` are:

- `aux_bce_weight = 0.05`
- `beta_nll_weight = 0.0`
- `proto_entropy_weight = 0.001`
- `consistency_weight = 0.05`

Important implication:

- the uncertainty head is still present in the network
- but the explicit beta-likelihood term is off by default in the current CLI
- so uncertainty is available structurally, though it is not being strongly
  supervised unless that weight is turned back on

### 11. Regularization Strategy

The model uses several forms of regularization, and they operate at different
levels:

- adapter dropout:
  regularizes the instance-level feature adapter
- head dropout:
  regularizes the bag-level trunk MLP
- instance dropout:
  randomly keeps only a subset of tiles during training, acting as bag-level
  data augmentation
- feature noise:
  small Gaussian noise added to adapted features during training
- consistency loss:
  encourages stable predictions under stochastic perturbations
- entropy regularization:
  prevents prototype usage collapse

In current `train_cli.py` defaults:

- `adapter_dropout = 0.20`
- `head_dropout = 0.50`
- `instance_dropout = 0.20`
- `feature_noise_std = 0.01`

These are relatively strong regularizers, especially the bag-level head
dropout, and they match the idea of a reinforced "v3-style" training protocol.

### 12. Optimization Protocol

The optimizer in `train_cli.py` is AdamW, but not in the naive "apply weight
decay to everything" form. The helper
`build_adamw_with_decay_exclusions()` separates:

- true weight tensors, which receive weight decay
- bias terms, norm parameters, and 1D parameters, which do not

This is a more modern setup because decaying normalization scales and biases is
usually not desirable.

Other important optimization details:

- learning rate default: `1e-4`
- weight decay default: `3e-3`
- scheduler: `ReduceLROnPlateau`
- early stopping based on validation loss

Gradient accumulation is supported through `--effective-batch-size`, but the
current defaults are:

- `batch_size = 128`
- `effective_batch_size = 32`

Because accumulation steps are clamped to at least 1, the default configuration
does not actually accumulate gradients. The support is there for memory-limited
setups, but it only changes behavior when the requested effective batch exceeds
the micro-batch actually used.

### 13. Validation and Why It Matters

The current training script makes validation more stable than a purely sampled
MIL regime:

- `val_use_all_tiles = True` by default
- `deterministic_val = True` by default
- validation batch size becomes `1` in all-tiles mode

Why this matters:

- sampled validation bags can inject a lot of noise into model selection
- if checkpointing depends on stochastic subsets of tiles, the "best model" can
  partly reflect sampling luck
- all-tiles validation makes the monitored metric closer to the full evidence
  available for each case

This is an important protocol improvement even though it does not change the
model architecture itself.

### 14. What the Model Is Actually Learning

From the ML viewpoint, the model is learning three things simultaneously:

- a task-adapted latent basis for pathology embeddings
- a distributional summary of tissue composition across a specimen
- a mapping from that composition to genomic purity

That is why the architecture is more elaborate than a simple "average the tiles
and run an MLP" baseline. Tumor purity is fundamentally a compositional
property. The model encodes that bias explicitly:

- KDE branch:
  how latent feature values are distributed
- prototype branch:
  what latent tissue archetypes are present and in what proportions
- moments:
  coarse global summary statistics
- auxiliary tumor head:
  extra supervision for separating tumor-bearing from normal bags

### 15. Why Heatmaps Are Possible Despite Bag-Level Supervision

The training target is bag-level, but `ml/enso_purity_mil/heatmap.py` still
produces tile-level scores by building a local neighborhood around each tile and
feeding that neighborhood through the same bag-level model.

In practice:

- for each tile, gather its `K = 81` nearest spatial neighbors
- treat that neighborhood as a small bag
- run the model on that local bag
- assign the output purity to the center tile location

So the heatmap is not a direct per-tile classifier. It is a localized bag-level
inference procedure. That distinction is important:

- the model is trained to score tissue neighborhoods, not isolated tiles
- spatial purity maps therefore reflect local context, not pointwise tile labels

### 16. Practical Summary

If you want the shortest accurate description of the current Enso MIL stack, it
is this:

- input:
  bags of Virchow tile embeddings
- task:
  bag-level purity regression with auxiliary tumor/normal discrimination
- core idea:
  use distribution pooling rather than simple averaging so purity is modeled as
  a tissue-composition problem
- architecture:
  nonlinear adapter + multi-scale KDE + prototype histogram + moments + multi-head MLP
- current regime:
  v2-style model structure with v3-style training and validation behavior

## Folder Roles in Plain Language

- `backend/` is where slide metadata becomes a clean manifest and where HTTP
  endpoints live.
- `ml/` is where the real training, evaluation, heatmap generation, and demo
  artifact production happen.
- `frontend/` is mostly a presentation layer over artifacts produced elsewhere.
- `data/` is the shared storage contract between scripts, training, and UI.
- `docs/` explains why the system is shaped this way.
- `third_party/` is reference material, not first-party production code.

## Quick Entry Points

Backend API:

```bash
source backend/.venv/bin/activate
cd backend
uvicorn enso_purity.api.main:app --host 0.0.0.0 --port 8000 --reload
```

Backend tests:

```bash
source backend/.venv/bin/activate
cd backend && python -m pytest -q
```

ML tests:

```bash
cd ml && python -m pytest tests/ -q
```

Canonical dataset rebuild:

```bash
source backend/.venv/bin/activate
python backend/scripts/build_wedge_dataset.py
```


## Repository Schema

```text
enso-purity-wedge-1/
|-- .cursor/                                      # Editor/agent workspace metadata
|-- .env.example                                  # Example environment variables for local setup
|-- AGENTS.md                                     # Repo-specific operating instructions and caveats
|-- README.md                                     # Root quickstart and project summary
|-- repo structure.md                             # Detailed structural guide for the repository
|
|-- backend/                                      # FastAPI package + data-linkage code
|   |-- pyproject.toml                            # Editable package definition for `enso_purity`
|   |-- requirements.txt                          # Backend/runtime Python dependencies
|   |
|   |-- scripts/                                  # One-off and canonical data-prep scripts
|   |   |-- build_wedge_dataset.py                # Canonical wedge manifest builder used by the current project
|   |   |-- build_purity_dataset.py               # Comparison pipeline: barcode linkage vs GDC linkage
|   |   |-- build_dataset.py                      # Older naive linkage/bootstrap script; not the main dataset builder
|   |   |-- download_slide_thumbnails.py          # Fetches lightweight GDC thumbnails for slide comparisons
|   |   `-- fetch_gdc_metadata.sh                 # Shell wrapper for refreshing slide metadata via GDC
|   |
|   |-- src/
|   |   `-- enso_purity/
|   |       |-- __init__.py                       # Package marker
|   |       |
|   |       |-- api/                              # HTTP-serving layer
|   |       |   |-- main.py                       # Minimal FastAPI MVP stub; validates H5 and returns placeholder outputs
|   |       |   `-- inference.py                  # Realer inference path; loads ML checkpoint and exposes purity/heatmap endpoints
|   |       |
|   |       |-- data/                             # TCGA parsing and slide-to-purity linkage logic
|   |       |   |-- tcga_barcode.py               # Barcode parsers for slide IDs and aliquot IDs
|   |       |   |-- slide_purity_matching.py      # Barcode-based linkage at sample-vial and portion levels
|   |       |   |-- gdc_linkage.py                # GDC biospecimen API linkage; resolves true slide/aliquot relationships
|   |       |   |-- naive_linkage.py              # Simplified match-rate utilities for quick sanity checks
|   |       |   `-- linkage_compare.py            # Plot/report helpers comparing linkage strategies
|   |       |
|   |       |-- models/                           # Lightweight model components kept with the backend package
|   |       |   |-- distribution_pooling.py       # Single-scale differentiable KDE distribution pooling
|   |       |   `-- mil_regressor.py              # Compact MIL regressor: projection -> KDE pooling -> regression head
|   |       |
|   |       `-- embeddings/
|   |           `-- virchow/                      # Embedding-generation and bucket-inspection utilities
|   |               |-- embedder_smartfilter_fp32.py   # WSI -> Virchow H5 embedding pipeline with smart tile filtering
|   |               |-- tcga_bucket_physical_scan.py   # Public bucket scan helper for SVS inventorying
|   |               `-- tcga_wsi_metadata_scan.py      # Metadata extraction helper for TCGA WSIs
|   |
|   `-- tests/                                    # Backend tests split by API, data, and model units
|       |-- api/
|       |   |-- test_health.py                    # Health endpoint test
|       |   `-- test_predict_stub.py              # Stub prediction endpoint contract test
|       |-- data/
|       |   |-- test_gdc_linkage.py               # GDC linkage behavior and payload parsing tests
|       |   `-- test_slide_purity_matching.py     # Barcode-based linkage tests
|       |-- models/
|       |   |-- test_kde_pooling.py               # Pooling layer shape/behavior tests
|       |   `-- test_mil_regressor.py             # Compact backend MIL model tests
|       `-- test_barcode_parsing.py               # TCGA barcode parser tests
|
|-- data/                                         # Project data area: raw inputs, cleaned manifests, reports
|   |-- raw/                                      # Raw upstream inputs
|   |   |-- TCGA_mastercalls.abs_tables_JSedit.fixed.txt   # ABSOLUTE genomic purity table
|   |   |-- slides_metadata_report(1).xlsx        # Slide metadata export used as the wedge input
|   |   `-- .gitkeep                              # Keeps empty dir tracked when raw files are absent
|   |-- interim/                                  # Scratch space for intermediate artifacts
|   |-- processed/                                # Canonical cleaned datasets consumed by ML and UI tooling
|   |   |-- wedge_mvp_dataset.xlsx                # Canonical TCGA frozen-section manifest used by the current retrain flow
|   |   |-- wedge_mvp_dataset(1).xlsx             # Historical tracked TCGA manifest variant kept for reference
|   |   |-- cptac_slides_ngs_purity_final.csv     # Canonical processed CPTAC tumour linkage table
|   |   |-- cptac_master_normals.csv              # Canonical processed CPTAC normal-slide table
|   |   `-- .gitkeep
|   |-- reports/                                  # Generated plots and JSON summary reports
|   |   `-- .gitkeep
|   `-- exclude_markers.txt                       # Manual exclusion list for demo/gallery slides with pen markers
|
|-- docs/                                         # Product, ML, dataset, serving, and UI specifications
|   |-- spec.md                                   # High-level MVP product spec
|   |-- dataset_linkage_spec.md                   # Detailed slide <-> purity linkage rules and deliverables
|   |-- ml_training_spec.md                       # Training goals, inputs, architecture, and evaluation spec
|   |-- serving_spec.md                           # Backend/frontend serving and deployment outline
|   |-- ui_spec.md                                # Frontend dashboard specification
|   |-- decisions.md                              # Decision log for major architectural choices
|   |-- vm_generate_artifacts.md                  # How to generate frontend/demo artifacts on the training VM
|   |-- model_runs/                               # Lightweight locked-run summaries for canonical model campaigns
|   |   `-- tcga_cptac_retrain_20260411T023052Z_retry1.md   # Final TCGA FS + CPTAC DX retrain registry
|   `-- plan_layout_hero_viewer_scatter.md        # UI implementation plan and what changed
|
|-- frontend/                                     # User-facing demo application and published artifacts
|   |-- README.md                                 # Frontend-specific run instructions and artifact workflow
|   |-- package.json                              # Next.js app dependencies and scripts
|   |-- next.config.mjs                           # Next.js configuration
|   |-- tailwind.config.ts                        # Tailwind configuration
|   |-- tsconfig.json                             # TypeScript configuration
|   |-- app.py                                    # Legacy Streamlit demo; separate from the current Next.js app
|   |
|   |-- app/                                      # Next.js app router entry points
|   |   |-- layout.tsx                            # Global page shell
|   |   |-- page.tsx                              # Main single-page demo with Case Explorer and Performance tabs
|   |   `-- globals.css                           # Global theme tokens and app-wide styling
|   |
|   |-- components/                               # Main UI building blocks
|   |   |-- CaseExplorer.tsx                      # Primary case browser / viewer shell
|   |   |-- PerformanceTab.tsx                    # Model-vs-pathologist performance dashboard
|   |   |-- ScatterChart.tsx                      # Interactive scatter plot for genomic vs predicted purity
|   |   |-- ThemeProvider.tsx                     # Theme state container
|   |   |-- ThemeToggle.tsx                       # Theme switcher control
|   |   `-- HeroTypewriter.tsx                    # Hero/branding animation component
|   |
|   |-- data/
|   |   `-- tcga_display_names.ts                 # TCGA short-code -> display-name mapping used by the UI
|   |
|   |-- gallery/                                  # Staging area for generated interactive viewers
|   |   |-- gallery_summary.csv                   # Metadata table used by the Case Explorer
|   |   `-- interactive_<uuid>.html               # Standalone HTML heatmap viewers generated by ML tooling
|   |
|   `-- public/                                   # Published static assets served by Next.js
|       |-- data/                                 # Statistics JSON/PNG used by the Performance tab
|       |-- gallery/                              # Published copy of the interactive viewer HTML files
|       |-- cases/                                # Static base-image and mask assets for case overlays
|       `-- enso-logo*.png                        # Brand assets used by the site header
|
|-- infra/                                        # Deployment/infrastructure placeholder; currently not populated
|
|-- logs/                                         # Ad hoc evaluation outputs and local experiment logs
|   |-- v3_allfolds_alltiles_predictions_191512.csv   # Historical TCGA tumour fold/prediction baseline used for direct-comparison audits
|   `-- test_cli_all_tiles.py                     # Local experimental evaluation helper
|
|-- ml/                                           # Training, evaluation, heatmaps, and demo artifact generation
|   |-- pyproject.toml                            # Editable package definition for `enso_purity_mil`
|   |-- training_v3_*.log*                        # Captured experiment logs
|   |
|   |-- enso_purity_mil/                          # Main ML package
|   |   |-- __init__.py                           # Package marker
|   |   |-- model.py                              # Main Enso MIL architecture used for training/checkpoints
|   |   |-- dataset.py                            # H5 bag loader with cache support and smart partial reads
|   |   |-- training.py                           # Epoch runner, regularized losses, and early stopping
|   |   |-- folds.py                              # Patient-level stratified fold generation
|   |   |-- train_cli.py                          # Main training entry point for fold-based experiments
|   |   |-- test_cli.py                           # Hold-out fold evaluation CLI with inference bagging
|   |   |-- heatmap.py                            # Tile-level scoring by local neighborhood inference
|   |   |-- heatmap_cli.py                        # CLI for writing heatmap PNG/NPZ outputs
|   |   |-- build_cache.py                        # Parallel bag-cache builder for local SSD / gcsfuse workflows
|   |   |-- manifest_io.py                        # Small loader for manifest tables across xlsx/csv/tsv formats
|   |   |-- build_tcga_cptac_retrain_manifest.py  # Builds the combined TCGA+CPTAC slide/bag manifests with preserved fold semantics
|   |   |-- build_union_h5_namespace.py           # Creates a unified H5 symlink namespace across TCGA and CPTAC embeddings
|   |   |-- plot_tcga_cptac_cv_test_eval.py       # Summarizes merged retrain fold outputs into composite evaluation artifacts
|   |   |-- plot_tcga_retrain_vs_original_cli.py  # Canonical original-v3 vs retrain comparison and audit plotting CLI
|   |   |-- predictions_utils.py                  # Helpers for working from prediction CSVs instead of live model runs
|   |   |-- statistical_tests.py                  # Global MIL-vs-PTN statistical comparison and plots
|   |   |-- per_cancer_statistical_tests.py       # Per-cancer breakdown for frontend tables
|   |   |-- build_demo_gallery.py                 # Curates gallery cases and launches HTML viewer generation
|   |   |-- interactive_viewer.py                 # Produces standalone thumbnail + purity-overlay HTML viewers
|   |   |-- export_static_cases.py                # Writes static JPG/PNG case assets for the frontend
|   |   `-- batch_demo_pack.py                    # Small demo bundle builder with selected aliquots and heatmaps
|   |
|   |-- tests/                                    # ML unit tests (CPU-only)
|   |   |-- test_model.py                         # Main architecture tests
|   |   |-- test_dataset.py                       # Dataset and sampling tests
|   |   |-- test_training.py                      # Training utility tests
|   |   |-- test_heatmap.py                       # Heatmap and neighborhood logic tests
|   |   |-- test_tcga_cptac_retrain.py            # Combined-manifest and preserved-fold semantics tests
|   |   `-- test_plot_tcga_retrain_vs_original_cli.py   # Audit/comparison plotting tests for retrain vs original-v3
|   |
|   |-- runs/                                     # Checkpoints and evaluation outputs
|   |-- runs_v2/                                  # Older experiment directory
|   |-- runs_v3/                                  # Newer experiment directory
|   `-- runs_v3_folds1to4_20260307_144740/        # Concrete multi-fold training run with per-fold histories
|
|-- prompts/                                      # Agent prompts used to scaffold individual project areas
|   |-- 00_repo_scaffold.md                       # Repo scaffold prompt
|   |-- 10_data_agent_gdc_linkage.md              # Data/linkage prompt
|   |-- 20_model_agent_mil_regressor.md           # Model/training prompt
|   |-- 30_api_agent_fastapi_heatmap.md           # API prompt
|   |-- 40_frontend_agent_next_dashboard.md       # Frontend prompt
|   `-- 50_infra_agent_gcp_deploy.md              # Infra/deployment prompt
|
|-- scripts/                                      # Top-level automation helpers spanning multiple subprojects
|   |-- run_on_vm_generate_all.sh                 # VM workflow to build stats/gallery artifacts end to end
|   |-- run_tcga_cptac_retrain_fold.sh            # Parameterized helper to launch one TCGA+CPTAC retrain fold
|   |-- queue_tcga_cptac_fold_test.sh             # Parameterized helper to run hold-out fold evaluation after training
|   |-- tcga_cptac_test_eval.py                   # Fold test evaluator that respects preassigned tumour folds
|   `-- update_frontend_data_from_predictions.py  # Refreshes frontend/public/data from a predictions CSV
|
`-- third_party/                                  # External reference code, assets, and layout inspirations
    |-- enso/                                     # Branding/reference image assets
    |-- cool_website_only_for_layout_rules/       # Layout inspiration HTML
    |-- nice_tables_website/                      # Table-layout inspiration HTML
    `-- oner/                                     # Upstream SRTPMs reference materials
        |-- README.md                             # Notes on the imported upstream baseline
        |-- UPSTREAM.md                           # Provenance documentation
        `-- upstream/SRTPMs-1.0.0/                # Imported reference implementation from Oner et al.
```
