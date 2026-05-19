"""Spatial heatmap inference: tile-level purity prediction using local KDE neighbourhoods.

Uses ``scipy.spatial.cKDTree`` to find K=81 nearest neighbours for each tile,
then runs batched inference through ``Adapter → KDE → Head`` to produce a
per-tile purity score.  Memory-safe: never allocates ``(M, K, dim)`` all at once.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree

from enso_purity_mil.model import EnsoMILModel


def build_neighborhood_indices(
    coords: np.ndarray,
    k: int = 81,
) -> np.ndarray:
    """Return (M, K) array of neighbour indices for every tile.

    If a slide has fewer than K tiles, neighbours are padded by repeating
    the closest available tiles.
    """
    M = len(coords)
    actual_k = min(k, M)
    tree = cKDTree(coords.astype(np.float64))
    _, indices = tree.query(coords, k=actual_k)
    if actual_k == 1:
        indices = indices.reshape(-1, 1)

    if actual_k < k:
        pad_needed = k - actual_k
        pad = np.tile(indices[:, :1], (1, pad_needed))
        indices = np.concatenate([indices, pad], axis=1)

    return indices  # (M, K)


@torch.no_grad()
def predict_tile_scores(
    model: EnsoMILModel,
    h5_path: str | Path,
    *,
    k: int = 81,
    batch_size: int = 1024,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Predict per-tile purity scores.

    Returns ``(scores, coords)`` where ``scores`` is shape ``(M,)``
    and ``coords`` is shape ``(M, 2)``.
    """
    model.eval()
    model = model.to(device)

    with h5py.File(h5_path, "r") as f:
        all_feats = f["features"][:]   # (M, D) — loaded once
        coords = f["coords"][:]        # (M, 2)

    M = len(all_feats)
    nb_idx = build_neighborhood_indices(coords, k=k)  # (M, K)
    scores = np.empty(M, dtype=np.float32)

    for start in range(0, M, batch_size):
        end = min(start + batch_size, M)
        batch_idx = nb_idx[start:end]          # (bs, K)
        batch_feats = all_feats[batch_idx]     # (bs, K, D) — sliced dynamically
        batch_t = torch.from_numpy(batch_feats).to(device)
        preds = model(batch_t).squeeze(-1)     # (bs,)
        preds = torch.clamp(preds, min=0.0, max=1.0)
        scores[start:end] = preds.cpu().numpy()

    return scores, coords
