"""Inference utilities for EnsoCellularity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
import torch

from enso_cellularity.dataset import (
    build_neighbor_indices,
    metadata_from_h5_attrs,
    read_h5_features_by_index,
)
from enso_cellularity.metrics import ordinal_count_bins, roi_count_summary
from enso_cellularity.model import EnsoCellularityConfig, EnsoCellularityModel


def load_cellularity_model(
    checkpoint_path: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[EnsoCellularityModel, dict[str, Any]]:
    """Load an EnsoCellularity checkpoint for inference."""

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("model_config", {})
    cfg = EnsoCellularityConfig(**cfg_dict)
    model = EnsoCellularityModel(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt


def predict_h5(
    model: EnsoCellularityModel,
    h5_path: Path | str,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 8192,
) -> pd.DataFrame:
    """Predict tile-level nuclei counts for one embedding H5 file."""

    h5_path = Path(h5_path)
    with h5py.File(h5_path, "r") as h5:
        coords = h5["coords"][:]
        coords_level0 = h5["coords_level0"][:] if "coords_level0" in h5 else np.zeros_like(coords)
        n_tiles = int(h5["features"].shape[0])
        metadata, exposure = metadata_from_h5_attrs(h5.attrs, n_tiles)

    if len(coords) != n_tiles:
        raise ValueError(f"coords/features length mismatch in {h5_path}: {len(coords)} vs {n_tiles}")

    embedding_index = np.arange(n_tiles, dtype=np.int64)
    neighbor_idx, valid9 = build_neighbor_indices(
        coords[:, 0],
        coords[:, 1],
        embedding_index,
    )

    chunks: list[pd.DataFrame] = []
    with torch.no_grad():
        for start in range(0, n_tiles, batch_size):
            end = min(start + batch_size, n_tiles)
            x9_np = read_h5_features_by_index(h5_path, neighbor_idx[start:end])
            x9 = torch.from_numpy(x9_np).to(device)
            valid = torch.from_numpy(valid9[start:end]).to(device)
            meta = torch.from_numpy(metadata[start:end]).to(device)
            exp = torch.from_numpy(exposure[start:end]).to(device)
            out = model.forward_outputs(x9, valid, meta, exp)
            quantiles = out["quantiles"].detach().cpu().numpy()
            quality_prob = torch.softmax(out["quality_logits"], dim=1).detach().cpu().numpy()
            pred_bin = ordinal_count_bins(out["ordinal_logits"]).detach().cpu().numpy()
            frame = pd.DataFrame(
                {
                    "embedding_index": embedding_index[start:end].astype(np.int64),
                    "tile_y": coords[start:end, 0].astype(np.int32),
                    "tile_x": coords[start:end, 1].astype(np.int32),
                    "tile_x_level0": coords_level0[start:end, 0].astype(np.int32),
                    "tile_y_level0": coords_level0[start:end, 1].astype(np.int32),
                    "pred_nuclei_count": out["mu"].detach().cpu().numpy().reshape(-1),
                    "pred_density_per_mm2": out["density_per_mm2"].detach().cpu().numpy().reshape(-1),
                    "pred_alpha": out["alpha"].detach().cpu().numpy().reshape(-1),
                    "pred_q05": quantiles[:, 0],
                    "pred_q50": quantiles[:, 1],
                    "pred_q95": quantiles[:, 2],
                    "pred_count_bin": pred_bin.astype(np.int16),
                    "quality_class": np.argmax(quality_prob, axis=1).astype(np.int16),
                    "quality_good_prob": quality_prob[:, 0],
                }
            )
            for j in range(1, quality_prob.shape[1]):
                frame[f"quality_prob_{j}"] = quality_prob[:, j]
            chunks.append(frame)
    return pd.concat(chunks, ignore_index=True)


def predict_h5_from_checkpoint(
    checkpoint_path: Path | str,
    h5_path: Path | str,
    *,
    device: str | torch.device = "cpu",
    batch_size: int = 8192,
) -> pd.DataFrame:
    model, _ = load_cellularity_model(checkpoint_path, device=device)
    return predict_h5(model, h5_path, device=device, batch_size=batch_size)


def save_prediction_frame(predictions: pd.DataFrame, out_path: Path | str) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".csv":
        predictions.to_csv(out, index=False)
    else:
        predictions.to_parquet(out, index=False)


def aggregate_roi_from_predictions(
    predictions: pd.DataFrame,
    *,
    coverage_col: str | None = None,
    tumor_fraction_col: str | None = None,
) -> dict[str, float]:
    """Aggregate a prediction table over an ROI.

    If no coverage column is supplied, every tile is counted as fully selected.
    """

    coverage = predictions[coverage_col] if coverage_col else None
    tumor_fraction = predictions[tumor_fraction_col] if tumor_fraction_col else None
    return roi_count_summary(
        predictions["pred_nuclei_count"],
        alpha=predictions["pred_alpha"],
        coverage=coverage,
        tumor_fraction=tumor_fraction,
    )

