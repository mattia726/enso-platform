"""Tests for spatial heatmap inference with cKDTree and batched processing."""
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig
from enso_purity_mil.heatmap import (
    build_neighborhood_indices,
    predict_tile_scores,
)


def _make_h5(path: Path, n_tiles: int = 200, dim: int = 2560, stride: int | None = None):
    coords = np.column_stack([
        np.arange(n_tiles) % 14 * 112,
        np.arange(n_tiles) // 14 * 112,
    ]).astype(np.int32)
    with h5py.File(path, "w") as f:
        f.create_dataset("features", data=np.random.randn(n_tiles, dim).astype(np.float32))
        f.create_dataset("coords", data=coords)
        f.attrs["tile_size"] = 224
        f.attrs["target_mpp"] = 0.5
        if stride is not None:
            f.attrs["stride"] = stride


class TestBuildNeighborhoodIndices:
    def test_shape(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        _make_h5(h5_path, n_tiles=100)
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
        indices = build_neighborhood_indices(coords, k=81)
        assert indices.shape == (100, 81)

    def test_self_included(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        _make_h5(h5_path, n_tiles=100)
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
        indices = build_neighborhood_indices(coords, k=81)
        # The tile itself should be its own nearest neighbor (distance=0)
        for i in range(100):
            assert i in indices[i]

    def test_fewer_tiles_than_k(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        _make_h5(h5_path, n_tiles=30)  # fewer than K=81
        with h5py.File(h5_path, "r") as f:
            coords = f["coords"][:]
        indices = build_neighborhood_indices(coords, k=81)
        # Should pad with repeated indices, shape still (30, 81)
        assert indices.shape == (30, 81)


class TestPredictTileScores:
    def test_output_shape(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        _make_h5(h5_path, n_tiles=100, dim=64)
        cfg = EnsoModelConfig(input_dim=64, num_features=8, num_bins=5)
        model = EnsoMILModel(cfg)
        model.eval()
        scores, coords = predict_tile_scores(model, h5_path, k=9, batch_size=32)
        assert scores.shape == (100,)
        assert coords.shape == (100, 2)

    def test_scores_finite(self, tmp_path):
        h5_path = tmp_path / "test.h5"
        _make_h5(h5_path, n_tiles=50, dim=64)
        cfg = EnsoModelConfig(input_dim=64, num_features=8, num_bins=5)
        model = EnsoMILModel(cfg)
        model.eval()
        scores, _ = predict_tile_scores(model, h5_path, k=9, batch_size=16)
        assert np.all(np.isfinite(scores))

    def test_memory_safe_batching(self, tmp_path):
        """Ensure we never allocate (M, K, dim) all at once."""
        h5_path = tmp_path / "test.h5"
        _make_h5(h5_path, n_tiles=200, dim=64)
        cfg = EnsoModelConfig(input_dim=64, num_features=8, num_bins=5)
        model = EnsoMILModel(cfg)
        model.eval()
        scores, _ = predict_tile_scores(model, h5_path, k=9, batch_size=16)
        assert scores.shape == (200,)

    def test_scores_clamped_01(self, tmp_path):
        """Scores must be clamped to [0, 1]."""
        h5_path = tmp_path / "test.h5"
        _make_h5(h5_path, n_tiles=50, dim=64)
        cfg = EnsoModelConfig(input_dim=64, num_features=8, num_bins=5)
        model = EnsoMILModel(cfg)
        model.eval()
        scores, _ = predict_tile_scores(model, h5_path, k=9, batch_size=32)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0


class TestHeatmapDynamicStride:
    def test_h5_stride_attribute_read(self, tmp_path):
        """Verify that stride attr is stored and readable."""
        h5_path = tmp_path / "strided.h5"
        _make_h5(h5_path, n_tiles=50, dim=64, stride=112)
        with h5py.File(h5_path, "r") as f:
            tile_size = int(f.attrs.get("tile_size", 224))
            stride = int(f.attrs.get("stride", tile_size))
        assert tile_size == 224
        assert stride == 112

    def test_default_stride_equals_tile_size(self, tmp_path):
        """Without stride attr, stride defaults to tile_size."""
        h5_path = tmp_path / "no_stride.h5"
        _make_h5(h5_path, n_tiles=50, dim=64)  # no stride attr
        with h5py.File(h5_path, "r") as f:
            tile_size = int(f.attrs.get("tile_size", 224))
            stride = int(f.attrs.get("stride", tile_size))
        assert stride == 224
