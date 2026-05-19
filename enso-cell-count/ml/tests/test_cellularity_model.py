import numpy as np
import pytest
import torch

from enso_cellularity.labels import (
    TileGridSpec,
    count_bin_indices,
    count_centroids_in_tiles,
    ordinal_targets_from_counts,
)
from enso_cellularity.losses import EnsoCellularityCompositeLoss
from enso_cellularity.model import EnsoCellularityConfig, EnsoCellularityModel


def test_count_centroids_in_non_overlapping_tiles():
    spec = TileGridSpec(
        grid_nx=3,
        grid_ny=2,
        stride_level0=100,
        tile_size_level0=100,
        pad_left_level0=10,
        pad_top_level0=20,
        mpp_x=0.5,
        mpp_y=0.5,
        target_mpp=0.5,
        tile_size=224,
    )
    # Kept tiles are row-major grid cells: (0,0), (0,1), (1,2).
    coords_level0 = np.array(
        [
            [-10, -20],
            [90, -20],
            [190, 80],
        ],
        dtype=np.int32,
    )
    x = np.array([0, 10, 99, 120, 220, 250, 260, 500], dtype=np.float32)
    y = np.array([0, 70, 79, 0, 100, 120, 179, 0], dtype=np.float32)
    counts = count_centroids_in_tiles(x, y, coords_level0, spec)
    assert counts.tolist() == [2, 2, 3]


def test_ordinal_targets_from_counts():
    counts = np.array([0, 1, 11, 151, 301], dtype=np.float32)
    targets = ordinal_targets_from_counts(counts)
    assert targets.shape == (5, 5)
    assert targets[0].tolist() == [False, False, False, False, False]
    assert targets[-1].tolist() == [True, True, True, True, True]
    assert count_bin_indices(counts).tolist() == [0, 1, 2, 4, 5]


def test_cellularity_model_forward_shapes_and_quantile_order():
    torch.manual_seed(7)
    cfg = EnsoCellularityConfig(input_dim=16, d_model=32, trunk_hidden_dim=64, metadata_dim=5)
    model = EnsoCellularityModel(cfg)
    x9 = torch.randn(4, 9, 16)
    valid9 = torch.ones(4, 9, dtype=torch.bool)
    valid9[0, 0] = False
    metadata = torch.randn(4, 5)
    exposure = torch.full((4,), 0.0125)

    out = model.forward_outputs(x9, valid9, metadata, exposure)
    assert out["mu"].shape == (4, 1)
    assert out["alpha"].shape == (4, 1)
    assert out["ordinal_logits"].shape == (4, 5)
    assert out["quality_logits"].shape == (4, 3)
    assert out["quantiles"].shape == (4, 3)
    assert torch.all(out["mu"] > 0)
    assert torch.all(out["alpha"] > 0)
    assert torch.all(out["quantiles"][:, 1] >= out["quantiles"][:, 0])
    assert torch.all(out["quantiles"][:, 2] >= out["quantiles"][:, 1])


def test_center_tile_must_be_valid():
    model = EnsoCellularityModel(EnsoCellularityConfig(input_dim=8, d_model=16, trunk_hidden_dim=32))
    x9 = torch.randn(2, 9, 8)
    valid9 = torch.ones(2, 9, dtype=torch.bool)
    valid9[1, 4] = False
    with pytest.raises(ValueError, match="Center tile"):
        model.forward_outputs(x9, valid9, torch.randn(2, 5), torch.ones(2))


def test_composite_loss_is_finite():
    torch.manual_seed(11)
    cfg = EnsoCellularityConfig(input_dim=8, d_model=16, trunk_hidden_dim=32, metadata_dim=5)
    model = EnsoCellularityModel(cfg)
    out = model.forward_outputs(
        torch.randn(3, 9, 8),
        torch.ones(3, 9, dtype=torch.bool),
        torch.randn(3, 5),
        torch.full((3,), 0.0125),
    )
    criterion = EnsoCellularityCompositeLoss()
    loss, parts = criterion(
        out,
        torch.tensor([0.0, 25.0, 180.0]),
        teacher_confidence=torch.tensor([1.0, 0.5, 1.0]),
        quality_target=torch.tensor([0, 0, 0]),
    )
    assert torch.isfinite(loss)
    assert set(parts) >= {"nb_nll", "smooth_l1_log", "ordinal_bce", "quantile_pinball"}
