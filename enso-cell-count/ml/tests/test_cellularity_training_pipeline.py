import sys

import numpy as np
import pandas as pd
import torch
import h5py

from enso_cellularity.dataset import (
    CellularitySlideTileDataset,
    build_slide_index,
    cellularity_collate,
)
from enso_cellularity.folds import assign_case_folds, split_for_fold
from enso_cellularity.inference import predict_h5
from enso_cellularity.losses import CellularityLossWeights, EnsoCellularityCompositeLoss
from enso_cellularity.model import EnsoCellularityConfig, EnsoCellularityModel
from enso_cellularity.train_cli import _parse_args
from enso_cellularity.training import TrainConfig, run_one_epoch


def _write_synthetic_slide(root, file_id, case_id, project_id="TCGA-TEST", input_dim=8):
    h5_dir = root / "h5"
    label_dir = root / "labels"
    h5_dir.mkdir(exist_ok=True)
    label_dir.mkdir(exist_ok=True)
    rows = []
    coords = []
    coords_l0 = []
    idx = 0
    for y in range(3):
        for x in range(3):
            coords.append([y, x])
            coords_l0.append([x * 100, y * 100])
            count = y * 10 + x
            rows.append(
                {
                    "file_uuid_original": file_id,
                    "barcode": f"{case_id}-01Z-00-DX1",
                    "project_id": project_id,
                    "case_id": case_id,
                    "embedding_index": idx,
                    "tile_y": y,
                    "tile_x": x,
                    "tile_x_level0": x * 100,
                    "tile_y_level0": y * 100,
                    "mpp_x": 0.5,
                    "mpp_y": 0.5,
                    "tile_area_mm2": 0.012544,
                    "tissue_fraction": 1.0,
                    "exposure_mm2": 0.012544,
                    "teacher_total_nuclei": count,
                    "teacher_confidence": 1.0,
                    "teacher_disagreement": 0.0,
                    "quality_flags": "pan_cancer_ann",
                    "source": "unit_test",
                    "count_bin": 1 if count > 0 else 0,
                }
            )
            idx += 1

    with h5py.File(h5_dir / f"{file_id}.h5", "w") as h5:
        h5.create_dataset("features", data=np.arange(9 * input_dim, dtype=np.float32).reshape(9, input_dim))
        h5.create_dataset("coords", data=np.asarray(coords, dtype=np.int32))
        h5.create_dataset("coords_level0", data=np.asarray(coords_l0, dtype=np.int32))
        h5.attrs["tile_size"] = 224
        h5.attrs["target_mpp"] = 0.5
        h5.attrs["mpp"] = 0.5
        h5.attrs["mpp_x"] = 0.5
        h5.attrs["mpp_y"] = 0.5
        h5.attrs["extracted_level0_size"] = 224
        h5.attrs["stride_level0"] = 224
        h5.attrs["grid_nx"] = 3
        h5.attrs["grid_ny"] = 3

    pd.DataFrame(rows).to_parquet(label_dir / f"{file_id}__{case_id}.parquet", index=False)
    return h5_dir, label_dir


def test_slide_dataset_gathers_3x3_neighbors(tmp_path):
    h5_dir, label_dir = _write_synthetic_slide(tmp_path, "slide-a", "CASE-A")
    index = build_slide_index(label_dir, h5_dir)
    ds = CellularitySlideTileDataset(index, tiles_per_slide=9, training=False, seed=1)
    item = ds[0]
    assert item["x9"].shape == (9, 9, 8)
    assert item["valid9"].shape == (9, 9)
    center_tile = 4
    assert item["valid9"][center_tile].all()
    assert torch.equal(item["x9"][center_tile, 4], torch.arange(32, 40, dtype=torch.float32))


def test_dataset_uses_full_tile_tissue_exposure(tmp_path):
    h5_dir, label_dir = _write_synthetic_slide(tmp_path, "slide-a", "CASE-A")
    parquet_path = next(label_dir.glob("*.parquet"))
    labels = pd.read_parquet(parquet_path)
    labels["tissue_fraction"] = 0.25
    labels["exposure_mm2"] = labels["tile_area_mm2"] * labels["tissue_fraction"]
    labels.to_parquet(parquet_path, index=False)

    index = build_slide_index(label_dir, h5_dir)
    ds = CellularitySlideTileDataset(index, tiles_per_slide=9, training=False, seed=1)
    item = ds[0]

    assert torch.allclose(item["metadata"][:, 2], torch.ones(9))
    assert torch.allclose(item["exposure_mm2"], torch.full((9,), 0.012544))


def test_eval_all_tiles_can_be_chunked(tmp_path):
    h5_dir, label_dir = _write_synthetic_slide(tmp_path, "slide-a", "CASE-A")
    index = build_slide_index(label_dir, h5_dir)
    ds = CellularitySlideTileDataset(
        index,
        tiles_per_slide=0,
        all_tiles_chunk_size=4,
        training=False,
        seed=1,
    )

    assert len(ds) == 3
    indices: list[int] = []
    for item in ds:
        indices.extend(item["embedding_index"])
        assert item["x9"].shape[0] <= 4
    assert indices == list(range(9))


def test_cellularity_training_defaults_disable_quality_and_eval_all_tiles():
    assert CellularityLossWeights().quality == 0.0
    assert TrainConfig().tiles_per_slide == 8192
    assert TrainConfig().eval_tiles_per_slide == 0
    assert TrainConfig().eval_tile_chunk_size == 8192
    assert TrainConfig().slide_batch_size == 1
    assert TrainConfig().lr == 3e-5
    assert TrainConfig().patience == 5
    assert TrainConfig().early_stop_metric == "val_mae_count"
    assert TrainConfig().scheduler_patience == 1


def test_train_cli_defaults_disable_quality_and_eval_all_tiles(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["train_cli"])
    args = _parse_args()
    assert args.tiles_per_slide == 8192
    assert args.slide_batch_size == 1
    assert args.prefetch_factor == 1
    assert args.lr == 3e-5
    assert args.patience == 5
    assert args.log_every == 10
    assert args.quality_weight == 0.0
    assert args.nb_weight == 0.30
    assert args.smooth_l1_weight == 1.00
    assert args.ordinal_weight == 0.05
    assert args.quantile_weight == 0.02
    assert args.eval_tiles_per_slide == 0
    assert args.eval_tile_chunk_size == 8192
    assert args.eval_slide_batch_size == 1
    assert args.scheduler_patience == 1
    assert args.early_stop_metric == "val_mae_count"


def test_training_epoch_and_inference(tmp_path):
    h5_dir, label_dir = _write_synthetic_slide(tmp_path, "slide-a", "CASE-A")
    _write_synthetic_slide(tmp_path, "slide-b", "CASE-B")
    index = build_slide_index(label_dir, h5_dir)
    index = assign_case_folds(index, n_folds=2, seed=2)
    split = split_for_fold(index, fold=0, n_folds=2, val_fold=1)
    assert set(split.train["case_id"]).isdisjoint(set(split.val["case_id"]))

    train_ds = CellularitySlideTileDataset(index, tiles_per_slide=4, training=True, seed=1)
    loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        collate_fn=cellularity_collate,
        shuffle=False,
        num_workers=0,
    )
    model = EnsoCellularityModel(
        EnsoCellularityConfig(input_dim=8, d_model=16, trunk_hidden_dim=32, metadata_dim=5)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = run_one_epoch(
        model,
        loader,
        EnsoCellularityCompositeLoss(),
        opt,
        device="cpu",
        train=True,
        log_every=0,
    )
    assert torch.isfinite(torch.tensor(metrics["loss"]))

    pred = predict_h5(model, h5_dir / "slide-a.h5", device="cpu", batch_size=4)
    assert len(pred) == 9
    assert {"pred_nuclei_count", "pred_q05", "pred_q95", "pred_count_bin"} <= set(pred.columns)
