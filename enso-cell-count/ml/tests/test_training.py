"""Tests for training loop, early stopping, cosine scheduler."""
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig
from enso_purity_mil.training import EarlyStopping, run_one_epoch, TrainConfig


class TestEarlyStopping:
    def test_no_stop_if_improving(self):
        es = EarlyStopping(patience=5, min_delta=0.0)
        for loss in [1.0, 0.9, 0.8, 0.7, 0.6]:
            assert not es(loss)

    def test_stops_after_patience(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es(0.5)  # best
        es(0.6)  # worse
        es(0.7)  # worse
        assert es(0.8)  # 3 epochs without improvement → stop

    def test_resets_on_improvement(self):
        es = EarlyStopping(patience=3, min_delta=0.0)
        es(0.5)
        es(0.6)
        assert not es(0.4)  # improvement resets counter
        es(0.5)  # bad 1
        assert not es(0.6)  # bad 2, patience=3 not yet hit

    def test_min_delta(self):
        es = EarlyStopping(patience=2, min_delta=0.01)
        es(0.50)
        assert not es(0.495)  # improvement < min_delta → counts as no improvement
        assert es(0.495)


class TestRunOneEpoch:
    @pytest.fixture()
    def setup(self):
        cfg = EnsoModelConfig(input_dim=64, num_features=8, num_bins=5)
        model = EnsoMILModel(cfg)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.L1Loss()
        # Synthetic data: 8 samples, bag size 32
        feats = torch.rand(8, 32, 64)
        labels = torch.rand(8)
        dataset = torch.utils.data.TensorDataset(feats, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=4)
        return model, optimizer, criterion, loader

    def test_train_epoch_returns_loss(self, setup):
        model, optimizer, criterion, loader = setup
        loss = run_one_epoch(model, loader, criterion, optimizer, device="cpu", train=True)
        assert isinstance(loss, float)
        assert loss > 0

    def test_val_epoch_returns_metrics_dict(self, setup):
        model, optimizer, criterion, loader = setup
        result = run_one_epoch(model, loader, criterion, optimizer=None, device="cpu", train=False)
        assert isinstance(result, dict)
        assert "loss" in result
        assert "r2" in result
        assert "spearman" in result
        assert result["loss"] > 0

    def test_train_updates_weights(self, setup):
        model, optimizer, criterion, loader = setup
        w_before = model.adapter.linear.weight.clone()
        run_one_epoch(model, loader, criterion, optimizer, device="cpu", train=True)
        w_after = model.adapter.linear.weight
        assert not torch.allclose(w_before, w_after), "Weights should change after training"


class TestTrainConfig:
    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.num_instances == 4096
        assert cfg.batch_size == 128
        assert cfg.lr == 1e-4
        assert cfg.patience == 20
        assert cfg.max_epochs == 200
        assert cfg.num_workers == 14
