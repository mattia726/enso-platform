"""Tests for the Enso purity MIL model: Adapter → KDE → Regression Head."""
import torch
import pytest

from enso_purity_mil.model import (
    VirchowAdapter,
    DistributionPoolingFilter,
    RegressionHead,
    EnsoMILModel,
    EnsoModelConfig,
)


# ── VirchowAdapter ───────────────────────────────────────────────
class TestVirchowAdapter:
    def test_output_shape(self):
        adapter = VirchowAdapter(input_dim=2560, hidden_dim=128)
        x = torch.randn(4, 200, 2560)
        out = adapter(x)
        assert out.shape == (4, 200, 128)

    def test_output_bounded_01(self):
        adapter = VirchowAdapter(input_dim=2560, hidden_dim=128)
        x = torch.randn(8, 100, 2560) * 10  # large inputs
        out = adapter(x)
        assert (out >= 0.0).all()
        assert (out <= 1.0).all()

    def test_sigmoid_saturation_not_all_same(self):
        adapter = VirchowAdapter(input_dim=2560, hidden_dim=128)
        x = torch.randn(2, 50, 2560)
        out = adapter(x)
        assert out.std() > 1e-4, "Adapter output should not be constant"


# ── DistributionPoolingFilter ────────────────────────────────────
class TestKDE:
    def test_output_shape(self):
        kde = DistributionPoolingFilter(num_bins=21, sigma=0.05)
        x = torch.rand(4, 200, 128)  # values in [0,1]
        out = kde(x)
        assert out.shape == (4, 128, 21)

    def test_normalized(self):
        kde = DistributionPoolingFilter(num_bins=21, sigma=0.05)
        x = torch.rand(2, 100, 32)
        out = kde(x)
        sums = out.sum(dim=2)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_deterministic(self):
        kde = DistributionPoolingFilter(num_bins=21, sigma=0.05)
        x = torch.rand(2, 50, 16)
        out1 = kde(x)
        out2 = kde(x)
        assert torch.allclose(out1, out2)


# ── RegressionHead ───────────────────────────────────────────────
class TestRegressionHead:
    def test_output_shape(self):
        head = RegressionHead(num_features=128, num_bins=21, num_classes=1)
        x = torch.randn(4, 128 * 21)
        out = head(x)
        assert out.shape == (4, 1)

    def test_dropout_train_vs_eval(self):
        head = RegressionHead(num_features=128, num_bins=21, num_classes=1)
        x = torch.randn(8, 128 * 21)
        head.train()
        out_train = [head(x) for _ in range(5)]
        differs = any(not torch.allclose(out_train[0], o) for o in out_train[1:])
        assert differs, "Dropout should produce different outputs in train mode"

        head.eval()
        out_eval = [head(x) for _ in range(3)]
        for o in out_eval[1:]:
            assert torch.allclose(out_eval[0], o)


# ── Full model ───────────────────────────────────────────────────
class TestEnsoMILModel:
    def test_forward_shape(self):
        cfg = EnsoModelConfig()
        model = EnsoMILModel(cfg)
        x = torch.randn(2, 4096, 2560)
        model.eval()
        out = model(x)
        assert out.shape == (2, 1)

    def test_forward_small_bag(self):
        cfg = EnsoModelConfig()
        model = EnsoMILModel(cfg)
        x = torch.randn(1, 50, 2560)
        model.eval()
        out = model(x)
        assert out.shape == (1, 1)

    def test_adapter_output_bounded(self):
        cfg = EnsoModelConfig()
        model = EnsoMILModel(cfg)
        x = torch.randn(2, 100, 2560) * 5
        adapted = model.adapter(x)
        assert (adapted >= 0).all() and (adapted <= 1).all()

    def test_heatmap_head_only(self):
        """Regression head can be called independently for heatmap inference."""
        cfg = EnsoModelConfig()
        model = EnsoMILModel(cfg)
        model.eval()
        # Simulate pre-computed KDE output flattened
        kde_flat = torch.rand(16, cfg.num_features * cfg.num_bins)
        out = model.head(kde_flat)
        assert out.shape == (16, 1)

    def test_config_defaults(self):
        cfg = EnsoModelConfig()
        assert cfg.input_dim == 2560
        assert cfg.num_features == 128
        assert cfg.num_bins == 21
        assert cfg.sigma == 0.05
        assert cfg.num_classes == 1
