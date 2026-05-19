import torch
from enso_purity.models.distribution_pooling import KDEDistributionPooling, KDEPoolingConfig


def test_kde_pooling_shape():
    cfg = KDEPoolingConfig(num_bins=21, sigma=0.05)
    pool = KDEDistributionPooling(cfg)
    x = torch.rand(2, 200, 128)  # [B,N,J]
    y = pool(x)
    assert y.shape == (2, 128 * 21)

def test_kde_pooling_finite():
    cfg = KDEPoolingConfig(num_bins=21, sigma=0.05)
    pool = KDEDistributionPooling(cfg)
    x = torch.rand(1, 5, 3)
    y = pool(x)
    assert torch.isfinite(y).all()
