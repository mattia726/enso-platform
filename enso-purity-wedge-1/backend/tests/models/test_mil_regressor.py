import torch
from enso_purity.models.mil_regressor import MILPurityRegressor, MILRegressorConfig


def test_mil_forward_shape_and_range():
    cfg = MILRegressorConfig(input_dim=1024)
    model = MILPurityRegressor(cfg)
    feats = torch.randn(4, 200, 1024)
    y = model(feats)
    assert y.shape == (4,)
    assert (y >= 0).all() and (y <= 1).all()
