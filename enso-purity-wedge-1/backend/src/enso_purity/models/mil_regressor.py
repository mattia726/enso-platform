from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn

from .distribution_pooling import KDEDistributionPooling, KDEPoolingConfig


@dataclass(frozen=True)
class MILRegressorConfig:
    input_dim: int
    proj_dim: int = 128
    bag_size: int = 200
    dropout: float = 0.5
    kde: KDEPoolingConfig = KDEPoolingConfig()
    hidden1: int = 384
    hidden2: int = 192


class MILPurityRegressor(nn.Module):
    def __init__(self, cfg: MILRegressorConfig):
        super().__init__()
        self.cfg = cfg

        self.proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.proj_dim),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.proj_dim, cfg.proj_dim),
            nn.ReLU(),
        )
        self.pool = KDEDistributionPooling(cfg.kde)
        pooled_dim = cfg.proj_dim * cfg.kde.num_bins
        self.regressor = nn.Sequential(
            nn.Dropout(cfg.dropout),
            nn.Linear(pooled_dim, cfg.hidden1),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden1, cfg.hidden2),
            nn.ReLU(),
            nn.Linear(cfg.hidden2, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats: [B,N,D]"""
        x = self.proj(feats)
        # squash to [0,1] for KDE bins; this is a practical adaptation
        x = torch.sigmoid(x)
        pooled = self.pool(x)
        y = self.regressor(pooled)
        return y.squeeze(-1)  # [B]
