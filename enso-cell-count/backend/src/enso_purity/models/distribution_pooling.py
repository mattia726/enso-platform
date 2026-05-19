from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn


@dataclass(frozen=True)
class KDEPoolingConfig:
    num_bins: int = 21
    sigma: float = 0.05
    value_range: Tuple[float, float] = (0.0, 1.0)  # assume features squashed to [0,1]


class KDEDistributionPooling(nn.Module):
    """
    Differentiable KDE-based distribution pooling per feature dimension.

    Input:  x [B, N, J] with values in [0,1] (recommended).
    Output: pooled [B, J * K] where K=num_bins.
    """

    def __init__(self, cfg: KDEPoolingConfig):
        super().__init__()
        self.cfg = cfg
        vmin, vmax = cfg.value_range
        # fixed bin centers
        centers = torch.linspace(vmin, vmax, cfg.num_bins)
        self.register_buffer("centers", centers, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected x shape [B,N,J], got {tuple(x.shape)}")
        B, N, J = x.shape
        K = self.centers.numel()
        sigma = self.cfg.sigma

        # x: [B,N,J] -> [B,N,J,1]
        x4 = x.unsqueeze(-1)
        # centers: [K] -> [1,1,1,K]
        c4 = self.centers.view(1, 1, 1, K)

        # Gaussian kernel
        # [B,N,J,K]
        z = (c4 - x4) / sigma
        w = torch.exp(-0.5 * z * z) / (sigma * math.sqrt(2.0 * math.pi))

        # average over instances N (uniform weights)
        # [B,J,K]
        p = w.mean(dim=1)

        # flatten to [B, J*K]
        return p.reshape(B, J * K)
