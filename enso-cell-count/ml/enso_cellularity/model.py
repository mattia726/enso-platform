"""EnsoCellularity tile-level count model."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EnsoCellularityConfig:
    input_dim: int = 2560
    d_model: int = 512
    metadata_dim: int = 5
    metadata_hidden_dim: int = 64
    metadata_out_dim: int = 128
    trunk_hidden_dim: int = 1024
    trunk_depth: int = 3
    attention_heads: int = 4
    num_quality_classes: int = 3
    num_count_bins: int = 6
    embed_dropout: float = 0.05
    trunk_dropout: float = 0.10
    eps: float = 1e-6


class ResidualMLPBlock(nn.Module):
    """Pre-norm residual MLP block."""

    def __init__(self, d_model: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.fc1(z)
        z = F.gelu(z)
        z = self.drop(z)
        z = self.fc2(z)
        z = self.drop(z)
        return x + z


class EnsoCellularityModel(nn.Module):
    """3x3 context model with NB count, ordinal, quantile, and quality heads."""

    center_index: int = 4

    def __init__(self, cfg: EnsoCellularityConfig | None = None):
        super().__init__()
        self.cfg = cfg or EnsoCellularityConfig()
        cfg = self.cfg

        self.embed_norm = nn.LayerNorm(cfg.input_dim)
        self.embed_proj = nn.Sequential(
            nn.Linear(cfg.input_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.embed_dropout),
        )
        self.pos_embed = nn.Parameter(torch.zeros(9, cfg.d_model))
        nn.init.normal_(self.pos_embed, std=0.02)

        self.context_attn = nn.MultiheadAttention(
            cfg.d_model,
            cfg.attention_heads,
            dropout=cfg.embed_dropout,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(cfg.d_model)

        self.meta_mlp = nn.Sequential(
            nn.LayerNorm(cfg.metadata_dim),
            nn.Linear(cfg.metadata_dim, cfg.metadata_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.metadata_hidden_dim, cfg.metadata_out_dim),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(cfg.d_model + cfg.metadata_out_dim, cfg.d_model),
            nn.GELU(),
            nn.LayerNorm(cfg.d_model),
        )
        self.trunk = nn.Sequential(
            *[
                ResidualMLPBlock(cfg.d_model, cfg.trunk_hidden_dim, cfg.trunk_dropout)
                for _ in range(cfg.trunk_depth)
            ]
        )

        self.density_head = nn.Linear(cfg.d_model, 1)
        self.dispersion_head = nn.Linear(cfg.d_model, 1)
        self.ordinal_head = nn.Linear(cfg.d_model, cfg.num_count_bins - 1)
        self.quantile_head = nn.Linear(cfg.d_model, 3)
        self.quality_head = nn.Linear(cfg.d_model, cfg.num_quality_classes)

    def forward_outputs(
        self,
        x9: torch.Tensor,
        valid9: torch.Tensor,
        metadata: torch.Tensor,
        exposure_mm2: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return all raw and transformed heads.

        Args:
            x9: ``[B, 9, 2560]`` center + 8-neighbor Virchow embeddings.
            valid9: ``[B, 9]`` boolean mask, True for existing/valid neighbor.
            metadata: ``[B, 5]`` normalized metadata.
            exposure_mm2: ``[B]`` or ``[B, 1]`` tile exposure.
        """

        if x9.ndim != 3 or x9.shape[1] != 9:
            raise ValueError(f"x9 must have shape [B, 9, D], got {tuple(x9.shape)}")
        if valid9.shape != x9.shape[:2]:
            raise ValueError(f"valid9 must have shape {tuple(x9.shape[:2])}, got {tuple(valid9.shape)}")

        valid9 = valid9.to(dtype=torch.bool, device=x9.device)
        center_valid = valid9[:, self.center_index]
        if not torch.all(center_valid):
            raise ValueError("Center tile must be valid for every sample.")

        tokens = self.embed_proj(self.embed_norm(x9))
        tokens = tokens + self.pos_embed.to(dtype=tokens.dtype, device=tokens.device).unsqueeze(0)
        q = tokens[:, self.center_index : self.center_index + 1, :]
        ctx, _ = self.context_attn(
            q,
            tokens,
            tokens,
            key_padding_mask=~valid9,
            need_weights=False,
        )
        h = self.context_norm(q.squeeze(1) + ctx.squeeze(1))

        m = self.meta_mlp(metadata.to(device=x9.device, dtype=x9.dtype))
        h = self.fuse(torch.cat([h, m], dim=-1))
        h = self.trunk(h)

        exposure = exposure_mm2.to(device=x9.device, dtype=x9.dtype).reshape(-1, 1).clamp_min(
            self.cfg.eps
        )
        density = F.softplus(self.density_head(h)) + self.cfg.eps
        mu = density * exposure
        alpha = F.softplus(self.dispersion_head(h)) + self.cfg.eps
        theta = 1.0 / alpha
        nb_logits = torch.log(mu.clamp_min(self.cfg.eps)) - torch.log(
            theta.clamp_min(self.cfg.eps)
        )

        q_raw = self.quantile_head(h)
        q05_log = q_raw[:, 0:1]
        q50_log = q05_log + F.softplus(q_raw[:, 1:2])
        q95_log = q50_log + F.softplus(q_raw[:, 2:3])
        quantiles_log1p = torch.cat([q05_log, q50_log, q95_log], dim=-1)
        quantiles = torch.expm1(quantiles_log1p).clamp_min(0.0)

        return {
            "mu": mu,
            "alpha": alpha,
            "theta": theta,
            "nb_logits": nb_logits,
            "density_per_mm2": density,
            "ordinal_logits": self.ordinal_head(h),
            "quality_logits": self.quality_head(h),
            "quantiles_log1p": quantiles_log1p,
            "quantiles": quantiles,
            "embedding": h,
        }

    def forward(
        self,
        x9: torch.Tensor,
        valid9: torch.Tensor,
        metadata: torch.Tensor,
        exposure_mm2: torch.Tensor,
    ) -> torch.Tensor:
        """Return expected nuclei count for backward-compatible inference."""

        return self.forward_outputs(x9, valid9, metadata, exposure_mm2)["mu"]
