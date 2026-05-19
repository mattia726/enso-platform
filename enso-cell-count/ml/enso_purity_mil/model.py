"""Enso MIL purity regressor v2.

Core design:
  VirchowAdapter (LayerNorm + low-rank MLP + bounded output)
  + Multi-scale KDE distribution pooling
  + Prototype histogram (joint-composition proxy)
  + Optional moment features
  + Multi-head prediction (purity mean, uncertainty, tumor-vs-normal)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class EnsoModelConfig:
    input_dim: int = 2560
    adapter_hidden_dim: int = 256
    num_features: int = 128
    num_bins: int = 21
    sigmas: tuple[float, ...] = (0.02, 0.05, 0.10)
    # Backward-compatibility fields from v1 checkpoints:
    sigma: float | None = None
    num_classes: int = 1
    prototype_k: int = 64
    prototype_temp: float = 0.10
    use_moments: bool = True
    adapter_dropout: float = 0.10
    head_dropout: float = 0.20
    instance_dropout: float = 0.20
    feature_noise_std: float = 0.01
    tau_init: float = 1.0
    learnable_tau: bool = True

    def __post_init__(self) -> None:
        if self.sigma is not None:
            self.sigmas = (float(self.sigma),)
        else:
            self.sigmas = tuple(float(s) for s in self.sigmas)


class VirchowAdapter(nn.Module):
    """LayerNorm → low-rank MLP → LayerNorm → bounded [0,1] features."""

    def __init__(
        self,
        input_dim: int = 2560,
        bottleneck_dim: int = 256,
        output_dim: int = 128,
        dropout: float = 0.10,
        tau_init: float = 1.0,
        learnable_tau: bool = True,
    ):
        super().__init__()
        self.in_norm = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, output_dim)
        self.out_norm = nn.LayerNorm(output_dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.learnable_tau = learnable_tau
        if learnable_tau:
            init = float(max(tau_init, 1e-4))
            # inverse softplus
            init_param = math.log(math.exp(init) - 1.0)
            self.tau_param = nn.Parameter(torch.tensor(init_param, dtype=torch.float32))
        else:
            self.register_buffer("tau_fixed", torch.tensor(float(tau_init), dtype=torch.float32))

    def _tau(self) -> torch.Tensor:
        if self.learnable_tau:
            tau = F.softplus(self.tau_param) + 1e-4
        else:
            tau = self.tau_fixed
        return torch.clamp(tau, min=0.5, max=2.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_norm(x)
        z = self.fc1(z)
        z = self.act(z)
        z = self.drop(z)
        z = self.fc2(z)
        z = self.out_norm(z)
        return torch.sigmoid(z / self._tau())


class MultiScaleDistributionPooling(nn.Module):
    """Gaussian KDE pooling over instance dimension for multiple sigmas.

    Input must be in [0, 1], output shape is ``(B, J * S, M)``
    where ``S`` is number of sigma values.
    """

    def __init__(self, num_bins: int = 21, sigmas: tuple[float, ...] = (0.02, 0.05, 0.10)):
        super().__init__()
        self.num_bins = num_bins
        self.sigmas = tuple(float(s) for s in sigmas)
        sample_points = torch.linspace(0, 1, steps=num_bins, dtype=torch.float32)
        self.register_buffer("sample_points", sample_points)

    def forward(self, data: torch.Tensor) -> torch.Tensor:
        B, N, J = data.size()
        sp = self.sample_points.view(1, 1, 1, self.num_bins)
        data_exp = data.unsqueeze(-1)  # (B, N, J, 1)
        pooled = []
        for sigma in self.sigmas:
            alfa = 1.0 / math.sqrt(2 * math.pi * sigma ** 2)
            beta = -1.0 / (2 * sigma ** 2)
            diff_sq = (sp - data_exp) ** 2
            kernels = alfa * torch.exp(beta * diff_sq)
            out_unnorm = kernels.sum(dim=1)  # (B, J, M)
            norm = out_unnorm.sum(dim=2, keepdim=True).clamp_min(1e-8)
            pooled.append(out_unnorm / norm)
        return torch.cat(pooled, dim=1)  # (B, J*S, M)


class DistributionPoolingFilter(MultiScaleDistributionPooling):
    """Backward-compatible single-sigma wrapper."""

    def __init__(self, num_bins: int = 21, sigma: float = 0.05):
        super().__init__(num_bins=num_bins, sigmas=(sigma,))


class PrototypeHistogram(nn.Module):
    """Soft-assignment histogram in feature space (joint proxy)."""

    def __init__(self, num_features: int = 128, num_prototypes: int = 64, temperature: float = 0.10):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_prototypes, num_features))
        init = float(max(temperature, 1e-3))
        self.log_temp = nn.Parameter(torch.log(torch.tensor(init, dtype=torch.float32)))

    def _temperature(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_temp), min=0.03, max=1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, J)
        x_n = F.normalize(x, dim=-1)
        p_n = F.normalize(self.prototypes, dim=-1)
        logits = torch.einsum("bnj,kj->bnk", x_n, p_n) / self._temperature()
        assign = torch.softmax(logits, dim=-1)
        return assign.mean(dim=1)  # (B, K)


class EnsoMILModel(nn.Module):
    """Full Enso MIL purity model with uncertainty + auxiliary head."""

    def __init__(self, cfg: EnsoModelConfig | None = None):
        super().__init__()
        if cfg is None:
            cfg = EnsoModelConfig()
        self.cfg = cfg
        self.adapter = VirchowAdapter(
            input_dim=cfg.input_dim,
            bottleneck_dim=cfg.adapter_hidden_dim,
            output_dim=cfg.num_features,
            dropout=cfg.adapter_dropout,
            tau_init=cfg.tau_init,
            learnable_tau=cfg.learnable_tau,
        )
        self.kde = MultiScaleDistributionPooling(cfg.num_bins, cfg.sigmas)
        self.prototype_hist = PrototypeHistogram(
            num_features=cfg.num_features,
            num_prototypes=cfg.prototype_k,
            temperature=cfg.prototype_temp,
        )
        total_dim = cfg.num_features * cfg.num_bins * len(cfg.sigmas) + cfg.prototype_k
        if cfg.use_moments:
            total_dim += 2 * cfg.num_features
        self.trunk = nn.Sequential(
            nn.Dropout(cfg.head_dropout),
            nn.Linear(total_dim, 512),
            nn.GELU(),
            nn.Dropout(cfg.head_dropout),
            nn.Linear(512, 128),
            nn.GELU(),
        )
        self.mu_head = nn.Linear(128, 1)
        self.kappa_head = nn.Linear(128, 1)
        self.tumor_head = nn.Linear(128, 1)

    def _instance_subsample(self, x: torch.Tensor) -> torch.Tensor:
        if (not self.training) or self.cfg.instance_dropout <= 0.0:
            return x
        B, N, J = x.shape
        if N <= 1:
            return x
        keep = int(round((1.0 - self.cfg.instance_dropout) * N))
        keep = max(1, min(N, keep))
        if keep == N:
            return x
        scores = torch.rand(B, N, device=x.device)
        idx = scores.topk(k=keep, dim=1).indices
        gather_idx = idx.unsqueeze(-1).expand(-1, -1, J)
        return torch.gather(x, dim=1, index=gather_idx)

    def _build_representation(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.adapter(x)  # (B, N, J), bounded
        if self.training and self.cfg.feature_noise_std > 0.0:
            z = z + torch.randn_like(z) * self.cfg.feature_noise_std
            z = torch.clamp(z, 0.0, 1.0)
        z = self._instance_subsample(z)
        kde_feat = self.kde(z).flatten(1)  # (B, J*S*M)
        proto_hist = self.prototype_hist(z)  # (B, K)
        parts = [kde_feat, proto_hist]
        if self.cfg.use_moments:
            mean = z.mean(dim=1)
            std = z.std(dim=1, unbiased=False)
            parts.extend([mean, std])
        rep = torch.cat(parts, dim=1)
        return rep, proto_hist

    def forward_outputs(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        rep, proto_hist = self._build_representation(x)
        h = self.trunk(rep)
        mu = torch.sigmoid(self.mu_head(h))
        kappa = F.softplus(self.kappa_head(h)) + 1e-4
        tumor_prob = torch.sigmoid(self.tumor_head(h))
        alpha = torch.clamp(mu * kappa, min=1e-4)
        beta = torch.clamp((1.0 - mu) * kappa, min=1e-4)
        return {
            "mu": mu,
            "kappa": kappa,
            "alpha": alpha,
            "beta": beta,
            "tumor_prob": tumor_prob,
            "proto_hist": proto_hist,
            "embedding": h,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Keep backward-compatible API: return purity mean only.
        return self.forward_outputs(x)["mu"]
