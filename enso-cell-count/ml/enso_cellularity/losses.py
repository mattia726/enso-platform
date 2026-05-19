"""Composite loss for EnsoCellularity."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CellularityLossWeights:
    nb_nll: float = 1.00
    smooth_l1_log: float = 0.25
    ordinal_bce: float = 0.20
    quantile_pinball: float = 0.10
    quality: float = 0.00


def negative_binomial_nll(
    y_count: torch.Tensor,
    theta: torch.Tensor,
    nb_logits: torch.Tensor,
) -> torch.Tensor:
    """Per-sample NB2 negative log likelihood."""

    y = y_count.reshape_as(theta).to(dtype=theta.dtype).clamp_min(0.0)
    dist = torch.distributions.NegativeBinomial(
        total_count=theta.clamp_min(1e-6),
        logits=nb_logits,
    )
    return -dist.log_prob(y)


def ordinal_targets(
    y_count: torch.Tensor,
    thresholds: torch.Tensor,
) -> torch.Tensor:
    """Cumulative targets for ordinal count bins."""

    return (y_count.reshape(-1, 1) > thresholds.reshape(1, -1)).to(dtype=torch.float32)


def quantile_pinball_loss(
    predictions: torch.Tensor,
    target: torch.Tensor,
    quantiles: torch.Tensor,
) -> torch.Tensor:
    """Elementwise pinball loss."""

    target = target.reshape(-1, 1).to(dtype=predictions.dtype)
    q = quantiles.reshape(1, -1).to(device=predictions.device, dtype=predictions.dtype)
    error = target - predictions
    return torch.maximum(q * error, (q - 1.0) * error)


class EnsoCellularityCompositeLoss:
    """Callable composite loss matching ``architecture.txt``."""

    def __init__(
        self,
        *,
        weights: CellularityLossWeights | None = None,
        count_bin_thresholds: tuple[float, ...] = (0.0, 10.0, 50.0, 150.0, 300.0),
        quantiles: tuple[float, ...] = (0.05, 0.50, 0.95),
    ):
        self.weights = weights or CellularityLossWeights()
        self.count_bin_thresholds = torch.tensor(count_bin_thresholds, dtype=torch.float32)
        self.quantiles = torch.tensor(quantiles, dtype=torch.float32)

    def __call__(
        self,
        outputs: dict[str, torch.Tensor],
        y_count: torch.Tensor,
        *,
        teacher_confidence: torch.Tensor | None = None,
        quality_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = y_count.reshape(-1, 1).to(device=outputs["mu"].device, dtype=outputs["mu"].dtype)
        if teacher_confidence is None:
            weight = torch.ones_like(y)
        else:
            weight = teacher_confidence.reshape_as(y).to(device=y.device, dtype=y.dtype)
            weight = weight.clamp_min(0.0)
        weight_denom = weight.sum().clamp_min(1e-6)

        nb = negative_binomial_nll(y, outputs["theta"], outputs["nb_logits"])
        nb_loss = (nb * weight).sum() / weight_denom

        smooth = F.smooth_l1_loss(
            torch.log1p(outputs["mu"].clamp_min(0.0)),
            torch.log1p(y),
            reduction="none",
        )
        smooth_loss = (smooth * weight).sum() / weight_denom

        thresholds = self.count_bin_thresholds.to(device=y.device, dtype=y.dtype)
        ord_target = ordinal_targets(y.reshape(-1), thresholds)
        ordinal = F.binary_cross_entropy_with_logits(
            outputs["ordinal_logits"],
            ord_target.to(dtype=outputs["ordinal_logits"].dtype),
            reduction="none",
        )
        ordinal_loss = (ordinal * weight).sum() / (weight_denom * ordinal.shape[1])

        q_loss = quantile_pinball_loss(
            outputs["quantiles_log1p"],
            torch.log1p(y),
            self.quantiles.to(device=y.device, dtype=y.dtype),
        )
        quantile_loss = (q_loss * weight).sum() / (weight_denom * q_loss.shape[1])

        if quality_target is None:
            quality_loss = torch.zeros((), device=y.device, dtype=y.dtype)
        elif outputs["quality_logits"].shape[1] == 1:
            quality_loss = F.binary_cross_entropy_with_logits(
                outputs["quality_logits"].reshape(-1),
                quality_target.to(device=y.device, dtype=y.dtype).reshape(-1),
            )
        else:
            quality_loss = F.cross_entropy(
                outputs["quality_logits"],
                quality_target.to(device=y.device, dtype=torch.long).reshape(-1),
            )

        total = (
            self.weights.nb_nll * nb_loss
            + self.weights.smooth_l1_log * smooth_loss
            + self.weights.ordinal_bce * ordinal_loss
            + self.weights.quantile_pinball * quantile_loss
            + self.weights.quality * quality_loss
        )
        parts = {
            "loss": total.detach(),
            "nb_nll": nb_loss.detach(),
            "smooth_l1_log": smooth_loss.detach(),
            "ordinal_bce": ordinal_loss.detach(),
            "quantile_pinball": quantile_loss.detach(),
            "quality": quality_loss.detach(),
        }
        return total, parts
