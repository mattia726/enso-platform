"""Metrics for tile and ROI-level EnsoCellularity evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class MetricAccumulator:
    """Accumulate predictions and targets on CPU for epoch metrics."""

    y_true: list[np.ndarray]
    y_mu: list[np.ndarray]
    q05: list[np.ndarray]
    q50: list[np.ndarray]
    q95: list[np.ndarray]
    pred_bin: list[np.ndarray]
    true_bin: list[np.ndarray]

    @classmethod
    def empty(cls) -> "MetricAccumulator":
        return cls([], [], [], [], [], [], [])

    def update(
        self,
        *,
        y_true: torch.Tensor,
        outputs: dict[str, torch.Tensor],
        true_bin: torch.Tensor | None = None,
    ) -> None:
        y = y_true.detach().float().reshape(-1).cpu().numpy()
        mu = outputs["mu"].detach().float().reshape(-1).cpu().numpy()
        q = outputs["quantiles"].detach().float().cpu().numpy()
        ordinal = outputs["ordinal_logits"].detach().float()
        pred_bin = ordinal_count_bins(ordinal).cpu().numpy()
        if true_bin is None:
            true_bin_np = count_bins_from_counts_np(y)
        else:
            true_bin_np = true_bin.detach().reshape(-1).cpu().numpy()

        self.y_true.append(y)
        self.y_mu.append(mu)
        self.q05.append(q[:, 0])
        self.q50.append(q[:, 1])
        self.q95.append(q[:, 2])
        self.pred_bin.append(pred_bin)
        self.true_bin.append(true_bin_np)

    def compute(self) -> dict[str, float]:
        if not self.y_true:
            return {}
        y = np.concatenate(self.y_true).astype(np.float64)
        mu = np.concatenate(self.y_mu).astype(np.float64)
        q05 = np.concatenate(self.q05).astype(np.float64)
        q95 = np.concatenate(self.q95).astype(np.float64)
        pred_bin = np.concatenate(self.pred_bin)
        true_bin = np.concatenate(self.true_bin)
        abs_err = np.abs(mu - y)
        log_err = np.abs(np.log1p(mu.clip(min=0.0)) - np.log1p(y.clip(min=0.0)))
        factor2 = within_factor(y, mu, factor=2.0)
        factor10 = within_factor(y, mu, factor=10.0)
        out = {
            "mae_count": float(np.mean(abs_err)),
            "mae_log1p": float(np.mean(log_err)),
            "rmse_log1p": float(np.sqrt(np.mean(log_err**2))),
            "within_factor2": float(np.mean(factor2)),
            "within_factor10": float(np.mean(factor10)),
            "bin_accuracy": float(np.mean(pred_bin == true_bin)),
            "interval_coverage_90": float(np.mean((y >= q05) & (y <= q95))),
            "mean_true_count": float(np.mean(y)),
            "mean_pred_count": float(np.mean(mu)),
            "spearman": spearman_np(y, mu),
        }
        return out


def count_bins_from_counts_np(counts: np.ndarray) -> np.ndarray:
    edges = np.asarray([0.0, 10.0, 50.0, 150.0, 300.0], dtype=np.float64)
    return np.digitize(np.asarray(counts, dtype=np.float64), edges, right=True).astype(np.int64)


def ordinal_count_bins(ordinal_logits: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    """Convert cumulative ordinal logits into bins 0..num_bins-1."""

    probs = torch.sigmoid(ordinal_logits)
    return (probs > threshold).sum(dim=1).to(dtype=torch.long)


def within_factor(y_true: np.ndarray, y_pred: np.ndarray, *, factor: float) -> np.ndarray:
    """Return per-sample within-factor accuracy on ``1 + count`` scale."""

    yt = np.asarray(y_true, dtype=np.float64) + 1.0
    yp = np.asarray(y_pred, dtype=np.float64) + 1.0
    ratio = np.maximum(yt / np.clip(yp, 1e-8, None), yp / np.clip(yt, 1e-8, None))
    return ratio <= factor


def spearman_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman correlation with an optional SciPy fast path."""

    if len(y_true) < 2:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        value, _ = spearmanr(y_true, y_pred)
        return float(value)
    except Exception:
        r_true = rankdata_np(y_true)
        r_pred = rankdata_np(y_pred)
        if np.std(r_true) == 0 or np.std(r_pred) == 0:
            return float("nan")
        return float(np.corrcoef(r_true, r_pred)[0, 1])


def rankdata_np(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def roi_count_summary(
    counts: Iterable[float],
    *,
    alpha: Iterable[float] | None = None,
    coverage: Iterable[float] | None = None,
    tumor_fraction: Iterable[float] | None = None,
) -> dict[str, float]:
    """Aggregate tile predictions over a selected region.

    Quantile heads are not summed. For uncertainty, use the NB variance
    approximation from ``mu`` and ``alpha``:

    ``Var[Y] = mu + alpha * mu^2``.
    """

    mu = np.asarray(list(counts), dtype=np.float64)
    cov = np.ones_like(mu) if coverage is None else np.asarray(list(coverage), dtype=np.float64)
    total = float(np.sum(cov * mu))
    out = {"estimated_total_nuclei": total}
    if alpha is not None:
        a = np.asarray(list(alpha), dtype=np.float64)
        var = np.sum((cov**2) * (mu + a * mu**2))
        sd = math.sqrt(max(float(var), 0.0))
        out["total_nuclei_sd"] = sd
        out["total_nuclei_q05_normal_approx"] = max(0.0, total - 1.645 * sd)
        out["total_nuclei_q95_normal_approx"] = max(0.0, total + 1.645 * sd)
    if tumor_fraction is not None:
        tf = np.asarray(list(tumor_fraction), dtype=np.float64)
        out["estimated_tumor_nuclei"] = float(np.sum(cov * mu * tf))
    return out

