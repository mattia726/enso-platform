"""Pass / borderline / fail adequacy labelling.

The adequacy decision is intentionally simple — a clinician should be able
to predict from the four threshold numbers exactly which label a given ROI
will receive. The MC ``adequacy_probability`` decides the broad bucket; the
borderline bands flag ROIs whose point estimates are close to the cliff so
that the UI can offer a soft warning even when the probability is high.

Reason strings are constructed deterministically from the inputs so the
report sheet can quote them verbatim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .roi import ROIMetrics
from .thresholds import ThresholdProfile


AdequacyLabel = Literal["pass", "borderline", "fail", "not_quantifiable"]


@dataclass(frozen=True)
class AdequacyVerdict:
    """Full adequacy decision: label, confidence, and reasons."""

    label: AdequacyLabel
    confidence: float
    reasons: list[str]
    thresholds: dict
    metrics_snapshot: dict

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "thresholds": dict(self.thresholds),
            "metrics_snapshot": dict(self.metrics_snapshot),
        }


def _purity_reasons(
    metrics: ROIMetrics, threshold: ThresholdProfile
) -> list[str]:
    reasons: list[str] = []
    if metrics.purity.median >= threshold.purity_min:
        margin = metrics.purity.median - threshold.purity_min
        if margin <= threshold.borderline_purity_band:
            reasons.append(
                f"Purity {metrics.purity.median:.0%} just above the {threshold.purity_min:.0%} threshold"
                f" (within the {threshold.borderline_purity_band:.0%} borderline band)."
            )
        else:
            reasons.append(
                f"Purity {metrics.purity.median:.0%} ≥ threshold {threshold.purity_min:.0%}."
            )
    else:
        reasons.append(
            f"Purity {metrics.purity.median:.0%} below threshold {threshold.purity_min:.0%}."
        )
    return reasons


def _tumor_cell_reasons(
    metrics: ROIMetrics, threshold: ThresholdProfile
) -> list[str]:
    reasons: list[str] = []
    if metrics.tumor_nuclei.median >= threshold.tumor_cells_min:
        margin = metrics.tumor_nuclei.median - threshold.tumor_cells_min
        if margin <= threshold.borderline_tumor_cells_band:
            reasons.append(
                f"Tumor nuclei {metrics.tumor_nuclei.median:,.0f} just above the {threshold.tumor_cells_min:,} threshold"
                f" (within the {threshold.borderline_tumor_cells_band:,}-cell borderline band)."
            )
        else:
            reasons.append(
                f"Tumor nuclei {metrics.tumor_nuclei.median:,.0f} ≥ threshold {threshold.tumor_cells_min:,}."
            )
    else:
        reasons.append(
            f"Tumor nuclei {metrics.tumor_nuclei.median:,.0f} below threshold {threshold.tumor_cells_min:,}."
        )
    return reasons


def _confidence_reason(metrics: ROIMetrics) -> str:
    return f"Adequacy confidence {metrics.adequacy_probability:.0%}."


def label_adequacy(
    metrics: ROIMetrics,
    threshold: ThresholdProfile,
) -> AdequacyVerdict:
    """Return a full adequacy verdict for an ROI."""

    if metrics.n_tiles == 0 or metrics.tiles_with_data == 0:
        return AdequacyVerdict(
            label="not_quantifiable",
            confidence=0.0,
            reasons=[
                "ROI does not overlap any tissue tile; no quantitative estimate possible.",
            ],
            thresholds=threshold.to_dict(),
            metrics_snapshot=metrics.to_dict(),
        )

    in_purity_band = (
        metrics.purity.median >= threshold.purity_min
        and metrics.purity.median - threshold.purity_min
        <= threshold.borderline_purity_band
    )
    in_cells_band = (
        metrics.tumor_nuclei.median >= threshold.tumor_cells_min
        and metrics.tumor_nuclei.median - threshold.tumor_cells_min
        <= threshold.borderline_tumor_cells_band
    )
    purity_ok = metrics.purity.median >= threshold.purity_min
    cells_ok = metrics.tumor_nuclei.median >= threshold.tumor_cells_min
    prob = metrics.adequacy_probability

    if prob >= threshold.pass_probability and purity_ok and cells_ok and not (
        in_purity_band or in_cells_band
    ):
        label: AdequacyLabel = "pass"
    elif prob >= threshold.borderline_probability and purity_ok and cells_ok:
        label = "borderline"
    elif prob >= threshold.borderline_probability:
        label = "borderline"
    else:
        label = "fail"

    reasons = (
        _purity_reasons(metrics, threshold)
        + _tumor_cell_reasons(metrics, threshold)
        + [_confidence_reason(metrics)]
    )

    return AdequacyVerdict(
        label=label,
        confidence=metrics.adequacy_probability,
        reasons=reasons,
        thresholds=threshold.to_dict(),
        metrics_snapshot=metrics.to_dict(),
    )
