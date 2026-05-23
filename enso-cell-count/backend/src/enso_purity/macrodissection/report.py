"""Build the data payload for the printable macrodissection sheet."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .adequacy import AdequacyVerdict
from .cases import CaseMeta
from .storage import ROIRecord
from .thresholds import ThresholdProfile


def build_report_payload(
    *,
    case_meta: CaseMeta,
    roi_record: ROIRecord,
    verdict: AdequacyVerdict,
    threshold: ThresholdProfile,
) -> dict[str, Any]:
    """Return a self-contained report dict for the frontend to render."""

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case": case_meta.to_dict(),
        "roi": {
            "roi_id": roi_record.roi_id,
            "label": roi_record.label,
            "geometry_thumb_px": roi_record.geometry_thumb_px,
            "locked": roi_record.locked,
            "created_at": roi_record.created_at,
            "updated_at": roi_record.updated_at,
            "user_id": roi_record.user_id,
            "revision": roi_record.revision,
            "notes": roi_record.notes,
        },
        "verdict": verdict.to_dict(),
        "threshold_profile": threshold.to_dict(),
        "models": roi_record.model_run,
        "disclaimer": (
            "AI-assisted estimate. The final macrodissection ROI must be "
            "selected and signed off by the reviewing pathologist; the "
            "EnsoPurity and EnsoCellularity outputs are decision support, "
            "not autonomous decision making."
        ),
    }
