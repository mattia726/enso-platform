"""Adequacy threshold profiles.

The thresholds that decide whether an ROI is *adequate* for downstream
molecular testing are not a model parameter — they are an assay / institution
policy. We expose them as named profiles so the same UI can be retargeted to
different labs without redeploying code.

The default :data:`PROFILES["humanitas_ngs"]` profile is the one used by the
Humanitas pilot transcript: a minimum 20% tumor fraction and at least one
thousand tumor nuclei in the macrodissected area. The narrow borderline band
gives the pathologist a soft warning before a hard fail.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass(frozen=True)
class ThresholdProfile:
    """Named adequacy threshold profile.

    Attributes:
        name: Stable machine-readable identifier (``humanitas_ngs``, ...).
        display_name: Human-readable label for the UI dropdown.
        purity_min: Minimum acceptable cellularity-weighted tumor purity.
        tumor_cells_min: Minimum acceptable estimated tumor-nuclei count
            inside the ROI.
        borderline_purity_band: Soft warning band immediately above
            ``purity_min``. ROIs whose median purity falls inside the band
            are flagged borderline even if the adequacy probability is high.
        borderline_tumor_cells_band: Same idea for the tumor-nuclei axis.
        pass_probability: Adequacy MC probability threshold for the *pass*
            label. ROIs below this are *borderline* (if above
            ``borderline_probability``) or *fail*.
        borderline_probability: Adequacy MC probability threshold for the
            *borderline* label.
        notes: Free-form description rendered in the report sheet.
    """

    name: str
    display_name: str
    purity_min: float
    tumor_cells_min: int
    borderline_purity_band: float = 0.05
    borderline_tumor_cells_band: int = 200
    pass_probability: float = 0.90
    borderline_probability: float = 0.50
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Return a plain JSON-serializable representation."""

        return asdict(self)


PROFILES: dict[str, ThresholdProfile] = {
    "humanitas_ngs": ThresholdProfile(
        name="humanitas_ngs",
        display_name="Humanitas NGS pilot",
        purity_min=0.20,
        tumor_cells_min=1000,
        borderline_purity_band=0.05,
        borderline_tumor_cells_band=200,
        pass_probability=0.90,
        borderline_probability=0.50,
        notes=(
            "Pilot profile based on the Humanitas macrodissection workflow: "
            "tumor area must be ≥20% pure and contain at least one thousand "
            "tumor nuclei for downstream NGS to be reliable."
        ),
    ),
    "research": ThresholdProfile(
        name="research",
        display_name="Research / exploratory",
        purity_min=0.10,
        tumor_cells_min=200,
        borderline_purity_band=0.05,
        borderline_tumor_cells_band=100,
        pass_probability=0.85,
        borderline_probability=0.40,
        notes=(
            "Relaxed thresholds intended for translational research where "
            "lower-yield specimens may still be informative."
        ),
    ),
    "strict_solid_tumor": ThresholdProfile(
        name="strict_solid_tumor",
        display_name="Strict solid tumor",
        purity_min=0.30,
        tumor_cells_min=2000,
        borderline_purity_band=0.05,
        borderline_tumor_cells_band=300,
        pass_probability=0.95,
        borderline_probability=0.70,
        notes=(
            "Conservative profile suited to deeply-sequenced solid tumor "
            "assays where false positives from contaminating normal tissue "
            "are particularly costly."
        ),
    ),
}


@dataclass(frozen=True)
class ThresholdOverride:
    """Per-call threshold override; values not present fall back to profile."""

    purity_min: float | None = None
    tumor_cells_min: int | None = None
    borderline_purity_band: float | None = None
    borderline_tumor_cells_band: int | None = None
    pass_probability: float | None = None
    borderline_probability: float | None = None


def resolve_thresholds(
    profile_name: str,
    override: ThresholdOverride | None = None,
) -> ThresholdProfile:
    """Return the threshold profile, applying an optional partial override.

    Raises:
        KeyError: if ``profile_name`` is not registered.
    """

    if profile_name not in PROFILES:
        raise KeyError(
            f"Unknown threshold profile '{profile_name}'. "
            f"Known profiles: {sorted(PROFILES)}"
        )
    base = PROFILES[profile_name]
    if override is None:
        return base
    fields: dict[str, Any] = base.to_dict()
    for key, value in vars(override).items():
        if value is not None:
            fields[key] = value
    # Keep ``name`` and ``display_name`` from the base profile.
    fields["name"] = base.name
    fields["display_name"] = base.display_name
    return ThresholdProfile(**fields)


def list_profiles() -> list[dict[str, Any]]:
    """Return a list of profile dictionaries suitable for the API response."""

    return [profile.to_dict() for profile in PROFILES.values()]
