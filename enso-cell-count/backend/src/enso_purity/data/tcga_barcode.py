from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlideBarcode:
    raw: str
    case_id: str         # TCGA-XX-YYYY
    sample_vial: str     # TCGA-XX-YYYY-01A
    portion: str         # e.g. 01
    section: str         # e.g. TS1 / BS1 / MS1 / DX1
    section_type: str    # e.g. TS / BS / MS / DX
    is_dx: bool

def parse_slide_barcode(slide_id: str) -> SlideBarcode:
    """Parse a TCGA slide submitter_id like: TCGA-CS-5394-01A-01-TS1"""
    parts = slide_id.split("-")
    if len(parts) < 6:
        raise ValueError(f"Not a TCGA slide barcode: {slide_id}")
    case_id = "-".join(parts[:3])
    sample_vial = "-".join(parts[:4])
    portion = parts[4]
    section = parts[5]
    section_type = section[:2]
    is_dx = section_type == "DX"
    return SlideBarcode(
        raw=slide_id,
        case_id=case_id,
        sample_vial=sample_vial,
        portion=portion,
        section=section,
        section_type=section_type,
        is_dx=is_dx,
    )


@dataclass(frozen=True)
class AliquotBarcode:
    raw: str
    patient_id: str      # TCGA-XX-YYYY
    sample_vial: str     # TCGA-XX-YYYY-01A
    portion: str         # e.g. 11 (from 11D)
    analyte: str         # e.g. D (from 11D)

def parse_aliquot_barcode(aliquot: str) -> AliquotBarcode:
    """Parse a TCGA aliquot barcode like: TCGA-CS-5394-01A-11D-1234-01"""
    parts = aliquot.split("-")
    if len(parts) < 5:
        raise ValueError(f"Not a TCGA aliquot barcode: {aliquot}")
    patient_id = "-".join(parts[:3])
    sample_vial = "-".join(parts[:4])
    portion_analyte = parts[4]
    portion = portion_analyte[:2]
    analyte = portion_analyte[2:3] if len(portion_analyte) >= 3 else ""
    return AliquotBarcode(
        raw=aliquot,
        patient_id=patient_id,
        sample_vial=sample_vial,
        portion=portion,
        analyte=analyte,
    )
