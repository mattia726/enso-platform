"""Helpers for the Pan-Cancer-Nuclei-Seg cellularity label source.

The Pan-Cancer-Nuclei-Seg DICOM release stores nuclei as DICOM Bulk Simple
Annotations (ANN).  Each ANN series corresponds to one TCGA diagnostic slide
and contains one annotation group whose ``NumberOfAnnotations`` is the
slide-level nuclei count.  Polygon coordinates live in bulk data fields and
can be downloaded later when building tile-level labels.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import pandas as pd

ANN_SOP_CLASS_UID = "1.2.840.10008.5.1.4.1.1.91.1"
PANCANCER_ANALYSIS_RESULT_ID = "Pan-Cancer-Nuclei-Seg-DICOM"
DICOMWEB_BASE_URL = (
    "https://proxy.imaging.datacommons.cancer.gov/current/"
    "viewer-only-no-downloads-see-tinyurl-dot-com-slash-3j3d9jyp/dicomWeb"
)

TAG_SOP_CLASS_UID = "00080016"
TAG_SOP_INSTANCE_UID = "00080018"
TAG_MODALITY = "00080060"
TAG_PATIENT_ID = "00100020"
TAG_PROJECT_ID = "00120020"
TAG_CONTAINER_IDENTIFIER = "00400512"
TAG_ANNOTATION_GROUP_SEQUENCE = "006A0002"
TAG_POINT_COORDINATES_DATA = "00660016"
TAG_LONG_PRIMITIVE_POINT_INDEX_LIST = "00660040"
TAG_MEASUREMENTS_SEQUENCE = "00660121"
TAG_MEASUREMENT_VALUES_SEQUENCE = "00660132"
TAG_FLOAT_POINT_VALUES = "00660125"
TAG_NUMBER_OF_ANNOTATIONS = "006A000C"
TAG_GRAPHIC_TYPE = "00700023"


@dataclass(frozen=True)
class TcgaSlideBarcode:
    """Parsed fields from a TCGA slide submitter barcode."""

    barcode: str
    case_id: str
    sample_vial: str
    sample_type_code: str
    vial: str
    portion_analyte: str
    portion_code: str
    analyte_code: str
    section: str
    section_type: str
    section_number: str


def normalize_text(value: Any) -> str:
    """Return a stable uppercase string, or ``""`` for null-ish values."""

    if value is None or pd.isna(value):
        return ""
    return str(value).strip().upper()


def parse_tcga_slide_barcode(value: Any) -> TcgaSlideBarcode:
    """Parse a TCGA slide barcode without requiring it to be valid.

    Examples:
    - ``TCGA-E2-A14S-01Z-00-DX1``
    - ``TCGA-08-0360-01A-01-TS1``
    """

    barcode = normalize_text(value)
    parts = barcode.split("-") if barcode else []
    sample_vial = parts[3] if len(parts) >= 4 else ""
    portion_analyte = parts[4] if len(parts) >= 5 else ""
    section = parts[5] if len(parts) >= 6 else ""
    section_type = "".join(ch for ch in section if ch.isalpha())
    section_number = "".join(ch for ch in section if ch.isdigit())

    return TcgaSlideBarcode(
        barcode=barcode,
        case_id="-".join(parts[:3]) if len(parts) >= 3 else barcode,
        sample_vial="-".join(parts[:4]) if len(parts) >= 4 else "",
        sample_type_code=sample_vial[:2],
        vial=sample_vial[2:],
        portion_analyte=portion_analyte,
        portion_code=portion_analyte[:2],
        analyte_code=portion_analyte[2:],
        section=section,
        section_type=section_type,
        section_number=section_number,
    )


def dicom_json_value(obj: Mapping[str, Any] | None, tag: str) -> Any:
    """Extract the first DICOM JSON ``Value`` for ``tag``."""

    if not obj:
        return None
    elem = obj.get(tag)
    if not isinstance(elem, Mapping):
        return None
    values = elem.get("Value")
    if not values:
        return None
    return values[0]


def dicom_json_bulk_uri(obj: Mapping[str, Any] | None, tag: str) -> str | None:
    """Extract a DICOM JSON ``BulkDataURI`` for ``tag`` if present."""

    if not obj:
        return None
    elem = obj.get(tag)
    if not isinstance(elem, Mapping):
        return None
    uri = elem.get("BulkDataURI")
    return str(uri) if uri else None


def first_annotation_group(instance: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the first annotation group from an ANN DICOM JSON instance."""

    seq = instance.get(TAG_ANNOTATION_GROUP_SEQUENCE)
    if not isinstance(seq, Mapping):
        return {}
    values = seq.get("Value")
    if not values:
        return {}
    group = values[0]
    return group if isinstance(group, Mapping) else {}


def find_first_bulk_uri(obj: Any, tag: str) -> str | None:
    """Recursively find the first ``BulkDataURI`` for ``tag``."""

    if isinstance(obj, Mapping):
        current = dicom_json_bulk_uri(obj, tag)
        if current:
            return current
        for value in obj.values():
            found = find_first_bulk_uri(value, tag)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = find_first_bulk_uri(value, tag)
            if found:
                return found
    return None


def dicomweb_metadata_url(study_instance_uid: str, series_instance_uid: str) -> str:
    """Build the IDC DICOMweb metadata URL for a series."""

    return (
        f"{DICOMWEB_BASE_URL}/studies/{study_instance_uid}"
        f"/series/{series_instance_uid}/metadata"
    )


def extract_ann_metadata_from_dicom_json(
    instance: Mapping[str, Any],
    *,
    collection_id: str | None = None,
    study_instance_uid: str | None = None,
    series_instance_uid: str | None = None,
    series_size_mb: float | None = None,
    series_aws_url: str | None = None,
) -> dict[str, Any]:
    """Extract a flat manifest row from a DICOM JSON ANN instance."""

    group = first_annotation_group(instance)
    slide_barcode = dicom_json_value(instance, TAG_CONTAINER_IDENTIFIER)
    parts = parse_tcga_slide_barcode(slide_barcode)
    study_uid = study_instance_uid or dicom_json_value(instance, "0020000D")
    series_uid = series_instance_uid or dicom_json_value(instance, "0020000E")
    project_id = dicom_json_value(instance, TAG_PROJECT_ID)

    return {
        "source": "pan_cancer_nuclei_seg",
        "collection_id": normalize_text(collection_id),
        "project_id": normalize_text(project_id),
        "slide_barcode": parts.barcode,
        "case_id": parts.case_id,
        "sample_vial": parts.sample_vial,
        "sample_type_code": parts.sample_type_code,
        "vial": parts.vial,
        "portion_analyte": parts.portion_analyte,
        "portion_code": parts.portion_code,
        "analyte_code": parts.analyte_code,
        "section": parts.section,
        "section_type": parts.section_type,
        "section_number": parts.section_number,
        "patient_id": normalize_text(dicom_json_value(instance, TAG_PATIENT_ID)),
        "nuclei_count_slide": dicom_json_value(group, TAG_NUMBER_OF_ANNOTATIONS),
        "ann_graphic_type": normalize_text(dicom_json_value(group, TAG_GRAPHIC_TYPE)),
        "modality": normalize_text(dicom_json_value(instance, TAG_MODALITY)),
        "sop_class_uid": normalize_text(dicom_json_value(instance, TAG_SOP_CLASS_UID)),
        "sop_instance_uid": normalize_text(dicom_json_value(instance, TAG_SOP_INSTANCE_UID)),
        "study_instance_uid": normalize_text(study_uid),
        "series_instance_uid": normalize_text(series_uid),
        "series_size_mb": series_size_mb,
        "series_aws_url": series_aws_url,
        "dicomweb_metadata_url": (
            dicomweb_metadata_url(normalize_text(study_uid), normalize_text(series_uid))
            if study_uid and series_uid
            else ""
        ),
        "point_coordinates_bulk_uri": dicom_json_bulk_uri(group, TAG_POINT_COORDINATES_DATA),
        "primitive_index_bulk_uri": dicom_json_bulk_uri(
            group, TAG_LONG_PRIMITIVE_POINT_INDEX_LIST
        ),
        "area_measurements_bulk_uri": find_first_bulk_uri(group, TAG_FLOAT_POINT_VALUES),
    }


def _project_from_collection(collection_id: str) -> str:
    collection = normalize_text(collection_id).replace("_", "-")
    return collection if collection.startswith("TCGA-") else ""


def normalize_ann_manifest(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize raw ANN metadata rows into the EnsoCellularity manifest schema."""

    out = df.copy()
    if "slide_barcode" not in out.columns and "container_identifier" in out.columns:
        out["slide_barcode"] = out["container_identifier"]
    if "nuclei_count_slide" not in out.columns and "annotation_count" in out.columns:
        out["nuclei_count_slide"] = out["annotation_count"]
    if "ann_graphic_type" not in out.columns and "graphic_type" in out.columns:
        out["ann_graphic_type"] = out["graphic_type"]

    barcode_parts = out["slide_barcode"].map(parse_tcga_slide_barcode)
    out["slide_barcode"] = [p.barcode for p in barcode_parts]
    out["case_id"] = [p.case_id for p in barcode_parts]
    out["sample_vial"] = [p.sample_vial for p in barcode_parts]
    out["sample_type_code"] = [p.sample_type_code for p in barcode_parts]
    out["vial"] = [p.vial for p in barcode_parts]
    out["portion_analyte"] = [p.portion_analyte for p in barcode_parts]
    out["portion_code"] = [p.portion_code for p in barcode_parts]
    out["analyte_code"] = [p.analyte_code for p in barcode_parts]
    out["section"] = [p.section for p in barcode_parts]
    out["section_type"] = [p.section_type for p in barcode_parts]
    out["section_number"] = [p.section_number for p in barcode_parts]

    for col in [
        "source",
        "collection_id",
        "project_id",
        "patient_id",
        "ann_graphic_type",
        "modality",
        "sop_class_uid",
        "sop_instance_uid",
        "study_instance_uid",
        "series_instance_uid",
        "series_aws_url",
        "dicomweb_metadata_url",
        "point_coordinates_bulk_uri",
        "primitive_index_bulk_uri",
        "area_measurements_bulk_uri",
    ]:
        if col not in out.columns:
            out[col] = ""

    out["source"] = out["source"].replace("", "pan_cancer_nuclei_seg")
    out["collection_id"] = out["collection_id"].map(normalize_text).str.replace("-", "_")
    out["project_id"] = out["project_id"].map(normalize_text)
    missing_project = out["project_id"].eq("")
    out.loc[missing_project, "project_id"] = out.loc[missing_project, "collection_id"].map(
        _project_from_collection
    )

    out["nuclei_count_slide"] = pd.to_numeric(out["nuclei_count_slide"], errors="coerce").astype(
        "Int64"
    )
    out["series_size_mb"] = pd.to_numeric(out.get("series_size_mb"), errors="coerce")

    missing_url = out["dicomweb_metadata_url"].map(normalize_text).eq("")
    valid_uids = out["study_instance_uid"].map(normalize_text).ne("") & out[
        "series_instance_uid"
    ].map(normalize_text).ne("")
    for idx in out.index[missing_url & valid_uids]:
        out.at[idx, "dicomweb_metadata_url"] = dicomweb_metadata_url(
            normalize_text(out.at[idx, "study_instance_uid"]),
            normalize_text(out.at[idx, "series_instance_uid"]),
        )

    columns = [
        "source",
        "collection_id",
        "project_id",
        "slide_barcode",
        "case_id",
        "sample_vial",
        "sample_type_code",
        "vial",
        "portion_analyte",
        "portion_code",
        "analyte_code",
        "section",
        "section_type",
        "section_number",
        "patient_id",
        "nuclei_count_slide",
        "ann_graphic_type",
        "modality",
        "sop_class_uid",
        "sop_instance_uid",
        "study_instance_uid",
        "series_instance_uid",
        "series_size_mb",
        "series_aws_url",
        "dicomweb_metadata_url",
        "point_coordinates_bulk_uri",
        "primitive_index_bulk_uri",
        "area_measurements_bulk_uri",
    ]
    return out[columns].sort_values(["project_id", "slide_barcode", "series_instance_uid"])


def add_tcga_barcode_columns(
    df: pd.DataFrame,
    *,
    source_col: str,
    prefix: str,
) -> pd.DataFrame:
    """Add parsed TCGA slide fields for overlap audits."""

    out = df.copy()
    parsed = out[source_col].map(parse_tcga_slide_barcode)
    out[f"{prefix}barcode"] = [p.barcode for p in parsed]
    out[f"{prefix}case_id"] = [p.case_id for p in parsed]
    out[f"{prefix}sample_vial"] = [p.sample_vial for p in parsed]
    out[f"{prefix}sample_type_code"] = [p.sample_type_code for p in parsed]
    out[f"{prefix}section_type"] = [p.section_type for p in parsed]
    return out


def case_sample_type_key(case_id: Any, sample_type_code: Any) -> str:
    """Return a normalized ``TCGA-case-sampletype`` key."""

    sample_type = normalize_text(sample_type_code)
    if sample_type.isdigit():
        sample_type = sample_type.zfill(2)
    return f"{normalize_text(case_id)}-{sample_type}"


def overlap_summary(ann_manifest: pd.DataFrame, wedge_manifest: pd.DataFrame) -> dict[str, Any]:
    """Compute the key overlap counts between Pan-Cancer ANN and wedge manifests."""

    ann = add_tcga_barcode_columns(ann_manifest, source_col="slide_barcode", prefix="ann_")
    wedge = add_tcga_barcode_columns(wedge_manifest, source_col="barcode", prefix="wedge_")

    ann_exact = set(ann["ann_barcode"])
    wedge_exact = set(wedge["wedge_barcode"])
    ann_cases = set(ann["ann_case_id"])
    wedge_cases = set(wedge["wedge_case_id"])
    ann_sample_vials = set(ann["ann_sample_vial"])
    wedge_sample_vials = set(wedge["wedge_sample_vial"])
    ann_case_sample_type = {
        case_sample_type_key(row.ann_case_id, row.ann_sample_type_code)
        for row in ann.itertuples(index=False)
    }
    wedge_case_sample_type = {
        case_sample_type_key(row.wedge_case_id, row.wedge_sample_type_code)
        for row in wedge.itertuples(index=False)
    }

    return {
        "ann_rows": int(len(ann)),
        "ann_unique_slides": int(ann["ann_barcode"].nunique()),
        "ann_unique_cases": int(ann["ann_case_id"].nunique()),
        "ann_total_nuclei": int(pd.to_numeric(ann["nuclei_count_slide"], errors="coerce").sum()),
        "wedge_rows": int(len(wedge)),
        "wedge_unique_slides": int(wedge["wedge_barcode"].nunique()),
        "wedge_unique_cases": int(wedge["wedge_case_id"].nunique()),
        "exact_slide_overlap": len(ann_exact & wedge_exact),
        "case_overlap": len(ann_cases & wedge_cases),
        "sample_vial_overlap": len(ann_sample_vials & wedge_sample_vials),
        "case_sample_type_overlap": len(ann_case_sample_type & wedge_case_sample_type),
        "ann_section_type_counts": ann["ann_section_type"].value_counts().sort_index().to_dict(),
        "ann_sample_type_counts": ann["ann_sample_type_code"].value_counts().sort_index().to_dict(),
    }


def unique_nonempty(values: Iterable[Any]) -> list[str]:
    """Sorted unique non-empty normalized strings."""

    return sorted({normalize_text(v) for v in values if normalize_text(v)})

