from __future__ import annotations

import pandas as pd
import numpy as np

from enso_cellularity.ann_dicom import (
    polygon_centroid,
    polygon_centroids,
    primitive_point_ranges_from_flat_indices,
)
from enso_cellularity.pancancer_nuclei import (
    ANN_SOP_CLASS_UID,
    extract_ann_metadata_from_dicom_json,
    normalize_ann_manifest,
    overlap_summary,
    parse_tcga_slide_barcode,
)


def test_parse_tcga_dx_slide_barcode():
    parsed = parse_tcga_slide_barcode("TCGA-E2-A14S-01Z-00-DX1")

    assert parsed.case_id == "TCGA-E2-A14S"
    assert parsed.sample_vial == "TCGA-E2-A14S-01Z"
    assert parsed.sample_type_code == "01"
    assert parsed.vial == "Z"
    assert parsed.portion_code == "00"
    assert parsed.section_type == "DX"
    assert parsed.section_number == "1"


def test_extract_ann_metadata_from_dicom_json():
    instance = {
        "00080016": {"vr": "UI", "Value": [ANN_SOP_CLASS_UID]},
        "00080018": {"vr": "UI", "Value": ["1.2.3"]},
        "00080060": {"vr": "CS", "Value": ["ANN"]},
        "00100020": {"vr": "LO", "Value": ["TCGA-E2-A14S"]},
        "00120020": {"vr": "LO", "Value": ["TCGA-BRCA"]},
        "00400512": {"vr": "LO", "Value": ["TCGA-E2-A14S-01Z-00-DX1"]},
        "006A0002": {
            "vr": "SQ",
            "Value": [
                {
                    "00660016": {"vr": "OF", "BulkDataURI": "https://example/coords"},
                    "00660040": {"vr": "OL", "BulkDataURI": "https://example/index"},
                    "006A000C": {"vr": "UL", "Value": [352609]},
                    "00700023": {"vr": "CS", "Value": ["POLYGON"]},
                    "00660121": {
                        "vr": "SQ",
                        "Value": [
                            {
                                "00660132": {
                                    "vr": "SQ",
                                    "Value": [
                                        {
                                            "00660125": {
                                                "vr": "OF",
                                                "BulkDataURI": "https://example/areas",
                                            }
                                        }
                                    ],
                                }
                            }
                        ],
                    },
                }
            ],
        },
    }

    row = extract_ann_metadata_from_dicom_json(
        instance,
        collection_id="tcga_brca",
        study_instance_uid="2.25.1",
        series_instance_uid="1.2.826.1",
        series_size_mb=123.4,
        series_aws_url="s3://idc-open-data/example/*",
    )

    assert row["slide_barcode"] == "TCGA-E2-A14S-01Z-00-DX1"
    assert row["case_id"] == "TCGA-E2-A14S"
    assert row["nuclei_count_slide"] == 352609
    assert row["ann_graphic_type"] == "POLYGON"
    assert row["point_coordinates_bulk_uri"] == "https://example/coords"
    assert row["primitive_index_bulk_uri"] == "https://example/index"
    assert row["area_measurements_bulk_uri"] == "https://example/areas"


def test_normalize_ann_manifest_and_overlap_summary():
    raw = pd.DataFrame(
        [
            {
                "collection_id": "tcga_brca",
                "project_id": "TCGA-BRCA",
                "container_identifier": "TCGA-E2-A14S-01Z-00-DX1",
                "annotation_count": 10,
                "graphic_type": "POLYGON",
            },
            {
                "collection_id": "tcga_brca",
                "project_id": "TCGA-BRCA",
                "container_identifier": "TCGA-AA-0001-01Z-00-DX1",
                "annotation_count": 20,
                "graphic_type": "POLYGON",
            },
        ]
    )
    manifest = normalize_ann_manifest(raw)
    wedge = pd.DataFrame(
        [
            {"barcode": "TCGA-E2-A14S-01A-01-TS1", "sample_type_code": 1},
            {"barcode": "TCGA-ZZ-9999-01A-01-TS1", "sample_type_code": 1},
        ]
    )

    summary = overlap_summary(manifest, wedge)

    assert manifest["nuclei_count_slide"].sum() == 30
    assert set(manifest["section_type"]) == {"DX"}
    assert summary["exact_slide_overlap"] == 0
    assert summary["case_overlap"] == 1
    assert summary["sample_vial_overlap"] == 0
    assert summary["case_sample_type_overlap"] == 1


def test_primitive_point_ranges_are_flat_coordinate_offsets():
    # Two 4-point polygons stored as x/y float pairs.  The second polygon starts
    # at the ninth float value, which is point offset 4 after reshaping.
    starts, ends = primitive_point_ranges_from_flat_indices(
        pd.Series([1, 9]).to_numpy(dtype="uint32"),
        num_float_values=16,
    )

    assert starts.tolist() == [0, 4]
    assert ends.tolist() == [4, 8]


def test_polygon_centroids_for_squares():
    square = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )

    assert polygon_centroid(square) == (1.0, 1.0)

    coords = np.array(
        [
            [0.0, 0.0],
            [2.0, 0.0],
            [2.0, 2.0],
            [0.0, 2.0],
            [10.0, 10.0],
            [12.0, 10.0],
            [12.0, 12.0],
            [10.0, 12.0],
        ],
        dtype=np.float32,
    )
    centroid_x, centroid_y, vertex_counts = polygon_centroids(
        coords,
        pd.Series([0, 4]).to_numpy(),
        pd.Series([4, 8]).to_numpy(),
    )

    assert centroid_x.tolist() == [1.0, 11.0]
    assert centroid_y.tolist() == [1.0, 11.0]
    assert vertex_counts.tolist() == [4, 4]
