from __future__ import annotations

import pandas as pd
import pytest

from enso_purity_mil.build_tcga_cptac_retrain_manifest import (
    DEFAULT_CPTAC_NORMALS_CSV,
    DEFAULT_CPTAC_TUMOUR_CSV,
    DEFAULT_TCGA_MANIFEST,
    DEFAULT_TCGA_PREDICTIONS_CSV,
    build_arg_parser,
    build_combined_manifest,
)
from enso_purity_mil.train_cli import _build_tumour_folds


def _touch_h5(directory, stem: str) -> None:
    (directory / f"{stem}.h5").write_bytes(b"")


def test_build_tumour_folds_uses_preassigned_column() -> None:
    tumour_df = pd.DataFrame(
        {
            "case_id": ["A", "A", "B", "C"],
            "aliquot_barcode": ["alq_a", "alq_a", "alq_b", "alq_c"],
            "project_id": ["TCGA-GBM", "TCGA-GBM", "TCGA-LUAD", "TCGA-LUAD"],
            "purity": [0.4, 0.4, 0.5, 0.6],
            "preassigned_fold": [0, 0, 2, 4],
        }
    )

    folds = _build_tumour_folds(tumour_df, seed=42, preassigned_fold_col="preassigned_fold")

    assert folds[0] == [0, 1]
    assert folds[1] == []
    assert folds[2] == [2]
    assert folds[3] == []
    assert folds[4] == [3]


def test_build_tumour_folds_rejects_patient_leakage() -> None:
    tumour_df = pd.DataFrame(
        {
            "case_id": ["A", "A"],
            "aliquot_barcode": ["alq_1", "alq_2"],
            "project_id": ["TCGA-GBM", "TCGA-GBM"],
            "purity": [0.4, 0.5],
            "preassigned_fold": [0, 1],
        }
    )

    with pytest.raises(ValueError, match="Patient leakage"):
        _build_tumour_folds(tumour_df, seed=42, preassigned_fold_col="preassigned_fold")


def test_build_arg_parser_uses_repo_native_defaults() -> None:
    parser = build_arg_parser()
    args = parser.parse_args(
        [
            "--tcga-h5-dir",
            "tcga_h5",
            "--cptac-h5-dir",
            "cptac_h5",
            "--out-dir",
            "out",
        ]
    )

    assert args.tcga_manifest == DEFAULT_TCGA_MANIFEST
    assert args.tcga_predictions_csv == DEFAULT_TCGA_PREDICTIONS_CSV
    assert args.cptac_tumour_csv == DEFAULT_CPTAC_TUMOUR_CSV
    assert args.cptac_normals_csv == DEFAULT_CPTAC_NORMALS_CSV
    for value in [
        str(args.tcga_manifest),
        str(args.tcga_predictions_csv),
        str(args.cptac_tumour_csv),
        str(args.cptac_normals_csv),
    ]:
        assert "/home/luca/" not in value


def test_build_combined_manifest_filters_and_maps_rows(tmp_path) -> None:
    tcga_manifest = pd.DataFrame(
        {
            "file_uuid_original": ["tcga_t1_a", "tcga_t1_b", "tcga_n1", "tcga_drop"],
            "barcode": [
                "TCGA-01-0001-01A-01-TS1",
                "TCGA-01-0001-01A-01-BS1",
                "TCGA-01-0001-11A-01-TS1",
                "TCGA-02-0002-01A-01-TS1",
            ],
            "aliquot_barcode": ["ALQ-1", "ALQ-1", "", "ALQ-2"],
            "gdc_match_type": ["same_portion", "same_portion", "normal_tissue", "same_portion"],
            "purity": [0.55, 0.55, 0.0, 1.0],
            "ploidy": [2.0, 2.0, 2.0, 2.0],
            "case_id": ["TCGA-01-0001", "TCGA-01-0001", "TCGA-01-0001", "TCGA-02-0002"],
            "project_id": ["TCGA-GBM", "TCGA-GBM", "TCGA-GBM", "TCGA-LUAD"],
        }
    )
    tcga_manifest_path = tmp_path / "tcga.tsv"
    tcga_manifest.to_csv(tcga_manifest_path, sep="\t", index=False)

    tcga_predictions = pd.DataFrame(
        {
            "model": ["v3", "v3"],
            "fold": [0, 1],
            "aliquot_barcode": ["ALQ-1", "ALQ-2"],
        }
    )
    tcga_predictions_path = tmp_path / "tcga_predictions.csv"
    tcga_predictions.to_csv(tcga_predictions_path, index=False)

    cptac_tumours = pd.DataFrame(
        {
            "file_id": ["cptac_luad", "cptac_conflict", "cptac_kirp", "cptac_drop", "cptac_drop_1"],
            "Case_ID": ["CPTAC-1", "CPTAC-2", "CPTAC-3", "CPTAC-4", "CPTAC-5"],
            "Slide_ID": ["S1", "S2", "S3", "S4", "S5"],
            "Tumor": ["LUAD", "PDA", "CCRCC", "CCRCC", "GBM"],
            "Disease_Type": [
                "Lung Adenocarcinoma",
                "Pancreatic Ductal Adenocarcinoma",
                "Non-Clear Cell Renal Cell Carcinoma",
                "Non-Clear Cell Renal Cell Carcinoma",
                "Glioblastoma",
            ],
            "Tumor_Histological_Type": [
                "",
                "",
                "Papillary Renal Cell Carcinoma",
                "papillary urothelial carcinoma",
                "",
            ],
            "genomic_purity_ngs_fraction": [0.41, 0.33, 0.52, 0.61, 1.0],
            "project_id": ["CPTAC-3", "CPTAC-3", "CPTAC-3", "CPTAC-3", "CPTAC-3"],
            "final_match_source": [
                "tree_matched",
                "legacy_sample_specific_only",
                "tree_matched",
                "tree_matched",
                "tree_matched",
            ],
            "audit_reason": ["", "conflicting_ngs_purity_values", "", "", ""],
            "legacy_tree_audit_reason": ["", "conflicting_ngs_purity_values", "", "", ""],
        }
    )
    cptac_tumours_path = tmp_path / "cptac_tumours.csv"
    cptac_tumours.to_csv(cptac_tumours_path, index=False)

    cptac_normals = pd.DataFrame(
        {
            "file_id": ["cptac_norm_keep", "cptac_norm_missing", "cptac_norm_orphan"],
            "Case_ID": ["CPTAC-1", "CPTAC-4", "CPTAC-99"],
            "Slide_ID": ["N1", "N4", "N99"],
            "project_id": ["CPTAC-3", "CPTAC-3", "CPTAC-3"],
            "normal_definition": ["normal_tissue", "normal_tissue", "normal_tissue"],
        }
    )
    cptac_normals_path = tmp_path / "cptac_normals.csv"
    cptac_normals.to_csv(cptac_normals_path, index=False)

    tcga_h5_dir = tmp_path / "tcga_h5"
    tcga_h5_dir.mkdir()
    for stem in ["tcga_t1_a", "tcga_t1_b", "tcga_n1"]:
        _touch_h5(tcga_h5_dir, stem)

    cptac_h5_dir = tmp_path / "cptac_h5"
    cptac_h5_dir.mkdir()
    for stem in ["cptac_luad", "cptac_conflict", "cptac_kirp", "cptac_norm_keep"]:
        _touch_h5(cptac_h5_dir, stem)

    out_dir = tmp_path / "out"
    metadata = build_combined_manifest(
        tcga_manifest_path=tcga_manifest_path,
        tcga_predictions_csv=tcga_predictions_path,
        cptac_tumour_csv=cptac_tumours_path,
        cptac_normals_csv=cptac_normals_path,
        tcga_h5_dir=tcga_h5_dir,
        cptac_h5_dir=cptac_h5_dir,
        out_dir=out_dir,
        seed=42,
    )

    slide_manifest = pd.read_csv(out_dir / "tcga_cptac_combined_slide_manifest_v1.tsv", sep="\t")
    bag_manifest = pd.read_csv(out_dir / "tcga_cptac_combined_bag_manifest_v1.tsv", sep="\t")

    assert "cptac_drop" not in slide_manifest["file_uuid_original"].tolist()
    assert "cptac_drop_1" not in slide_manifest["file_uuid_original"].tolist()
    assert "cptac_norm_missing" not in slide_manifest["file_uuid_original"].tolist()
    assert "cptac_norm_orphan" not in slide_manifest["file_uuid_original"].tolist()

    kirp_row = slide_manifest.loc[slide_manifest["file_uuid_original"] == "cptac_kirp"].iloc[0]
    assert kirp_row["project_id"] == "TCGA-KIRP"
    assert kirp_row["source_dataset"] == "cptac_dx"

    conflict_row = slide_manifest.loc[slide_manifest["file_uuid_original"] == "cptac_conflict"].iloc[0]
    assert conflict_row["audit_reason"] == "conflicting_ngs_purity_values"

    normal_row = slide_manifest.loc[slide_manifest["file_uuid_original"] == "cptac_norm_keep"].iloc[0]
    tumour_fold = int(
        slide_manifest.loc[slide_manifest["case_id"] == "CPTAC-1", "preassigned_fold"]
        .dropna()
        .astype(int)
        .iloc[0]
    )
    assert int(normal_row["preassigned_fold"]) == tumour_fold
    assert float(normal_row["purity"]) == 0.0
    assert normal_row["gdc_match_type"] == "normal_tissue"

    assert metadata["tcga_tumour_bags"] == 1
    assert metadata["tcga_normal_bags"] == 1
    assert metadata["cptac_tumour_bags"] == 3
    assert metadata["cptac_normal_bags"] == 1
    assert metadata["expected_cache_bag_count"] == len(bag_manifest) == 6
    assert metadata["cptac_source_rows"] == 5
    assert metadata["cptac_source_legacy_fallback_rows"] == 1
    assert metadata["cptac_source_conflicting_rows"] == 1
    assert metadata["cptac_source_missing_slide_tree_rows"] == 0
    assert metadata["cptac_papillary_kirp_count"] == 1
    assert metadata["cptac_chromophobe_kich_count"] == 0
