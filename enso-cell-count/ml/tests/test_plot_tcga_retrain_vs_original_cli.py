from __future__ import annotations

import json
import sys

import pandas as pd
import pytest

from enso_purity_mil.plot_tcga_retrain_vs_original_cli import (
    _binned_mae,
    _build_comparison_frames,
    _load_cptac_rows,
    _validate_same_bag_universe,
)


def test_build_comparison_frames_accepts_excel_inputs(tmp_path) -> None:
    original_df = pd.DataFrame(
        {
            "fold": [0, 0, 1, 2],
            "bag_id": [
                "tumor_TCGA-A",
                "tumor_TCGA-B",
                "tumor_TCGA-C",
                "tumor_TCGA-D",
            ],
            "true_purity": [0.25, 1.0, 0.65, 0.45],
            "predicted_purity": [0.30, 0.95, 0.60, 0.50],
        }
    )
    retrain_df = pd.DataFrame(
        {
            "fold": [0, 0, 1, 2, 1],
            "bag_id": [
                "tumor_TCGA-A",
                "tumor_TCGA-B",
                "tumor_TCGA-C",
                "tumor_TCGA-D",
                "tumor_CPTAC-X",
            ],
            "true_purity": [0.25, 1.0, 0.65, 0.45, 0.35],
            "predicted_purity": [0.28, 0.91, 0.63, 0.52, 0.40],
        }
    )

    original_path = tmp_path / "original.xlsx"
    retrain_path = tmp_path / "retrain.xlsx"
    with pd.ExcelWriter(original_path, engine="openpyxl") as writer:
        original_df.to_excel(writer, sheet_name="all_available_folds", index=False)
    with pd.ExcelWriter(retrain_path, engine="openpyxl") as writer:
        retrain_df.to_excel(writer, sheet_name="all_available_folds", index=False)

    orig_plot_df, retrain_plot_df, metadata, orig_label, retrain_label, merged = _build_comparison_frames(
        mode="bag_matched_tcga",
        new_predictions_csv=retrain_path,
        original_predictions_csv=original_path,
        folds=[0, 1, 2, 3],
        exclude_purity_ge=1.0,
        new_sheet="all_available_folds",
        original_sheet="all_available_folds",
    )

    assert orig_label == "original_v3"
    assert retrain_label == "retrain_tcga_subset"
    assert metadata["mode"] == "bag_matched_tcga"
    assert metadata["comparison_contract"] == "bag_matched_tcga_purity_lt_1"
    assert metadata["unit_label"] == "bags"
    assert metadata["new_sheet"] == "all_available_folds"
    assert metadata["original_sheet"] == "all_available_folds"
    assert merged is not None
    assert merged["bag_id"].tolist() == ["tumor_TCGA-A", "tumor_TCGA-C", "tumor_TCGA-D"]
    assert orig_plot_df["bag_id"].tolist() == ["tumor_TCGA-A", "tumor_TCGA-C", "tumor_TCGA-D"]
    assert retrain_plot_df["bag_id"].tolist() == ["tumor_TCGA-A", "tumor_TCGA-C", "tumor_TCGA-D"]
    assert "tumor_CPTAC-X" not in retrain_plot_df["bag_id"].tolist()


def test_load_cptac_rows_filters_to_cptac_only(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "fold": [0, 1, 2, 2],
            "bag_id": ["tumor_TCGA-A", "tumor_CPTAC__X", "tumor_CPTAC__Y", "tumor_CPTAC__Z"],
            "true_purity": [0.4, 0.2, 0.7, 1.0],
            "predicted_purity": [0.41, 0.25, 0.65, 0.9],
        }
    )
    path = tmp_path / "retrain.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="all_available_folds", index=False)

    cptac_df = _load_cptac_rows(
        path,
        folds=[0, 1, 2, 3],
        exclude_purity_ge=1.0,
        sheet_name="all_available_folds",
    )

    assert cptac_df["bag_id"].tolist() == ["tumor_CPTAC__X", "tumor_CPTAC__Y"]


def test_validate_same_bag_universe_rejects_mismatched_cptac_sets() -> None:
    ref = pd.DataFrame(
        {
            "fold": [0, 1],
            "bag_id": ["tumor_CPTAC__X", "tumor_CPTAC__Y"],
            "true_purity": [0.2, 0.7],
            "pred_purity": [0.25, 0.65],
        }
    )
    cand = pd.DataFrame(
        {
            "fold": [0, 1],
            "bag_id": ["tumor_CPTAC__X", "tumor_CPTAC__Z"],
            "true_purity": [0.2, 0.7],
            "pred_purity": [0.24, 0.63],
        }
    )

    with pytest.raises(ValueError, match="same bag universe"):
        _validate_same_bag_universe(
            ref,
            cand,
            reference_label="retrain CPTAC test rows",
            candidate_label="original-v3 CPTAC test rows",
        )


def test_binned_mae_uses_fixed_purity_bins() -> None:
    df = pd.DataFrame(
        {
            "true_purity": [0.02, 0.08, 0.11, 0.18, 0.91],
            "pred_purity": [0.12, 0.03, 0.10, 0.28, 0.81],
        }
    )

    bins = _binned_mae(df, bin_width=0.1)

    assert bins["bin_left"].tolist() == [0.0, 0.1, 0.9]
    assert bins["count"].tolist() == [2, 2, 1]
    assert bins["mae"].tolist() == pytest.approx([0.075, 0.055, 0.10], abs=1e-8)


def test_main_writes_audit_outputs_with_explicit_contracts(monkeypatch, tmp_path) -> None:
    from enso_purity_mil import plot_tcga_retrain_vs_original_cli as mod

    original_df = pd.DataFrame(
        {
            "fold": [0, 0, 1, 2],
            "bag_id": [
                "tumor_TCGA-A",
                "tumor_TCGA-B",
                "tumor_TCGA-C",
                "tumor_TCGA-D",
            ],
            "case_id": ["TCGA-A", "TCGA-B", "TCGA-C", "TCGA-D"],
            "true_purity": [0.25, 1.0, 0.65, 0.45],
            "predicted_purity": [0.30, 0.95, 0.60, 0.50],
        }
    )
    retrain_df = pd.DataFrame(
        {
            "fold": [0, 0, 1, 2, 1, 2],
            "bag_id": [
                "tumor_TCGA-A",
                "tumor_TCGA-B",
                "tumor_TCGA-C",
                "tumor_TCGA-D",
                "tumor_CPTAC__X",
                "tumor_CPTAC__Y",
            ],
            "true_purity": [0.25, 1.0, 0.65, 0.45, 0.35, 0.55],
            "predicted_purity": [0.28, 0.91, 0.63, 0.52, 0.40, 0.58],
        }
    )
    original_cptac_df = pd.DataFrame(
        {
            "fold": [1, 2],
            "bag_id": ["tumor_CPTAC__X", "tumor_CPTAC__Y"],
            "true_purity": [0.35, 0.55],
            "predicted_purity": [0.37, 0.59],
        }
    )

    original_path = tmp_path / "original.xlsx"
    retrain_path = tmp_path / "retrain.xlsx"
    original_cptac_path = tmp_path / "original_cptac.xlsx"
    with pd.ExcelWriter(original_path, engine="openpyxl") as writer:
        original_df.to_excel(writer, sheet_name="all_available_folds", index=False)
    with pd.ExcelWriter(retrain_path, engine="openpyxl") as writer:
        retrain_df.to_excel(writer, sheet_name="all_available_folds", index=False)
    with pd.ExcelWriter(original_cptac_path, engine="openpyxl") as writer:
        original_cptac_df.to_excel(writer, sheet_name="all_available_folds", index=False)

    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_tcga_retrain_vs_original_cli.py",
            "--new-predictions-csv",
            str(retrain_path),
            "--new-sheet",
            "all_available_folds",
            "--original-predictions-csv",
            str(original_path),
            "--original-sheet",
            "all_available_folds",
            "--audit-raw-original-predictions-csv",
            str(original_path),
            "--audit-raw-original-sheet",
            "all_available_folds",
            "--include-cptac-row",
            "--include-original-cptac-row",
            "--include-merged-test-rows",
            "--write-overlay-mae-pairs",
            "--original-cptac-predictions-csv",
            str(original_cptac_path),
            "--original-cptac-sheet",
            "all_available_folds",
            "--expected-raw-original-bags",
            "4",
            "--expected-raw-original-cases",
            "4",
            "--expected-filtered-original-bags",
            "3",
            "--expected-filtered-original-cases",
            "3",
            "--expected-retrain-tcga-bags",
            "3",
            "--expected-matched-bags",
            "3",
            "--expected-cptac-bags",
            "2",
            "--out-dir",
            str(out_dir),
        ],
    )

    mod.main()

    metrics = json.loads((out_dir / "tcga_folds_0_3_retrain_vs_original_metrics.json").read_text())
    assert metrics["comparison_contract"] == "bag_matched_tcga_purity_lt_1"
    assert metrics["unit_label"] == "bags"
    assert metrics["audit"]["raw_original_bags"] == 4
    assert metrics["audit"]["filtered_original_bags"] == 3
    assert metrics["audit"]["matched_bags"] == 3
    assert metrics["audit"]["cptac_retrain_bags"] == 2
    assert metrics["original_v3"]["comparison_contract"] == "bag_matched_tcga_purity_lt_1"
    assert metrics["retrain_tcga_subset"]["comparison_contract"] == "bag_matched_tcga_purity_lt_1"
    assert metrics["original_v3_cptac_test_only"]["comparison_contract"] == "current_fold_specific_cptac_tests"
    assert metrics["original_v3_tcga_cptac_test_only"]["comparison_contract"] == "merged_tcga_cptac_test_bags"
    assert metrics["retrain_tcga_cptac_test_only"]["comparison_contract"] == "merged_tcga_cptac_test_bags"
    assert metrics["original_v3_tcga_cptac_test_only"]["n_items"] == 5
    assert metrics["retrain_tcga_cptac_test_only"]["n_items"] == 5

    audit_tsv = pd.read_csv(
        out_dir / "tcga_folds_0_3_retrain_vs_original_audit.tsv",
        sep="\t",
    )
    assert audit_tsv.loc[0, "raw_original_bags"] == 4
    assert audit_tsv.loc[0, "matched_bags"] == 3

    metrics_tsv = pd.read_csv(
        out_dir / "tcga_folds_0_3_retrain_vs_original_metrics.tsv",
        sep="\t",
    )
    assert "comparison_contract" in metrics_tsv.columns
    assert set(metrics_tsv["comparison_contract"]) == {
        "bag_matched_tcga_purity_lt_1",
        "current_fold_specific_cptac_tests",
        "merged_tcga_cptac_test_bags",
    }

    audit_note = (out_dir / "tcga_folds_0_3_retrain_vs_original_audit_note.txt").read_text()
    assert "same current CPTAC test bags" in audit_note
    assert "rho ~0.3" in audit_note

    overlay_pairs = pd.read_csv(out_dir / "tcga_folds_0_3_retrain_vs_original_overlay_mae_pairs.csv")
    assert set(overlay_pairs["dataset"]) == {
        "tcga_matched",
        "cptac_current_test",
        "merged_tcga_cptac",
    }
    assert (out_dir / "tcga_folds_0_3_retrain_vs_original_overlay_mae_pairs.png").exists()
    overlay_binned = pd.read_csv(
        out_dir / "tcga_folds_0_3_retrain_vs_original_overlay_mae_pairs_fixed_bins.csv"
    )
    assert set(overlay_binned["dataset"]) == {
        "tcga_matched",
        "cptac_current_test",
        "merged_tcga_cptac",
    }
    assert set(overlay_binned["method"]) == {"original", "retrain"}
    assert (out_dir / "tcga_folds_0_3_retrain_vs_original_overlay_mae_pairs_fixed_bins.png").exists()


def test_main_requires_original_cptac_source_when_requested(monkeypatch, tmp_path) -> None:
    from enso_purity_mil import plot_tcga_retrain_vs_original_cli as mod

    csv_path = tmp_path / "rows.csv"
    pd.DataFrame(
        {
            "fold": [0],
            "bag_id": ["tumor_TCGA-A"],
            "true_purity": [0.4],
            "pred_purity": [0.5],
        }
    ).to_csv(csv_path, index=False)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_tcga_retrain_vs_original_cli.py",
            "--new-predictions-csv",
            str(csv_path),
            "--original-predictions-csv",
            str(csv_path),
            "--out-dir",
            str(tmp_path / "out"),
            "--include-original-cptac-row",
        ],
    )

    with pytest.raises(ValueError, match="original-cptac-predictions-csv"):
        mod.main()
