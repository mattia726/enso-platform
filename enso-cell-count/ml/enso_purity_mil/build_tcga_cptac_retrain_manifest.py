"""Build a combined TCGA FS + CPTAC DX retraining manifest."""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from enso_purity_mil.folds import generate_stratified_folds
from enso_purity_mil.manifest_io import load_manifest_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TCGA_MANIFEST = Path("data/processed/wedge_mvp_dataset.xlsx")
DEFAULT_TCGA_PREDICTIONS_CSV = Path("logs/v3_allfolds_alltiles_predictions_191512.csv")
DEFAULT_CPTAC_TUMOUR_CSV = Path("data/processed/cptac_slides_ngs_purity_final.csv")
DEFAULT_CPTAC_NORMALS_CSV = Path("data/processed/cptac_master_normals.csv")

SIMPLE_CPTAC_PROJECT_MAP = {
    "LUAD": "TCGA-LUAD",
    "LSCC": "TCGA-LUSC",
    "HNSCC": "TCGA-HNSC",
    "PDA": "TCGA-PAAD",
    "GBM": "TCGA-GBM",
    "UCEC": "TCGA-UCEC",
    "STAD": "TCGA-STAD",
}


def _require_columns(df: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _list_h5_stems(h5_dir: Path) -> set[str]:
    if not h5_dir.exists():
        raise FileNotFoundError(f"H5 directory does not exist: {h5_dir}")
    return {path.stem for path in h5_dir.glob("*.h5")}


def _first_non_empty(*values: object) -> str:
    for value in values:
        if pd.notna(value) and str(value).strip():
            return str(value)
    return ""


def _build_tcga_fold_map(predictions_csv: Path) -> dict[str, int]:
    pred = pd.read_csv(predictions_csv)
    if "model" in pred.columns:
        pred = pred[pred["model"] == "v3"].copy()
    _require_columns(pred, ["aliquot_barcode", "fold"], "TCGA predictions CSV")

    fold_df = pred[["aliquot_barcode", "fold"]].dropna().drop_duplicates().copy()
    dup = fold_df.groupby("aliquot_barcode")["fold"].nunique()
    leaking = dup[dup > 1]
    if not leaking.empty:
        raise ValueError(
            "Historical TCGA predictions contain aliquots assigned to multiple folds: "
            f"{leaking.index.tolist()[:10]}"
        )

    fold_df["fold"] = fold_df["fold"].astype(int)
    return dict(zip(fold_df["aliquot_barcode"], fold_df["fold"]))


def _canonical_cptac_project(row: pd.Series) -> str | None:
    tumor = str(row.get("Tumor", "")).strip()
    disease = str(row.get("Disease_Type", "")).strip()
    histology = str(row.get("Tumor_Histological_Type", "")).strip().lower()

    if tumor in SIMPLE_CPTAC_PROJECT_MAP:
        return SIMPLE_CPTAC_PROJECT_MAP[tumor]

    if tumor != "CCRCC":
        return None
    if disease == "Clear Cell Renal Cell Carcinoma":
        return "TCGA-KIRC"
    if disease != "Non-Clear Cell Renal Cell Carcinoma":
        return None
    if "chromophobe" in histology:
        return "TCGA-KICH"
    if "papillary" in histology and "urothelial" not in histology:
        return "TCGA-KIRP"
    return None


def _normalize_tcga_rows(
    tcga_manifest: pd.DataFrame,
    tcga_fold_map: dict[str, int],
    tcga_h5_stems: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(
        tcga_manifest,
        ["file_uuid_original", "barcode", "aliquot_barcode", "gdc_match_type", "purity", "case_id", "project_id"],
        "TCGA wedge manifest",
    )

    manifest = tcga_manifest[tcga_manifest["purity"].notna()].copy()
    tumour_df = manifest[manifest["gdc_match_type"] != "normal_tissue"].copy()
    normal_df = manifest[manifest["gdc_match_type"] == "normal_tissue"].copy()

    tumour_df = tumour_df[tumour_df["purity"] < 1.0].copy()
    tumour_df["preassigned_fold"] = tumour_df["aliquot_barcode"].map(tcga_fold_map)
    if tumour_df["preassigned_fold"].isna().any():
        missing = tumour_df.loc[tumour_df["preassigned_fold"].isna(), ["case_id", "aliquot_barcode"]]
        raise ValueError(
            "Some retained TCGA tumour rows do not have a historical fold assignment. "
            f"Examples:\n{missing.head(10).to_string(index=False)}"
        )

    case_fold_map = (
        tumour_df[["case_id", "preassigned_fold"]]
        .drop_duplicates()
        .groupby("case_id")["preassigned_fold"]
        .agg(lambda values: sorted(set(int(v) for v in values)))
    )
    leaking_cases = case_fold_map[case_fold_map.map(len) > 1]
    if not leaking_cases.empty:
        raise ValueError(
            "Some TCGA tumour cases appear in multiple historical folds. "
            f"Examples: {leaking_cases.index.tolist()[:10]}"
        )
    tcga_case_to_fold = {case_id: folds[0] for case_id, folds in case_fold_map.items()}

    missing_tcga_h5 = sorted(set(tumour_df["file_uuid_original"].astype(str)) - tcga_h5_stems)
    if missing_tcga_h5:
        raise FileNotFoundError(
            "Retained TCGA tumour rows are missing H5 embeddings. "
            f"Examples: {missing_tcga_h5[:10]}"
        )

    missing_tcga_normal_h5 = sorted(set(normal_df["file_uuid_original"].astype(str)) - tcga_h5_stems)
    if missing_tcga_normal_h5:
        raise FileNotFoundError(
            "TCGA normal rows are missing H5 embeddings. "
            f"Examples: {missing_tcga_normal_h5[:10]}"
        )

    tumour_rows = pd.DataFrame(
        {
            "file_uuid_original": tumour_df["file_uuid_original"].astype(str),
            "aliquot_barcode": tumour_df["aliquot_barcode"].astype(str),
            "case_id": tumour_df["case_id"].astype(str),
            "project_id": tumour_df["project_id"].astype(str),
            "purity": tumour_df["purity"].astype(float),
            "gdc_match_type": tumour_df["gdc_match_type"].astype(str),
            "preassigned_fold": tumour_df["preassigned_fold"].astype(int),
            "source_dataset": "tcga_fs",
            "source_slide_id": tumour_df["barcode"].astype(str),
            "source_file_id": tumour_df["file_uuid_original"].astype(str),
            "source_project_id": tumour_df["project_id"].astype(str),
            "canonical_project_id": tumour_df["project_id"].astype(str),
            "match_source": tumour_df["gdc_match_type"].astype(str),
            "audit_reason": "",
        }
    )

    normal_rows = pd.DataFrame(
        {
            "file_uuid_original": normal_df["file_uuid_original"].astype(str),
            "aliquot_barcode": normal_df["aliquot_barcode"].fillna("").astype(str),
            "case_id": normal_df["case_id"].astype(str),
            "project_id": normal_df["project_id"].astype(str),
            "purity": normal_df["purity"].astype(float),
            "gdc_match_type": normal_df["gdc_match_type"].astype(str),
            "preassigned_fold": normal_df["case_id"].map(tcga_case_to_fold),
            "source_dataset": "tcga_fs",
            "source_slide_id": normal_df["barcode"].astype(str),
            "source_file_id": normal_df["file_uuid_original"].astype(str),
            "source_project_id": normal_df["project_id"].astype(str),
            "canonical_project_id": normal_df["project_id"].astype(str),
            "match_source": normal_df["gdc_match_type"].astype(str),
            "audit_reason": "",
        }
    )
    return tumour_rows, normal_rows


def _build_cptac_tumour_rows(
    cptac_tumour_manifest: pd.DataFrame,
    cptac_h5_stems: set[str],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, str], dict[str, int]]:
    _require_columns(
        cptac_tumour_manifest,
        [
            "file_id",
            "Case_ID",
            "Slide_ID",
            "Tumor",
            "Disease_Type",
            "Tumor_Histological_Type",
            "genomic_purity_ngs_fraction",
            "project_id",
            "final_match_source",
        ],
        "CPTAC tumour manifest",
    )

    tumour_df = cptac_tumour_manifest.copy()
    tumour_df = tumour_df[tumour_df["genomic_purity_ngs_fraction"].notna()].copy()
    tumour_df = tumour_df[tumour_df["genomic_purity_ngs_fraction"] < 1.0].copy()
    tumour_df["canonical_project_id"] = tumour_df.apply(_canonical_cptac_project, axis=1)
    tumour_df = tumour_df[tumour_df["canonical_project_id"].notna()].copy()

    missing_h5 = sorted(set(tumour_df["file_id"].astype(str)) - cptac_h5_stems)
    if missing_h5:
        raise FileNotFoundError(
            "Retained CPTAC tumour rows are missing H5 embeddings. "
            f"Examples: {missing_h5[:10]}"
        )

    case_project_counts = tumour_df.groupby("Case_ID")["canonical_project_id"].nunique()
    ambiguous_cases = case_project_counts[case_project_counts > 1]
    if not ambiguous_cases.empty:
        raise ValueError(
            "Some CPTAC tumour cases map to multiple canonical projects. "
            f"Examples: {ambiguous_cases.index.tolist()[:10]}"
        )

    strat_df = pd.DataFrame(
        {
            "case_id": tumour_df["Case_ID"].astype(str).tolist(),
            "Slide_ID": tumour_df["Slide_ID"].astype(str).tolist(),
            "file_id": tumour_df["file_id"].astype(str).tolist(),
            "purity": tumour_df["genomic_purity_ngs_fraction"].astype(float).tolist(),
            "project_id": tumour_df["canonical_project_id"].astype(str).tolist(),
        }
    )
    folds = generate_stratified_folds(strat_df, n_folds=5, seed=seed, cancer_col="project_id")

    row_to_fold: dict[int, int] = {}
    for fold_id, indices in enumerate(folds):
        for idx in indices:
            row_to_fold[idx] = fold_id

    strat_df["preassigned_fold"] = strat_df.index.map(row_to_fold).astype(int)
    case_fold_counts = strat_df.groupby("case_id")["preassigned_fold"].nunique()
    leaking_cases = case_fold_counts[case_fold_counts > 1]
    if not leaking_cases.empty:
        raise ValueError(
            "CPTAC tumour stratification leaked cases across folds. "
            f"Examples: {leaking_cases.index.tolist()[:10]}"
        )

    case_to_fold = (
        strat_df[["case_id", "preassigned_fold"]]
        .drop_duplicates()
        .set_index("case_id")["preassigned_fold"]
        .astype(int)
        .to_dict()
    )
    case_to_project = (
        strat_df[["case_id", "project_id"]]
        .drop_duplicates()
        .set_index("case_id")["project_id"]
        .astype(str)
        .to_dict()
    )

    tumour_rows = pd.DataFrame(
        {
            "file_uuid_original": strat_df["file_id"].astype(str),
            "aliquot_barcode": "CPTAC__" + strat_df["Slide_ID"].astype(str),
            "case_id": strat_df["case_id"].astype(str),
            "project_id": strat_df["project_id"].astype(str),
            "purity": strat_df["purity"].astype(float),
            "gdc_match_type": "cptac_dx",
            "preassigned_fold": strat_df["preassigned_fold"].astype(int),
            "source_dataset": "cptac_dx",
            "source_slide_id": strat_df["Slide_ID"].astype(str),
            "source_file_id": strat_df["file_id"].astype(str),
            "source_project_id": tumour_df["project_id"].astype(str).reset_index(drop=True),
            "canonical_project_id": strat_df["project_id"].astype(str),
            "match_source": tumour_df["final_match_source"].astype(str).reset_index(drop=True),
            "audit_reason": tumour_df.apply(
                lambda row: _first_non_empty(row.get("audit_reason"), row.get("legacy_tree_audit_reason")),
                axis=1,
            ).reset_index(drop=True),
        }
    )

    renal_stats = {
        "papillary_kirp_count": int(
            (
                tumour_df["Tumor"].eq("CCRCC")
                & tumour_df["Disease_Type"].eq("Non-Clear Cell Renal Cell Carcinoma")
                & tumour_df["canonical_project_id"].eq("TCGA-KIRP")
            ).sum()
        ),
        "chromophobe_kich_count": int(
            (
                tumour_df["Tumor"].eq("CCRCC")
                & tumour_df["Disease_Type"].eq("Non-Clear Cell Renal Cell Carcinoma")
                & tumour_df["canonical_project_id"].eq("TCGA-KICH")
            ).sum()
        ),
    }
    return tumour_rows, case_to_fold, case_to_project, renal_stats


def _build_cptac_normal_rows(
    cptac_normal_manifest: pd.DataFrame,
    cptac_h5_stems: set[str],
    case_to_fold: dict[str, int],
    case_to_project: dict[str, str],
) -> pd.DataFrame:
    _require_columns(
        cptac_normal_manifest,
        ["file_id", "Case_ID", "Slide_ID", "normal_definition"],
        "CPTAC normals manifest",
    )

    normal_df = cptac_normal_manifest.copy()
    normal_df = normal_df[normal_df["file_id"].astype(str).isin(cptac_h5_stems)].copy()
    normal_df = normal_df[normal_df["Case_ID"].astype(str).isin(case_to_fold)].copy()

    if normal_df.empty:
        return pd.DataFrame(
            columns=[
                "file_uuid_original",
                "aliquot_barcode",
                "case_id",
                "project_id",
                "purity",
                "gdc_match_type",
                "preassigned_fold",
                "source_dataset",
                "source_slide_id",
                "source_file_id",
                "source_project_id",
                "canonical_project_id",
                "match_source",
                "audit_reason",
            ]
        )

    normal_df = normal_df.reset_index(drop=True)
    case_series = normal_df["Case_ID"].astype(str)
    project_series = case_series.map(case_to_project)
    fold_series = case_series.map(case_to_fold)
    if project_series.isna().any() or fold_series.isna().any():
        raise ValueError("CPTAC normals contain rows without tumour case anchors after filtering.")

    return pd.DataFrame(
        {
            "file_uuid_original": normal_df["file_id"].astype(str),
            "aliquot_barcode": "CPTAC_NORMAL__" + normal_df["Slide_ID"].astype(str),
            "case_id": case_series,
            "project_id": project_series.astype(str),
            "purity": 0.0,
            "gdc_match_type": "normal_tissue",
            "preassigned_fold": fold_series.astype(int),
            "source_dataset": "cptac_dx",
            "source_slide_id": normal_df["Slide_ID"].astype(str),
            "source_file_id": normal_df["file_id"].astype(str),
            "source_project_id": normal_df.apply(
                lambda row: _first_non_empty(
                    row.get("collection"),
                    row.get("Tumor"),
                    row.get("Disease_Type"),
                ),
                axis=1,
            ),
            "canonical_project_id": project_series.astype(str),
            "match_source": normal_df["normal_definition"].fillna("normal_tissue").astype(str),
            "audit_reason": "",
        }
    )


def _bag_id_for_row(row: pd.Series) -> str:
    if str(row["gdc_match_type"]) == "normal_tissue":
        return f"normal_{row['file_uuid_original']}"
    return f"tumor_{row['aliquot_barcode']}"


def _build_bag_manifest(slide_manifest: pd.DataFrame) -> pd.DataFrame:
    df = slide_manifest.copy()
    df["bag_id"] = df.apply(_bag_id_for_row, axis=1)
    grouped = df.groupby("bag_id", sort=True)

    bag_rows = []
    for bag_id, sub in grouped:
        bag_rows.append(
            {
                "bag_id": bag_id,
                "case_id": str(sub["case_id"].iloc[0]),
                "project_id": str(sub["project_id"].iloc[0]),
                "source_dataset": str(sub["source_dataset"].iloc[0]),
                "gdc_match_type": str(sub["gdc_match_type"].iloc[0]),
                "preassigned_fold": sub["preassigned_fold"].iloc[0],
                "purity": float(sub["purity"].iloc[0]),
                "file_count": int(len(sub)),
                "file_uuid_originals": ";".join(sub["file_uuid_original"].astype(str).tolist()),
                "source_slide_ids": ";".join(sub["source_slide_id"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(bag_rows).sort_values(["source_dataset", "gdc_match_type", "case_id", "bag_id"]).reset_index(drop=True)


def _build_fold_summary(slide_manifest: pd.DataFrame, bag_manifest: pd.DataFrame) -> pd.DataFrame:
    bag_df = bag_manifest.copy()
    bag_df["is_tumor"] = bag_df["gdc_match_type"] != "normal_tissue"
    summary = (
        bag_df.groupby(["source_dataset", "gdc_match_type", "preassigned_fold"], dropna=False)
        .agg(
            bag_count=("bag_id", "nunique"),
            patient_count=("case_id", "nunique"),
            mean_purity=("purity", "mean"),
        )
        .reset_index()
        .sort_values(["source_dataset", "gdc_match_type", "preassigned_fold"])
        .reset_index(drop=True)
    )
    return summary


def build_combined_manifest(
    *,
    tcga_manifest_path: Path,
    tcga_predictions_csv: Path,
    cptac_tumour_csv: Path,
    cptac_normals_csv: Path,
    tcga_h5_dir: Path,
    cptac_h5_dir: Path,
    out_dir: Path,
    seed: int = 42,
) -> dict[str, object]:
    tcga_manifest = load_manifest_table(tcga_manifest_path)
    cptac_tumour = load_manifest_table(cptac_tumour_csv)
    cptac_normals = load_manifest_table(cptac_normals_csv)

    tcga_h5_stems = _list_h5_stems(tcga_h5_dir)
    cptac_h5_stems = _list_h5_stems(cptac_h5_dir)
    tcga_fold_map = _build_tcga_fold_map(tcga_predictions_csv)

    tcga_tumour_rows, tcga_normal_rows = _normalize_tcga_rows(tcga_manifest, tcga_fold_map, tcga_h5_stems)
    cptac_tumour_rows, case_to_fold, case_to_project, renal_stats = _build_cptac_tumour_rows(
        cptac_tumour, cptac_h5_stems, seed=seed
    )
    cptac_normal_rows = _build_cptac_normal_rows(cptac_normals, cptac_h5_stems, case_to_fold, case_to_project)

    slide_manifest = pd.concat(
        [tcga_tumour_rows, tcga_normal_rows, cptac_tumour_rows, cptac_normal_rows],
        ignore_index=True,
    )
    slide_manifest["bag_id"] = slide_manifest.apply(_bag_id_for_row, axis=1)
    slide_manifest = slide_manifest.sort_values(
        ["source_dataset", "gdc_match_type", "case_id", "aliquot_barcode", "file_uuid_original"]
    ).reset_index(drop=True)

    # Coverage validation for retained rows.
    missing_tcga = slide_manifest.loc[
        slide_manifest["source_dataset"].eq("tcga_fs")
        & ~slide_manifest["file_uuid_original"].isin(tcga_h5_stems),
        "file_uuid_original",
    ]
    missing_cptac = slide_manifest.loc[
        slide_manifest["source_dataset"].eq("cptac_dx")
        & ~slide_manifest["file_uuid_original"].isin(cptac_h5_stems),
        "file_uuid_original",
    ]
    if not missing_tcga.empty or not missing_cptac.empty:
        raise FileNotFoundError(
            "Combined manifest contains retained rows without backing H5 files. "
            f"Missing TCGA={sorted(set(missing_tcga))[:5]} Missing CPTAC={sorted(set(missing_cptac))[:5]}"
        )

    bag_manifest = _build_bag_manifest(slide_manifest)
    fold_summary = _build_fold_summary(slide_manifest, bag_manifest)

    out_dir.mkdir(parents=True, exist_ok=True)
    slide_path = out_dir / "tcga_cptac_combined_slide_manifest_v1.tsv"
    bag_path = out_dir / "tcga_cptac_combined_bag_manifest_v1.tsv"
    fold_path = out_dir / "tcga_cptac_fold_summary_v1.tsv"
    meta_path = out_dir / "tcga_cptac_combined_metadata_v1.json"

    slide_manifest.to_csv(slide_path, sep="\t", index=False)
    bag_manifest.to_csv(bag_path, sep="\t", index=False)
    fold_summary.to_csv(fold_path, sep="\t", index=False)

    tumour_bag_count = int((bag_manifest["gdc_match_type"] != "normal_tissue").sum())
    normal_bag_count = int((bag_manifest["gdc_match_type"] == "normal_tissue").sum())
    cptac_audit_series = cptac_tumour.get("audit_reason")
    if cptac_audit_series is None:
        cptac_audit_series = pd.Series([""] * len(cptac_tumour), index=cptac_tumour.index)
    legacy_audit_series = cptac_tumour.get("legacy_tree_audit_reason")
    if legacy_audit_series is None:
        legacy_audit_series = pd.Series([""] * len(cptac_tumour), index=cptac_tumour.index)
    cptac_audit_series = cptac_audit_series.fillna(legacy_audit_series).fillna("").astype(str)

    metadata = {
        "seed": seed,
        "slide_manifest": str(slide_path),
        "bag_manifest": str(bag_path),
        "fold_summary": str(fold_path),
        "tumour_bag_count": tumour_bag_count,
        "normal_bag_count": normal_bag_count,
        "expected_cache_bag_count": int(len(bag_manifest)),
        "tcga_tumour_bags": int((bag_manifest["source_dataset"].eq("tcga_fs") & bag_manifest["gdc_match_type"].ne("normal_tissue")).sum()),
        "tcga_normal_bags": int((bag_manifest["source_dataset"].eq("tcga_fs") & bag_manifest["gdc_match_type"].eq("normal_tissue")).sum()),
        "cptac_tumour_bags": int((bag_manifest["source_dataset"].eq("cptac_dx") & bag_manifest["gdc_match_type"].ne("normal_tissue")).sum()),
        "cptac_normal_bags": int((bag_manifest["source_dataset"].eq("cptac_dx") & bag_manifest["gdc_match_type"].eq("normal_tissue")).sum()),
        "cptac_source_rows": int(len(cptac_tumour)),
        "cptac_source_legacy_fallback_rows": int(
            cptac_tumour["final_match_source"].fillna("").astype(str).eq("legacy_sample_specific_only").sum()
        ),
        "cptac_source_conflicting_rows": int(
            cptac_audit_series.eq("conflicting_ngs_purity_values").sum()
        ),
        "cptac_source_missing_slide_tree_rows": int(
            cptac_audit_series.eq("missing_slide_tree").sum()
        ),
        "cptac_papillary_kirp_count": renal_stats["papillary_kirp_count"],
        "cptac_chromophobe_kich_count": renal_stats["chromophobe_kich_count"],
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Build combined TCGA FS + CPTAC DX retraining manifests.")
    ap.add_argument("--tcga-manifest", type=Path, default=DEFAULT_TCGA_MANIFEST)
    ap.add_argument(
        "--tcga-predictions-csv",
        type=Path,
        default=DEFAULT_TCGA_PREDICTIONS_CSV,
    )
    ap.add_argument(
        "--cptac-tumour-csv",
        type=Path,
        default=DEFAULT_CPTAC_TUMOUR_CSV,
    )
    ap.add_argument(
        "--cptac-normals-csv",
        type=Path,
        default=DEFAULT_CPTAC_NORMALS_CSV,
    )
    ap.add_argument("--tcga-h5-dir", type=Path, required=True)
    ap.add_argument("--cptac-h5-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()

    metadata = build_combined_manifest(
        tcga_manifest_path=args.tcga_manifest,
        tcga_predictions_csv=args.tcga_predictions_csv,
        cptac_tumour_csv=args.cptac_tumour_csv,
        cptac_normals_csv=args.cptac_normals_csv,
        tcga_h5_dir=args.tcga_h5_dir,
        cptac_h5_dir=args.cptac_h5_dir,
        out_dir=args.out_dir,
        seed=args.seed,
    )
    logger.info("Wrote combined retrain manifests to %s", args.out_dir)
    logger.info(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
