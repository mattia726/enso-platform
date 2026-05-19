"""Build a unified H5 namespace from TCGA and CPTAC embedding stores."""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import pandas as pd

from enso_purity_mil.manifest_io import load_manifest_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def build_union_h5_namespace(
    *,
    manifest_path: Path,
    tcga_h5_dir: Path,
    cptac_h5_dir: Path,
    out_dir: Path,
    clean: bool = False,
) -> dict[str, int]:
    manifest = load_manifest_table(manifest_path)
    required = {"file_uuid_original", "source_dataset"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    if clean and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unique_files = manifest[["file_uuid_original", "source_dataset"]].drop_duplicates().copy()
    source_counts = unique_files.groupby("file_uuid_original")["source_dataset"].nunique()
    collisions = source_counts[source_counts > 1]
    if not collisions.empty:
        raise ValueError(
            "The same H5 stem appears under multiple source datasets. "
            f"Examples: {collisions.index.tolist()[:10]}"
        )

    linked = 0
    for row in unique_files.itertuples(index=False):
        file_id = str(row.file_uuid_original)
        dataset = str(row.source_dataset)
        if dataset == "tcga_fs":
            source_root = tcga_h5_dir
        elif dataset == "cptac_dx":
            source_root = cptac_h5_dir
        else:
            raise ValueError(f"Unsupported source_dataset in manifest: {dataset}")

        source = source_root / f"{file_id}.h5"
        if not source.exists():
            raise FileNotFoundError(f"Missing source H5: {source}")

        dest = out_dir / f"{file_id}.h5"
        if dest.exists():
            if dest.is_symlink() and dest.resolve() == source.resolve():
                continue
            raise FileExistsError(f"Destination already exists and does not match source: {dest}")
        dest.symlink_to(source)
        linked += 1

    summary = {
        "linked_h5_count": linked,
        "total_manifest_unique_h5": int(len(unique_files)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="Create a unified H5 namespace for TCGA+CPTAC retraining.")
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--tcga-h5-dir", type=Path, required=True)
    ap.add_argument("--cptac-h5-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    summary = build_union_h5_namespace(
        manifest_path=args.manifest,
        tcga_h5_dir=args.tcga_h5_dir,
        cptac_h5_dir=args.cptac_h5_dir,
        out_dir=args.out_dir,
        clean=args.clean,
    )
    logger.info(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
