"""Build the slide ↔ ABSOLUTE purity dataset using both matching methods.

Usage
-----
    python -m scripts.build_purity_dataset \
        --slides-xlsx "data/raw/slides_metadata_report(1).xlsx" \
        --abs-tsv    "data/raw/TCGA_mastercalls.abs_tables_JSedit.fixed.txt" \
        --out-dir    data/processed \
        --reports-dir data/reports \
        [--skip-gdc]     # skip the live GDC API call (useful for CI)

Outputs
-------
  * ``data/processed/slides_metadata_purity_barcode.xlsx``   — Method 1 (vial-level match)
  * ``data/processed/slides_metadata_purity_barcode_portion.xlsx`` — Method 1 (portion-level)
  * ``data/processed/slides_metadata_purity_gdc.xlsx``       — Method 2 (GDC biospecimen)
  * ``data/reports/*.png``                                    — comparison plots
  * ``data/reports/linkage_stats.json``                       — machine-readable summary
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from enso_purity.data.slide_purity_matching import (
    build_merged_dataset,
    load_and_filter_slides,
    load_and_filter_absolute,
)
from enso_purity.data.gdc_linkage import match_slides_to_purity_via_gdc
from enso_purity.data.linkage_compare import save_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Build TCGA slide ↔ ABSOLUTE purity datasets (barcode + GDC methods).",
    )
    ap.add_argument(
        "--slides-xlsx", type=Path,
        default=Path("data/raw/slides_metadata_report(1).xlsx"),
        help="Path to the slides metadata Excel.",
    )
    ap.add_argument(
        "--abs-tsv", type=Path,
        default=Path("data/raw/TCGA_mastercalls.abs_tables_JSedit.fixed.txt"),
        help="Path to the ABSOLUTE purity TSV.",
    )
    ap.add_argument("--out-dir", type=Path, default=Path("data/processed"))
    ap.add_argument("--reports-dir", type=Path, default=Path("data/reports"))
    ap.add_argument(
        "--skip-gdc", action="store_true",
        help="Skip the live GDC API call (for CI or offline use).",
    )
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.reports_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────
    logger.info("Loading slides from %s", args.slides_xlsx)
    slides_raw = pd.read_excel(args.slides_xlsx)

    logger.info("Loading ABSOLUTE from %s", args.abs_tsv)
    abs_raw = pd.read_csv(args.abs_tsv, sep="\t")

    # ── Method 1: barcode matching ────────────────────────────────
    logger.info("=== Method 1: barcode-based matching ===")
    result = build_merged_dataset(slides_raw, abs_raw)
    stats = result["stats"]

    vial_path = args.out_dir / "slides_metadata_purity_barcode.xlsx"
    result["vial"].to_excel(vial_path, index=False, engine="openpyxl")
    logger.info("Wrote %s  (%d rows)", vial_path, len(result["vial"]))

    portion_path = args.out_dir / "slides_metadata_purity_barcode_portion.xlsx"
    result["portion"].to_excel(portion_path, index=False, engine="openpyxl")
    logger.info("Wrote %s  (%d rows)", portion_path, len(result["portion"]))

    # ── Method 2: GDC API ─────────────────────────────────────────
    gdc_merged: pd.DataFrame | None = None
    if not args.skip_gdc:
        logger.info("=== Method 2: GDC biospecimen matching ===")
        slides_filtered = load_and_filter_slides(slides_raw)
        try:
            gdc_merged = match_slides_to_purity_via_gdc(slides_filtered, abs_raw)
            gdc_path = args.out_dir / "slides_metadata_purity_gdc.xlsx"
            gdc_merged.to_excel(gdc_path, index=False, engine="openpyxl")
            logger.info("Wrote %s  (%d rows)", gdc_path, len(gdc_merged))
            stats["n_matched_gdc"] = len(gdc_merged)
        except Exception:
            logger.exception("GDC API matching failed — continuing without it")
    else:
        logger.info("Skipping GDC API (--skip-gdc)")

    # ── plots ─────────────────────────────────────────────────────
    logger.info("=== Generating plots ===")
    slides_filtered = load_and_filter_slides(slides_raw)
    plot_paths = save_all_plots(
        slides_filtered,
        stats,
        result["vial"],
        gdc_merged,
        out_dir=args.reports_dir,
    )
    for p in plot_paths:
        logger.info("  plot: %s", p)

    # ── stats JSON ────────────────────────────────────────────────
    stats_path = args.reports_dir / "linkage_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    logger.info("Wrote %s", stats_path)

    # ── summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("LINKAGE SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k:40s} {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
