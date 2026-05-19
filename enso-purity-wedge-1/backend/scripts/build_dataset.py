from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from enso_purity.data.naive_linkage import (
    build_naive_keys_from_abs,
    build_naive_keys_from_slides,
    naive_match_rates,
)


def main():
    ap = argparse.ArgumentParser(description="Build naive TCGA slide↔ABS linkage dataset")
    ap.add_argument("--slides-xlsx", type=Path, required=True)
    ap.add_argument("--abs-tsv", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("outputs"))
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    slides = pd.read_excel(args.slides_xlsx)
    # Parse slide_id from full_path
    slides["file_name"] = slides["full_path"].astype(str).str.split("/").str[-1]
    slides["slide_id"] = slides["file_name"].str.split(".").str[0]

    abs_df = pd.read_csv(args.abs_tsv, sep="\t")

    slides_k = build_naive_keys_from_slides(slides, slide_col="slide_id")
    abs_k = build_naive_keys_from_abs(abs_df, aliquot_col="sample")

    # Exclude DX
    slides_k = slides_k[~slides_k["is_dx"]].copy()

    rates = naive_match_rates(slides_k, abs_k)
    report_path = args.out_dir / "mismatch_report.json"
    report_path.write_text(pd.Series(rates).to_json(indent=2), encoding="utf-8")

    print("Naive match rates:")
    for k, v in rates.items():
        print(f"  {k}: {v:.4f}")

    # Placeholder dataset output
    ds_path = args.out_dir / "dataset.parquet"
    slides_k.head(100).to_parquet(ds_path, index=False)
    print(f"Wrote {ds_path} (placeholder head(100))")


if __name__ == "__main__":
    main()
