"""Regenerate frontend/public/data from a precomputed predictions CSV.

This keeps the existing VM workflow intact and adds a local CSV-driven path.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--predictions-csv",
        type=Path,
        default=Path("logs/v3_allfolds_alltiles_predictions_191512.csv"),
    )
    ap.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/wedge_mvp_dataset.xlsx"),
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ml/runs/all_tiles_stats_from_csv"),
    )
    ap.add_argument(
        "--frontend-data-dir",
        type=Path,
        default=Path("frontend/public/data"),
    )
    ap.add_argument(
        "--pred-fold",
        type=int,
        default=None,
        help="Optional fold filter when using a multi-fold predictions CSV.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    ml_dir = repo_root / "ml"
    out_dir = (repo_root / args.out_dir).resolve()
    frontend_data_dir = (repo_root / args.frontend_data_dir).resolve()
    pred_csv = (repo_root / args.predictions_csv).resolve()
    manifest = (repo_root / args.manifest).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    frontend_data_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--manifest",
        str(manifest),
        "--predictions-csv",
        str(pred_csv),
        "--out-dir",
        str(out_dir),
    ]
    if args.pred_fold is not None:
        common.extend(["--pred-fold", str(args.pred_fold)])

    _run([sys.executable, "-m", "enso_purity_mil.statistical_tests", *common], cwd=ml_dir)
    _run([sys.executable, "-m", "enso_purity_mil.per_cancer_statistical_tests", *common], cwd=ml_dir)

    generated = [
        "statistical_tests.json",
        "per_cancer_stats.json",
        "scatter_data.json",
        "scatter_mil_vs_ptn.png",
        "error_difference_histogram.png",
    ]
    for name in generated:
        src = out_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing expected output: {src}")
        shutil.copy2(src, frontend_data_dir / name)

    print("Updated frontend/public/data from predictions CSV.")


if __name__ == "__main__":
    main()
