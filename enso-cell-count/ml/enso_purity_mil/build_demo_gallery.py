"""Build a curated gallery for the investor/doctor demo.

Two modes:
  - Default: n_cases slides with diverse cancer types and purity; prefer model beats pathologist.
  - One-per-cancer (--one-per-cancer): one slide per project_id (target 32–34 types);
    prefer slides where Enso is closer than pathologist (err_mil < err_ptn) within --err-limit.
    Use --exclude-markers to pass a file listing file_uuid_original or barcode (one per line)
    to exclude slides that contain markers (identify manually).

For each case: fetches GDC thumbnail, generates interactive HTML viewer,
saves gallery_summary.csv for the Next.js frontend.

Usage:
    python -m enso_purity_mil.build_demo_gallery \
        --model-path ml/runs/fold0/best_model.pth \
        --manifest data/processed/wedge_mvp_dataset.xlsx \
        --h5-dir ~/bucket_embeddings/embeddings_fp32 \
        --cache-dir ~/enso_workspace/data/cache \
        --out-dir frontend/gallery --n-cases 50

    # One slide per cancer type (32–34), closer than pathologist, exclude marker slides:
    python -m enso_purity_mil.build_demo_gallery ... --one-per-cancer \\
        --err-limit 0.15 --exclude-markers data/exclude_markers.txt
"""
from __future__ import annotations

import argparse
import csv
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from enso_purity_mil.dataset import EmbeddingBagDataset, custom_collate_fn
from enso_purity_mil.folds import generate_stratified_folds
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/wedge_mvp_dataset.xlsx"))
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("frontend/gallery"))
    ap.add_argument("--n-cases", type=int, default=50)
    ap.add_argument("--one-per-cancer", action="store_true", help="Select one slide per project_id (target 32–34 types); prefer closer than pathologist")
    ap.add_argument("--err-limit", type=float, default=0.15, help="Max allowed |predicted - expected| for selection when using --one-per-cancer")
    ap.add_argument("--exclude-markers", type=Path, default=None, help="Text file: one file_uuid_original or barcode per line to exclude (e.g. slides with markers)")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Expand all paths so ~ and relative paths resolve (fixes "every viewer failed" when passing ~/bucket_...)
    args.model_path = Path(args.model_path).expanduser().resolve()
    args.manifest = Path(args.manifest).expanduser().resolve()
    args.h5_dir = Path(args.h5_dir).expanduser().resolve()
    args.cache_dir = Path(args.cache_dir).expanduser().resolve()
    args.out_dir = Path(args.out_dir).expanduser().resolve()
    if args.exclude_markers is not None:
        args.exclude_markers = Path(args.exclude_markers).expanduser().resolve()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest: use default or fallback to wedge_mvp_dataset(1).xlsx if present (e.g. in data/processed)
    if not args.manifest.exists():
        alt = args.manifest.parent / "wedge_mvp_dataset(1).xlsx"
        if alt.exists():
            args.manifest = alt
            logger.info("Using manifest: %s", args.manifest)
        else:
            raise FileNotFoundError(f"Manifest not found: {args.manifest} (tried {alt})")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Load data ────────────────────────────────────────────────
    manifest = pd.read_excel(args.manifest)
    manifest = manifest[manifest["purity"].notna()].copy()

    tumour_df = manifest[manifest["gdc_match_type"] != "normal_tissue"].copy().reset_index(drop=True)
    folds = generate_stratified_folds(tumour_df, n_folds=5, seed=args.seed, cancer_col="project_id")
    test_indices = folds[args.fold]
    test_df = tumour_df.iloc[test_indices].reset_index(drop=True)

    # Filter to slides with PTN
    valid = test_df[test_df["percent_tumor_nuclei"].notna()].copy()
    logger.info("Test slides with PTN: %d", len(valid))

    # ── Load model and predict per-aliquot ───────────────────────
    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    model = EnsoMILModel(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    ds = EmbeddingBagDataset(valid, args.h5_dir, num_instances=4096, cache_dir=args.cache_dir)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                          num_workers=args.num_workers, collate_fn=custom_collate_fn,
                                          pin_memory=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels in loader:
            preds = torch.clamp(model(feats.to(args.device)).squeeze(-1), 0, 1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.tolist())

    # Map predictions back to aliquots
    aliquot_preds = {}
    for i, g in enumerate(ds.groups):
        if i < len(all_preds):
            aliquot_preds[g["bag_id"]] = all_preds[i]

    # Build per-slide prediction table
    rows = []
    for _, slide in valid.iterrows():
        aliquot = slide["aliquot_barcode"]
        bag_id = f"tumor_{aliquot}"
        pred = aliquot_preds.get(bag_id, float("nan"))
        if np.isnan(pred):
            continue
        ptn = slide["percent_tumor_nuclei"] / 100.0
        expected = slide["purity"]
        rows.append({
            "file_uuid_original": slide["file_uuid_original"],
            "file_uuid_new": slide.get("file_uuid_new", None),
            "barcode": slide["barcode"],
            "project_id": slide["project_id"],
            "aliquot_barcode": aliquot,
            "expected": expected,
            "predicted": pred,
            "ptn": ptn,
            "err_mil": abs(expected - pred),
            "err_ptn": abs(expected - ptn),
            "area": slide.get("area", 0),
        })

    df = pd.DataFrame(rows)
    logger.info("Slides with predictions: %d", len(df))

    # Exclude marker slides if list provided
    exclude_set = set()
    if args.exclude_markers and args.exclude_markers.exists():
        for line in args.exclude_markers.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                exclude_set.add(line)
        df = df[~df["file_uuid_original"].isin(exclude_set) & ~df["barcode"].isin(exclude_set)].copy()
        logger.info("After excluding %d marker slide(s): %d slides", len(exclude_set), len(df))

    if args.one_per_cancer:
        # One slide per project_id: prefer err_mil < err_ptn and err_mil < err_limit
        within_limit = df[df["err_mil"] <= args.err_limit].copy()
        within_limit["closer_than_ptn"] = within_limit["err_mil"] < within_limit["err_ptn"]
        # Per project: pick best (prefer closer than PTN, then smallest err_mil)
        selected = []
        for project_id, grp in within_limit.groupby("project_id"):
            best = grp.sort_values(by=["closer_than_ptn", "err_mil"], ascending=[False, True]).iloc[0]
            selected.append(best)
        # If some projects have no slide within err_limit, add best available per project
        used_projects = {s["project_id"] for s in selected}
        for project_id in df["project_id"].unique():
            if project_id in used_projects:
                continue
            fallback = df[df["project_id"] == project_id].sort_values("err_mil").iloc[0]
            selected.append(fallback)
        gallery = pd.DataFrame(selected).reset_index(drop=True)
        gallery = gallery.sort_values("expected").reset_index(drop=True)
        logger.info("One-per-cancer gallery: %d cases across %d cancer types", len(gallery), gallery["project_id"].nunique())
    else:
        # Original logic: stunning filter and diverse bins
        stunning = df[(df["err_mil"] < 0.15) & (df["err_mil"] < df["err_ptn"])].copy()
        logger.info("Stunning (MIL err<0.15 & beats PTN): %d slides", len(stunning))
        stunning["advantage"] = stunning["err_ptn"] - stunning["err_mil"]
        stunning["purity_bin"] = (stunning["expected"] * 10).astype(int).clip(0, 9)
        selected = []
        target_per_bin = max(3, args.n_cases // 10)
        for pbin in range(10):
            bin_df = stunning[stunning["purity_bin"] == pbin].sort_values("advantage", ascending=False)
            if len(bin_df) == 0:
                continue
            used_projects_in_bin = set()
            count = 0
            for _, row in bin_df.iterrows():
                if count >= target_per_bin:
                    break
                if row.name in [s.name for s in selected]:
                    continue
                if row["project_id"] not in used_projects_in_bin or count < 2:
                    selected.append(row)
                    used_projects_in_bin.add(row["project_id"])
                    count += 1
        if len(selected) < args.n_cases:
            remaining = stunning[~stunning.index.isin([s.name for s in selected])]
            remaining = remaining.sort_values("advantage", ascending=False)
            for _, row in remaining.iterrows():
                if len(selected) >= args.n_cases:
                    break
                selected.append(row)
        gallery = pd.DataFrame(selected).reset_index(drop=True)
        gallery = gallery.sort_values("expected").reset_index(drop=True)
    logger.info("Gallery: %d cases across %d cancer types, purity range %.2f–%.2f",
                len(gallery), gallery["project_id"].nunique(),
                gallery["expected"].min(), gallery["expected"].max())

    # ── Generate interactive viewers ─────────────────────────────
    # H5 path: prefer file_uuid_original if it exists in embeddings_fp32, else file_uuid_new.
    # SVS UUID: pass original and fallback so viewer can fetch GDC thumbnail (original may 404 after rehost).
    # Output HTML: interactive_<file_uuid_original>.html so frontend's first HEAD request finds it.
    fail_dir = args.out_dir / "viewer_failures"
    for idx, row in gallery.iterrows():
        orig_h5 = args.h5_dir / f"{row['file_uuid_original']}.h5"
        new_h5 = (args.h5_dir / f"{row['file_uuid_new']}.h5") if pd.notna(row.get("file_uuid_new")) and row.get("file_uuid_new") else None
        if orig_h5.exists():
            h5_path = orig_h5
        elif new_h5 and new_h5.exists():
            h5_path = new_h5
        else:
            h5_path = orig_h5  # viewer will fail with clear FileNotFoundError

        logger.info("[%d/%d] %s %s purity=%.2f pred=%.2f ptn=%.2f",
                     idx + 1, len(gallery), row["project_id"], row["barcode"],
                     row["expected"], row["predicted"], row["ptn"])

        cmd = [
            sys.executable, "-m", "enso_purity_mil.interactive_viewer",
            "--model-path", str(args.model_path),
            "--h5-path", str(h5_path),
            "--expected", str(row["expected"]),
            "--predicted", str(row["predicted"]),
            "--out-dir", str(args.out_dir),
            "--device", args.device,
            "--out-uuid", str(row["file_uuid_original"]),
        ]
        if row.get("file_uuid_original"):
            cmd.extend(["--svs-uuid", str(row["file_uuid_original"])])
        if pd.notna(row.get("file_uuid_new")) and row.get("file_uuid_new"):
            cmd.extend(["--svs-uuid-fallback", str(row["file_uuid_new"])])

        res = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if res.returncode != 0:
            fail_dir.mkdir(parents=True, exist_ok=True)
            log_path = fail_dir / f"{row['file_uuid_original']}.log"
            log_path.write_text(
                f"CMD: {' '.join(cmd)}\n\nSTDOUT:\n{res.stdout}\n\nSTDERR:\n{res.stderr}\n"
            )
            logger.warning("  viewer failed (see %s): %s", log_path, res.stderr[:200] if res.stderr else res.returncode)

    # ── Save gallery CSV ─────────────────────────────────────────
    out_cols = ["file_uuid_original", "file_uuid_new", "barcode", "project_id", "aliquot_barcode",
                "expected", "predicted", "ptn", "err_mil", "err_ptn"]
    export_cols = [c for c in out_cols if c in gallery.columns]
    if "file_uuid_new" in export_cols and (gallery["file_uuid_new"].isna().all() if "file_uuid_new" in gallery.columns else True):
        export_cols = [c for c in export_cols if c != "file_uuid_new"]
    gallery[export_cols].to_csv(args.out_dir / "gallery_summary.csv", index=False)

    logger.info("Gallery saved: %s (%d cases)", args.out_dir / "gallery_summary.csv", len(gallery))
    logger.info("Mean |Δ MIL|: %.3f  Mean |Δ PTN|: %.3f  Advantage: %.3f",
                gallery["err_mil"].mean(), gallery["err_ptn"].mean(),
                (gallery["err_ptn"] - gallery["err_mil"]).mean())


if __name__ == "__main__":
    main()
