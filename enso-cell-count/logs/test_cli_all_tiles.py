#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr


def build_test_bags(manifest: pd.DataFrame) -> list[dict]:
    groups: list[dict] = []
    tumours = manifest[manifest["gdc_match_type"] != "normal_tissue"]
    for aliquot, sub in tumours.groupby("aliquot_barcode"):
        r0 = sub.iloc[0]
        groups.append(
            {
                "bag_id": f"tumor_{aliquot}",
                "aliquot_barcode": str(aliquot),
                "case_id": str(r0.get("case_id", "")),
                "project_id": str(r0.get("project_id", "")),
                "file_ids": sub["file_uuid_original"].tolist(),
                "label": float(sub["purity"].iloc[0]),
            }
        )
    return groups


def load_full_pool_h5(h5_dir: Path, file_ids: list[str]) -> np.ndarray:
    chunks = []
    for fid in file_ids:
        with h5py.File(h5_dir / f"{fid}.h5", "r") as f:
            chunks.append(f["features"][:])
    if not chunks:
        raise ValueError("No file_ids in bag")
    return np.concatenate(chunks, axis=0)


def load_all_tiles_for_bag(bag: dict, h5_dir: Path, cache_dir: Path | None) -> torch.Tensor:
    bag_id = bag["bag_id"]
    if cache_dir is not None:
        cache_path = cache_dir / f"{bag_id}.pt"
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
            feats_pool = payload.get("feats_pool")
            if isinstance(feats_pool, torch.Tensor):
                return feats_pool.to(torch.float32)
    arr = load_full_pool_h5(h5_dir, bag["file_ids"])
    return torch.from_numpy(arr.astype(np.float32, copy=False))


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate fold test set on all tiles per bag")
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--pred-out", type=Path, default=None,
                    help="Optional CSV output with per-bag predictions.")
    ap.add_argument("--model-name", type=str, required=True)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    repo_root = args.repo_root.resolve()
    sys.path.insert(0, str(repo_root / "ml"))

    from enso_purity_mil.folds import generate_stratified_folds
    from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig

    manifest_path = args.manifest or (repo_root / "data/processed/wedge_mvp_dataset.xlsx")
    manifest = pd.read_excel(manifest_path)
    manifest = manifest[manifest["purity"].notna()].copy()
    tumour_df = manifest[manifest["gdc_match_type"] != "normal_tissue"].copy().reset_index(drop=True)

    folds = generate_stratified_folds(tumour_df, n_folds=5, seed=args.seed, cancer_col="project_id")
    test_df = tumour_df.iloc[folds[args.fold]].reset_index(drop=True)
    bags = build_test_bags(test_df)

    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    model = EnsoMILModel(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    pred_rows = []

    with torch.no_grad():
        for bag in bags:
            feats = load_all_tiles_for_bag(bag, args.h5_dir, args.cache_dir)
            feats = feats.unsqueeze(0).to(args.device, non_blocking=True)
            pred = model(feats).squeeze().item()
            y_pred.append(float(pred))
            y_true.append(float(bag["label"]))
            pred_rows.append(
                {
                    "model": args.model_name,
                    "fold": args.fold,
                    "bag_id": bag["bag_id"],
                    "aliquot_barcode": bag.get("aliquot_barcode", ""),
                    "case_id": bag.get("case_id", ""),
                    "project_id": bag.get("project_id", ""),
                    "file_ids": ";".join(str(x) for x in bag["file_ids"]),
                    "true_purity": float(bag["label"]),
                    "pred_purity": float(pred),
                }
            )

    y_true_np = np.array(y_true, dtype=np.float64)
    y_pred_np = np.array(y_pred, dtype=np.float64)

    l1 = float(np.mean(np.abs(y_pred_np - y_true_np)))
    ss_res = float(np.sum((y_true_np - y_pred_np) ** 2))
    ss_tot = float(np.sum((y_true_np - y_true_np.mean()) ** 2))
    r2 = float("nan") if ss_tot <= 0 else float(1.0 - ss_res / ss_tot)
    sp, _ = spearmanr(y_true_np, y_pred_np)
    spearman = float(sp)

    result = {
        "model": args.model_name,
        "repo_root": str(repo_root),
        "checkpoint": str(args.model_path.resolve()),
        "manifest": str(Path(manifest_path).resolve()),
        "h5_dir": str(args.h5_dir.resolve()),
        "cache_dir": str(args.cache_dir.resolve()) if args.cache_dir else None,
        "fold": args.fold,
        "seed": args.seed,
        "device": args.device,
        "all_tiles": True,
        "test_bags": int(len(bags)),
        "checkpoint_epoch": int(ckpt.get("epoch", -1)) if isinstance(ckpt.get("epoch", -1), (int, float)) else -1,
        "checkpoint_val_loss": float(ckpt.get("val_loss", float("nan"))),
        "test_l1": l1,
        "test_r2": r2,
        "test_spearman": spearman,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))

    if args.pred_out is not None:
        args.pred_out.parent.mkdir(parents=True, exist_ok=True)
        with args.pred_out.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "model", "fold", "bag_id", "aliquot_barcode", "case_id",
                    "project_id", "file_ids", "true_purity", "pred_purity",
                ],
            )
            writer.writeheader()
            writer.writerows(pred_rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
