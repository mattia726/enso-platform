"""Generate a demo pack: aliquot-level predictions + slide heatmaps.

Picks 6–8 aliquots spanning purity range and cancer types, runs scalar
prediction from the cached bag, generates a heatmap for one representative
slide per aliquot, and writes summary.csv + index.html.

Usage:
    python -m enso_purity_mil.batch_demo_pack \
        --model-path ml/runs/fold0/best_model.pth \
        --manifest data/processed/wedge_mvp_dataset.xlsx \
        --h5-dir ~/bucket_embeddings/embeddings_fp32 \
        --cache-dir ~/enso_workspace/data/cache \
        --out-dir heatmaps/demo_pack
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from enso_purity_mil.dataset import _sample_tensor_rows
from enso_purity_mil.heatmap import predict_tile_scores
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _pick_aliquots(manifest: pd.DataFrame, targets: list[float]) -> pd.DataFrame:
    """Pick one aliquot per target purity, preferring unique project_ids."""
    tumours = manifest[manifest["gdc_match_type"] == "same_portion"].copy()
    aliquot_df = tumours.groupby("aliquot_barcode").agg(
        purity=("purity", "first"),
        project_id=("project_id", "first"),
        n_slides=("barcode", "nunique"),
        best_slide_uuid=("file_uuid_original", "first"),
        best_barcode=("barcode", "first"),
        max_area=("area", "max"),
    ).reset_index()

    chosen: list[pd.Series] = []
    used_projects: set[str] = set()

    for target in targets:
        aliquot_df["_dist"] = (aliquot_df["purity"] - target).abs()
        candidates = aliquot_df.sort_values("_dist")
        picked = False
        for _, row in candidates.iterrows():
            if row["aliquot_barcode"] in [c["aliquot_barcode"] for c in chosen]:
                continue
            if row["project_id"] not in used_projects:
                chosen.append(row)
                used_projects.add(row["project_id"])
                picked = True
                break
        if not picked:
            for _, row in candidates.iterrows():
                if row["aliquot_barcode"] not in [c["aliquot_barcode"] for c in chosen]:
                    chosen.append(row)
                    break

    # For each chosen aliquot, pick the largest slide
    result = pd.DataFrame(chosen)
    for i, row in result.iterrows():
        slides = tumours[tumours["aliquot_barcode"] == row["aliquot_barcode"]]
        best = slides.sort_values("area", ascending=False).iloc[0]
        result.at[i, "best_slide_uuid"] = best["file_uuid_original"]
        result.at[i, "best_barcode"] = best["barcode"]

    return result.drop(columns=["_dist"], errors="ignore")


def _predict_aliquot(model: EnsoMILModel, cache_dir: Path, aliquot: str, device: str) -> float:
    """Load cached bag and predict scalar purity."""
    bag_path = cache_dir / f"tumor_{aliquot}.pt"
    if not bag_path.exists():
        return float("nan")
    payload = torch.load(bag_path, map_location="cpu", weights_only=True)
    feats = payload.get("feats")
    feats_pool = payload.get("feats_pool")
    if isinstance(feats_pool, torch.Tensor):
        feats = _sample_tensor_rows(feats_pool, num_instances=4096)
    elif isinstance(feats, torch.Tensor) and int(feats.shape[0]) != 4096:
        feats = _sample_tensor_rows(feats, num_instances=4096)
    if not isinstance(feats, torch.Tensor):
        return float("nan")

    feats = feats.to(torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(feats).squeeze().item()
    return max(0.0, min(1.0, pred))


def _generate_heatmap(
    model: EnsoMILModel,
    h5_path: Path,
    out_path: Path,
    title: str,
    device: str,
) -> None:
    """Generate and save a heatmap PNG."""
    import h5py
    scores, coords = predict_tile_scores(model, h5_path, k=81, batch_size=1024, device=device)

    with h5py.File(h5_path, "r") as f:
        tile_size = int(f.attrs.get("tile_size", 224))
        stride = int(f.attrs.get("stride", tile_size))

    x_grid = coords[:, 0] // stride
    y_grid = coords[:, 1] // stride
    nx = int(x_grid.max()) + 1
    ny = int(y_grid.max()) + 1

    heatmap = np.full((ny, nx), np.nan, dtype=np.float32)
    for i in range(len(scores)):
        heatmap[int(y_grid[i]), int(x_grid[i])] = scores[i]

    fig, ax = plt.subplots(figsize=(max(6, nx / 10), max(4, ny / 10)))
    im = ax.imshow(heatmap, cmap="RdYlBu_r", vmin=0, vmax=1, interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Predicted purity", shrink=0.8)
    ax.set_title(title, fontsize=9)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return scores, coords


def _write_index_html(out_dir: Path, rows: list[dict]) -> None:
    html = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
            '<title>Enso Purity — Demo Pack</title>',
            '<style>body{font-family:sans-serif;max-width:1200px;margin:auto;padding:20px}',
            '.card{border:1px solid #ddd;border-radius:8px;margin:16px 0;padding:16px;display:flex;gap:16px;align-items:start}',
            '.card img{max-width:600px;border-radius:4px}',
            '.meta{min-width:280px}',
            'table{border-collapse:collapse;width:100%}td,th{padding:6px 10px;text-align:left;border-bottom:1px solid #eee}',
            'h1{color:#1a1a2e}h3{margin:0 0 8px}</style></head><body>',
            '<h1>Enso Biosciences — Purity Prediction Demo Pack</h1>',
            '<p>MIL model: VirchowAdapter → KDE(σ=0.05, M=21) → RegressionHead | Fold 0</p>',
            '<table><tr><th>#</th><th>Project</th><th>Aliquot</th><th>Expected</th><th>Predicted</th><th>Δ</th><th>Slides</th></tr>']

    for i, r in enumerate(rows):
        delta = abs(r["expected"] - r["predicted"])
        html.append(f'<tr><td>{i+1}</td><td>{r["project_id"]}</td><td style="font-size:0.8em">{r["aliquot"]}</td>'
                     f'<td><b>{r["expected"]:.2f}</b></td><td><b>{r["predicted"]:.2f}</b></td>'
                     f'<td>{delta:.2f}</td><td>{r["n_slides"]}</td></tr>')
    html.append('</table><hr>')

    for i, r in enumerate(rows):
        png = f'{r["slide_uuid"]}_heatmap.png'
        html.append(f'<div class="card"><img src="{png}" alt="heatmap">'
                     f'<div class="meta"><h3>#{i+1} {r["project_id"]}</h3>'
                     f'<p>Expected: <b>{r["expected"]:.2f}</b> | Predicted: <b>{r["predicted"]:.2f}</b></p>'
                     f'<p>Slide: <code>{r["barcode"]}</code></p>'
                     f'<p>Aliquot: <code>{r["aliquot"]}</code></p>'
                     f'<p>Tiles: {r["n_tiles"]} | Slides: {r["n_slides"]}</p></div></div>')

    html.append('</body></html>')
    (out_dir / "index.html").write_text("\n".join(html))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, default=Path("data/processed/wedge_mvp_dataset.xlsx"))
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("heatmaps/demo_pack"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_excel(args.manifest)
    manifest = manifest[manifest["purity"].notna()].copy()

    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    model = EnsoMILModel(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Model loaded (epoch %d, val_loss=%.4f)", ckpt["epoch"], ckpt["val_loss"])

    targets = [0.15, 0.35, 0.55, 0.75, 0.95, 1.00]
    chosen = _pick_aliquots(manifest, targets)
    logger.info("Selected %d aliquots for demo pack", len(chosen))

    summary_rows: list[dict] = []
    for _, row in chosen.iterrows():
        aliquot = row["aliquot_barcode"]
        slide_uuid = row["best_slide_uuid"]
        barcode = row["best_barcode"]
        expected = row["purity"]
        project = row["project_id"]
        n_slides = int(row["n_slides"])

        predicted = _predict_aliquot(model, args.cache_dir, aliquot, args.device)
        logger.info("  %s | %s | expected=%.2f predicted=%.2f", project, aliquot, expected, predicted)

        h5_path = args.h5_dir / f"{slide_uuid}.h5"
        title = f"{project} | Expected: {expected:.2f} | Predicted: {predicted:.2f} | {barcode}"
        out_png = args.out_dir / f"{slide_uuid}_heatmap.png"

        n_tiles = 0
        if h5_path.exists():
            scores, coords = _generate_heatmap(model, h5_path, out_png, title, args.device)
            n_tiles = len(scores)
            logger.info("    heatmap: %d tiles, score range [%.3f, %.3f]", n_tiles, scores.min(), scores.max())
        else:
            logger.warning("    H5 not found: %s", h5_path)

        summary_rows.append({
            "aliquot": aliquot, "project_id": project,
            "expected": expected, "predicted": predicted,
            "slide_uuid": slide_uuid, "barcode": barcode,
            "n_tiles": n_tiles, "n_slides": n_slides,
        })

    with open(args.out_dir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=summary_rows[0].keys())
        w.writeheader()
        w.writerows(summary_rows)

    _write_index_html(args.out_dir, summary_rows)
    logger.info("Demo pack written to %s", args.out_dir)


if __name__ == "__main__":
    main()
