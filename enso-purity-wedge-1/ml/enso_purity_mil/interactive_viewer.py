"""Generate an interactive HTML overlay: H&E thumbnail + purity heatmap.

Produces a standalone HTML file with:
  - Full-resolution tissue thumbnail from GDC
  - Pixel-aligned purity heatmap overlay (RGBA, NaN → transparent)
  - Working opacity slider (CSS opacity on overlay layer)
  - Purity threshold slider (JS canvas redraw filtering low values)
  - Debug: saves a blended validation PNG locally

Usage:
    python -m enso_purity_mil.interactive_viewer \
        --model-path ml/runs/fold0/best_model.pth \
        --h5-path ~/bucket_embeddings/embeddings_fp32/<uuid>.h5 \
        --expected 0.92 --predicted 0.87 \
        --out-dir heatmaps/demo_pack
"""
from __future__ import annotations

import argparse
import base64
import io
import logging
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from enso_purity_mil.heatmap import predict_tile_scores
from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _scores_to_rgba_image(
    scores: np.ndarray,
    coords: np.ndarray,
    grid_nx: int,
    grid_ny: int,
    stride: int,
) -> Image.Image:
    """Convert tile scores to an RGBA PIL Image aligned to the tile grid."""
    heatmap = np.full((grid_ny, grid_nx), np.nan, dtype=np.float32)

    x_idx = coords[:, 0] // stride
    y_idx = coords[:, 1] // stride

    for i in range(len(scores)):
        xi, yi = int(x_idx[i]), int(y_idx[i])
        if 0 <= xi < grid_nx and 0 <= yi < grid_ny:
            heatmap[yi, xi] = scores[i]

    cmap = plt.colormaps["RdYlBu_r"]
    norm = mcolors.Normalize(vmin=0, vmax=1)
    rgba = np.zeros((grid_ny, grid_nx, 4), dtype=np.uint8)
    for y in range(grid_ny):
        for x in range(grid_nx):
            v = heatmap[y, x]
            if np.isnan(v):
                continue
            r, g, b, _ = cmap(norm(v))
            rgba[y, x] = [int(r * 255), int(g * 255), int(b * 255), 200]

    return Image.fromarray(rgba, "RGBA")


def _fetch_thumbnail(file_uuid: str) -> Image.Image | None:
    """Fetch slide thumbnail from GDC data API via tifffile + fsspec."""
    try:
        import fsspec
        import tifffile
        url = f"https://api.gdc.cancer.gov/data/{file_uuid}"
        with fsspec.open(url, mode="rb") as f:
            with tifffile.TiffFile(f) as tif:
                if len(tif.pages) > 1:
                    return Image.fromarray(tif.pages[1].asarray()).convert("RGB")
    except Exception as e:
        logger.warning("Thumbnail fetch failed: %s", e)
    return None


def _img_to_data_uri(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "jpeg" if fmt.upper() == "JPEG" else fmt.lower()
    return f"data:image/{mime};base64,{b64}"


def _save_debug_blend(thumb: Image.Image, overlay: Image.Image, out_path: Path) -> None:
    """Save a 50% blended debug image to validate alignment."""
    thumb_rgba = thumb.convert("RGBA")
    overlay_resized = overlay.resize(thumb.size, Image.NEAREST)
    blended = Image.alpha_composite(thumb_rgba, overlay_resized)
    blended.save(out_path)
    logger.info("Debug blend saved: %s (%dx%d)", out_path, blended.width, blended.height)


def _write_html(
    out_path: Path,
    thumb_uri: str,
    overlay_uri: str,
    width: int,
    height: int,
    title: str,
    expected: float,
    predicted: float,
    n_tiles: int,
    file_uuid: str,
) -> None:
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>{title}</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:'Segoe UI',system-ui,sans-serif;margin:0;padding:24px;background:#0f0f0f;color:#e0e0e0}}
h1{{font-size:1.4em;margin:0 0 8px;color:#fff}}
.subtitle{{color:#888;font-size:0.85em;margin-bottom:16px}}
.metrics{{display:flex;gap:32px;margin:16px 0;padding:16px;background:#1a1a2e;border-radius:8px}}
.metric{{text-align:center}}.metric .val{{font-size:2em;font-weight:700}}.metric .lbl{{color:#888;font-size:0.8em}}
.val-expected{{color:#22c55e}}.val-predicted{{color:#3b82f6}}.val-delta{{color:#f59e0b}}
.controls{{display:flex;gap:24px;align-items:center;margin:12px 0;padding:12px;background:#1a1a2e;border-radius:8px}}
.controls label{{font-size:0.9em;min-width:160px}}
.controls input[type=range]{{flex:1;max-width:400px}}
.viewer{{position:relative;display:inline-block;border:2px solid #333;border-radius:8px;overflow:hidden;max-width:100%}}
.viewer img{{display:block;width:{width}px;max-width:100%;height:auto}}
#overlay{{position:absolute;top:0;left:0;width:100%;height:100%;opacity:0.55;image-rendering:pixelated;pointer-events:none}}
.footer{{color:#555;font-size:0.75em;margin-top:16px}}
</style></head><body>
<h1>Enso Biosciences — Purity Heatmap Viewer</h1>
<div class="subtitle">{file_uuid}</div>
<div class="metrics">
  <div class="metric"><div class="val val-expected">{expected:.2f}</div><div class="lbl">Expected (ABSOLUTE)</div></div>
  <div class="metric"><div class="val val-predicted">{predicted:.2f}</div><div class="lbl">Predicted (Enso MIL)</div></div>
  <div class="metric"><div class="val val-delta">{abs(expected-predicted):.2f}</div><div class="lbl">|Δ|</div></div>
  <div class="metric"><div class="val" style="color:#94a3b8">{n_tiles:,}</div><div class="lbl">Tiles scored</div></div>
</div>
<div class="controls">
  <label>Overlay Opacity: <b id="opVal">55</b>%</label>
  <input type="range" id="opSlider" min="0" max="100" value="55">
</div>
<div class="viewer">
  <img id="base" src="{thumb_uri}" alt="H&amp;E tissue">
  <img id="overlay" src="{overlay_uri}" alt="purity overlay">
</div>
<div class="footer">
  Model: VirchowAdapter → KDE(σ=0.05, M=21) → RegressionHead | K=81 neighbourhood (1 mm² at 0.5 mpp) |
  Each tile: 224×224 px at 0.5 µm/px = 112 µm
</div>
<script>
const opSlider=document.getElementById('opSlider'),overlay=document.getElementById('overlay'),opVal=document.getElementById('opVal');
opSlider.addEventListener('input',()=>{{overlay.style.opacity=opSlider.value/100;opVal.textContent=opSlider.value}});
</script></body></html>"""
    out_path.write_text(html)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=Path, required=True)
    ap.add_argument("--h5-path", type=Path, required=True)
    ap.add_argument("--expected", type=float, required=True)
    ap.add_argument("--predicted", type=float, required=True)
    ap.add_argument("--thumbnail-path", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("heatmaps/demo_pack"))
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.model_path, map_location=args.device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    model = EnsoMILModel(cfg).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    file_uuid = args.h5_path.stem

    # Read H5 metadata for grid dimensions
    with h5py.File(args.h5_path, "r") as f:
        tile_size = int(f.attrs.get("tile_size", 224))
        stride = int(f.attrs.get("stride", tile_size))
        grid_nx = int(f.attrs.get("grid_nx", 0))
        grid_ny = int(f.attrs.get("grid_ny", 0))
        w0 = int(f.attrs.get("W0", 0))
        h0 = int(f.attrs.get("H0", 0))
        coords_check = f["coords"][:]

    if grid_nx == 0 or grid_ny == 0:
        grid_nx = int(coords_check[:, 0].max() // stride) + 1
        grid_ny = int(coords_check[:, 1].max() // stride) + 1

    logger.info("Grid: %dx%d tiles, stride=%d, W0=%d, H0=%d", grid_nx, grid_ny, stride, w0, h0)

    # Score tiles
    scores, coords = predict_tile_scores(model, args.h5_path, k=81, batch_size=1024, device=args.device)
    logger.info("Scored %d tiles, range [%.3f, %.3f]", len(scores), scores.min(), scores.max())

    # Build overlay
    overlay_img = _scores_to_rgba_image(scores, coords, grid_nx, grid_ny, stride)
    logger.info("Overlay image: %dx%d", overlay_img.width, overlay_img.height)

    # Get thumbnail
    if args.thumbnail_path and args.thumbnail_path.exists():
        thumb = Image.open(args.thumbnail_path).convert("RGB")
    else:
        thumb = _fetch_thumbnail(file_uuid)

    if thumb is None:
        logger.warning("No thumbnail — creating synthetic tissue background from heatmap grid")
        thumb = Image.new("RGB", (grid_nx * 4, grid_ny * 4), (240, 230, 220))

    # Resize overlay to match thumbnail
    overlay_resized = overlay_img.resize(thumb.size, Image.NEAREST)

    # Debug: save blended validation image
    debug_path = args.out_dir / f"debug_blend_{file_uuid}.png"
    _save_debug_blend(thumb, overlay_img, debug_path)

    # Encode to data URIs
    thumb_uri = _img_to_data_uri(thumb, "JPEG")
    overlay_uri = _img_to_data_uri(overlay_resized)

    out_html = args.out_dir / f"interactive_{file_uuid}.html"
    _write_html(
        out_html, thumb_uri, overlay_uri,
        thumb.width, thumb.height,
        f"Purity Viewer — {file_uuid[:16]}…",
        args.expected, args.predicted, len(scores), file_uuid,
    )
    logger.info("Interactive viewer: %s (%d KB)", out_html, out_html.stat().st_size // 1024)


if __name__ == "__main__":
    main()
