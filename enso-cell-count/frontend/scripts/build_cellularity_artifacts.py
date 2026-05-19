"""Build frontend-ready EnsoCellularity case artifacts.

The current purity viewer consumes static base JPGs plus RGBA heatmap masks from
``frontend/public/cases``. This script creates a parallel folder for the future
EnsoCellularity section without touching the purity assets:

  frontend/public/cellularity-cases/
    case_1_base.jpg
    case_1_cellularity_mask.png
    previews/case_1_cellularity_preview.jpg
    cellularity_case_summary.csv
    cellularity_case_summary.json
    cellularity_asset_manifest.json
    cellularity_legend.png
    cellularity_contact_sheet.jpg

It reuses the existing cell-count masks as the source signal and remaps the
purity-style colors into a cellularity display palette. Masks are always
resized with nearest-neighbor sampling so individual tile calls stay crisp.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont


CASE_RE = re.compile(r"case_(\d+)_base\.jpg$")
FRONTEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = FRONTEND_ROOT.parent

SOURCE_PURITY_STOPS = [
    (0.00, "#313695"),
    (0.17, "#4575b4"),
    (0.34, "#74add1"),
    (0.50, "#fee08b"),
    (0.67, "#fdae61"),
    (0.84, "#f46d43"),
    (1.00, "#a50026"),
]

VIRIDESCENCE_STOPS = [
    (0.00, "#12183d"),
    (0.14, "#203a78"),
    (0.28, "#136f9a"),
    (0.43, "#09a78f"),
    (0.58, "#35c96f"),
    (0.75, "#a8e34f"),
    (1.00, "#fff7a8"),
]

VIRIDIS_STOPS = [
    (0.00, "#440154"),
    (0.20, "#414487"),
    (0.40, "#2a788e"),
    (0.60, "#22a884"),
    (0.80, "#7ad151"),
    (1.00, "#fde725"),
]

VIRIDIS_ORANGE_STOPS = [
    (0.00, "#440154"),
    (0.16, "#3b528b"),
    (0.32, "#21908d"),
    (0.50, "#35b779"),
    (0.66, "#90d743"),
    (0.80, "#fde725"),
    (0.91, "#ffb627"),
    (1.00, "#f97316"),
]

AURORA_EMBER_STOPS = [
    (0.00, "#0b102f"),
    (0.14, "#24306e"),
    (0.28, "#0f6f9a"),
    (0.43, "#00a99d"),
    (0.58, "#4bd96c"),
    (0.73, "#c9ef5a"),
    (0.88, "#ffbf42"),
    (1.00, "#ff6f2c"),
]

MOSS_FIRE_STOPS = [
    (0.00, "#061b2b"),
    (0.15, "#123d57"),
    (0.30, "#096c73"),
    (0.46, "#159567"),
    (0.62, "#68c854"),
    (0.78, "#d9df54"),
    (0.91, "#f7a13a"),
    (1.00, "#e4572e"),
]

PALETTES = {
    "viridescence": {
        "stops": VIRIDESCENCE_STOPS,
        "label": "EnsoCellularity viridescence map",
    },
    "viridis": {
        "stops": VIRIDIS_STOPS,
        "label": "EnsoCellularity viridis map",
    },
    "viridis-orange": {
        "stops": VIRIDIS_ORANGE_STOPS,
        "label": "EnsoCellularity viridis-orange map",
    },
    "aurora-ember": {
        "stops": AURORA_EMBER_STOPS,
        "label": "EnsoCellularity aurora-ember map",
    },
    "moss-fire": {
        "stops": MOSS_FIRE_STOPS,
        "label": "EnsoCellularity moss-fire map",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build EnsoCellularity frontend case assets.")
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=FRONTEND_ROOT / "public" / "cases",
        help="Folder containing case_N_base.jpg and case_N_cell_count_mask.png files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=FRONTEND_ROOT / "public" / "cellularity-cases",
        help="Output folder for the cellularity asset bundle.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Optional cell-count summary CSV. Defaults to SOURCE_DIR/cell_count_case_summary.csv.",
    )
    parser.add_argument(
        "--palette",
        choices=tuple(PALETTES),
        default="aurora-ember",
        help="Output color map. Aurora-ember is the default EnsoCellularity display palette.",
    )
    parser.add_argument(
        "--preview-opacity",
        type=float,
        default=0.70,
        help="Opacity used for generated preview composites.",
    )
    parser.add_argument(
        "--color-gamma",
        type=float,
        default=1.20,
        help="Power transform on normalized display values. Values >1 reserve orange for denser tiles.",
    )
    parser.add_argument(
        "--preview-max-edge",
        type=int,
        default=1800,
        help="Maximum width or height for per-case preview JPGs.",
    )
    parser.add_argument(
        "--contact-thumb-width",
        type=int,
        default=420,
        help="Thumbnail width for the generated contact sheet.",
    )
    return parser.parse_args()


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)


def make_palette(stops: Iterable[tuple[float, str]], n: int = 256) -> np.ndarray:
    positions = np.array([p for p, _ in stops], dtype=np.float32)
    colors = np.array([hex_to_rgb(c) for _, c in stops], dtype=np.float32)
    x = np.linspace(0.0, 1.0, n, dtype=np.float32)
    channels = [np.interp(x, positions, colors[:, i]) for i in range(3)]
    return np.stack(channels, axis=1).round().clip(0, 255).astype(np.uint8)


def discover_case_ids(source_dir: Path) -> list[int]:
    ids: list[int] = []
    for path in source_dir.glob("case_*_base.jpg"):
        match = CASE_RE.match(path.name)
        if match:
            ids.append(int(match.group(1)))
    return sorted(ids)


def load_summary(summary_csv: Path | None) -> dict[int, dict[str, str]]:
    if summary_csv is None or not summary_csv.exists():
        return {}
    with summary_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {int(row["case_id"]): row for row in rows if row.get("case_id")}


def source_mask_for_case(source_dir: Path, case_id: int) -> Path:
    preferred = source_dir / f"case_{case_id}_cell_count_mask.png"
    if preferred.exists():
        return preferred
    fallback = source_dir / f"case_{case_id}_mask.png"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"No source mask found for case_{case_id} in {source_dir}")


def remap_mask(
    mask: Image.Image,
    source_palette: np.ndarray,
    target_palette: np.ndarray,
    color_gamma: float,
) -> Image.Image:
    rgba = np.asarray(mask.convert("RGBA"))
    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    valid = alpha > 0

    out = np.zeros_like(rgba)
    out[..., 3] = alpha
    if not np.any(valid):
        return Image.fromarray(out, "RGBA")

    codes = (
        (rgb[..., 0].astype(np.uint32) << 16)
        | (rgb[..., 1].astype(np.uint32) << 8)
        | rgb[..., 2].astype(np.uint32)
    )
    unique_codes, inverse = np.unique(codes[valid], return_inverse=True)
    unique_rgb = np.stack(
        [
            (unique_codes >> 16) & 255,
            (unique_codes >> 8) & 255,
            unique_codes & 255,
        ],
        axis=1,
    ).astype(np.int16)

    palette_i16 = source_palette.astype(np.int16)
    distances = ((unique_rgb[:, None, :] - palette_i16[None, :, :]) ** 2).sum(axis=2)
    source_indices = distances.argmin(axis=1)

    values = source_indices.astype(np.float32) / float(len(source_palette) - 1)
    values = np.power(values, max(float(color_gamma), 1e-6))
    target_indices = np.rint(values * (len(target_palette) - 1)).astype(np.int16)
    remapped_unique = target_palette[target_indices]

    out_rgb = out[..., :3]
    out_rgb[valid] = remapped_unique[inverse]
    return Image.fromarray(out, "RGBA")


def blend_preview(base: Image.Image, mask: Image.Image, opacity: float, max_edge: int) -> Image.Image:
    base_rgb = base.convert("RGB")
    if mask.size != base_rgb.size:
        mask = mask.resize(base_rgb.size, Image.Resampling.NEAREST)

    longest = max(base_rgb.size)
    if max_edge > 0 and longest > max_edge:
        scale = max_edge / float(longest)
        preview_size = (
            max(1, round(base_rgb.width * scale)),
            max(1, round(base_rgb.height * scale)),
        )
        base_rgb = base_rgb.resize(preview_size, Image.Resampling.LANCZOS)
        mask = mask.resize(preview_size, Image.Resampling.NEAREST)

    base_arr = np.asarray(base_rgb).astype(np.float32)
    mask_arr = np.asarray(mask.convert("RGBA")).astype(np.float32)
    alpha = (mask_arr[..., 3:4] / 255.0) * max(0.0, min(float(opacity), 1.0))
    blended = (base_arr * (1.0 - alpha) + mask_arr[..., :3] * alpha).round().clip(0, 255).astype(np.uint8)
    return Image.fromarray(blended, "RGB")


def draw_legend(target_palette: np.ndarray, out_path: Path, palette_label: str) -> None:
    width, height = 1120, 132
    margin_x = 56
    bar_y, bar_h = 34, 32
    img = Image.new("RGB", (width, height), "#07110f")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    gradient = np.zeros((bar_h, width - margin_x * 2, 3), dtype=np.uint8)
    idx = np.linspace(0, len(target_palette) - 1, gradient.shape[1]).round().astype(int)
    gradient[:, :, :] = target_palette[idx][None, :, :]
    img.paste(Image.fromarray(gradient, "RGB"), (margin_x, bar_y))

    draw.rounded_rectangle(
        (margin_x - 1, bar_y - 1, width - margin_x + 1, bar_y + bar_h + 1),
        radius=8,
        outline="#d8ffe8",
        width=1,
    )
    tick_y = bar_y + bar_h + 12
    for label, x in [
        ("Low cellularity", margin_x),
        ("Moderate", width // 2),
        ("High cellularity", width - margin_x),
    ]:
        draw.line((x, bar_y + bar_h + 2, x, tick_y - 3), fill="#d8ffe8", width=1)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        if x == margin_x:
            tx = x
        elif x == width - margin_x:
            tx = x - text_w
        else:
            tx = x - text_w // 2
        draw.text((tx, tick_y), label, fill="#eafff4", font=font)

    draw.text((margin_x, 94), palette_label, fill="#7ff5b7", font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")


def make_contact_sheet(previews: list[tuple[int, Path, dict[str, str]]], out_path: Path, thumb_width: int) -> None:
    if not previews:
        return

    cols = 4
    gap = 18
    label_h = 42
    bg = "#07110f"
    text = "#eafff4"
    muted = "#93c7ad"
    font = ImageFont.load_default()

    thumbs: list[tuple[Image.Image, str, str]] = []
    for case_id, preview_path, summary in previews:
        img = Image.open(preview_path).convert("RGB")
        scale = thumb_width / float(img.width)
        thumb = img.resize((thumb_width, max(1, round(img.height * scale))), Image.Resampling.LANCZOS)
        project_id = summary.get("project_id", "")
        mean_count = summary.get("mean_pred_nuclei_count", "")
        label = f"case_{case_id}"
        detail = project_id
        if mean_count:
            try:
                detail = f"{project_id}  mean {float(mean_count):.1f}"
            except ValueError:
                pass
        thumbs.append((thumb, label, detail))

    cell_h = max(t.height for t, _, _ in thumbs) + label_h
    rows = math.ceil(len(thumbs) / cols)
    width = cols * thumb_width + (cols + 1) * gap
    height = rows * cell_h + (rows + 1) * gap
    sheet = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(sheet)

    for idx, (thumb, label, detail) in enumerate(thumbs):
        row = idx // cols
        col = idx % cols
        x = gap + col * (thumb_width + gap)
        y = gap + row * (cell_h + gap)
        sheet.paste(thumb, (x, y))
        draw.text((x, y + thumb.height + 8), label, fill=text, font=font)
        draw.text((x, y + thumb.height + 24), detail, fill=muted, font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, "JPEG", quality=88, optimize=True)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_readme(out_dir: Path, route_prefix: str, preview_opacity: float) -> None:
    readme = f"""# EnsoCellularity Case Assets

This folder is a frontend-ready cellularity artifact bundle for the same demo
slides used by the current EnsoPurity case explorer.

Use `{route_prefix}/case_N_base.jpg` as the H&E image and
`{route_prefix}/case_N_cellularity_mask.png` as the RGBA cellularity overlay.
The preview composites in `{route_prefix}/previews/` use {preview_opacity:.0%} overlay opacity
and are included only for visual review.

The default palette is aurora-ember: deep blue for low cellularity, teal/green
through the midrange, and a controlled amber-orange accent for dense cellular
regions.

Regenerate from the repository root with:

```bash
python frontend/scripts/build_cellularity_artifacts.py
```
"""
    (out_dir / "README.md").write_text(readme, encoding="utf-8")


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def main() -> None:
    args = parse_args()
    source_dir = args.source_dir.resolve()
    out_dir = args.out_dir.resolve()
    summary_csv = args.summary_csv.resolve() if args.summary_csv else source_dir / "cell_count_case_summary.csv"
    preview_dir = out_dir / "previews"

    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    source_palette = make_palette(SOURCE_PURITY_STOPS)
    target_stops = PALETTES[args.palette]["stops"]
    palette_label = PALETTES[args.palette]["label"]
    target_palette = make_palette(target_stops)
    summaries = load_summary(summary_csv)
    case_ids = discover_case_ids(source_dir)
    if not case_ids:
        raise FileNotFoundError(f"No case_N_base.jpg files found in {source_dir}")

    rows: list[dict[str, object]] = []
    preview_records: list[tuple[int, Path, dict[str, str]]] = []

    for case_id in case_ids:
        base_src = source_dir / f"case_{case_id}_base.jpg"
        mask_src = source_mask_for_case(source_dir, case_id)
        base_out = out_dir / base_src.name
        mask_out = out_dir / f"case_{case_id}_cellularity_mask.png"
        preview_out = preview_dir / f"case_{case_id}_cellularity_preview.jpg"

        shutil.copy2(base_src, base_out)
        with Image.open(base_src) as base_img, Image.open(mask_src) as source_mask:
            remapped = remap_mask(source_mask, source_palette, target_palette, args.color_gamma)
            if remapped.size != base_img.size:
                remapped = remapped.resize(base_img.size, Image.Resampling.NEAREST)
            remapped.save(mask_out, "PNG")
            preview = blend_preview(base_img, remapped, args.preview_opacity, args.preview_max_edge)
            preview.save(preview_out, "JPEG", quality=90, optimize=True)

        summary = summaries.get(case_id, {})
        row: dict[str, object] = {
            **summary,
            "case_id": case_id,
            "base_file": base_out.name,
            "source_mask_file": mask_src.name,
            "mask_file": mask_out.name,
            "preview_file": f"previews/{preview_out.name}",
            "palette": args.palette,
            "default_overlay_opacity": args.preview_opacity,
        }
        with Image.open(mask_out) as out_mask:
            row["mask_width"] = out_mask.width
            row["mask_height"] = out_mask.height
        rows.append(row)
        preview_records.append((case_id, preview_out, summary))
        print(f"case_{case_id}: {mask_src.name} -> {mask_out.relative_to(out_dir.parent)}")

    legend_path = out_dir / "cellularity_legend.png"
    contact_sheet_path = out_dir / "cellularity_contact_sheet.jpg"
    draw_legend(target_palette, legend_path, palette_label)
    make_contact_sheet(preview_records, contact_sheet_path, args.contact_thumb_width)

    route_prefix = "/" + out_dir.name
    manifest = {
        "name": "EnsoCellularity static case assets",
        "route_prefix": route_prefix,
        "case_count": len(rows),
        "base_pattern": f"{route_prefix}/case_{{case_id}}_base.jpg",
        "mask_pattern": f"{route_prefix}/case_{{case_id}}_cellularity_mask.png",
        "preview_pattern": f"{route_prefix}/previews/case_{{case_id}}_cellularity_preview.jpg",
        "legend_file": f"{route_prefix}/{legend_path.name}",
        "contact_sheet_file": f"{route_prefix}/{contact_sheet_path.name}",
        "palette": {
            "name": args.palette,
            "stops": [{"position": p, "color": c} for p, c in target_stops],
            "low_label": "Low cellularity",
            "high_label": "High cellularity",
            "display_gamma": args.color_gamma,
        },
        "source": {
            "source_dir": portable_path(source_dir),
            "source_summary_csv": portable_path(summary_csv),
            "source_mask_preference": "case_N_cell_count_mask.png, then case_N_mask.png",
        },
        "default_overlay_opacity": args.preview_opacity,
        "color_gamma": args.color_gamma,
        "cases": rows,
    }
    (out_dir / "cellularity_asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "cellularity_case_summary.json").write_text(
        json.dumps({"cases": rows, "palette": manifest["palette"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(out_dir / "cellularity_case_summary.csv", rows)
    write_readme(out_dir, route_prefix, args.preview_opacity)

    print(f"Wrote {len(rows)} cellularity case bundles to {out_dir}")
    print(f"Manifest: {out_dir / 'cellularity_asset_manifest.json'}")


if __name__ == "__main__":
    main()
