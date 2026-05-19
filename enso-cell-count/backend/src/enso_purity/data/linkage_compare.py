"""Comparison and visualisation utilities for slide ↔ purity linkage methods."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_REPORT_DIR = Path("data/reports")


def save_section_type_histogram(
    slides: pd.DataFrame,
    *,
    out_dir: Path = _REPORT_DIR,
) -> Path:
    """Bar chart of TS / MS / BS slide counts in the processed bucket."""
    counts = slides["section_type"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(counts.index, counts.values, color=["#4c72b0", "#dd8452", "#55a868"])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 50, str(v),
                ha="center", va="bottom", fontsize=10)
    ax.set_title("Frozen-section slides in bucket (mpp < 2)")
    ax.set_ylabel("Count")
    ax.set_xlabel("Section type")
    fig.tight_layout()
    out = out_dir / "section_type_histogram.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def save_dataset_size_comparison(
    stats: dict[str, Any],
    *,
    n_gdc: int | None = None,
    out_dir: Path = _REPORT_DIR,
) -> Path:
    """Bar chart comparing dataset sizes from each matching method."""
    labels = ["Frozen slides\n(bucket)", "Barcode\n(vial)", "Barcode\n(portion)"]
    values = [
        stats["n_frozen_mpp_ok"],
        stats["n_matched_vial"],
        stats["n_matched_portion"],
    ]
    if n_gdc is not None:
        labels.append("GDC API")
        values.append(n_gdc)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ["#999999", "#4c72b0", "#dd8452", "#55a868"][:len(labels)]
    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 100, str(v),
                ha="center", va="bottom", fontsize=9)
    ax.set_title("Dataset sizes by matching method")
    ax.set_ylabel("Number of slide–purity rows")
    fig.tight_layout()
    out = out_dir / "dataset_size_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def save_multi_slide_histogram(
    merged: pd.DataFrame,
    *,
    group_col: str = "sample_vial",
    title_suffix: str = "",
    out_dir: Path = _REPORT_DIR,
) -> Path:
    """Histogram of aliquots/sample_vials with >1 frozen slide."""
    slide_counts = merged.groupby(group_col)["barcode"].nunique()
    multi = slide_counts[slide_counts > 1]

    fig, ax = plt.subplots(figsize=(6, 4))
    if len(multi) > 0:
        bins = np.arange(1.5, multi.max() + 1.5, 1)
        ax.hist(multi.values, bins=bins, color="#4c72b0", edgecolor="white", rwidth=0.8)
    ax.set_title(f"Portions with >1 TS/MS/BS slide{title_suffix}")
    ax.set_xlabel("Number of slides per sample_vial")
    ax.set_ylabel("Count of sample_vials")
    fig.tight_layout()
    name = f"multi_slide_histogram{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    out = out_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def save_purity_distribution(
    merged: pd.DataFrame,
    *,
    title_suffix: str = "",
    out_dir: Path = _REPORT_DIR,
) -> Path:
    """Histogram of purity values in the matched dataset."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(merged["purity"].dropna(), bins=30, color="#4c72b0",
            edgecolor="white", alpha=0.85)
    ax.set_title(f"ABSOLUTE purity distribution{title_suffix}")
    ax.set_xlabel("Purity")
    ax.set_ylabel("Count")
    fig.tight_layout()
    name = f"purity_distribution{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    out = out_dir / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def save_gdc_match_type_pie(
    gdc_merged: pd.DataFrame,
    *,
    out_dir: Path = _REPORT_DIR,
) -> Path:
    """Pie chart of same_portion vs same_sample match types from GDC."""
    counts = gdc_merged["gdc_match_type"].value_counts()
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(counts.values, labels=counts.index, autopct="%1.1f%%",
           colors=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_title("GDC match types")
    fig.tight_layout()
    out = out_dir / "gdc_match_type_pie.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    logger.info("Saved %s", out)
    return out


def save_all_plots(
    slides_filtered: pd.DataFrame,
    barcode_stats: dict[str, Any],
    vial_merged: pd.DataFrame,
    gdc_merged: pd.DataFrame | None = None,
    *,
    out_dir: Path = _REPORT_DIR,
) -> list[Path]:
    """Generate and save all comparison plots. Returns list of paths."""
    paths: list[Path] = []
    paths.append(save_section_type_histogram(slides_filtered, out_dir=out_dir))
    paths.append(save_dataset_size_comparison(
        barcode_stats,
        n_gdc=len(gdc_merged) if gdc_merged is not None else None,
        out_dir=out_dir,
    ))
    paths.append(save_multi_slide_histogram(
        vial_merged, title_suffix=" (barcode-vial)", out_dir=out_dir,
    ))
    paths.append(save_purity_distribution(
        vial_merged, title_suffix=" (barcode-vial)", out_dir=out_dir,
    ))
    if gdc_merged is not None and len(gdc_merged) > 0:
        paths.append(save_multi_slide_histogram(
            gdc_merged, title_suffix=" (GDC)", out_dir=out_dir,
        ))
        paths.append(save_purity_distribution(
            gdc_merged, title_suffix=" (GDC)", out_dir=out_dir,
        ))
        paths.append(save_gdc_match_type_pie(gdc_merged, out_dir=out_dir))
    return paths
