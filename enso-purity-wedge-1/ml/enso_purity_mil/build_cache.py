"""Hyper-fast parallel bag cache builder.

Uses ``ProcessPoolExecutor`` to read H5 files from gcsfuse in true
parallel (bypasses HDF5 C-library's internal GIL-like lock that
serializes ThreadPoolExecutor).  Each worker is a separate process
with its own HDF5 state.

Typical throughput: ~10+ bags/s with 16 processes on a g2-standard-16 VM.

Usage:
    python -m enso_purity_mil.build_cache \
        --manifest data/processed/wedge_mvp_dataset.xlsx \
        --h5-dir /path/to/embeddings_fp32 \
        --cache-dir /path/to/cache \
        --cache-dtype fp32 \
        --threads 32
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from enso_purity_mil.dataset import EmbeddingBagDataset, _load_full_pool
from enso_purity_mil.manifest_io import load_manifest_table

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
logger = logging.getLogger(__name__)


def _cache_one_bag(
    bag_id: str,
    file_ids: list[str],
    label: float,
    is_tumor: float,
    h5_dir: Path,
    cache_dir: Path,
    cache_dtype: str,
    _num_instances: int,
) -> str | None:
    """Cache a single bag. Returns bag_id on success, None if skipped/failed."""
    out_path = cache_dir / f"{bag_id}.pt"
    if out_path.exists():
        return None  # already cached

    try:
        # Store full bag pool so training can re-sample a fresh subset each epoch.
        pool = _load_full_pool(h5_dir, file_ids)
        pool_t = torch.from_numpy(pool)
        if cache_dtype == "fp16":
            pool_t = pool_t.to(torch.float16)
        else:
            pool_t = pool_t.to(torch.float32)
        torch.save(
            {
                "feats_pool": pool_t,
                "label": float(label),
                "is_tumor": float(is_tumor),
                "n_tiles": int(pool_t.shape[0]),
                "cache_dtype": cache_dtype,
            },
            out_path,
        )
        return bag_id
    except Exception as e:
        logger.warning("Failed %s: %s", bag_id, e)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Build local bag cache (parallel)")
    ap.add_argument("--manifest", type=Path,
                    default=Path("data/processed/wedge_mvp_dataset.xlsx"))
    ap.add_argument("--h5-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument(
        "--cache-dtype",
        type=str,
        choices=["fp32", "fp16"],
        default="fp32",
        help="Tensor dtype to store in cache files (default: fp32).",
    )
    ap.add_argument(
        "--num-instances",
        type=int,
        default=4096,
        help="Deprecated, kept for backward compatibility (ignored in pool-cache mode).",
    )
    ap.add_argument("--threads", type=int, default=16,
                    help="Number of worker processes (default 16 for 64GB RAM)")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    logger.info("Loading manifest: %s", args.manifest)
    manifest = load_manifest_table(args.manifest)
    manifest = manifest[manifest["purity"].notna()].copy()
    logger.info("Slides with purity: %d", len(manifest))

    ds = EmbeddingBagDataset(manifest, args.h5_dir, num_instances=args.num_instances)
    groups = ds.groups
    n = len(groups)

    already = sum(1 for g in groups if (args.cache_dir / f"{g['bag_id']}.pt").exists())
    logger.info("Total bags: %d, already cached: %d, to build: %d", n, already, n - already)
    logger.info(
        "Cache format: storing full bag pools (feats_pool), dtype=%s, for epoch-wise re-sampling",
        args.cache_dtype,
    )

    t0 = time.time()
    built = 0

    with ProcessPoolExecutor(max_workers=min(args.threads, 16)) as executor:
        futures = {
            executor.submit(
                _cache_one_bag,
                g["bag_id"], g["file_ids"], g["label"],
                g.get("is_tumor", 1.0),
                args.h5_dir, args.cache_dir, args.cache_dtype, args.num_instances,
            ): g["bag_id"]
            for g in groups
        }

        with tqdm(total=n, desc="Caching bags", unit="bag") as pbar:
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    built += 1
                pbar.update(1)

    elapsed = time.time() - t0
    total_cached = sum(1 for _ in args.cache_dir.glob("*.pt"))
    logger.info("Done. Built %d new bags in %.1f min (%.1f bags/s). Total cached: %d",
                built, elapsed / 60, built / max(elapsed, 0.001), total_cached)


if __name__ == "__main__":
    main()
