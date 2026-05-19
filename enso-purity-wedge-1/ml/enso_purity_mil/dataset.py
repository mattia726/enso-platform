"""H5-based embedding bag dataset for Enso MIL training.

A **bag** corresponds to an **aliquot** (for tumours) or a single normal
slide. Each bag has a stable ``bag_id`` (derived from aliquot barcode or
file UUID) so cached ``.pt`` files are fold-invariant.

Supports two I/O modes:
  * **cache_dir** — pre-built ``.pt`` files on local SSD.
    - New cache format stores full bag pools (``feats_pool``) and draws a
      fresh random ``num_instances`` sample on every ``__getitem__`` call.
    - Legacy cache format stores a fixed ``feats`` sample. For true
      epoch-wise resampling, loader falls back to H5 sampling when possible.
  * **h5_dir** — smart partial reads from H5 files (gcsfuse-compatible).
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data

_log = logging.getLogger(__name__)


def _build_aliquot_groups(
    manifest: pd.DataFrame,
    file_id_col: str,
    label_col: str,
) -> list[dict]:
    """Group manifest rows by aliquot (tumours) or by slide (normals).

    Each group gets a stable ``bag_id`` for cache file naming.
    """
    groups: list[dict] = []
    tumours = manifest[manifest["gdc_match_type"] != "normal_tissue"]
    normals = manifest[manifest["gdc_match_type"] == "normal_tissue"]

    for aliquot, sub in tumours.groupby("aliquot_barcode"):
        groups.append({
            "bag_id": f"tumor_{aliquot}",
            "file_ids": sub[file_id_col].tolist(),
            "label": float(sub[label_col].iloc[0]),
            "is_tumor": 1.0,
        })

    for _, row in normals.iterrows():
        groups.append({
            "bag_id": f"normal_{row[file_id_col]}",
            "file_ids": [row[file_id_col]],
            "label": float(row[label_col]),
            "is_tumor": 0.0,
        })

    return groups


def _sample_smart(
    h5_dir: Path,
    file_ids: list[str],
    num_instances: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample ``num_instances`` tile embeddings with minimal I/O."""
    handles_and_counts: list[tuple[str, int]] = []
    for fid in file_ids:
        with h5py.File(h5_dir / f"{fid}.h5", "r") as f:
            handles_and_counts.append((fid, f["features"].shape[0]))

    total = sum(c for _, c in handles_and_counts)

    if total <= num_instances:
        pool = _load_full_pool(h5_dir, file_ids)
        return _sample_array_rows(pool, num_instances, rng=rng)

    tile_counts = [c for _, c in handles_and_counts]
    fractions = np.array(tile_counts, dtype=np.float64) / total
    allocs = np.round(fractions * num_instances).astype(int)
    diff = num_instances - allocs.sum()
    if diff != 0:
        order = np.argsort(-fractions)
        for j in range(abs(diff)):
            allocs[order[j % len(order)]] += np.sign(diff)

    chunks = []
    for (fid, n_tiles), n_sample in zip(handles_and_counts, allocs):
        if n_sample <= 0:
            continue
        n_sample = min(n_sample, n_tiles)
        if rng is None:
            indices = np.sort(np.random.choice(n_tiles, size=n_sample, replace=False))
        else:
            indices = np.sort(rng.choice(n_tiles, size=n_sample, replace=False))
        with h5py.File(h5_dir / f"{fid}.h5", "r") as f:
            chunks.append(f["features"][indices])

    return np.concatenate(chunks, axis=0)


def _load_full_pool(
    h5_dir: Path,
    file_ids: list[str],
) -> np.ndarray:
    """Load all tile features for a bag across one or more slide files."""
    chunks = []
    for fid in file_ids:
        with h5py.File(h5_dir / f"{fid}.h5", "r") as f:
            chunks.append(f["features"][:])
    if not chunks:
        raise ValueError("No slide files provided for bag.")
    return np.concatenate(chunks, axis=0)


def _sample_array_rows(
    arr: np.ndarray,
    num_instances: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Sample rows from a numpy array with replacement when needed."""
    n_rows = int(arr.shape[0])
    if n_rows <= 0:
        raise ValueError("Cannot sample from empty feature array.")
    if rng is None:
        indices = np.random.choice(n_rows, size=num_instances, replace=n_rows < num_instances)
    else:
        indices = rng.choice(n_rows, size=num_instances, replace=n_rows < num_instances)
    return arr[indices]


def _sample_tensor_rows(
    feats: torch.Tensor,
    num_instances: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> torch.Tensor:
    """Sample rows from a tensor with replacement when needed."""
    if feats.ndim != 2:
        raise ValueError(f"Expected 2D feature tensor, got shape={tuple(feats.shape)}")
    n_rows = int(feats.shape[0])
    if n_rows <= 0:
        raise ValueError("Cannot sample from empty feature tensor.")
    if rng is None:
        indices = np.random.choice(n_rows, size=num_instances, replace=n_rows < num_instances)
    else:
        indices = rng.choice(n_rows, size=num_instances, replace=n_rows < num_instances)
    idx_t = torch.from_numpy(indices).to(dtype=torch.int64)
    return feats.index_select(0, idx_t)


class EmbeddingBagDataset(torch.utils.data.Dataset):
    """Loads pre-computed Virchow tile embeddings.

    If ``cache_dir`` is provided and contains ``{bag_id}.pt`` files,
    cache entries are used when possible, but sampling remains stochastic
    per access so each epoch sees fresh tile subsets.
    """

    def __init__(
        self,
        manifest: pd.DataFrame,
        h5_dir: Path | str,
        num_instances: int = 4096,
        *,
        cache_dir: Optional[Path | str] = None,
        use_all_tiles: bool = False,
        deterministic: bool = False,
        deterministic_seed: int = 42,
        file_id_col: str = "file_uuid_original",
        label_col: str = "purity",
    ):
        self.h5_dir = Path(h5_dir)
        self.num_instances = num_instances
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_all_tiles = use_all_tiles
        self.deterministic = deterministic
        self.deterministic_seed = int(deterministic_seed)
        self.groups = _build_aliquot_groups(manifest, file_id_col, label_col)

    def _bag_rng(self, bag_id: str) -> Optional[np.random.Generator]:
        if not self.deterministic:
            return None
        token = f"{self.deterministic_seed}:{bag_id}".encode("utf-8")
        digest = hashlib.blake2b(token, digest_size=8).digest()
        seed = int.from_bytes(digest, byteorder="little", signed=False)
        return np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, float, float]:
        group = self.groups[idx]
        bag_id = group["bag_id"]
        label = float(group["label"])
        is_tumor = float(group.get("is_tumor", 1.0))
        rng = self._bag_rng(bag_id)

        # Full-bag mode for deterministic evaluation: no sub-sampling.
        if self.use_all_tiles:
            if self.cache_dir is not None:
                cache_path = self.cache_dir / f"{bag_id}.pt"
                if cache_path.exists():
                    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                    label = float(payload.get("label", label))
                    is_tumor = float(payload.get("is_tumor", is_tumor))
                    feats_pool = payload.get("feats_pool")
                    if isinstance(feats_pool, torch.Tensor):
                        return feats_pool.to(torch.float32), label, is_tumor
            feats = _load_full_pool(self.h5_dir, group["file_ids"])
            return torch.from_numpy(feats), label, is_tumor

        if self.cache_dir is not None:
            cache_path = self.cache_dir / f"{bag_id}.pt"
            if cache_path.exists():
                payload = torch.load(cache_path, map_location="cpu", weights_only=True)
                label = float(payload.get("label", label))
                is_tumor = float(payload.get("is_tumor", is_tumor))

                # New cache format: full bag pool -> sample fresh rows every access.
                feats_pool = payload.get("feats_pool")
                if isinstance(feats_pool, torch.Tensor):
                    sampled = _sample_tensor_rows(feats_pool, self.num_instances, rng=rng)
                    return sampled.to(torch.float32), label, is_tumor

                # Legacy cache format: fixed sample under "feats".
                feats = payload.get("feats")
                if isinstance(feats, torch.Tensor):
                    # If cache length differs, at least resample from cached rows.
                    if int(feats.shape[0]) != self.num_instances:
                        sampled = _sample_tensor_rows(feats, self.num_instances, rng=rng)
                        return sampled.to(torch.float32), label, is_tumor

                    # If exactly num_instances, resample from H5 for epoch-wise refresh.
                    try:
                        arr = _sample_smart(
                            self.h5_dir, group["file_ids"], self.num_instances, rng=rng
                        )
                        return torch.from_numpy(arr), label, is_tumor
                    except Exception as e:
                        _log.warning(
                            "Bag %s: H5 resample failed (%s), using legacy cached feats.",
                            bag_id, e,
                        )
                        return feats.to(torch.float32), label, is_tumor

        feats = _sample_smart(self.h5_dir, group["file_ids"], self.num_instances, rng=rng)
        return torch.from_numpy(feats), label, is_tumor


def custom_collate_fn(
    batch: list[tuple[torch.Tensor, float, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    feats_list, labels_list, is_tumor_list = zip(*batch)
    return (
        torch.stack(feats_list, dim=0),
        torch.tensor(labels_list, dtype=torch.float32),
        torch.tensor(is_tumor_list, dtype=torch.float32),
    )
