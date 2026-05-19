"""Datasets for EnsoCellularity training and inference.

The full Pan-Cancer label set is too large to concatenate into one in-memory
table. The training dataset therefore samples from one slide at a time:

* one Parquet file stores tile labels for one slide,
* one H5 file stores Virchow embeddings for the same slide,
* each ``__getitem__`` samples ``tiles_per_slide`` center tiles,
* 3x3 neighbor embeddings are gathered dynamically from the H5.
"""

from __future__ import annotations

import bisect
import logging
import os
import shutil
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd
import torch
import torch.utils.data

from enso_cellularity.labels import tile_grid_spec_from_h5_attrs

_log = logging.getLogger(__name__)

NEIGHBOR_OFFSETS: tuple[tuple[int, int], ...] = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 0),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

SLIDE_INDEX_COLUMNS = [
    "file_uuid_original",
    "barcode",
    "project_id",
    "case_id",
    "label_path",
    "h5_path",
    "label_url",
    "h5_url",
    "num_tiles",
    "has_h5",
]

TRAIN_LABEL_COLUMNS = [
    "file_uuid_original",
    "barcode",
    "project_id",
    "case_id",
    "embedding_index",
    "tile_y",
    "tile_x",
    "tile_x_level0",
    "tile_y_level0",
    "mpp_x",
    "mpp_y",
    "tile_area_mm2",
    "tissue_fraction",
    "exposure_mm2",
    "teacher_total_nuclei",
    "teacher_confidence",
    "teacher_disagreement",
    "quality_target",
    "quality_flags",
    "source",
    "count_bin",
]


@dataclass(frozen=True)
class SlideRecord:
    file_id: str
    barcode: str
    project_id: str
    case_id: str
    label_path: Path
    h5_path: Path
    num_tiles: int
    label_url: str = ""
    h5_url: str = ""


def _read_parquet_first_row(path: Path, columns: list[str]) -> tuple[dict[str, Any], int]:
    """Read the first row plus row count without loading a whole slide table."""

    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        num_rows = int(pf.metadata.num_rows)
        present = [c for c in columns if c in pf.schema.names]
        table = pf.read_row_group(0, columns=present).slice(0, 1)
        rows = table.to_pandas()
        return rows.iloc[0].to_dict(), num_rows
    except Exception:
        df = pd.read_parquet(path, columns=columns)
        return df.iloc[0].to_dict(), int(len(df))


def list_label_parquets(label_dir: Path | str) -> list[Path]:
    """Return one-slide label Parquets under a directory."""

    root = Path(label_dir)
    if not root.exists():
        raise FileNotFoundError(f"Label directory does not exist: {root}")
    paths = sorted(root.glob("*.parquet"))
    if not paths:
        paths = sorted((root / "by_slide").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No Parquet labels found under {root}")
    return paths


def build_slide_index(
    label_dir: Path | str,
    h5_dir: Path | str,
    *,
    max_slides: int | None = None,
) -> pd.DataFrame:
    """Build a compact one-row-per-slide index from per-slide labels."""

    h5_root = Path(h5_dir)
    rows: list[dict[str, Any]] = []
    for label_path in list_label_parquets(label_dir):
        first, num_tiles = _read_parquet_first_row(
            label_path,
            ["file_uuid_original", "barcode", "project_id", "case_id"],
        )
        file_id = str(first["file_uuid_original"])
        h5_path = h5_root / f"{file_id}.h5"
        rows.append(
            {
                "file_uuid_original": file_id,
                "barcode": str(first.get("barcode", "")),
                "project_id": str(first.get("project_id", "")),
                "case_id": str(first.get("case_id", "")),
                "label_path": str(label_path),
                "h5_path": str(h5_path),
                "label_url": "",
                "h5_url": "",
                "num_tiles": int(num_tiles),
                "has_h5": bool(h5_path.exists()),
            }
        )
        if max_slides is not None and len(rows) >= max_slides:
            break
    return pd.DataFrame(rows, columns=SLIDE_INDEX_COLUMNS)


def build_slide_index_from_completed_tsv(
    completed_tsv: Path | str,
    *,
    base_url: str = "https://vmshareddisk.blob.core.windows.net/data",
    h5_prefix: str = "embeddings_fp32",
) -> pd.DataFrame:
    """Build a direct-Blob slide index from processing ``completed.tsv``.

    The direct label builder's state file already contains the uploaded label
    Parquet URL and row count. This reconstructs the matching H5 URL so training
    can use ``CellularityBlobSlideTileDataset`` without blobfuse.
    """

    df = pd.read_csv(completed_tsv, sep="\t")
    if "status" in df.columns:
        df = df[df["status"] == "completed"].copy()
    df = df.drop_duplicates(subset=["file_id"], keep="last").reset_index(drop=True)
    base = base_url.rstrip("/")
    h5_prefix = h5_prefix.strip("/")

    rows: list[dict[str, Any]] = []
    for row in df.itertuples(index=False):
        file_id = str(getattr(row, "file_id"))
        barcode = str(getattr(row, "slide_barcode"))
        case_id = "-".join(barcode.split("-")[:3])
        rows.append(
            {
                "file_uuid_original": file_id,
                "barcode": barcode,
                "project_id": str(getattr(row, "project_id")),
                "case_id": case_id,
                "label_path": "",
                "h5_path": "",
                "label_url": str(getattr(row, "out_path")),
                "h5_url": f"{base}/{h5_prefix}/{file_id}.h5",
                "num_tiles": int(getattr(row, "rows")),
                "has_h5": True,
            }
        )
    return pd.DataFrame(rows, columns=SLIDE_INDEX_COLUMNS)


def load_slide_index(path: Path | str) -> pd.DataFrame:
    """Load a slide index from CSV or Parquet."""

    p = Path(path)
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)


def write_slide_index(index: pd.DataFrame, path: Path | str) -> None:
    """Write a slide index, choosing format from suffix."""

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        index.to_parquet(p, index=False)
    else:
        index.to_csv(p, index=False)


def _records_from_index(index: pd.DataFrame) -> list[SlideRecord]:
    records: list[SlideRecord] = []
    for row in index.itertuples(index=False):
        records.append(
            SlideRecord(
                file_id=str(getattr(row, "file_uuid_original")),
                barcode=str(getattr(row, "barcode")),
                project_id=str(getattr(row, "project_id")),
                case_id=str(getattr(row, "case_id")),
                label_path=Path(str(getattr(row, "label_path"))),
                h5_path=Path(str(getattr(row, "h5_path"))),
                num_tiles=int(getattr(row, "num_tiles")),
                label_url="" if pd.isna(getattr(row, "label_url", "")) else str(getattr(row, "label_url", "")),
                h5_url="" if pd.isna(getattr(row, "h5_url", "")) else str(getattr(row, "h5_url", "")),
            )
        )
    return records


def _available_columns(path: Path) -> set[str]:
    try:
        import pyarrow.parquet as pq

        return set(pq.ParquetFile(path).schema.names)
    except Exception:
        return set(pd.read_parquet(path).columns)


def _read_label_table(path: Path) -> pd.DataFrame:
    available = _available_columns(path)
    columns = [c for c in TRAIN_LABEL_COLUMNS if c in available]
    return pd.read_parquet(path, columns=columns)


def _sample_indices(
    labels: pd.DataFrame,
    n: int | None,
    *,
    rng: np.random.Generator,
    strategy: str,
) -> np.ndarray:
    total = len(labels)
    if n is None or n <= 0 or n >= total:
        return np.arange(total, dtype=np.int64)

    if strategy == "balanced_bins" and "count_bin" in labels.columns:
        groups = labels.groupby("count_bin", sort=True).indices
        bins = [np.asarray(idx, dtype=np.int64) for idx in groups.values() if len(idx) > 0]
        if bins:
            per_bin = max(1, int(np.ceil(n / len(bins))))
            chosen: list[np.ndarray] = []
            for idx in bins:
                size = min(per_bin, len(idx))
                chosen.append(rng.choice(idx, size=size, replace=False))
            out = np.concatenate(chosen)
            if len(out) < n:
                remaining = np.setdiff1d(np.arange(total, dtype=np.int64), out, assume_unique=False)
                extra_pool = remaining if len(remaining) else np.arange(total, dtype=np.int64)
                extra = rng.choice(extra_pool, size=n - len(out), replace=len(extra_pool) < n - len(out))
                out = np.concatenate([out, extra])
            rng.shuffle(out)
            return out[:n].astype(np.int64)

    return rng.choice(total, size=n, replace=total < n).astype(np.int64)


def _metadata_from_labels(labels: pd.DataFrame) -> np.ndarray:
    eps = 1e-8
    ref_mpp = 0.5
    ref_area = 0.012544
    mpp_x = labels["mpp_x"].to_numpy(dtype=np.float32, copy=False)
    mpp_y = labels["mpp_y"].to_numpy(dtype=np.float32, copy=False)
    tile_area = labels["tile_area_mm2"].to_numpy(dtype=np.float32, copy=False)
    tissue_fraction = np.ones(len(labels), dtype=np.float32)
    exposure = _full_tile_exposure_from_labels(labels)
    return np.stack(
        [
            np.log(np.clip(mpp_x, eps, None) / ref_mpp),
            np.log(np.clip(mpp_y, eps, None) / ref_mpp),
            np.clip(tissue_fraction, 0.0, 1.0),
            np.log(np.clip(tile_area, eps, None) / ref_area),
            np.log(np.clip(exposure, eps, None) / ref_area),
        ],
        axis=1,
    ).astype(np.float32)


def _full_tile_exposure_from_labels(labels: pd.DataFrame) -> np.ndarray:
    """Use full-tile exposure until true tissue fractions are available."""

    return labels["tile_area_mm2"].to_numpy(dtype=np.float32, copy=False)


def metadata_from_h5_attrs(attrs: Any, n_rows: int) -> tuple[np.ndarray, np.ndarray]:
    """Build normalized metadata and raw exposure for unlabeled H5 inference."""

    spec = tile_grid_spec_from_h5_attrs(attrs)
    tile_area = float(spec.tile_area_mm2)
    labels = pd.DataFrame(
        {
            "mpp_x": np.full(n_rows, spec.mpp_x, dtype=np.float32),
            "mpp_y": np.full(n_rows, spec.mpp_y, dtype=np.float32),
            "tissue_fraction": np.ones(n_rows, dtype=np.float32),
            "tile_area_mm2": np.full(n_rows, tile_area, dtype=np.float32),
            "exposure_mm2": np.full(n_rows, tile_area, dtype=np.float32),
        }
    )
    return _metadata_from_labels(labels), _full_tile_exposure_from_labels(labels)


def _quality_targets(labels: pd.DataFrame) -> np.ndarray:
    if "quality_target" in labels.columns:
        return labels["quality_target"].to_numpy(dtype=np.int64, copy=False)
    if "quality_flags" not in labels.columns:
        return np.zeros(len(labels), dtype=np.int64)
    flags = labels["quality_flags"].fillna("").astype(str).str.lower()
    targets = np.zeros(len(labels), dtype=np.int64)
    targets[flags.str.contains("background|empty", regex=True).to_numpy()] = 1
    targets[flags.str.contains("artifact|blur|necrosis|fold", regex=True).to_numpy()] = 2
    return targets


def build_neighbor_indices(
    tile_y: np.ndarray,
    tile_x: np.ndarray,
    embedding_index: np.ndarray,
    *,
    selected_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return H5 feature row indices and validity masks for 3x3 neighborhoods."""

    tile_y = np.asarray(tile_y, dtype=np.int64)
    tile_x = np.asarray(tile_x, dtype=np.int64)
    embedding_index = np.asarray(embedding_index, dtype=np.int64)
    if selected_positions is None:
        selected_positions = np.arange(len(tile_y), dtype=np.int64)
    else:
        selected_positions = np.asarray(selected_positions, dtype=np.int64)

    n = len(selected_positions)
    centers = embedding_index[selected_positions]
    neighbor_idx = np.repeat(centers[:, None], 9, axis=1)
    valid = np.zeros((n, 9), dtype=bool)

    min_y = int(tile_y.min(initial=0))
    max_y = int(tile_y.max(initial=0))
    min_x = int(tile_x.min(initial=0))
    max_x = int(tile_x.max(initial=0))
    height = max_y - min_y + 1
    width = max_x - min_x + 1

    # Pan-Cancer tile grids are compact WSI grids, so a dense lookup table is
    # much faster than rebuilding a Python dict for every sampled slide.
    if height > 0 and width > 0 and height * width <= max(len(tile_y) * 4, 1_000_000):
        grid = np.full((height, width), -1, dtype=np.int64)
        grid[tile_y - min_y, tile_x - min_x] = embedding_index
        sel_y = tile_y[selected_positions] - min_y
        sel_x = tile_x[selected_positions] - min_x
        for j, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
            yy = sel_y + dy
            xx = sel_x + dx
            in_bounds = (yy >= 0) & (yy < height) & (xx >= 0) & (xx < width)
            found = np.full(n, -1, dtype=np.int64)
            if in_bounds.any():
                found[in_bounds] = grid[yy[in_bounds], xx[in_bounds]]
            ok = found >= 0
            neighbor_idx[ok, j] = found[ok]
            valid[:, j] = ok
    else:
        lookup = {
            (int(y), int(x)): int(ei)
            for y, x, ei in zip(tile_y.tolist(), tile_x.tolist(), embedding_index.tolist())
        }
        for out_i, pos in enumerate(selected_positions):
            cy = int(tile_y[pos])
            cx = int(tile_x[pos])
            center = int(embedding_index[pos])
            for j, (dy, dx) in enumerate(NEIGHBOR_OFFSETS):
                found = lookup.get((cy + dy, cx + dx))
                if found is not None:
                    neighbor_idx[out_i, j] = found
                    valid[out_i, j] = True

    neighbor_idx[:, 4] = centers
    valid[:, 4] = True
    return neighbor_idx, valid


def read_h5_features_by_index(h5_path: Path, indices: np.ndarray) -> np.ndarray:
    """Read arbitrary H5 feature rows using one sorted unique selection."""

    flat = np.asarray(indices, dtype=np.int64).reshape(-1)
    unique, inverse = np.unique(flat, return_inverse=True)
    with h5py.File(h5_path, "r") as h5:
        features = h5["features"]
        if unique.min(initial=0) < 0 or unique.max(initial=0) >= int(features.shape[0]):
            raise IndexError(f"Feature index out of range for {h5_path}")
        values = features[unique]
    return values[inverse].reshape(*indices.shape, values.shape[-1]).astype(np.float32, copy=False)


class CellularitySlideTileDataset(torch.utils.data.Dataset):
    """Sample training tiles from per-slide Parquet labels and H5 embeddings."""

    def __init__(
        self,
        slide_index: pd.DataFrame,
        *,
        tiles_per_slide: int | None = 8192,
        all_tiles_chunk_size: int | None = None,
        sample_strategy: str = "balanced_bins",
        training: bool = True,
        seed: int = 42,
        label_cache_size: int = 2,
    ):
        index = slide_index.copy()
        if "has_h5" in index.columns:
            index = index[index["has_h5"]].copy()
        self.records = _records_from_index(index.reset_index(drop=True))
        if not self.records:
            raise ValueError("CellularitySlideTileDataset received no usable slides.")
        self.tiles_per_slide = tiles_per_slide
        self.all_tiles_chunk_size = all_tiles_chunk_size
        self.sample_strategy = sample_strategy
        self.training = training
        self.seed = int(seed)
        self.epoch = 0
        self.label_cache_size = max(0, int(label_cache_size))
        self._label_cache: OrderedDict[Path, pd.DataFrame] = OrderedDict()
        self.items = self._build_items()

    def _build_items(self) -> list[tuple[int, int | None, int | None]]:
        if self.training or self.tiles_per_slide is None or self.tiles_per_slide > 0:
            return [(i, None, None) for i in range(len(self.records))]
        chunk_size = 0 if self.all_tiles_chunk_size is None else int(self.all_tiles_chunk_size)
        if chunk_size <= 0:
            return [(i, None, None) for i in range(len(self.records))]
        items: list[tuple[int, int | None, int | None]] = []
        for i, record in enumerate(self.records):
            for start in range(0, int(record.num_tiles), chunk_size):
                items.append((i, start, min(start + chunk_size, int(record.num_tiles))))
        return items

    def __len__(self) -> int:
        return len(self.items)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        seed = self.seed + self.epoch * 1_000_003 + idx * 9_176 + worker_id * 101
        return np.random.default_rng(seed)

    def _labels(self, path: Path) -> pd.DataFrame:
        if self.label_cache_size > 0 and path in self._label_cache:
            labels = self._label_cache.pop(path)
            self._label_cache[path] = labels
            return labels
        labels = _read_label_table(path)
        if self.label_cache_size > 0:
            self._label_cache[path] = labels
            while len(self._label_cache) > self.label_cache_size:
                self._label_cache.popitem(last=False)
        return labels

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record_idx, chunk_start, chunk_stop = self.items[idx]
        record = self.records[record_idx]
        labels = self._labels(record.label_path)
        if chunk_start is not None and chunk_stop is not None:
            start = min(int(chunk_start), len(labels))
            stop = min(int(chunk_stop), len(labels))
            selected = np.arange(start, stop, dtype=np.int64)
        else:
            rng = self._rng(record_idx)
            selected = _sample_indices(
                labels,
                self.tiles_per_slide,
                rng=rng,
                strategy=self.sample_strategy if self.training else "uniform",
            )
        selected_labels = labels.iloc[selected].reset_index(drop=True)

        neighbor_idx, valid9 = build_neighbor_indices(
            labels["tile_y"].to_numpy(),
            labels["tile_x"].to_numpy(),
            labels["embedding_index"].to_numpy(),
            selected_positions=selected,
        )
        x9 = read_h5_features_by_index(record.h5_path, neighbor_idx)
        metadata = _metadata_from_labels(selected_labels)
        exposure = _full_tile_exposure_from_labels(selected_labels)
        y_count = selected_labels["teacher_total_nuclei"].to_numpy(dtype=np.float32, copy=False)
        teacher_confidence = selected_labels.get(
            "teacher_confidence",
            pd.Series(1.0, index=selected_labels.index),
        ).to_numpy(dtype=np.float32, copy=False)
        quality_target = _quality_targets(selected_labels)
        count_bin = selected_labels.get(
            "count_bin",
            pd.Series(0, index=selected_labels.index),
        ).to_numpy(dtype=np.int64, copy=False)

        return {
            "x9": torch.from_numpy(x9),
            "valid9": torch.from_numpy(valid9),
            "metadata": torch.from_numpy(metadata),
            "exposure_mm2": torch.from_numpy(exposure.astype(np.float32, copy=False)),
            "y_count": torch.from_numpy(y_count.astype(np.float32, copy=False)),
            "teacher_confidence": torch.from_numpy(teacher_confidence.astype(np.float32, copy=False)),
            "quality_target": torch.from_numpy(quality_target.astype(np.int64, copy=False)),
            "count_bin": torch.from_numpy(count_bin.astype(np.int64, copy=False)),
            "file_id": [record.file_id] * len(selected_labels),
            "barcode": [record.barcode] * len(selected_labels),
            "embedding_index": selected_labels["embedding_index"].astype(int).tolist(),
        }


class CellularityBlobSlideTileDataset(torch.utils.data.Dataset):
    """Sample slides through direct Azure Blob endpoint downloads.

    This is the no-blobfuse path for massive remote runs. Each item downloads
    one slide's label Parquet and H5 to local scratch through the direct Blob
    endpoint, samples the 3x3 training tensors locally, and then removes the
    scratch files unless ``keep_cache`` is enabled.
    """

    def __init__(
        self,
        slide_index: pd.DataFrame,
        *,
        scratch_dir: Path | str,
        tiles_per_slide: int | None = 8192,
        sample_strategy: str = "balanced_bins",
        training: bool = True,
        seed: int = 42,
        azcopy_bin: str = "azcopy",
        azcopy_auto_login_type: str = "MSI",
        transfer_mode: str = "azcopy",
        sdk_max_concurrency: int = 12,
        keep_cache: bool = False,
    ):
        self.records = _records_from_index(slide_index.reset_index(drop=True))
        if not self.records:
            raise ValueError("CellularityBlobSlideTileDataset received no slides.")
        missing_urls = [r.file_id for r in self.records if not r.label_url or not r.h5_url]
        if missing_urls:
            raise ValueError(
                "Blob dataset requires label_url and h5_url columns. "
                f"Missing examples: {missing_urls[:5]}"
            )
        self.scratch_dir = Path(scratch_dir)
        self.tiles_per_slide = tiles_per_slide
        self.sample_strategy = sample_strategy
        self.training = training
        self.seed = int(seed)
        self.epoch = 0
        self.azcopy_bin = azcopy_bin
        self.azcopy_auto_login_type = azcopy_auto_login_type
        self.transfer_mode = transfer_mode
        self.sdk_max_concurrency = max(1, int(sdk_max_concurrency))
        self.keep_cache = keep_cache
        self._sdk_credential = None

    def __len__(self) -> int:
        return len(self.records)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _rng(self, idx: int) -> np.random.Generator:
        worker = torch.utils.data.get_worker_info()
        worker_id = 0 if worker is None else int(worker.id)
        seed = self.seed + self.epoch * 1_000_003 + idx * 9_176 + worker_id * 101
        return np.random.default_rng(seed)

    def _azcopy(self, src: str, dest: Path) -> None:
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        if self.azcopy_auto_login_type:
            env["AZCOPY_AUTO_LOGIN_TYPE"] = self.azcopy_auto_login_type
        proc = subprocess.run(
            [self.azcopy_bin, "copy", src, str(dest), "--overwrite=true", "--log-level=ERROR"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
        if proc.returncode != 0:
            tail = proc.stdout[-4000:] if proc.stdout else ""
            raise RuntimeError(f"azcopy failed for {src} -> {dest}\n{tail}")

    def _sdk_download(self, src: str, dest: Path) -> None:
        if dest.exists():
            return
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
            from azure.storage.blob import BlobClient
        except ImportError as exc:
            raise RuntimeError(
                "Azure SDK transfer mode requires azure-identity and azure-storage-blob. "
                "Install them or use transfer_mode='azcopy'."
            ) from exc

        if self._sdk_credential is None:
            if self.azcopy_auto_login_type.upper() == "MSI":
                self._sdk_credential = ManagedIdentityCredential()
            else:
                self._sdk_credential = DefaultAzureCredential(exclude_interactive_browser_credential=True)

        tmp = dest.with_name(f".{dest.name}.{os.getpid()}.part")
        try:
            blob = BlobClient.from_blob_url(src, credential=self._sdk_credential)
            with open(tmp, "wb") as f:
                stream = blob.download_blob(max_concurrency=self.sdk_max_concurrency)
                stream.readinto(f)
            tmp.replace(dest)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def _download(self, src: str, dest: Path) -> None:
        if self.transfer_mode == "azcopy":
            self._azcopy(src, dest)
        elif self.transfer_mode == "sdk":
            self._sdk_download(src, dest)
        else:
            raise ValueError(f"Unknown Blob transfer mode: {self.transfer_mode}")

    def _local_record(self, record: SlideRecord) -> SlideRecord:
        local_dir = self.scratch_dir / record.file_id
        label_path = local_dir / record.label_url.rstrip("/").split("/")[-1]
        h5_path = local_dir / f"{record.file_id}.h5"
        self._download(record.label_url, label_path)
        self._download(record.h5_url, h5_path)
        return SlideRecord(
            file_id=record.file_id,
            barcode=record.barcode,
            project_id=record.project_id,
            case_id=record.case_id,
            label_path=label_path,
            h5_path=h5_path,
            num_tiles=record.num_tiles,
            label_url=record.label_url,
            h5_url=record.h5_url,
        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        local_record = self._local_record(record)
        try:
            labels = _read_label_table(local_record.label_path)
            rng = self._rng(idx)
            selected = _sample_indices(
                labels,
                self.tiles_per_slide,
                rng=rng,
                strategy=self.sample_strategy if self.training else "uniform",
            )
            selected_labels = labels.iloc[selected].reset_index(drop=True)

            neighbor_idx, valid9 = build_neighbor_indices(
                labels["tile_y"].to_numpy(),
                labels["tile_x"].to_numpy(),
                labels["embedding_index"].to_numpy(),
                selected_positions=selected,
            )
            x9 = read_h5_features_by_index(local_record.h5_path, neighbor_idx)
            metadata = _metadata_from_labels(selected_labels)
            exposure = _full_tile_exposure_from_labels(selected_labels)
            y_count = selected_labels["teacher_total_nuclei"].to_numpy(dtype=np.float32, copy=False)
            teacher_confidence = selected_labels.get(
                "teacher_confidence",
                pd.Series(1.0, index=selected_labels.index),
            ).to_numpy(dtype=np.float32, copy=False)
            quality_target = _quality_targets(selected_labels)
            count_bin = selected_labels.get(
                "count_bin",
                pd.Series(0, index=selected_labels.index),
            ).to_numpy(dtype=np.int64, copy=False)

            return {
                "x9": torch.from_numpy(x9),
                "valid9": torch.from_numpy(valid9),
                "metadata": torch.from_numpy(metadata),
                "exposure_mm2": torch.from_numpy(exposure.astype(np.float32, copy=False)),
                "y_count": torch.from_numpy(y_count.astype(np.float32, copy=False)),
                "teacher_confidence": torch.from_numpy(
                    teacher_confidence.astype(np.float32, copy=False)
                ),
                "quality_target": torch.from_numpy(quality_target.astype(np.int64, copy=False)),
                "count_bin": torch.from_numpy(count_bin.astype(np.int64, copy=False)),
                "file_id": [record.file_id] * len(selected_labels),
                "barcode": [record.barcode] * len(selected_labels),
                "embedding_index": selected_labels["embedding_index"].astype(int).tolist(),
            }
        finally:
            if not self.keep_cache:
                shutil.rmtree(self.scratch_dir / record.file_id, ignore_errors=True)


class CellularityTileDataset(torch.utils.data.Dataset):
    """Map-style tile dataset for small tests/debugging.

    This dataset indexes every tile row globally and is not intended for the
    full 100M+ row Pan-Cancer training set.
    """

    def __init__(self, slide_index: pd.DataFrame):
        index = slide_index.copy()
        if "has_h5" in index.columns:
            index = index[index["has_h5"]].copy()
        self.records = _records_from_index(index.reset_index(drop=True))
        self.cumulative: list[int] = []
        total = 0
        for record in self.records:
            total += int(record.num_tiles)
            self.cumulative.append(total)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, idx: int) -> dict[str, Any]:
        slide_i = bisect.bisect_right(self.cumulative, idx)
        prev = 0 if slide_i == 0 else self.cumulative[slide_i - 1]
        local = idx - prev
        ds = CellularitySlideTileDataset(
            pd.DataFrame([self.records[slide_i].__dict__]).rename(
                columns={"file_id": "file_uuid_original"}
            ),
            tiles_per_slide=1,
            training=False,
        )
        ds._labels(self.records[slide_i].label_path)
        labels = ds._labels(self.records[slide_i].label_path)
        ds.tiles_per_slide = 1
        selected = np.array([local], dtype=np.int64)
        record = self.records[slide_i]
        neighbor_idx, valid9 = build_neighbor_indices(
            labels["tile_y"].to_numpy(),
            labels["tile_x"].to_numpy(),
            labels["embedding_index"].to_numpy(),
            selected_positions=selected,
        )
        row = labels.iloc[selected].reset_index(drop=True)
        x9 = read_h5_features_by_index(record.h5_path, neighbor_idx)
        return cellularity_collate(
            [
                {
                    "x9": torch.from_numpy(x9),
                    "valid9": torch.from_numpy(valid9),
                    "metadata": torch.from_numpy(_metadata_from_labels(row)),
                    "exposure_mm2": torch.from_numpy(_full_tile_exposure_from_labels(row)),
                    "y_count": torch.from_numpy(
                        row["teacher_total_nuclei"].to_numpy(dtype=np.float32, copy=False)
                    ),
                    "teacher_confidence": torch.ones(1, dtype=torch.float32),
                    "quality_target": torch.from_numpy(_quality_targets(row)),
                    "count_bin": torch.from_numpy(
                        row.get("count_bin", pd.Series(0, index=row.index)).to_numpy(dtype=np.int64)
                    ),
                    "file_id": [record.file_id],
                    "barcode": [record.barcode],
                    "embedding_index": row["embedding_index"].astype(int).tolist(),
                }
            ]
        )


def cellularity_collate(batch: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Flatten slide-sampled items into one tile batch."""

    items = list(batch)
    if len(items) == 1:
        return dict(items[0])

    tensor_keys = [
        "x9",
        "valid9",
        "metadata",
        "exposure_mm2",
        "y_count",
        "teacher_confidence",
        "quality_target",
        "count_bin",
    ]
    out: dict[str, Any] = {}
    for key in tensor_keys:
        out[key] = torch.cat([item[key] for item in items], dim=0)
    for key in ["file_id", "barcode", "embedding_index"]:
        values: list[Any] = []
        for item in items:
            values.extend(item[key])
        out[key] = values
    return out
