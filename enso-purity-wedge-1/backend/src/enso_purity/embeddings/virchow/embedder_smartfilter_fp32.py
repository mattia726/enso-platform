
import os
import time
import math
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import h5py
import numpy as np
import pandas as pd
import torch
import openslide
import cv2
import timm
from timm.data import resolve_data_config
from timm.layers import SwiGLUPacked
from huggingface_hub import login
from google.cloud import storage

# ============================ USER CONFIG ============================

# TCGA public bucket
SOURCE_BUCKET_NAME = "gdc-tcga-phs000178-open"
MASTER_CSV_NAME = "bucket_physical_scan.csv"

# Model
MODEL_NAME = "hf-hub:paige-ai/Virchow"

# Model geometry
TILE_SIZE = 224
TARGET_MPP = 0.5

# ---------------- Filter C: Smart Saturation + Intensity ----------------
# Keep if: (Value < VAL_THRESH) OR (Saturation > SAT_THRESH)
# Keep tile if at least KEEP_FRAC of the (padded) tile is "interesting".
VAL_THRESH = 205
SAT_THRESH = 15
KEEP_FRAC = 0.02

# Thumbnail settings for prefilter
THUMB_MAX_DIM = 6000                  # max side length of thumbnail
THUMB_PREFILTER_MAX_MPX = 1500.0      # if chosen level MPx > this, skip thumbnail prefilter (tile-wise filter fallback)
THUMB_STRIPE_MAX_MPX = 60.0           # cap stripe pixels at chosen level (memory/time knob)

# Tile count sanity
MIN_TILES = 50

# MPP policy
MPP_TOL = 0.02          # ±2% band around 0.5 where we skip resampling for speed
MAX_MPP_ALLOWED = 0.51 # no-upscaling guardrail (override via CLI if needed)

# Runtime
BATCH_SIZE = 256
NUM_WORKERS = 8

# GCS chunk size: must be multiple of 256 KiB (API requirement)
GCS_CHUNK_SIZE = 8 * 1024 * 1024  # 8 MiB

# Log upload
LOG_UPLOAD_EVERY = 5  # upload log every N messages (small file, overwrite)

# =====================================================================

# Avoid OpenCV thread oversubscription with multi-worker DataLoader
try:
    cv2.setNumThreads(1)
except Exception:
    pass


# ============================ SIMPLE RUN LOGGER =============================

class RunLogger:
    """
    Minimal tee-logger:
      - prints to stdout
      - appends to a local file
      - periodically uploads the file to GCS (overwrite remote, effectively "append" semantics)
    """
    def __init__(self, local_path: Path, gcs_bucket=None, gcs_key: Optional[str] = None, upload_every: int = LOG_UPLOAD_EVERY):
        self.local_path = Path(local_path)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(self.local_path, "a", buffering=1, encoding="utf-8")
        self.bucket = gcs_bucket
        self.gcs_key = gcs_key
        self.upload_every = max(1, int(upload_every))
        self._n = 0

    def _upload(self):
        if self.bucket is None or self.gcs_key is None:
            return
        try:
            blob = self.bucket.blob(self.gcs_key)
            blob.chunk_size = GCS_CHUNK_SIZE
            blob.upload_from_filename(str(self.local_path))
        except Exception as e:
            # avoid recursive logging errors
            print(f"⚠️ LOG UPLOAD ERR: {e}", flush=True)

    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{ts} {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self._n += 1
        if self._n % self.upload_every == 0:
            self._upload()

    def close(self):
        try:
            self._upload()
        finally:
            try:
                self.f.close()
            except Exception:
                pass


class SummaryCSV:
    """
    Append-only CSV summary with periodic upload to GCS.
    """
    def __init__(self, local_path: Path, header: Tuple[str, ...], gcs_bucket=None, gcs_key: Optional[str] = None, upload_every_rows: int = 25):
        self.local_path = Path(local_path)
        self.local_path.parent.mkdir(parents=True, exist_ok=True)
        self.header = header
        self.bucket = gcs_bucket
        self.gcs_key = gcs_key
        self.upload_every_rows = max(1, int(upload_every_rows))
        self._rows = 0

        # create file with header if absent
        if not self.local_path.exists():
            with open(self.local_path, "w", encoding="utf-8") as f:
                f.write(",".join(self.header) + "\n")

        self.f = open(self.local_path, "a", buffering=1, encoding="utf-8")

    def _upload(self):
        if self.bucket is None or self.gcs_key is None:
            return
        try:
            blob = self.bucket.blob(self.gcs_key)
            blob.chunk_size = GCS_CHUNK_SIZE
            blob.upload_from_filename(str(self.local_path))
        except Exception as e:
            print(f"⚠️ SUMMARY UPLOAD ERR: {e}", flush=True)

    def append(self, row: Dict[str, Any]):
        values = []
        for k in self.header:
            v = row.get(k, "")
            if isinstance(v, float):
                values.append(f"{v:.6g}")
            else:
                values.append(str(v))
        self.f.write(",".join(values) + "\n")
        self._rows += 1
        if self._rows % self.upload_every_rows == 0:
            self._upload()

    def close(self):
        try:
            self._upload()
        finally:
            try:
                self.f.close()
            except Exception:
                pass


# ============================ GCS UTILS =============================

def get_shard_batch(df: pd.DataFrame, num_shards: int, shard_id: int) -> pd.DataFrame:
    if shard_id >= num_shards:
        raise ValueError(f"Shard ID {shard_id} >= Num Shards {num_shards}")
    shards = np.array_split(df, num_shards)
    return shards[shard_id]


def download_master_csv_if_needed(bucket_out, local_path: str) -> None:
    if os.path.exists(local_path):
        return
    blob = bucket_out.blob(MASTER_CSV_NAME)
    if not blob.exists():
        raise FileNotFoundError(f"❌ Missing {MASTER_CSV_NAME} in output bucket.")
    blob.download_to_filename(local_path)


def blob_exists(bucket, remote_path: str, client: storage.Client) -> bool:
    return bucket.blob(remote_path).exists(client)


def download_blob_to_file(bucket, blob_path: str, local_path: str) -> Tuple[float, float]:
    t0 = time.time()
    blob = bucket.blob(blob_path)
    blob.chunk_size = GCS_CHUNK_SIZE
    blob.download_to_filename(local_path)
    dt = time.time() - t0

    size_mb = os.path.getsize(local_path) / (1024 * 1024) if os.path.exists(local_path) else 0.0
    return size_mb, dt


def upload_file_to_bucket(bucket, local_path: str, remote_path: str) -> None:
    blob = bucket.blob(remote_path)
    blob.chunk_size = GCS_CHUNK_SIZE
    blob.upload_from_filename(local_path)


# ============================ SLIDE METADATA =========================

def _try_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def get_slide_mpp(slide: openslide.OpenSlide) -> Optional[float]:
    """
    Prefer OpenSlide standard keys; fall back to common vendor keys.
    Returns scalar mpp (assumes isotropic; user verified mpp.x == mpp.y for this bucket).
    """
    props = slide.properties
    mx = _try_float(props.get("openslide.mpp-x"))
    my = _try_float(props.get("openslide.mpp-y"))

    if mx is None or my is None:
        alt = _try_float(props.get("aperio.MPP"))
        if alt is not None:
            mx = my = alt

    if mx is None and my is None:
        return None
    if mx is None:
        return my
    if my is None:
        return mx

    return 0.5 * (mx + my)


def get_objective_power(slide: openslide.OpenSlide) -> Optional[float]:
    props = slide.properties
    v = _try_float(props.get("openslide.objective-power"))
    if v is not None:
        return v
    v = _try_float(props.get("aperio.AppMag"))
    if v is not None:
        return v
    return None


# ============================ TILE PLANNING ===========================

@dataclass(frozen=True)
class TilePlan:
    slide_mpp: float
    target_mpp: float
    tol: float
    max_mpp_allowed: float
    needs_downsample: bool
    extracted_level0_size: int   # size read from L0 (square)
    output_size: int             # model input size (224)
    stride_level0: int           # non-overlapping stride in L0 pixels
    effective_mpp: float         # mpp after (optional) downsample


def plan_tile_read(
    slide_mpp: float,
    *,
    target_mpp: float = TARGET_MPP,
    output_size: int = TILE_SIZE,
    tol: float = MPP_TOL,
    max_mpp_allowed: float = MAX_MPP_ALLOWED,
) -> Optional[TilePlan]:
    """
    Rules:
      - Never upsample.
      - If slide_mpp > max_mpp_allowed: skip.
      - If slide_mpp >= target_mpp OR within ±tol of target_mpp -> keep as-is (no resampling).
      - Else: downsample to target_mpp by reading a larger window at level 0 and resizing down with INTER_AREA.
    """
    if slide_mpp is None:
        return None
    if slide_mpp > max_mpp_allowed:
        return None

    lo = target_mpp * (1.0 - tol)
    hi = target_mpp * (1.0 + tol)

    within_band = (lo <= slide_mpp <= hi)

    if slide_mpp >= target_mpp or within_band:
        extracted = output_size
        needs = False
        eff = slide_mpp
    else:
        # slide_mpp is finer than target: we can downsample
        scale = target_mpp / slide_mpp  # > 1
        extracted = int(round(output_size * scale))
        extracted = max(extracted, output_size)

        needs = True
        eff = slide_mpp * (extracted / output_size)

        # Guard: if rounding pushed eff slightly above max_mpp_allowed, decrement extracted
        while extracted > output_size and eff > max_mpp_allowed:
            extracted -= 1
            eff = slide_mpp * (extracted / output_size)

    return TilePlan(
        slide_mpp=float(slide_mpp),
        target_mpp=float(target_mpp),
        tol=float(tol),
        max_mpp_allowed=float(max_mpp_allowed),
        needs_downsample=needs,
        extracted_level0_size=int(extracted),
        output_size=int(output_size),
        stride_level0=int(extracted),
        effective_mpp=float(eff),
    )


# ============================ PADDING GRID ===========================

@dataclass(frozen=True)
class GridSpec:
    stride_L0: int
    pad_left_L0: int
    pad_top_L0: int
    W0: int
    H0: int
    Wpad: int
    Hpad: int
    nx: int
    ny: int


def compute_symmetric_padding_grid(W0: int, H0: int, stride: int) -> GridSpec:
    Wpad = int(math.ceil(W0 / stride) * stride)
    Hpad = int(math.ceil(H0 / stride) * stride)

    pad_w = Wpad - W0
    pad_h = Hpad - H0

    pad_left = pad_w // 2
    pad_top = pad_h // 2

    nx = Wpad // stride
    ny = Hpad // stride

    return GridSpec(
        stride_L0=int(stride),
        pad_left_L0=int(pad_left),
        pad_top_L0=int(pad_top),
        W0=int(W0),
        H0=int(H0),
        Wpad=int(Wpad),
        Hpad=int(Hpad),
        nx=int(nx),
        ny=int(ny),
    )


def rgba_to_rgb_composite_white(rgba: np.ndarray) -> np.ndarray:
    """
    Fast alpha compositing onto white background.
    Input: HxWx4 uint8 (OpenSlide read_region RGBA)
    Output: HxWx3 uint8
    """
    if rgba.ndim != 3 or rgba.shape[2] not in (3, 4):
        raise ValueError(f"Expected HxWx3/4, got {rgba.shape}")

    if rgba.shape[2] == 3:
        return rgba

    rgb = rgba[..., :3].astype(np.float32)
    a = rgba[..., 3:4].astype(np.float32) / 255.0
    out = rgb * a + 255.0 * (1.0 - a)
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


# ============================ THUMBNAIL (FAST, INTER_AREA) ===========================

def choose_level_no_upsample(slide: openslide.OpenSlide, downsample_req: float) -> int:
    """
    Choose the coarsest pyramid level whose downsample <= downsample_req.
    This avoids upsampling when resizing to the requested thumbnail size.
    """
    ds = [float(d) for d in slide.level_downsamples]
    candidates = [i for i, d in enumerate(ds) if d <= downsample_req]
    return max(candidates) if candidates else 0


def make_thumbnail_inter_area_stream(
    slide: openslide.OpenSlide,
    max_dim: int = THUMB_MAX_DIM,
    max_level_mpx: float = THUMB_PREFILTER_MAX_MPX,
    stripe_max_mpx: float = THUMB_STRIPE_MAX_MPX,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """
    Build an RGB thumbnail using:
      - choose best pyramid level without upsampling for requested (Tw,Th)
      - stream read in horizontal stripes
      - resize each stripe with cv2.INTER_AREA to output

    Returns:
      thumb_rgb (Ht, Wt, 3) uint8 or None if skipped
      info dict containing chosen level and MPx estimates
    """
    W0, H0 = slide.dimensions
    scale = min(float(max_dim) / float(max(W0, H0)), 1.0)
    Tw = max(1, int(round(W0 * scale)))
    Th = max(1, int(round(H0 * scale)))

    downsample_req = max(W0 / Tw, H0 / Th)

    lvl = choose_level_no_upsample(slide, downsample_req)
    lvl_w, lvl_h = slide.level_dimensions[lvl]
    lvl_down = float(slide.level_downsamples[lvl])
    lvl_mpx = (float(lvl_w) * float(lvl_h)) / 1e6

    info = {
        "thumb_max_dim": int(max_dim),
        "thumb_out_w": int(Tw),
        "thumb_out_h": int(Th),
        "thumb_downsample_req": float(downsample_req),
        "thumb_level": int(lvl),
        "thumb_level_w": int(lvl_w),
        "thumb_level_h": int(lvl_h),
        "thumb_level_downsample": float(lvl_down),
        "thumb_level_mpx": float(lvl_mpx),
        "thumb_skipped": False,
        "thumb_skip_reason": "",
    }

    # If chosen level is too huge, skip thumbnail prefilter (fallback to tilewise filter)
    if lvl_mpx > float(max_level_mpx):
        info["thumb_skipped"] = True
        info["thumb_skip_reason"] = f"level_mpx>{max_level_mpx}"
        return None, info

    # Adaptive stripe height to cap per-stripe pixels (memory)
    stripe_max_px = max(1.0, float(stripe_max_mpx) * 1e6)
    stripe_h = int(max(1, min(lvl_h, int(stripe_max_px / max(lvl_w, 1)))))
    # Safety: avoid pathological tiny stripes unless unavoidable
    stripe_h = max(64, stripe_h) if lvl_h >= 64 else stripe_h
    stripe_h = min(stripe_h, lvl_h)

    out = np.empty((Th, Tw, 3), dtype=np.uint8)

    y_lvl = 0
    while y_lvl < lvl_h:
        h = min(stripe_h, lvl_h - y_lvl)

        # map y in chosen level -> y in level0 coords (read_region location is in level0 coords)
        y0_l0 = int(round(y_lvl * lvl_down))

        pil_rgba = slide.read_region((0, y0_l0), lvl, (lvl_w, h))
        rgba = np.asarray(pil_rgba, dtype=np.uint8)  # (h, lvl_w, 4)
        rgb = rgba_to_rgb_composite_white(rgba)

        # map stripe -> output rows (avoid gaps/overlap)
        y0_out = (y_lvl * Th) // lvl_h
        y1_out = ((y_lvl + h) * Th) // lvl_h
        if y_lvl + h >= lvl_h:
            y1_out = Th
        out_h = int(max(1, y1_out - y0_out))

        # INTER_AREA for downsampling; if upsampling (rare), use INTER_LINEAR
        interp = cv2.INTER_AREA
        if out_h > h or Tw > lvl_w:
            interp = cv2.INTER_LINEAR

        strip_resized = cv2.resize(rgb, (Tw, out_h), interpolation=interp)
        out[y0_out:y0_out + out_h, :, :] = strip_resized

        # cleanup
        del pil_rgba, rgba, rgb, strip_resized
        y_lvl += h

    return out, info


# ============================ FILTER INTEGRAL UTILS ===========================

def integral_image_2d(mask_bool: np.ndarray) -> np.ndarray:
    """
    mask_bool: HxW boolean
    returns padded integral image uint32 of shape (H+1, W+1)
    """
    m = mask_bool.astype(np.uint32)
    S = np.pad(m, ((1, 0), (1, 0)), mode="constant", constant_values=0)
    return S.cumsum(axis=0).cumsum(axis=1)


def rect_sum_integral(S: np.ndarray, x0: np.ndarray, y0: np.ndarray, x1: np.ndarray, y1: np.ndarray) -> np.ndarray:
    """
    Vectorized rectangle sum for integral image.
    Inputs x0,x1,y0,y1 are arrays of same shape, coords in [0,W]/[0,H] for original image.
    S must be padded integral of shape (H+1, W+1).
    """
    return S[y1, x1] - S[y0, x1] - S[y1, x0] + S[y0, x0]


# ============================ COORD GENERATION ===========================

def generate_all_coords_with_padding(grid: GridSpec) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Returns all tile top-left coordinates on padded grid:
      - coords_l0: (N,2) int32 (x_level0, y_level0) can be negative due to symmetric padding
      - coords_target: (N,2) int32 (row_px, col_px) in virtual target space (row-major)
      - total_grids: nx*ny
    """
    stride = grid.stride_L0
    nx, ny = grid.nx, grid.ny

    cols = np.arange(nx, dtype=np.int64)
    rows = np.arange(ny, dtype=np.int64)

    x_l0 = cols * stride - grid.pad_left_L0  # (nx,)
    y_l0 = rows * stride - grid.pad_top_L0   # (ny,)

    xx, yy = np.meshgrid(x_l0, y_l0)  # (ny,nx)
    coords_l0 = np.stack([xx, yy], axis=-1).reshape(-1, 2).astype(np.int32)

    row_px = (rows * TILE_SIZE).astype(np.int64)
    col_px = (cols * TILE_SIZE).astype(np.int64)
    rr, cc = np.meshgrid(row_px, col_px, indexing="ij")  # (ny,nx)
    coords_target = np.stack([rr, cc], axis=-1).reshape(-1, 2).astype(np.int32)

    return coords_l0, coords_target, int(nx * ny)


def generate_coords_with_padding_smart_filter(
    slide: openslide.OpenSlide,
    grid: GridSpec,
    thumb_rgb: np.ndarray,
    val_thresh: int = VAL_THRESH,
    sat_thresh: int = SAT_THRESH,
    keep_frac: float = KEEP_FRAC,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Prefilter on thumbnail using Filter C (Smart Sat+Int) with integral images.
    Returns kept coords only.
    """
    W0, H0 = slide.dimensions
    assert W0 == grid.W0 and H0 == grid.H0

    Ht, Wt, _ = thumb_rgb.shape
    sx = Wt / W0
    sy = Ht / H0

    # Build mask_keep on thumbnail (pixel-wise)
    hsv = cv2.cvtColor(thumb_rgb, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    mask_keep = (val < int(val_thresh)) | (sat > int(sat_thresh))

    S = integral_image_2d(mask_keep)

    stride = grid.stride_L0
    total_area_l0 = float(stride * stride)

    cols = np.arange(grid.nx, dtype=np.int64)
    x_l0 = cols * stride - grid.pad_left_L0

    # Inside region in L0 for x (vectorized)
    x0 = np.clip(x_l0, 0, W0).astype(np.int64)
    x1 = np.clip(x_l0 + stride, 0, W0).astype(np.int64)
    inside_w = (x1 - x0).astype(np.int64)

    # Map to thumbnail coords (inside region)
    tx0 = np.floor(x0 * sx).astype(np.int64)
    tx1 = np.floor(x1 * sx).astype(np.int64)
    tx0 = np.clip(tx0, 0, Wt)
    tx1 = np.clip(tx1, 0, Wt)
    force_w = (inside_w > 0) & (tx1 <= tx0)
    tx1[force_w] = np.minimum(tx0[force_w] + 1, Wt)
    tx1 = np.clip(tx1, 0, Wt)

    coords_l0_list = []
    coords_target_list = []

    for r in range(grid.ny):
        y_l0 = r * stride - grid.pad_top_L0

        y0 = int(min(max(y_l0, 0), H0))
        y1 = int(min(max(y_l0 + stride, 0), H0))
        inside_h = y1 - y0
        if inside_h <= 0:
            continue

        inside_areas = inside_w.astype(np.float32) * float(inside_h)

        # Map to thumbnail coords
        ty0 = int(math.floor(y0 * sy))
        ty1 = int(math.floor(y1 * sy))
        ty0 = max(0, min(ty0, Ht))
        ty1 = max(0, min(ty1, Ht))
        if ty1 <= ty0:
            ty1 = min(Ht, ty0 + 1)
            if ty1 <= ty0:
                continue

        # Rectangle sum on mask_keep
        # Need vector y0/y1 arrays for broadcasting over x arrays
        y0_arr = np.full(tx0.shape, ty0, dtype=np.int64)
        y1_arr = np.full(tx0.shape, ty1, dtype=np.int64)

        good_px = rect_sum_integral(S, tx0, y0_arr, tx1, y1_arr).astype(np.float32)
        areas_th = np.maximum((tx1 - tx0) * (ty1 - ty0), 1).astype(np.float32)

        # Fraction of "interesting" pixels in the whole (padded) tile
        frac_total = (good_px / areas_th) * (inside_areas / total_area_l0)

        keep = (frac_total >= float(keep_frac)) & (inside_w > 0)
        if not np.any(keep):
            continue

        kept_cols = cols[keep]

        xs = x_l0[keep].astype(np.int32)
        ys = np.full(xs.shape, y_l0, dtype=np.int32)
        coords_l0_list.append(np.stack([xs, ys], axis=1))

        # Virtual target coords (row_px, col_px) row-major
        row_px = np.full(xs.shape, r * TILE_SIZE, dtype=np.int32)
        col_px = (kept_cols * TILE_SIZE).astype(np.int32)
        coords_target_list.append(np.stack([row_px, col_px], axis=1))

    if coords_l0_list:
        coords_l0 = np.concatenate(coords_l0_list, axis=0).astype(np.int32)
        coords_target = np.concatenate(coords_target_list, axis=0).astype(np.int32)
    else:
        coords_l0 = np.zeros((0, 2), dtype=np.int32)
        coords_target = np.zeros((0, 2), dtype=np.int32)

    total_grids = int(grid.nx * grid.ny)
    return coords_l0, coords_target, total_grids


# ============================ TILE DATASET ============================

class WSITileDataset(torch.utils.data.Dataset):
    """
    Reads level-0 tiles (possibly larger than 224), downsamples with INTER_AREA if needed,
    composites alpha onto white, returns normalized torch tensor ready for Virchow.

    Optionally computes tile-wise keep flag using Filter C (only used for fallback when thumbnail is skipped).
    """
    def __init__(
        self,
        slide_path: str,
        coords_l0: np.ndarray,        # (N,2) int32, x/y in L0 (can be negative)
        coords_target: np.ndarray,    # (N,2) int32, row_px/col_px in target space
        extracted_size: int,
        needs_downsample: bool,
        mean: Tuple[float, float, float],
        std: Tuple[float, float, float],
        compute_keep: bool = False,
        val_thresh: int = VAL_THRESH,
        sat_thresh: int = SAT_THRESH,
        keep_frac: float = KEEP_FRAC,
    ):
        self.slide_path = slide_path
        self.coords_l0 = coords_l0
        self.coords_target = coords_target
        self.extracted_size = int(extracted_size)
        self.needs_downsample = bool(needs_downsample)

        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

        self.compute_keep = bool(compute_keep)
        self.val_thresh = int(val_thresh)
        self.sat_thresh = int(sat_thresh)
        self.keep_frac = float(keep_frac)

        self.slide = None

    def _get_slide(self):
        if self.slide is None:
            self.slide = openslide.OpenSlide(self.slide_path)
        return self.slide

    def __len__(self):
        return int(self.coords_l0.shape[0])

    def __getitem__(self, idx: int):
        slide = self._get_slide()

        x, y = self.coords_l0[idx]
        x = int(x); y = int(y)

        pil_rgba = slide.read_region((x, y), 0, (self.extracted_size, self.extracted_size))
        rgba = np.asarray(pil_rgba, dtype=np.uint8)  # HxWx4

        if self.needs_downsample:
            # Downsample to 224 using INTER_AREA on RGBA then composite
            rgba = cv2.resize(rgba, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_AREA)

        rgb = rgba_to_rgb_composite_white(rgba)  # 224x224x3 uint8

        keep = True
        if self.compute_keep:
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
            sat = hsv[:, :, 1]
            val = hsv[:, :, 2]
            mask_keep = (val < self.val_thresh) | (sat > self.sat_thresh)
            frac = float(mask_keep.mean())
            keep = (frac >= self.keep_frac)

        # Convert to CHW float32 [0,1], normalize
        t = torch.from_numpy(rgb).permute(2, 0, 1).to(torch.float32) / 255.0
        t = (t - self.mean) / self.std

        coord_tgt = torch.from_numpy(self.coords_target[idx].astype(np.int32))
        coord_l0 = torch.from_numpy(self.coords_l0[idx].astype(np.int32))
        keep_t = torch.tensor(keep, dtype=torch.bool)

        return t, coord_tgt, coord_l0, keep_t

    def __del__(self):
        try:
            if self.slide is not None:
                self.slide.close()
        except Exception:
            pass


# ============================ INFERENCE ===============================

def infer_embedding_dim(model: torch.nn.Module, device: torch.device) -> int:
    dummy = torch.zeros(1, 3, TILE_SIZE, TILE_SIZE, device=device, dtype=torch.float32)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(dummy)
        return int(out.shape[-1] * 2)


# ============================ CORE PROCESSING ===============================

def process_single_slide(
    row: pd.Series,
    work_dir: Path,
    source_bucket,
    out_bucket,
    client: storage.Client,
    model: torch.nn.Module,
    device: torch.device,
    mean: Tuple[float, float, float],
    std: Tuple[float, float, float],
    emb_dim: int,
    progress: str,
    logger: RunLogger,
    summary: SummaryCSV,
    *,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    skip_existing: bool = True,
    mpp_tol: float = MPP_TOL,
    max_mpp_allowed: float = MAX_MPP_ALLOWED,
    thumb_max_dim: int = THUMB_MAX_DIM,
    thumb_max_mpx: float = THUMB_PREFILTER_MAX_MPX,
    stripe_max_mpx: float = THUMB_STRIPE_MAX_MPX,
) -> None:
    file_id = row["file_id"]
    full_path = row["full_path"]

    out_key = f"embeddings_fp32/{file_id}.h5"
    if skip_existing and blob_exists(out_bucket, out_key, client):
        logger.log(f"⏩ {progress} {file_id[:10]}... already done.")
        summary.append({"file_id": file_id, "status": "SKIP", "reason": "already_done"})
        return

    # Use original extension if present (safer than forcing .svs)
    suffix = Path(full_path).suffix
    if suffix == "":
        suffix = ".svs"
    local_wsi = work_dir / f"{file_id}{suffix}"
    local_h5 = work_dir / f"{file_id}.h5"

    t0_all = time.time()

    # Prepare summary row base
    sum_row: Dict[str, Any] = {"file_id": file_id, "full_path": full_path}

    try:
        # ----------------- download -----------------
        size_mb, dt = download_blob_to_file(source_bucket, full_path, str(local_wsi))
        dl_speed = size_mb / (dt + 1e-6)

        sum_row.update({"dl_size_mb": size_mb, "dl_time_s": dt, "dl_speed_mb_s": dl_speed})

        # ----------------- open slide + metadata filters -----------------
        slide = openslide.OpenSlide(str(local_wsi))
        try:
            mpp = get_slide_mpp(slide)
            obj = get_objective_power(slide)

            sum_row.update({"mpp": mpp if mpp is not None else "", "objective_power": obj if obj is not None else ""})

            # Mandatory filters:
            if obj is None:
                logger.log(f"⚠️ {progress} SKIP {file_id[:10]}: objective power missing.")
                sum_row.update({"status": "SKIP", "reason": "objective_missing"})
                summary.append(sum_row)
                return
            if abs(obj - 5.0) < 0.25:
                logger.log(f"⚠️ {progress} SKIP {file_id[:10]}: objective power ~5x ({obj}).")
                sum_row.update({"status": "SKIP", "reason": "objective_5x"})
                summary.append(sum_row)
                return

            if mpp is None:
                logger.log(f"⚠️ {progress} SKIP {file_id[:10]}: MPP missing.")
                sum_row.update({"status": "SKIP", "reason": "mpp_missing"})
                summary.append(sum_row)
                return
            if mpp > 1.0:
                logger.log(f"⚠️ {progress} SKIP {file_id[:10]}: MPP too large ({mpp:.3f}).")
                sum_row.update({"status": "SKIP", "reason": "mpp_gt_1"})
                summary.append(sum_row)
                return

            plan = plan_tile_read(
                mpp,
                target_mpp=TARGET_MPP,
                output_size=TILE_SIZE,
                tol=mpp_tol,
                max_mpp_allowed=max_mpp_allowed,
            )
            if plan is None:
                logger.log(f"⚠️ {progress} SKIP {file_id[:10]}: mpp={mpp:.4f} outside allowed range (max={max_mpp_allowed}).")
                sum_row.update({"status": "SKIP", "reason": "mpp_outside_allowed"})
                summary.append(sum_row)
                return

            sum_row.update({
                "extracted_level0_size": plan.extracted_level0_size,
                "needs_downsample": int(plan.needs_downsample),
                "effective_mpp": plan.effective_mpp,
                "stride_level0": plan.stride_level0,
            })

            W0, H0 = slide.dimensions
            grid = compute_symmetric_padding_grid(W0, H0, plan.stride_level0)
            sum_row.update({
                "W0": grid.W0, "H0": grid.H0, "Wpad": grid.Wpad, "Hpad": grid.Hpad,
                "pad_left_L0": grid.pad_left_L0, "pad_top_L0": grid.pad_top_L0,
                "grid_nx": grid.nx, "grid_ny": grid.ny,
            })

            # ----------------- thumbnail + smart filter prefilter -----------------
            thumb_rgb, thumb_info = make_thumbnail_inter_area_stream(
                slide,
                max_dim=thumb_max_dim,
                max_level_mpx=thumb_max_mpx,
                stripe_max_mpx=stripe_max_mpx,
            )
            sum_row.update(thumb_info)

            if thumb_rgb is not None:
                coords_l0, coords_target, total_grids = generate_coords_with_padding_smart_filter(
                    slide, grid, thumb_rgb,
                    val_thresh=VAL_THRESH, sat_thresh=SAT_THRESH, keep_frac=KEEP_FRAC
                )
                prefilter_strategy = "thumbnail"
                compute_keep_tilewise = False
            else:
                # Fallback: no thumbnail (too huge). We will filter tile-wise during embedding.
                coords_l0, coords_target, total_grids = generate_all_coords_with_padding(grid)
                prefilter_strategy = "tilewise"
                compute_keep_tilewise = True

            kept = int(coords_l0.shape[0]) if prefilter_strategy == "thumbnail" else ""
            ratio = (kept / total_grids * 100.0) if (prefilter_strategy == "thumbnail" and total_grids > 0) else ""

            sum_row.update({
                "prefilter_strategy": prefilter_strategy,
                "total_grids": total_grids,
                "kept_tiles_prefilter": kept,
                "kept_ratio_pct_prefilter": ratio,
            })

            if prefilter_strategy == "thumbnail" and int(coords_l0.shape[0]) < MIN_TILES:
                logger.log(f"⚠️ {progress} SKIP {file_id[:10]}: only {int(coords_l0.shape[0])} tiles (<{MIN_TILES}) after prefilter.")
                sum_row.update({"status": "SKIP", "reason": "too_few_tiles"})
                summary.append(sum_row)
                return

            
            thumb_mpx = thumb_info.get("thumb_level_mpx", None)
            thumb_mpx_str = ""
            try:
                if thumb_mpx is not None:
                    thumb_mpx_str = f"{float(thumb_mpx):.2f}"
            except Exception:
                thumb_mpx_str = str(thumb_mpx)

            resize_msg = (
                f"| mpp={mpp:.4f} obj={obj:.1f}x "
                f"| ext={plan.extracted_level0_size}->{TILE_SIZE} "
                f"| downsample={'Y' if plan.needs_downsample else 'N'} "
                f"| eff_mpp={plan.effective_mpp:.4f} "
                f"| grid={grid.nx}x{grid.ny} padL={grid.pad_left_L0} padT={grid.pad_top_L0} "
                f"| prefilter={prefilter_strategy} "
                f"| thumb_lvl={thumb_info.get('thumb_level','')} MPx={thumb_mpx_str} "
                f"| thumb_out={thumb_info.get('thumb_out_w','')}x{thumb_info.get('thumb_out_h','')}"
            )


        finally:
            slide.close()

        # ----------------- dataloader -----------------
        dataset = WSITileDataset(
            slide_path=str(local_wsi),
            coords_l0=coords_l0,
            coords_target=coords_target,
            extracted_size=plan.extracted_level0_size,
            needs_downsample=plan.needs_downsample,
            mean=mean,
            std=std,
            compute_keep=compute_keep_tilewise,
            val_thresh=VAL_THRESH,
            sat_thresh=SAT_THRESH,
            keep_frac=KEEP_FRAC,
        )

        dl_kwargs = dict(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        if num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = 2

        loader = torch.utils.data.DataLoader(dataset, **dl_kwargs)

        # ----------------- HDF5 write -----------------
        t0_gpu = time.time()

        if prefilter_strategy == "thumbnail":
            kept = int(len(dataset))
            ratio = (kept / total_grids * 100.0) if total_grids > 0 else 0.0

            with h5py.File(local_h5, "w") as f:
                chunk = min(4096, kept) if kept > 0 else 1

                feat_ds = f.create_dataset(
                    "features",
                    shape=(kept, emb_dim),
                    dtype=np.float32,
                    chunks=(chunk, emb_dim),
                )
                # coords in virtual target space (row_px, col_px)
                f.create_dataset("coords", data=coords_target.astype(np.int32), dtype=np.int32, chunks=(chunk, 2))
                # original L0 coords (x_L0, y_L0)
                f.create_dataset("coords_level0", data=coords_l0.astype(np.int32), dtype=np.int32, chunks=(chunk, 2))

                # Metadata
                f.attrs["slide_id"] = file_id
                f.attrs["source_full_path"] = full_path
                f.attrs["target_mpp"] = float(TARGET_MPP)
                f.attrs["tile_size"] = int(TILE_SIZE)

                f.attrs["mpp"] = float(mpp)
                f.attrs["objective_power"] = float(obj)

                f.attrs["extracted_level0_size"] = int(plan.extracted_level0_size)
                f.attrs["needs_downsample"] = int(plan.needs_downsample)
                f.attrs["effective_mpp"] = float(plan.effective_mpp)
                f.attrs["stride_level0"] = int(plan.stride_level0)

                f.attrs["W0"] = int(grid.W0)
                f.attrs["H0"] = int(grid.H0)
                f.attrs["Wpad"] = int(grid.Wpad)
                f.attrs["Hpad"] = int(grid.Hpad)
                f.attrs["pad_left_L0"] = int(grid.pad_left_L0)
                f.attrs["pad_top_L0"] = int(grid.pad_top_L0)
                f.attrs["grid_nx"] = int(grid.nx)
                f.attrs["grid_ny"] = int(grid.ny)
                f.attrs["total_grids"] = int(total_grids)
                f.attrs["kept_tiles"] = int(kept)
                f.attrs["kept_ratio_pct"] = float(ratio)

                f.attrs["prefilter_strategy"] = prefilter_strategy
                f.attrs["filter_val_thresh"] = int(VAL_THRESH)
                f.attrs["filter_sat_thresh"] = int(SAT_THRESH)
                f.attrs["filter_keep_frac"] = float(KEEP_FRAC)

                # thumb info
                for k, v in thumb_info.items():
                    try:
                        f.attrs[k] = v
                    except Exception:
                        pass

                # Inference (features streaming)
                idx0 = 0
                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    for batch, _coord_tgt, _coord_l0, _keep in loader:
                        batch = batch.to(device, non_blocking=True)
                        out = model(batch)  # (B, 257, 1280)
                        cls = out[:, 0]
                        patch = out[:, 1:].mean(1)
                        emb = torch.cat([cls, patch], dim=-1)

                        bsz = emb.shape[0]
                        feat_ds[idx0:idx0+bsz] = emb.float().cpu().numpy()
                        idx0 += bsz

            gpu_speed = kept / (time.time() - t0_gpu + 1e-6)
            total_time = time.time() - t0_all

            upload_file_to_bucket(out_bucket, str(local_h5), out_key)

            logger.log(
                f"✅ {progress} OK | DL:{dl_speed:.0f}MB/s | Tiles:{kept} ({ratio:.1f}%) "
                f"| GPU:{gpu_speed:.0f} t/s | Total:{total_time:.1f}s {resize_msg}"
            )

            sum_row.update({
                "status": "OK",
                "reason": "",
                "kept_tiles": kept,
                "kept_ratio_pct": ratio,
                "gpu_tiles_per_s": gpu_speed,
                "total_time_s": total_time,
            })
            summary.append(sum_row)

        else:
            # Tile-wise filter fallback: dynamic number of kept tiles
            total_grids = int(total_grids)
            written = 0

            with h5py.File(local_h5, "w") as f:
                chunk = 4096

                feat_ds = f.create_dataset(
                    "features",
                    shape=(0, emb_dim),
                    maxshape=(None, emb_dim),
                    dtype=np.float32,
                    chunks=(chunk, emb_dim),
                )
                coord_ds = f.create_dataset(
                    "coords",
                    shape=(0, 2),
                    maxshape=(None, 2),
                    dtype=np.int32,
                    chunks=(chunk, 2),
                )
                coord_l0_ds = f.create_dataset(
                    "coords_level0",
                    shape=(0, 2),
                    maxshape=(None, 2),
                    dtype=np.int32,
                    chunks=(chunk, 2),
                )

                # Metadata (same + note about fallback)
                f.attrs["slide_id"] = file_id
                f.attrs["source_full_path"] = full_path
                f.attrs["target_mpp"] = float(TARGET_MPP)
                f.attrs["tile_size"] = int(TILE_SIZE)

                f.attrs["mpp"] = float(mpp)
                f.attrs["objective_power"] = float(obj)

                f.attrs["extracted_level0_size"] = int(plan.extracted_level0_size)
                f.attrs["needs_downsample"] = int(plan.needs_downsample)
                f.attrs["effective_mpp"] = float(plan.effective_mpp)
                f.attrs["stride_level0"] = int(plan.stride_level0)

                f.attrs["W0"] = int(grid.W0)
                f.attrs["H0"] = int(grid.H0)
                f.attrs["Wpad"] = int(grid.Wpad)
                f.attrs["Hpad"] = int(grid.Hpad)
                f.attrs["pad_left_L0"] = int(grid.pad_left_L0)
                f.attrs["pad_top_L0"] = int(grid.pad_top_L0)
                f.attrs["grid_nx"] = int(grid.nx)
                f.attrs["grid_ny"] = int(grid.ny)
                f.attrs["total_grids"] = int(total_grids)

                f.attrs["prefilter_strategy"] = prefilter_strategy
                f.attrs["filter_val_thresh"] = int(VAL_THRESH)
                f.attrs["filter_sat_thresh"] = int(SAT_THRESH)
                f.attrs["filter_keep_frac"] = float(KEEP_FRAC)

                # thumb info (even though thumb skipped)
                for k, v in thumb_info.items():
                    try:
                        f.attrs[k] = v
                    except Exception:
                        pass

                with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                    for batch, coord_tgt, coord_l0, keep_t in loader:
                        keep = keep_t  # torch.bool mask
                        if not bool(keep.any()):
                            continue

                        batch_kept = batch[keep].to(device, non_blocking=True)
                        coord_tgt_kept = coord_tgt[keep].numpy().astype(np.int32)
                        coord_l0_kept = coord_l0[keep].numpy().astype(np.int32)

                        out = model(batch_kept)
                        cls = out[:, 0]
                        patch = out[:, 1:].mean(1)
                        emb = torch.cat([cls, patch], dim=-1)
                        emb_np = emb.float().cpu().numpy()

                        k = emb_np.shape[0]
                        new_size = written + k
                        feat_ds.resize((new_size, emb_dim))
                        coord_ds.resize((new_size, 2))
                        coord_l0_ds.resize((new_size, 2))

                        feat_ds[written:new_size] = emb_np
                        coord_ds[written:new_size] = coord_tgt_kept
                        coord_l0_ds[written:new_size] = coord_l0_kept
                        written = new_size

                # finalize attrs
                ratio = (written / total_grids * 100.0) if total_grids > 0 else 0.0
                f.attrs["kept_tiles"] = int(written)
                f.attrs["kept_ratio_pct"] = float(ratio)

            # If extremely few tiles, skip upload to avoid junk output
            if written < MIN_TILES:
                logger.log(f"⚠️ {progress} SKIP {file_id[:10]}: tilewise kept {written} (<{MIN_TILES}). (thumb skipped)")
                sum_row.update({"status": "SKIP", "reason": "too_few_tiles_tilewise", "kept_tiles": written, "kept_ratio_pct": ratio})
                summary.append(sum_row)
                return

            gpu_speed = written / (time.time() - t0_gpu + 1e-6)
            total_time = time.time() - t0_all

            upload_file_to_bucket(out_bucket, str(local_h5), out_key)

            logger.log(
                f"✅ {progress} OK(tilewise) | DL:{dl_speed:.0f}MB/s | Tiles:{written} ({ratio:.1f}%) "
                f"| GPU:{gpu_speed:.0f} t/s | Total:{total_time:.1f}s "
                f"| thumb_skipped={thumb_info.get('thumb_skip_reason','')} "
                f"| thumb_lvl={thumb_info.get('thumb_level','')} MPx={thumb_info.get('thumb_level_mpx','')}"
            )

            sum_row.update({
                "status": "OK",
                "reason": "",
                "kept_tiles": written,
                "kept_ratio_pct": ratio,
                "gpu_tiles_per_s": gpu_speed,
                "total_time_s": total_time,
            })
            summary.append(sum_row)

    except torch.cuda.OutOfMemoryError:
        logger.log(f"❌ {progress} CUDA OOM on {file_id}. Lower --batch_size (128/64).")
        sum_row.update({"status": "ERR", "reason": "cuda_oom"})
        summary.append(sum_row)
        raise
    except Exception as e:
        logger.log(f"❌ {progress} ERR {file_id[:10]}: {e}")
        sum_row.update({"status": "ERR", "reason": f"{type(e).__name__}:{e}"})
        summary.append(sum_row)
    finally:
        try:
            if local_wsi.exists():
                local_wsi.unlink()
        except Exception:
            pass
        try:
            if local_h5.exists():
                local_h5.unlink()
        except Exception:
            pass


# ============================ MAIN ===================================

def maybe_download_existing_log(bucket_out, remote_key: str, local_path: Path):
    """
    If remote log exists, download it so we can truly append locally.
    """
    try:
        blob = bucket_out.blob(remote_key)
        if blob.exists():
            blob.chunk_size = GCS_CHUNK_SIZE
            blob.download_to_filename(str(local_path))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bucket_out", type=str, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--shard_id", type=int, required=True)

    # runtime overrides
    parser.add_argument("--work_dir", type=str, default="scratch_vm")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num_workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--force", action="store_true", help="Recompute even if output exists")

    # mpp overrides
    parser.add_argument("--mpp_tol", type=float, default=MPP_TOL)
    parser.add_argument("--max_mpp_allowed", type=float, default=MAX_MPP_ALLOWED)

    # thumb/filter overrides
    parser.add_argument("--thumb_max_dim", type=int, default=THUMB_MAX_DIM)
    parser.add_argument("--thumb_max_mpx", type=float, default=THUMB_PREFILTER_MAX_MPX)
    parser.add_argument("--thumb_stripe_max_mpx", type=float, default=THUMB_STRIPE_MAX_MPX)

    parser.add_argument("--compile", action="store_true", help="torch.compile(model) (optional)")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Virchow is intended for GPU inference.")

    # Hugging Face auth (once in main)
    login(token=HF_TOKEN)

    work_dir = Path(args.work_dir)
    work_dir.mkdir(exist_ok=True)

    # GCS
    client = storage.Client()
    source_bucket = client.bucket(SOURCE_BUCKET_NAME)
    out_bucket = client.bucket(args.bucket_out)

    # logs in bucket (unique per shard)
    remote_log_key = f"logs_fp32/embedder_shard_{args.shard_id}.log"
    remote_csv_key = f"logs_fp32/embedder_shard_{args.shard_id}_summary.csv"

    local_log = work_dir / f"embedder_shard_{args.shard_id}.log"
    local_csv = work_dir / f"embedder_shard_{args.shard_id}_summary.csv"

    # If exists in bucket, download so we can truly append
    maybe_download_existing_log(out_bucket, remote_log_key, local_log)
    maybe_download_existing_log(out_bucket, remote_csv_key, local_csv)

    logger = RunLogger(local_log, gcs_bucket=out_bucket, gcs_key=remote_log_key, upload_every=LOG_UPLOAD_EVERY)

    summary_header = (
        "file_id","full_path","status","reason",
        "mpp","objective_power",
        "extracted_level0_size","needs_downsample","effective_mpp","stride_level0",
        "W0","H0","Wpad","Hpad","pad_left_L0","pad_top_L0","grid_nx","grid_ny",
        "prefilter_strategy","total_grids","kept_tiles_prefilter","kept_ratio_pct_prefilter",
        "kept_tiles","kept_ratio_pct",
        "thumb_max_dim","thumb_out_w","thumb_out_h","thumb_downsample_req",
        "thumb_level","thumb_level_w","thumb_level_h","thumb_level_downsample","thumb_level_mpx","thumb_skipped","thumb_skip_reason",
        "dl_size_mb","dl_time_s","dl_speed_mb_s",
        "gpu_tiles_per_s","total_time_s",
    )
    summary = SummaryCSV(local_csv, header=summary_header, gcs_bucket=out_bucket, gcs_key=remote_csv_key, upload_every_rows=25)

    # CSV
    csv_local = work_dir / MASTER_CSV_NAME
    download_master_csv_if_needed(out_bucket, str(csv_local))
    df = pd.read_csv(csv_local)

    my_batch = get_shard_batch(df, args.num_shards, args.shard_id)
    logger.log(f"🤖 SHARD {args.shard_id + 1}/{args.num_shards} | {len(my_batch)} slides")

    # Model
    device = torch.device("cuda")
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        mlp_layer=SwiGLUPacked,
        act_layer=torch.nn.SiLU,
    ).eval().to(device)

    cfg = resolve_data_config(model.pretrained_cfg, model=model)
    mean = tuple(float(x) for x in cfg["mean"])
    std = tuple(float(x) for x in cfg["std"])

    if args.compile:
        model = torch.compile(model)

    emb_dim = infer_embedding_dim(model, device)
    logger.log(f"🧠 Virchow embedding dim = {emb_dim}")

    # Process slides
    for i, (_, row) in enumerate(my_batch.iterrows()):
        process_single_slide(
            row=row,
            work_dir=work_dir,
            source_bucket=source_bucket,
            out_bucket=out_bucket,
            client=client,
            model=model,
            device=device,
            mean=mean,
            std=std,
            emb_dim=emb_dim,
            progress=f"[{i+1}/{len(my_batch)}]",
            logger=logger,
            summary=summary,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            skip_existing=(not args.force),
            mpp_tol=args.mpp_tol,
            max_mpp_allowed=args.max_mpp_allowed,
            thumb_max_dim=args.thumb_max_dim,
            thumb_max_mpx=args.thumb_max_mpx,
            stripe_max_mpx=args.thumb_stripe_max_mpx,
        )

    logger.log(f"🎉 DONE SHARD {args.shard_id}")
    summary.close()
    logger.close()


if __name__ == "__main__":
    main()
