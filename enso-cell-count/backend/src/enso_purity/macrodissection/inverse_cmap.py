"""Inverse colormap decoding for the pre-rendered tile heatmaps.

The artifact-building pipeline can either (a) consume raw scalar tile grids
emitted directly by the ML inference scripts, or (b) recover those scalars
from the pre-rendered RGBA PNG masks. Path (b) is what the demo build runs
on the public CI VM, where the trained model weights are not mounted but the
PNG masks are checked in under ``frontend/public/cases/``.

The decoder is implementation-correct in the sense that, for *any* color
produced by the forward pipeline, it returns a value that round-trips into
the same color modulo nearest-neighbour LUT quantization. We assert that
property in :mod:`tests.macrodissection.test_inverse_cmap`.

Two palettes are supported:

* ``RdYlBu_r`` — matplotlib's blue→yellow→red, 256 entries, range ``[0, 1]``
  (used by the EnsoPurity overlay).
* ``purity_no_white`` — custom seven-stop linear segmented colormap defined
  by ``enso_cellularity.export_static_cases._get_colormap``, range
  ``[0, vmax]``, applied after a ``pow(value/vmax, gamma)`` warp (used by the
  EnsoCellularity overlay).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# Stop colors for the EnsoCellularity custom palette. Copied verbatim from
# ``enso_cellularity.export_static_cases._get_colormap`` so we stay in sync.
_PURITY_NO_WHITE_STOPS: tuple[str, ...] = (
    "#313695",
    "#4575b4",
    "#74add1",
    "#fee08b",
    "#fdae61",
    "#f46d43",
    "#a50026",
)


def _build_lut(cmap: mcolors.Colormap, n: int = 256) -> np.ndarray:
    """Return an ``(n, 3)`` uint8 RGB LUT for the supplied colormap."""

    samples = np.linspace(0.0, 1.0, n, dtype=np.float64)
    rgba = cmap(samples)
    return (rgba[:, :3] * 255.0 + 0.5).astype(np.uint8)


@lru_cache(maxsize=8)
def _get_cmap(name: str) -> mcolors.Colormap:
    if name == "purity_no_white":
        return mcolors.LinearSegmentedColormap.from_list(
            "purity_no_white",
            list(_PURITY_NO_WHITE_STOPS),
            N=256,
        )
    if name == "RdYlBu_r":
        return plt.colormaps["RdYlBu_r"]
    raise KeyError(
        f"Unknown colormap '{name}'. Supported: RdYlBu_r, purity_no_white"
    )


@dataclass(frozen=True)
class ColormapSpec:
    """Encoding parameters for one rendered overlay."""

    cmap_name: str
    vmin: float
    vmax: float
    gamma: float = 1.0  # forward warp applied to (value-vmin)/(vmax-vmin)
    n_lut: int = 256

    @property
    def lut(self) -> np.ndarray:
        """RGB LUT of shape ``(n_lut, 3)`` in uint8."""

        return _build_lut(_get_cmap(self.cmap_name), n=self.n_lut)


# ----- Canonical specs used by the build pipeline ---------------------------

PURITY_SPEC = ColormapSpec(cmap_name="RdYlBu_r", vmin=0.0, vmax=1.0, gamma=1.0)
CELLULARITY_SPEC = ColormapSpec(
    cmap_name="purity_no_white", vmin=0.0, vmax=180.0, gamma=0.65
)


# ---------------------------------------------------------------------------


def encode_value(value: float, spec: ColormapSpec) -> tuple[int, int, int]:
    """Forward encoding: a scalar → uint8 RGB triple.

    Useful in tests to confirm the inverse function is the right inverse.
    """

    if not np.isfinite(value):
        return (0, 0, 0)
    span = max(spec.vmax - spec.vmin, 1e-12)
    norm = (value - spec.vmin) / span
    norm = float(np.clip(norm, 0.0, 1.0))
    warped = norm ** max(spec.gamma, 1e-6)
    idx = int(round(warped * (spec.n_lut - 1)))
    r, g, b = spec.lut[idx]
    return int(r), int(g), int(b)


def decode_rgb(
    rgb: np.ndarray,
    spec: ColormapSpec,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse decoding: ``(..., 3)`` uint8 RGB → scalar values + match dist.

    Parameters
    ----------
    rgb
        Array of shape ``(..., 3)`` containing uint8 RGB triples.
    spec
        The forward encoding parameters.

    Returns
    -------
    values
        Array of shape ``(...,)`` (same leading dims as ``rgb``) of float32
        scalar estimates in the original ``[vmin, vmax]`` range.
    distances
        Array of shape ``(...,)`` containing the L2 RGB distance between the
        observed color and the nearest LUT entry. Large distances are a
        signal that the pixel is *not* a colormap-rendered tile (e.g. it was
        anti-aliased by NEAREST resize at a tile boundary). Callers can use
        the value to gate decoding decisions.
    """

    if rgb.dtype != np.uint8:
        rgb = rgb.astype(np.uint8)

    lut = spec.lut.astype(np.int32)  # (n_lut, 3)
    flat = rgb.reshape(-1, 3).astype(np.int32)
    # Squared distances to every LUT entry: (N, n_lut)
    diff = flat[:, None, :] - lut[None, :, :]
    sq = (diff * diff).sum(axis=2)
    idx = np.argmin(sq, axis=1)
    dist = np.sqrt(sq[np.arange(flat.shape[0]), idx]).astype(np.float32)

    norm = idx.astype(np.float64) / max(spec.n_lut - 1, 1)
    # Invert the gamma warp.
    invg = 1.0 / max(spec.gamma, 1e-6)
    raw = np.power(np.clip(norm, 0.0, 1.0), invg)
    values = (raw * (spec.vmax - spec.vmin) + spec.vmin).astype(np.float32)

    out_shape = rgb.shape[:-1]
    return values.reshape(out_shape), dist.reshape(out_shape)


def decode_rgba_image(
    rgba: np.ndarray,
    spec: ColormapSpec,
    *,
    alpha_threshold: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode an entire ``(H, W, 4)`` uint8 RGBA mask.

    Pixels with alpha below ``alpha_threshold`` are treated as missing (NaN
    in the returned ``values`` array) and the ``tissue_mask`` flags pixels
    that *do* carry a usable value.

    Returns
    -------
    values
        Float32 array of shape ``(H, W)`` with NaN where the pixel was
        transparent.
    distances
        Float32 array of shape ``(H, W)``; the L2 distance to the closest
        LUT entry for opaque pixels, ``np.inf`` for transparent pixels.
    tissue_mask
        Boolean array of shape ``(H, W)`` flagging opaque pixels.
    """

    if rgba.ndim != 3 or rgba.shape[-1] != 4:
        raise ValueError(
            f"Expected RGBA array of shape (H, W, 4), got {rgba.shape}"
        )
    if rgba.dtype != np.uint8:
        rgba = rgba.astype(np.uint8)

    rgb = rgba[..., :3]
    alpha = rgba[..., 3]
    tissue_mask = alpha >= alpha_threshold
    values_raw, dist_raw = decode_rgb(rgb, spec)
    values = np.where(tissue_mask, values_raw, np.float32(np.nan))
    distances = np.where(tissue_mask, dist_raw, np.float32(np.inf))
    return values, distances, tissue_mask
