"""Unit tests for the inverse colormap decoder."""

from __future__ import annotations

import numpy as np
import pytest

from enso_purity.macrodissection.inverse_cmap import (
    CELLULARITY_SPEC,
    PURITY_SPEC,
    decode_rgb,
    decode_rgba_image,
    encode_value,
)


def test_purity_roundtrip_grid():
    """For 256 evenly spaced values, encode → decode is within one LUT step."""

    samples = np.linspace(0.0, 1.0, 256)
    rgb = np.array([encode_value(v, PURITY_SPEC) for v in samples], dtype=np.uint8)
    values, distances = decode_rgb(rgb, PURITY_SPEC)
    assert np.all(distances < 1.0), "Forward-encoded colors must hit the LUT exactly."
    err = np.abs(values - samples.astype(np.float32))
    # One LUT step is 1/255 ≈ 0.004 on the [0, 1] range.
    assert err.max() <= 1.0 / (PURITY_SPEC.n_lut - 1) + 1e-6


def test_cellularity_roundtrip_grid():
    """Round-trip for the cellularity palette across [0, vmax]."""

    samples = np.linspace(0.0, CELLULARITY_SPEC.vmax, 257)
    rgb = np.array(
        [encode_value(v, CELLULARITY_SPEC) for v in samples],
        dtype=np.uint8,
    )
    values, distances = decode_rgb(rgb, CELLULARITY_SPEC)
    assert np.all(distances < 1.0)
    # The gamma warp compresses the high end; tolerate a small absolute error.
    np.testing.assert_allclose(values, samples.astype(np.float32), atol=3.0)


def test_decode_rgba_handles_alpha():
    """Transparent pixels decode to NaN; opaque pixels keep their value."""

    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[0, 0] = (*encode_value(0.0, PURITY_SPEC), 255)
    rgba[0, 1] = (*encode_value(0.5, PURITY_SPEC), 255)
    rgba[1, 0] = (*encode_value(1.0, PURITY_SPEC), 255)
    rgba[1, 1] = (0, 0, 0, 0)  # transparent

    values, distances, mask = decode_rgba_image(rgba, PURITY_SPEC)
    assert np.isnan(values[1, 1])
    assert np.isinf(distances[1, 1])
    assert not mask[1, 1]
    assert mask[0, 0] and mask[0, 1] and mask[1, 0]
    np.testing.assert_allclose(values[0, 0], 0.0, atol=1e-3)
    np.testing.assert_allclose(values[0, 1], 0.5, atol=1.0 / 255)
    np.testing.assert_allclose(values[1, 0], 1.0, atol=1.0 / 255)


def test_decode_rgb_rejects_out_of_palette_pixels():
    """Pure white is not in either palette → high LUT distance."""

    white = np.array([[255, 255, 255]], dtype=np.uint8)
    _, dist = decode_rgb(white, PURITY_SPEC)
    assert dist[0] > 10.0
