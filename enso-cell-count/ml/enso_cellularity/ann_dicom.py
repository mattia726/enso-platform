"""Read DICOM ANN nuclei polygons into centroid tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def primitive_point_ranges_from_flat_indices(
    flat_indices_one_based: np.ndarray,
    *,
    num_float_values: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert DICOM ANN primitive indices to point start/end offsets.

    DICOM ANN ``PointCoordinatesData`` stores coordinates as a flat float array:
    ``x0, y0, x1, y1, ...``.  ``LongPrimitivePointIndexList`` stores one-based
    offsets into that flat array, not into the reshaped point array.
    """

    if flat_indices_one_based.ndim != 1:
        raise ValueError("Primitive index list must be one-dimensional.")
    if len(flat_indices_one_based) == 0:
        raise ValueError("Primitive index list is empty.")

    starts_flat = flat_indices_one_based.astype(np.int64) - 1
    ends_flat = np.concatenate([starts_flat[1:], np.array([num_float_values], dtype=np.int64)])
    lengths_flat = ends_flat - starts_flat

    if np.any(starts_flat < 0):
        raise ValueError("DICOM ANN primitive indices must be one-based positive offsets.")
    if np.any(lengths_flat <= 0):
        raise ValueError("DICOM ANN primitive ranges must be strictly increasing.")
    if np.any(starts_flat % 2 != 0) or np.any(ends_flat % 2 != 0):
        raise ValueError("DICOM ANN primitive ranges must align to x/y coordinate pairs.")

    return starts_flat // 2, ends_flat // 2


def polygon_centroid(points: np.ndarray) -> tuple[float, float]:
    """Return the area centroid of a polygon, falling back to vertex mean."""

    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(f"Expected polygon points shaped (N, 2), got {points.shape}")
    if len(points) == 0:
        raise ValueError("Cannot compute centroid for an empty polygon.")

    x = points[:, 0].astype(np.float64, copy=False)
    y = points[:, 1].astype(np.float64, copy=False)
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    cross_sum = float(cross.sum())

    if abs(cross_sum) < 1e-8:
        mean = points.mean(axis=0)
        return float(mean[0]), float(mean[1])

    cx = float(((x + x_next) * cross).sum() / (3.0 * cross_sum))
    cy = float(((y + y_next) * cross).sum() / (3.0 * cross_sum))
    return cx, cy


def polygon_centroids(
    coords: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centroid and vertex count arrays for variable-length polygons."""

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords shaped (N, 2), got {coords.shape}")
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length.")

    centroids = np.empty((len(starts), 2), dtype=np.float32)
    vertex_counts = np.empty(len(starts), dtype=np.int32)
    for i, (start, end) in enumerate(zip(starts, ends)):
        polygon = coords[int(start) : int(end)]
        centroids[i] = polygon_centroid(polygon)
        vertex_counts[i] = len(polygon)
    return centroids[:, 0], centroids[:, 1], vertex_counts


def polygon_vertex_mean_centroids(
    coords: np.ndarray,
    starts: np.ndarray,
    ends: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fast centroid approximation for large WSI-scale ANN files.

    Area centroids are more exact, but looping over millions of nucleus
    polygons per slide is too slow for label generation. The vertex mean is
    stable for the compact nucleus polygons in Pan-Cancer-Nuclei-Seg and is
    sufficient for assigning each nucleus to a non-overlapping tile by centroid.
    """

    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError(f"Expected coords shaped (N, 2), got {coords.shape}")
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length.")
    lengths = (ends - starts).astype(np.int32)
    if np.any(lengths <= 0):
        raise ValueError("Polygon ranges must be non-empty.")
    sum_x = np.add.reduceat(coords[:, 0].astype(np.float64, copy=False), starts)
    sum_y = np.add.reduceat(coords[:, 1].astype(np.float64, copy=False), starts)
    return (
        (sum_x / lengths).astype(np.float32),
        (sum_y / lengths).astype(np.float32),
        lengths,
    )


def _as_float32_array(value) -> np.ndarray:
    if isinstance(value, bytes):
        return np.frombuffer(value, dtype="<f4")
    return np.asarray(value, dtype=np.float32)


def _as_uint32_array(value) -> np.ndarray:
    if isinstance(value, bytes):
        return np.frombuffer(value, dtype="<u4")
    return np.asarray(value, dtype=np.uint32)


def _first_area_measurement(group) -> np.ndarray | None:
    for measurement in getattr(group, "MeasurementsSequence", []):
        for values in getattr(measurement, "MeasurementValuesSequence", []):
            if hasattr(values, "FloatingPointValues"):
                return _as_float32_array(values.FloatingPointValues)
    return None


def read_ann_centroids(path: str | Path, *, fast_vertex_mean: bool = False) -> pd.DataFrame:
    """Read a Pan-Cancer-Nuclei-Seg ANN DICOM file into nucleus centroids.

    Returns one row per nucleus polygon with ``centroid_x`` and ``centroid_y`` in
    the ANN coordinate system, plus optional ``area_um2`` when present.
    """

    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "pydicom is required to read ANN DICOM files. "
            "Install it with: python -m pip install pydicom"
        ) from exc

    ds = pydicom.dcmread(path)
    if not hasattr(ds, "AnnotationGroupSequence") or len(ds.AnnotationGroupSequence) == 0:
        raise ValueError(f"No AnnotationGroupSequence found in {path}")
    group = ds.AnnotationGroupSequence[0]

    flat_coords = _as_float32_array(group.PointCoordinatesData)
    coords = flat_coords.reshape(-1, 2)
    flat_indices = _as_uint32_array(group.LongPrimitivePointIndexList)
    starts, ends = primitive_point_ranges_from_flat_indices(
        flat_indices,
        num_float_values=len(flat_coords),
    )

    expected = int(group.NumberOfAnnotations)
    if len(starts) != expected:
        raise ValueError(
            f"Expected {expected} annotations from DICOM metadata, found {len(starts)} polygons."
        )

    if fast_vertex_mean:
        centroid_x, centroid_y, vertex_counts = polygon_vertex_mean_centroids(coords, starts, ends)
    else:
        centroid_x, centroid_y, vertex_counts = polygon_centroids(coords, starts, ends)
    out = pd.DataFrame(
        {
            "annotation_index": np.arange(expected, dtype=np.int32),
            "centroid_x": centroid_x,
            "centroid_y": centroid_y,
            "vertex_count": vertex_counts,
        }
    )

    areas = _first_area_measurement(group)
    if areas is not None and len(areas) == expected:
        out["area_um2"] = areas.astype(np.float32, copy=False)

    for attr, col in [
        ("PatientID", "patient_id"),
        ("ContainerIdentifier", "slide_barcode"),
        ("StudyInstanceUID", "study_instance_uid"),
        ("SeriesInstanceUID", "series_instance_uid"),
        ("SOPInstanceUID", "sop_instance_uid"),
    ]:
        if hasattr(ds, attr):
            out[col] = str(getattr(ds, attr))

    return out
