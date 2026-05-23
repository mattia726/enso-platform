"""Macrodissection workbench backend.

This package turns the raw EnsoPurity and EnsoCellularity outputs into the
structured artifacts required by the macrodissection workbench UI:

* per-tile predictions packaged as JSON + Float32 binary grids,
* ROI (region of interest) geometry math (polygon × tile area weights,
  Monte-Carlo uncertainty propagation, pass/borderline/fail adequacy),
* threshold profiles (Humanitas NGS, research, custom),
* candidate-ROI auto-suggestion,
* file-backed append-only ROI storage with audit metadata,
* a FastAPI router exposing the workbench endpoints.

The math is implemented to be *identical* to the corresponding TypeScript
implementation that ships with the Next.js frontend; the Python side acts as
the authoritative recompute layer when a pathologist locks a region.

Nothing in this package mutates state at import time. The package is safe to
import from the FastAPI app or from a one-shot CLI build script.
"""

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:
    __version__ = _metadata.version("enso-purity")
except _metadata.PackageNotFoundError:  # pragma: no cover - dev mode
    __version__ = "0.0.0+dev"
