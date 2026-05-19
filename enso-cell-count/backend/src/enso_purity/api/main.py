from __future__ import annotations

import base64
import io
import os
from pathlib import Path

import h5py
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from enso_purity.macrodissection.router import build_router


app = FastAPI(title="Enso Purity MVP", version="0.2.0")

# Allow the Next.js dev server (default ports 3000 / 3001) to call the API.
_default_origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:3001",
]
_extra_origins = os.environ.get("ENSO_API_CORS_ORIGINS", "").split(",")
_origins = [o.strip() for o in _default_origins + _extra_origins if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins or ["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    purity_wsi: float = Field(..., ge=0.0, le=1.0)
    purity_ta: float = Field(..., ge=0.0, le=1.0)
    heatmap_png_base64: str


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


def _dummy_heatmap_png_base64() -> str:
    # Placeholder: a 1×1 transparent PNG.
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABDQottAAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(png).decode("ascii")


@app.post("/predict_purity", response_model=PredictResponse)
async def predict_purity(file: UploadFile = File(...)) -> PredictResponse:
    """Legacy MVP endpoint kept for backward compatibility with old clients."""

    raw = await file.read()
    with h5py.File(io.BytesIO(raw), "r") as f:
        if "features" not in f:
            raise ValueError("Missing 'features' dataset in H5.")
        feats = f["features"][:]
        _ = np.asarray(feats)
    return PredictResponse(
        purity_wsi=0.5,
        purity_ta=0.8,
        heatmap_png_base64=_dummy_heatmap_png_base64(),
    )


# --------- Macrodissection workbench router --------------------------------

# The cases directory is configurable via ``ENSO_CASES_DIR`` so the API can
# run from anywhere (repo root, tests, container). Default points to the
# Next.js public folder used by the demo deployment.
_cases_dir = Path(
    os.environ.get("ENSO_CASES_DIR", str(Path.cwd() / "frontend" / "public" / "cases"))
)
_rois_dir = Path(
    os.environ.get("ENSO_ROIS_DIR", str(Path.cwd() / "backend" / ".runtime" / "rois"))
)
app.include_router(build_router(cases_dir=_cases_dir, rois_dir=_rois_dir))
