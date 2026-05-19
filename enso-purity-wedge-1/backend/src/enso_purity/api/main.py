from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

app = FastAPI(title="Enso Purity MVP", version="0.1.0")


class PredictResponse(BaseModel):
    purity_wsi: float = Field(..., ge=0.0, le=1.0)
    purity_ta: float = Field(..., ge=0.0, le=1.0)
    heatmap_png_base64: str


@app.get("/health")
def health():
    return {"status": "ok"}


def _dummy_heatmap_png_base64() -> str:
    # Placeholder: return a 1x1 transparent PNG
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR4nGNgYGBgAAAABQABDQottAAAAABJRU5ErkJggg=="
    )
    return base64.b64encode(png).decode("ascii")


@app.post("/predict_purity", response_model=PredictResponse)
async def predict_purity(file: UploadFile = File(...)):
    # MVP stub: parse file and return placeholder outputs
    # Real implementation will load model weights and run inference.
    raw = await file.read()

    # sanity check: ensure it's a readable HDF5
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
