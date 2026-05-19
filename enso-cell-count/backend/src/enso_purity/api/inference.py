"""On-demand GPU inference endpoint for purity prediction.

Accepts an uploaded H5 embedding file, runs the MIL model, and returns:
  - Global purity prediction
  - Per-tile purity scores + coordinates (for heatmap rendering)

The model checkpoint is loaded once at startup and cached in memory.
For production, this runs on a GCE VM that auto-starts on request
and auto-stops after an idle timeout.
"""
from __future__ import annotations

import io
import logging
import tempfile
from pathlib import Path

import h5py
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Lazy-loaded model singleton
_model = None
_device = None
_model_path = Path("ml/runs/fold0/best_model.pth")


class PurityResult(BaseModel):
    purity_global: float = Field(..., ge=0.0, le=1.0)
    n_tiles: int
    tile_scores_min: float
    tile_scores_max: float
    tile_scores_mean: float


class HeatmapResult(BaseModel):
    purity_global: float = Field(..., ge=0.0, le=1.0)
    n_tiles: int
    scores: list[float]
    coords_x: list[int]
    coords_y: list[int]


def _load_model():
    """Load model checkpoint once and cache."""
    global _model, _device
    if _model is not None:
        return _model, _device

    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "ml"))

    from enso_purity_mil.model import EnsoMILModel, EnsoModelConfig

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model from %s on %s", _model_path, _device)

    if not _model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {_model_path}")

    ckpt = torch.load(_model_path, map_location=_device, weights_only=False)
    cfg = EnsoModelConfig(**ckpt["config"])
    _model = EnsoMILModel(cfg).to(_device)
    _model.load_state_dict(ckpt["model_state_dict"])
    _model.eval()
    logger.info("Model loaded (epoch %d, val_loss=%.4f)", ckpt["epoch"], ckpt["val_loss"])
    return _model, _device


app = FastAPI(title="Enso Purity Inference API", version="0.2.0")


@app.get("/health")
def health():
    return {"status": "ok", "gpu": torch.cuda.is_available()}


@app.post("/predict_purity", response_model=PurityResult)
async def predict_purity(file: UploadFile = File(...)):
    """Upload an H5 embedding file and get global purity + tile stats."""
    model, device = _load_model()

    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=True) as tmp:
        tmp.write(raw)
        tmp.flush()

        with h5py.File(tmp.name, "r") as f:
            if "features" not in f:
                raise HTTPException(400, "Missing 'features' dataset in H5")
            feats = f["features"][:]

        n = len(feats)
        idx = np.random.choice(n, size=min(4096, n), replace=n < 4096)
        bag = torch.from_numpy(feats[idx]).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.clamp(model(bag).squeeze(), 0, 1).item()

    return PurityResult(
        purity_global=round(pred, 4),
        n_tiles=n,
        tile_scores_min=0.0,
        tile_scores_max=0.0,
        tile_scores_mean=pred,
    )


@app.post("/predict_heatmap", response_model=HeatmapResult)
async def predict_heatmap(file: UploadFile = File(...)):
    """Upload an H5 embedding file and get per-tile purity scores."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent / "ml"))
    from enso_purity_mil.heatmap import predict_tile_scores

    model, device = _load_model()

    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as tmp:
        tmp.write(raw)
        tmp_path = tmp.name

    try:
        scores, coords = predict_tile_scores(model, tmp_path, k=81, batch_size=1024, device=device)

        n = len(scores)
        idx = np.random.choice(n, size=min(4096, n), replace=n < 4096)
        bag = torch.from_numpy(
            h5py.File(tmp_path, "r")["features"][idx]
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_global = torch.clamp(model(bag).squeeze(), 0, 1).item()
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return HeatmapResult(
        purity_global=round(pred_global, 4),
        n_tiles=len(scores),
        scores=scores.tolist(),
        coords_x=coords[:, 0].tolist(),
        coords_y=coords[:, 1].tolist(),
    )
