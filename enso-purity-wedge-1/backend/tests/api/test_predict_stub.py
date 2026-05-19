import io
import h5py
import numpy as np
from fastapi.testclient import TestClient
from enso_purity.api.main import app


def make_h5_bytes():
    bio = io.BytesIO()
    with h5py.File(bio, "w") as f:
        f.create_dataset("features", data=np.random.randn(10, 32).astype("float32"))
        f.create_dataset("coords", data=np.zeros((10,2), dtype="int32"))
    bio.seek(0)
    return bio.getvalue()

def test_predict_stub_returns_fields():
    c = TestClient(app)
    payload = make_h5_bytes()
    r = c.post("/predict_purity", files={"file": ("test.h5", payload, "application/octet-stream")})
    assert r.status_code == 200
    j = r.json()
    assert "purity_wsi" in j and "purity_ta" in j and "heatmap_png_base64" in j
