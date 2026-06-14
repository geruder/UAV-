"""FastAPI web service for UAV intent estimation.

Endpoints (all under /api):
  GET  /api/metrics    -> model metrics, class names, torque envelope
  POST /api/simulate   -> generate a track of a class + its prediction (demo)
  POST /api/classify   -> classify an arbitrary track
  POST /api/risk       -> geographic risk heatmap for a track
GET / serves the dashboard frontend.

Run:  python -m api.server     (or: uvicorn api.server:app --reload)
"""

from __future__ import annotations

import io
import json
import os

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse
from pydantic import BaseModel, Field

from core.config import (
    Intent, INTENT_NAMES, INTENT_COLORS, PARAMETERS, TRACK_LENGTH, TORQUE_ENVELOPE_PCT,
)
from api.predict import predict
from api.geo import (
    GRID_N, MAP_CENTER, MAP_SPAN, strategic_value_grid, synthesize_ground_path,
    risk_heatmap, locations_view, latlon_path_to_grid,
)
from api.synth import SLIDERS, synth_from_levels
from data.generator import _generate

app = FastAPI(title="UAV Intent Estimation", version="1.0")

_HERE = os.path.dirname(__file__)
_WEB = os.path.join(_HERE, "..", "web", "index.html")
_METRICS = os.path.join(_HERE, "..", "model", "metrics.json")


# --- request / response models ------------------------------------------------

class Track(BaseModel):
    altitude: list[float]
    pitch_angle: list[float]
    rotor_rpm: list[float]
    velocity: list[float]
    torque_load: list[float]
    object_count: int = Field(1, ge=1)
    # Optional user-drawn ground path as [[lat,lon],...] (Configure/Upload modes).
    path: list[list[float]] | None = None

    def to_dict(self) -> dict[str, np.ndarray]:
        return {p: np.asarray(getattr(self, p), dtype=float) for p in PARAMETERS}


class SimulateRequest(BaseModel):
    intent_id: int = Field(..., ge=0, le=2)
    hard: bool = False


class SynthRequest(BaseModel):
    """Slider levels -> a synthesized track. Keys match api.synth.SLIDERS."""
    altitude: float = 90
    pitch: float = 6
    rotor_rpm: float = 5200
    velocity: float = 11
    torque: float = 60
    erratic: float = 0.3
    object_count: int = Field(1, ge=1)


def _bundle(track: dict[str, np.ndarray], object_count: int, **extra) -> dict:
    """Common response: the track + its prediction (+ optional ground-truth)."""
    out = {
        "object_count": int(object_count),
        "track": {p: np.asarray(track[p], float).round(3).tolist() for p in PARAMETERS},
        "prediction": predict(track, object_count),
    }
    out.update(extra)
    return out


def _track_from_table(df: pd.DataFrame, object_count: int) -> tuple[dict, int]:
    """Build a track dict from a dataframe with the 5 parameter columns."""
    missing = [p for p in PARAMETERS if p not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing column(s): {missing}. Required: {PARAMETERS}")
    track = {p: df[p].to_numpy(dtype=float) for p in PARAMETERS}
    if any(len(track[p]) < 3 for p in PARAMETERS):
        raise HTTPException(400, "Each parameter needs at least 3 timesteps.")
    if "object_count" in df.columns and len(df["object_count"]):
        object_count = int(df["object_count"].iloc[0])
    return track, max(1, int(object_count))


# --- helpers ------------------------------------------------------------------

def _classify_track(track: dict[str, np.ndarray], object_count: int) -> dict:
    return predict(track, object_count)


# --- endpoints ----------------------------------------------------------------

@app.get("/api/metrics")
def get_metrics():
    out = {
        "class_names": [INTENT_NAMES[i] for i in Intent],
        "class_colors": [INTENT_COLORS[i] for i in Intent],
        "parameters": PARAMETERS,
        "track_length": TRACK_LENGTH,
        "torque_envelope": TORQUE_ENVELOPE_PCT,
    }
    try:
        with open(_METRICS) as f:
            out["model"] = json.load(f)
    except Exception:
        out["model"] = None
    return out


@app.get("/api/sliders")
def get_sliders():
    """Slider definitions for the Configure panel (single source of truth)."""
    return {"sliders": SLIDERS}


@app.post("/api/simulate")
def simulate(req: SimulateRequest):
    """Generate a track of the requested class and classify it (demo convenience)."""
    rng = np.random.default_rng()  # fresh randomness each call
    intent = Intent(req.intent_id)
    track, object_count = _generate(rng, intent, req.hard)
    return _bundle(track, object_count, true_intent=INTENT_NAMES[intent],
                   true_intent_id=int(intent), hard=req.hard)


@app.post("/api/synthesize")
def synthesize(req: SynthRequest):
    """Build a track from slider levels and classify it (live configurator)."""
    track, object_count = synth_from_levels(req.model_dump())
    return _bundle(track, object_count, source="sliders")


@app.post("/api/classify_file")
async def classify_file(file: UploadFile = File(...), object_count: int = Form(1)):
    """Classify a user-supplied track uploaded as CSV or JSON."""
    raw = (await file.read())
    name = (file.filename or "").lower()
    try:
        if name.endswith(".json") or raw.lstrip()[:1] in (b"{", b"["):
            obj = json.loads(raw.decode("utf-8"))
            df = pd.DataFrame({p: obj[p] for p in PARAMETERS if p in obj})
            if "object_count" in obj:
                df["object_count"] = int(obj["object_count"])
        else:
            df = pd.read_csv(io.BytesIO(raw))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not parse file: {e}")
    track, oc = _track_from_table(df, object_count)
    return _bundle(track, oc, source=file.filename)


@app.post("/api/classify")
def classify(track: Track):
    return _classify_track(track.to_dict(), track.object_count)


@app.get("/api/sample")
def sample(fmt: str = "csv", intent_id: int = 2):
    """Downloadable sample track in the accepted upload format."""
    rng = np.random.default_rng(1)
    track, oc = _generate(rng, Intent(intent_id), hard=False)
    if fmt == "json":
        body = json.dumps({**{p: track[p].round(3).tolist() for p in PARAMETERS},
                           "object_count": oc}, indent=2)
        return PlainTextResponse(body, media_type="application/json",
                                 headers={"Content-Disposition": "attachment; filename=sample_track.json"})
    df = pd.DataFrame({p: track[p].round(3) for p in PARAMETERS})
    df["object_count"] = oc
    return PlainTextResponse(df.to_csv(index=False), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=sample_track.csv"})


_VALUE_GRID = strategic_value_grid()  # static -> compute once


@app.post("/api/risk")
def risk(track: Track):
    """Risk heatmap = strategic value x proximity-to-path x P(hostile).

    The trajectory is diversified and intent-aware: a deterministic seed derived
    from the track keeps it stable for a given track, while different tracks route
    through different sectors (hostile drones favor strategic sites)."""
    td = track.to_dict()
    pred = _classify_track(td, track.object_count)
    hostile_prob = pred["probabilities"][INTENT_NAMES[Intent.HOSTILE]]
    if track.path and len(track.path) >= 2:
        # user-drawn trajectory (Configure / Upload modes)
        path = latlon_path_to_grid(track.path, n=GRID_N)
        target = {"name": "user-drawn path", "type": "drawn"}
    else:
        seed = int(abs(np.sum(td["torque_load"]) * 7 + np.mean(td["altitude"]) * 13)) % (2 ** 32)
        path, target = synthesize_ground_path(td, n=GRID_N, seed=seed, hostile_prob=hostile_prob)
    risk_grid = risk_heatmap(_VALUE_GRID, path, hostile_prob)
    return {
        "grid_n": GRID_N,
        "center": MAP_CENTER,
        "span": MAP_SPAN,
        "hostile_prob": round(float(hostile_prob), 4),
        "risk_grid": risk_grid.round(4).tolist(),
        "path": path.round(2).tolist(),
        "target": {"name": target["name"], "type": target["type"]},
        "locations": locations_view(),
        "prediction": pred,
    }


@app.get("/")
def index():
    return FileResponse(_WEB)


if __name__ == "__main__":
    import uvicorn
    # Honor the platform-provided $PORT (Render/Railway/Fly); default to 8000 locally.
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run(app, host=host, port=port)
