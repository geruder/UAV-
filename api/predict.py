"""Inference: raw track -> intent + confidence + explanation.

Imports the SAME core.extract_features used in training, loads the saved KAN and
scaler, and returns a per-prediction "why". This is the clean boundary the API /
dashboard calls -- it cannot disagree with training because both go through core.

Run a self-test:  python -m api.predict
"""

from __future__ import annotations

import json
import os
import warnings

import numpy as np
import torch

from core.config import Intent, INTENT_NAMES, TORQUE_ENVELOPE_PCT
from core.features import FEATURE_NAMES, extract_features

_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")
_CKPT = os.path.join(_MODEL_DIR, "kan_ckpt")
_SCALER = os.path.join(_MODEL_DIR, "scaler.npz")
_METRICS = os.path.join(_MODEL_DIR, "metrics.json")

# Cached singletons (load once).
_model = None
_mean = None
_std = None
_importance = None


def _load():
    """Lazy-load model, scaler, and global feature importance."""
    global _model, _mean, _std, _importance
    if _model is not None:
        return
    from kan import KAN  # imported here so importing this module is cheap
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # single-row std warnings on forward
        _model = KAN.loadckpt(_CKPT)
    _model.eval()
    sc = np.load(_SCALER, allow_pickle=True)
    _mean, _std = sc["mean"].astype(np.float32), sc["std"].astype(np.float32)
    try:
        with open(_METRICS) as f:
            imp = json.load(f).get("feature_importance", {})
        _importance = np.array([imp.get(n, 0.0) for n in FEATURE_NAMES], dtype=np.float32)
        if _importance.sum() == 0:
            _importance = np.ones(len(FEATURE_NAMES), dtype=np.float32)
    except Exception:
        _importance = np.ones(len(FEATURE_NAMES), dtype=np.float32)


def _explain(vec: np.ndarray, track: dict[str, np.ndarray]) -> dict:
    """Per-prediction 'why': features that are both unusual (|z|) and important."""
    z = (vec - _mean) / _std
    score = np.abs(z) * _importance
    order = np.argsort(score)[::-1][:3]
    top = [
        {
            "feature": FEATURE_NAMES[i],
            "value": round(float(vec[i]), 3),
            "direction": "high" if z[i] > 0 else "low",
            "importance": round(float(_importance[i]), 3),
        }
        for i in order
    ]
    torque = np.asarray(track["torque_load"], dtype=float)
    frac_above = float((torque > TORQUE_ENVELOPE_PCT).mean())
    breached = frac_above > 0.0
    note = (
        f"Breached civilian torque envelope {frac_above:.0%} of flight"
        if breached else "Stayed within the civilian torque envelope"
    )
    return {"top_features": top, "envelope_breached": breached,
            "envelope_fraction": round(frac_above, 3), "note": note}


def predict_from_features(vec: np.ndarray) -> dict:
    """Predict from an already-extracted feature vector (len == N features)."""
    _load()
    x = torch.tensor(((np.asarray(vec, float) - _mean) / _std)[None, :], dtype=torch.float32)
    with torch.no_grad(), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        probs = torch.softmax(_model(x), dim=1).numpy()[0]
    cls = int(probs.argmax())
    return {
        "intent": INTENT_NAMES[Intent(cls)],
        "intent_id": cls,
        "confidence": round(float(probs[cls]), 4),
        "probabilities": {INTENT_NAMES[Intent(i)]: round(float(p), 4) for i, p in enumerate(probs)},
    }


def warmup() -> None:
    """Eagerly load the model + scaler and run one dummy forward, so the first
    real request (and the user's teacher's visit) is instant rather than paying
    the checkpoint-load cost on demand."""
    _load()
    try:
        predict_from_features(np.zeros(len(FEATURE_NAMES), dtype=float))
    except Exception:
        pass  # warmup is best-effort; never block startup on it


def predict(track: dict[str, np.ndarray], object_count: int) -> dict:
    """Predict intent for one raw track (the main entry point).

    Returns {intent, intent_id, confidence, probabilities, explanation}.
    """
    _load()
    vec = extract_features(track, object_count)
    out = predict_from_features(vec)
    out["explanation"] = _explain(vec, track)
    return out


if __name__ == "__main__":
    # Self-test / verification gate: predict on known tracks from the saved dataset.
    from core.config import PARAMETERS
    d = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "raw_tracks.npz"),
                allow_pickle=True)
    raw, labels, oc, hard = d["raw"], d["label"], d["object_count"], d["hard"]

    print("Self-test: predict one EASY track of each true class\n")
    correct = 0
    for intent in Intent:
        idx = np.where((labels == int(intent)) & (hard == 0))[0][0]
        track = {p: raw[idx, j, :] for j, p in enumerate(PARAMETERS)}
        r = predict(track, int(oc[idx]))
        correct += (r["intent_id"] == int(intent))
        print(f"true={INTENT_NAMES[intent]:9s} -> pred={r['intent']:9s} "
              f"conf={r['confidence']:.2f}  | {r['explanation']['note']}")
        tf = r["explanation"]["top_features"][0]
        print(f"    top feature: {tf['feature']} ({tf['direction']})")
    print(f"\n{correct}/3 easy tracks correct.")
