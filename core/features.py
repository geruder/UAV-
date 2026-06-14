"""Feature extraction: raw time series -> model feature vector.

THIS IS THE SINGLE SOURCE OF TRUTH for what the model sees. The data generator,
the training pipeline, and the inference API all call extract_features(), so they
can never disagree about feature meaning or order.

A "track" is a dict mapping each parameter name in core.config.PARAMETERS to a
1-D numpy array of length TRACK_LENGTH, plus a scalar scene-level object_count.
"""

from __future__ import annotations

import numpy as np

from core.config import PARAMETERS, TORQUE_ENVELOPE_PCT


def _feature_names() -> list[str]:
    """Canonical, ordered feature names. Order here == order in the vector."""
    names: list[str] = []
    # Per-parameter summary statistics (mean / max / std), in PARAMETERS order.
    for p in PARAMETERS:
        names += [f"{p}_mean", f"{p}_max", f"{p}_std"]
    # Actuator-envelope features (the spine): how the track relates to the
    # civilian torque limit. These are what the UI's "why" panel highlights.
    names += ["torque_time_above_envelope", "torque_mean_exceedance"]
    # Scene context.
    names += ["object_count"]
    return names


FEATURE_NAMES: list[str] = _feature_names()
N_FEATURES: int = len(FEATURE_NAMES)


def extract_features(track: dict[str, np.ndarray], object_count: int) -> np.ndarray:
    """Turn one raw track into a fixed-order feature vector.

    Args:
        track: {parameter_name: 1-D array(TRACK_LENGTH)} for every name in
            core.config.PARAMETERS.
        object_count: scene-level swarm size for this track.

    Returns:
        1-D float array of length N_FEATURES, ordered to match FEATURE_NAMES.
    """
    feats: list[float] = []
    for p in PARAMETERS:
        series = np.asarray(track[p], dtype=float)
        feats += [float(series.mean()), float(series.max()), float(series.std())]

    # Envelope features derived from the torque series and the one shared limit.
    torque = np.asarray(track["torque_load"], dtype=float)
    above = torque > TORQUE_ENVELOPE_PCT
    time_above = float(above.mean())  # fraction of flight above the civilian limit
    # Mean amount by which torque exceeds the envelope while above it (0 if never).
    exceed = float((torque[above] - TORQUE_ENVELOPE_PCT).mean()) if above.any() else 0.0
    feats += [time_above, exceed]

    feats += [float(object_count)]

    vec = np.asarray(feats, dtype=float)
    assert vec.shape[0] == N_FEATURES, "feature vector length drifted from FEATURE_NAMES"
    return vec


def extract_features_batch(
    tracks: list[dict[str, np.ndarray]], object_counts: list[int]
) -> np.ndarray:
    """Vectorize extract_features over many tracks -> (n_tracks, N_FEATURES)."""
    return np.stack(
        [extract_features(t, oc) for t, oc in zip(tracks, object_counts)], axis=0
    )
