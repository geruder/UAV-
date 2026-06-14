"""Phase 1: synthetic UAV track generator.

Produces labeled raw time series for the five per-drone parameters plus a
scene-level object_count, following the class x parameter behavior table in
ROADMAP.md. Raw series are saved for visualization; the core feature extractor
derives the model's feature vectors. Fully seeded -> reproducible.

Run:  python -m data.generator
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

from core.config import (
    SEED,
    Intent,
    INTENT_NAMES,
    PARAMETERS,
    TRACK_LENGTH,
    TORQUE_ENVELOPE_PCT,
    OBJECT_COUNT_MAX,
)
from core.features import FEATURE_NAMES, extract_features

N_PER_CLASS = 2000
T = TRACK_LENGTH
DATA_DIR = os.path.dirname(__file__)

# Fraction of "hard"/ambiguous tracks per class. These deliberately blur the
# class boundaries so the model cannot win on the torque-envelope shortcut alone:
#   - hostile (hard): a careful adversary that mostly STAYS WITHIN the envelope
#   - benign  (hard): heavy payload / wind -> occasional torque spikes near the limit
#   - hobbyist(hard): pushed toward both neighbors in pitch / rpm / velocity
HARD_FRACTION = 0.25


# --- low-level signal building blocks -----------------------------------------

def _smooth(x: np.ndarray, window: int = 7) -> np.ndarray:
    """Moving-average smoothing so series look like physical motion, not white noise."""
    if window <= 1:
        return x
    k = np.ones(window) / window
    return np.convolve(x, k, mode="same")


def _random_walk(rng, scale: float) -> np.ndarray:
    """Smoothed cumulative random walk centered at zero."""
    steps = rng.normal(0.0, scale, size=T)
    return _smooth(np.cumsum(steps) - np.cumsum(steps).mean())


# --- per-class track generation -----------------------------------------------

# --- continuous latent model --------------------------------------------------
# Every track is generated from two latent factors that the SAME synthesis
# function turns into signals:
#   agg = aggression  -> pitch amplitude, rotor spikes, velocity burst, dive depth,
#                        and torque level (high agg breaches the civilian envelope)
#   err = erraticness -> jitter / variability across all parameters
# The three classes are OVERLAPPING distributions in this (agg, err) space, so a
# calm hostile really can look benign and an aggressive hobbyist really can look
# hostile. That is what produces realistic, adjacent-class confusion.

_MU_AGG = {Intent.BENIGN: 0.18, Intent.HOBBYIST: 0.38, Intent.HOSTILE: 0.68}
_MU_ERR = {Intent.BENIGN: 0.18, Intent.HOBBYIST: 0.72, Intent.HOSTILE: 0.40}


def _draw_latents(rng, intent: Intent, hard: bool) -> tuple[float, float]:
    """Sample (agg, err) for a track. Hard tracks are wider and pulled toward a
    neighboring class, making them genuinely ambiguous."""
    if hard:
        s_agg, s_err = 0.20, 0.20
        neighbor = rng.choice([i for i in Intent if i != intent])
        w = rng.uniform(0.3, 0.7)
        mu_agg = (1 - w) * _MU_AGG[intent] + w * _MU_AGG[neighbor]
        mu_err = (1 - w) * _MU_ERR[intent] + w * _MU_ERR[neighbor]
    else:
        s_agg, s_err = 0.09, 0.11
        mu_agg, mu_err = _MU_AGG[intent], _MU_ERR[intent]
    agg = float(np.clip(rng.normal(mu_agg, s_agg), 0.0, 1.0))
    err = float(np.clip(rng.normal(mu_err, s_err), 0.0, 1.0))
    return agg, err


def _synth(rng, agg: float, err: float) -> dict[str, np.ndarray]:
    """Turn latent (agg, err) into the five raw parameter time series."""
    lin = np.linspace(0, 1, T)
    # Altitude: random base; the more aggressive, the more likely a terrain-hug dive.
    altitude = rng.uniform(35, 110) + _random_walk(rng, 3 + 5 * err) + rng.normal(0, 2 + 3 * err, T)
    if rng.random() < agg:
        altitude = altitude - lin * rng.uniform(20, 80) * agg
    altitude = np.clip(altitude, 8, 130)
    # Pitch: amplitude from aggression, extra jitter from erraticness.
    pitch = _smooth(rng.normal(0, 2 + 26 * agg, T)) + rng.normal(0, 8 * err, T)
    # Rotor RPM: base + aggression-scaled spikes; variability from erraticness.
    rotor_rpm = (4700 + 1600 * agg) + (500 + 1500 * agg) * np.abs(
        np.sin(rng.uniform(4, 8) * lin * np.pi)) + rng.normal(0, 80 + 320 * err, T)
    # Velocity: burst-loiter grows with aggression; noise from erraticness.
    velocity = (9 + 3 * agg) + (1 + 12 * agg) * np.abs(
        np.sin(rng.uniform(3, 6) * lin * np.pi)) - 6 * agg + rng.normal(0, 1 + 2.5 * err, T)
    velocity = np.clip(velocity, 0.5, 22)
    # Torque tied to aggression: high aggression breaches the 100% civilian envelope.
    torque = (45 + 58 * agg) + (4 + 14 * agg) * np.sin(
        rng.uniform(3, 5) * lin * np.pi) + rng.normal(0, 4 + 6 * err, T)
    return _pack(altitude, pitch, rotor_rpm, velocity, torque)


def _object_count(rng, intent: Intent) -> int:
    """Scene swarm size: class-conditioned but overlapping (benign can be busy)."""
    if intent == Intent.HOSTILE:
        return 1 if rng.random() < 0.5 else int(rng.integers(4, OBJECT_COUNT_MAX + 1))
    if intent == Intent.BENIGN:
        return 1 if rng.random() < 0.7 else int(rng.integers(2, OBJECT_COUNT_MAX + 1))
    return 1 if rng.random() < 0.85 else int(rng.integers(2, 5))  # hobbyist


def _generate(rng, intent: Intent, hard: bool) -> tuple[dict[str, np.ndarray], int]:
    agg, err = _draw_latents(rng, intent, hard)
    return _synth(rng, agg, err), _object_count(rng, intent)


def _pack(altitude, pitch, rotor_rpm, velocity, torque) -> dict[str, np.ndarray]:
    """Assemble a track dict keyed by the canonical PARAMETERS names."""
    return {
        "altitude": altitude,
        "pitch_angle": pitch,
        "rotor_rpm": rotor_rpm,
        "velocity": velocity,
        "torque_load": torque,
    }


# --- dataset assembly ---------------------------------------------------------

def generate_dataset(seed: int = SEED, n_per_class: int = N_PER_CLASS,
                     hard_fraction: float = HARD_FRACTION):
    """Generate the full dataset. Returns (raw, object_counts, labels, hard, features)."""
    rng = np.random.default_rng(seed)
    n = len(Intent) * n_per_class
    raw = np.zeros((n, len(PARAMETERS), T), dtype=np.float32)
    object_counts = np.zeros(n, dtype=np.int16)
    labels = np.zeros(n, dtype=np.int8)
    hard = np.zeros(n, dtype=np.int8)
    feats = np.zeros((n, len(FEATURE_NAMES)), dtype=np.float32)

    i = 0
    for intent in Intent:
        for _ in range(n_per_class):
            is_hard = rng.random() < hard_fraction
            track, oc = _generate(rng, intent, is_hard)
            for j, p in enumerate(PARAMETERS):
                raw[i, j, :] = track[p]
            object_counts[i] = oc
            labels[i] = int(intent)
            hard[i] = int(is_hard)
            feats[i, :] = extract_features(track, oc)
            i += 1
    return raw, object_counts, labels, hard, feats


def save_dataset():
    raw, object_counts, labels, hard, feats = generate_dataset()

    # Raw series (for visualization / animation) -> compressed npz.
    raw_path = os.path.join(DATA_DIR, "raw_tracks.npz")
    np.savez_compressed(
        raw_path, raw=raw, object_count=object_counts, label=labels, hard=hard,
        parameters=np.array(PARAMETERS),
    )

    # Derived features (for the model) -> parquet.
    df = pd.DataFrame(feats, columns=FEATURE_NAMES)
    df["intent"] = labels
    df["hard"] = hard
    feat_path = os.path.join(DATA_DIR, "features.parquet")
    df.to_parquet(feat_path, index=False)

    return raw_path, feat_path, df


if __name__ == "__main__":
    raw_path, feat_path, df = save_dataset()
    print(f"Saved raw series  -> {raw_path}")
    print(f"Saved features    -> {feat_path}")
    print(f"\nDataset: {len(df)} tracks, {len(FEATURE_NAMES)} features")
    print(f"Hard/ambiguous tracks: {df['hard'].sum()} ({df['hard'].mean():.0%})")
    print("Class balance:")
    for v, name in INTENT_NAMES.items():
        print(f"  {name:9s}: {(df['intent'] == int(v)).sum()}")
    # Quick sanity on the spine: hostile should breach the envelope, others should not.
    # With hard cases, hostile time-above drops (careful adversary) and benign rises
    # (payload spikes) -- the torque-only shortcut is now intentionally weakened.
    print(f"\nActuator envelope = {TORQUE_ENVELOPE_PCT:.0f}%  | mean time-above-envelope:")
    for v, name in INTENT_NAMES.items():
        sub = df[df["intent"] == int(v)]
        easy = sub.loc[sub["hard"] == 0, "torque_time_above_envelope"].mean()
        hardm = sub.loc[sub["hard"] == 1, "torque_time_above_envelope"].mean()
        print(f"  {name:9s}: easy={easy:.3f}  hard={hardm:.3f}")
