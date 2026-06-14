"""Synthesize a track from explicit parameter LEVELS (for the slider UI).

Unlike data.generator (which samples from latent class distributions to build the
training set), this builds a track directly from user-chosen levels so the slider
panel is a "what drives intent?" explorer. Seeded -> dragging is smooth/deterministic:
the same slider positions always produce the same track and classification.
"""

from __future__ import annotations

import numpy as np

from core.config import PARAMETERS, TRACK_LENGTH

T = TRACK_LENGTH
_LIN = np.linspace(0, 1, T)


def _smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def _walk(rng, scale: float) -> np.ndarray:
    c = np.cumsum(rng.normal(0, scale, T))
    return _smooth(c - c.mean())


# Slider definitions: (key, label, min, max, default, step). Frontend reads this
# via /api/sliders so the UI and synthesis never drift apart.
SLIDERS = [
    {"key": "altitude",    "label": "Altitude (m)",          "min": 10,   "max": 130,  "default": 90,   "step": 1},
    {"key": "pitch",       "label": "Pitch aggressiveness (deg)", "min": 0, "max": 35, "default": 6,    "step": 1},
    {"key": "rotor_rpm",   "label": "Rotor RPM",             "min": 4500, "max": 8500, "default": 5200, "step": 50},
    {"key": "velocity",    "label": "Velocity (m/s)",        "min": 0,    "max": 20,   "default": 11,   "step": 0.5},
    {"key": "torque",      "label": "Torque load (%)",       "min": 40,   "max": 130,  "default": 60,   "step": 1},
    {"key": "erratic",     "label": "Erraticness",           "min": 0,    "max": 1,    "default": 0.3,  "step": 0.05},
    {"key": "object_count","label": "Swarm size",            "min": 1,    "max": 12,   "default": 1,    "step": 1},
]


def synth_from_levels(levels: dict) -> tuple[dict[str, np.ndarray], int]:
    """Build a track (5 time series) + object_count from slider levels."""
    rng = np.random.default_rng(0)  # fixed -> deterministic for a given config
    e = float(levels.get("erratic", 0.3))
    agg = float(levels.get("pitch", 6)) / 35.0  # pitch slider doubles as "aggression"

    altitude = float(levels["altitude"]) + _walk(rng, 2 + 6 * e) + rng.normal(0, 1 + 3 * e, T)
    # the more aggressive, the more likely a descent toward a target
    altitude = altitude - _LIN * (40 * agg)
    altitude = np.clip(altitude, 5, 135)

    pitch = _smooth(rng.normal(0, max(0.5, float(levels["pitch"])), T))

    rotor_rpm = (float(levels["rotor_rpm"])
                 + (150 + 700 * agg) * np.abs(np.sin(5 * _LIN * np.pi))
                 + rng.normal(0, 60 + 260 * e, T))

    v = float(levels["velocity"])
    velocity = (v + (1 + 11 * agg) * np.abs(np.sin(4 * _LIN * np.pi)) - 5 * agg
                + rng.normal(0, 1 + 2.5 * e, T))
    velocity = np.clip(velocity, 0, 22)

    torque = (float(levels["torque"]) + (3 + 9 * agg) * np.sin(3 * _LIN * np.pi)
              + rng.normal(0, 2 + 5 * e, T))

    track = {
        "altitude": altitude,
        "pitch_angle": pitch,
        "rotor_rpm": rotor_rpm,
        "velocity": velocity,
        "torque_load": torque,
    }
    return track, int(levels.get("object_count", 1))
