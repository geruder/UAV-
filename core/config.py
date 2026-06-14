"""Frozen project definitions: classes, parameters, the actuator envelope.

These are the contracts the whole project depends on. Treat them as frozen once
Phase 0 is approved -- changing a name or order here ripples through data, model,
api, and app, so do it deliberately and in one place.
"""

from enum import IntEnum

# Global random seed -- used everywhere a generator/split is seeded, so the whole
# pipeline (data -> model -> metrics) is reproducible.
SEED = 42


class Intent(IntEnum):
    """The three intent classes. Integer values are the model's label encoding;
    keep the order stable -- the confusion matrix and UI colors rely on it."""

    BENIGN = 0      # commercial delivery: smooth, lawful, within hardware limits
    HOBBYIST = 1    # erratic recreational flight: jittery but still within limits
    HOSTILE = 2     # loitering / strike profile: aggressive, exceeds civilian envelope


# Human-readable names, indexed by Intent value.
INTENT_NAMES = {
    Intent.BENIGN: "benign",
    Intent.HOBBYIST: "hobbyist",
    Intent.HOSTILE: "hostile",
}

# Display colors, indexed by Intent value. Used identically in plots and the UI.
INTENT_COLORS = {
    Intent.BENIGN: "#2ca02c",    # green
    Intent.HOBBYIST: "#ff7f0e",  # orange
    Intent.HOSTILE: "#d62728",   # red
}

# The five per-drone parameters, each generated as a time series value(t).
# Order is canonical -- raw-series arrays and plots follow it.
PARAMETERS = [
    "altitude",      # meters above ground (micro-UAV domain, < 330 m)
    "pitch_angle",   # degrees; aggressive maneuvering -> high magnitude
    "rotor_rpm",     # rotor revolutions per minute (acoustic / mechatronic signature)
    "velocity",      # m/s ground speed
    "torque_load",   # % of actuator capacity -- the un-spoofable envelope signal
]

# Number of timesteps per track (one flight sample). Fixed so every raw series
# has the same shape and the feature extractor is deterministic.
TRACK_LENGTH = 120  # e.g. ~120 samples over a flight window

# --- The actuator envelope: the conceptual spine -------------------------------
# Civilian/commercial UAV hardware saturates near 100% torque load. Sustained
# demand ABOVE this line is physically revealing of a non-civilian airframe and
# cannot be spoofed the way GPS/RF telemetry can. The generator uses this to make
# hostile tracks breach the line; the model sees it via peak_torque /
# time_above_envelope; the UI explains flags with it. One number, used everywhere.
TORQUE_ENVELOPE_PCT = 100.0

# Scene-level swarm size bounds (object_count). Scene context attached to each
# track -- correlates with hostility but is deliberately NOT deterministic.
OBJECT_COUNT_MAX = 12
