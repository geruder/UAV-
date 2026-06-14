"""Idealized / CONCEPTUAL archetype track per class.

These are clean canonical exemplars that illustrate what each class *means*
(matching the written class definitions). They are NOT samples of the training
data -- the real training data is hard and overlapping (see plot_separation.py).
Use this only for the "class definitions / concept" section of the report.

Run:  python -m data.plot_archetypes_ideal
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core.config import PARAMETERS, INTENT_NAMES, INTENT_COLORS, Intent, TORQUE_ENVELOPE_PCT, TRACK_LENGTH

DATA_DIR = os.path.dirname(__file__)
T = TRACK_LENGTH
LIN = np.linspace(0, 1, T)


def _ideal(intent: Intent, rng) -> dict[str, np.ndarray]:
    """Clean canonical signals per the written class definitions (low noise)."""
    if intent == Intent.BENIGN:
        return {
            "altitude": 100 + 3 * np.sin(2 * np.pi * LIN) + rng.normal(0, 0.8, T),   # steady ~100 m
            "pitch_angle": 1.5 * np.sin(4 * np.pi * LIN) + rng.normal(0, 0.6, T),     # almost straight
            "rotor_rpm": 5000 + 30 * np.sin(3 * np.pi * LIN) + rng.normal(0, 20, T),  # stable
            "velocity": 12 + rng.normal(0, 0.25, T),                                  # constant cruise
            "torque_load": 55 + 6 * np.sin(3 * np.pi * LIN) + rng.normal(0, 1.5, T),  # well within envelope
        }
    if intent == Intent.HOBBYIST:
        walk = np.cumsum(rng.normal(0, 1.2, T)); walk -= walk.mean()
        return {
            "altitude": np.clip(45 + walk + rng.normal(0, 2, T), 15, 90),            # low, erratic
            "pitch_angle": 9 * np.sin(10 * np.pi * LIN) + rng.normal(0, 3, T),        # slight vibrations
            "rotor_rpm": 5200 + 400 * np.sin(9 * np.pi * LIN) + rng.normal(0, 120, T),# fluctuating
            "velocity": np.clip(8 + 3 * np.sin(7 * np.pi * LIN) + rng.normal(0, 1.2, T), 2, 15),  # variable
            "torque_load": np.clip(70 + 10 * np.sin(6 * np.pi * LIN) + rng.normal(0, 3, T), 45, 95),  # < 100
        }
    # HOSTILE
    return {
        "altitude": np.clip(np.linspace(95, 20, T) + 4 * np.sin(6 * np.pi * LIN), 10, 100),  # terrain-hug dive
        "pitch_angle": 30 * np.sin(8 * np.pi * LIN) + rng.normal(0, 3, T),                   # aggressive, high-angle
        "rotor_rpm": 6500 + 1500 * np.abs(np.sin(7 * np.pi * LIN)) + rng.normal(0, 80, T),   # high + peaks
        "velocity": 17 * np.abs(np.sin(5 * np.pi * LIN)) + rng.normal(0, 1.0, T),            # burst -> loiter
        "torque_load": 95 + 25 * np.abs(np.sin(6 * np.pi * LIN)) + rng.normal(0, 2, T),      # crosses envelope
    }


def main():
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(len(PARAMETERS), 1, figsize=(10, 12), sharex=True)
    for intent in Intent:
        track = _ideal(intent, rng)
        for j, p in enumerate(PARAMETERS):
            axes[j].plot(track[p], color=INTENT_COLORS[intent], label=INTENT_NAMES[intent], lw=1.8)
            axes[j].set_ylabel(p)
    t_idx = PARAMETERS.index("torque_load")
    axes[t_idx].axhline(TORQUE_ENVELOPE_PCT, color="black", ls="--", lw=1,
                        label=f"civilian envelope ({TORQUE_ENVELOPE_PCT:.0f}%)")
    axes[0].set_title("IDEALIZED / CONCEPTUAL archetypes (class definitions, NOT training data)")
    axes[t_idx].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.tight_layout()
    out = os.path.join(DATA_DIR, "archetypes_ideal.png")
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
