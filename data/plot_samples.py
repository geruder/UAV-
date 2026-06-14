"""Phase 1 verification: plot one raw track per class for the five parameters.

Run:  python -m data.plot_samples
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from core.config import PARAMETERS, INTENT_NAMES, INTENT_COLORS, Intent, TORQUE_ENVELOPE_PCT

DATA_DIR = os.path.dirname(__file__)


def main():
    d = np.load(os.path.join(DATA_DIR, "raw_tracks.npz"), allow_pickle=True)
    raw, labels, oc = d["raw"], d["label"], d["object_count"]

    fig, axes = plt.subplots(len(PARAMETERS), 1, figsize=(10, 12), sharex=True)
    for intent in Intent:
        idx = np.where(labels == int(intent))[0][0]  # first example of this class
        color = INTENT_COLORS[intent]
        name = INTENT_NAMES[intent]
        for j, p in enumerate(PARAMETERS):
            axes[j].plot(raw[idx, j, :], color=color, label=f"{name} (swarm={oc[idx]})", lw=1.6)
            axes[j].set_ylabel(p)

    # Mark the actuator envelope on the torque subplot (the spine).
    t_idx = PARAMETERS.index("torque_load")
    axes[t_idx].axhline(TORQUE_ENVELOPE_PCT, color="black", ls="--", lw=1,
                        label=f"civilian envelope ({TORQUE_ENVELOPE_PCT:.0f}%)")

    axes[0].set_title("One sample track per class")
    axes[t_idx].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig.tight_layout()
    out = os.path.join(DATA_DIR, "sample_tracks.png")
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
