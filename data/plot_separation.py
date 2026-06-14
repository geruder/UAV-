"""Visualize class separation two honest ways:

1) Representative ("archetype") track per class -- the real track whose feature
   vector is closest to its class mean. Clean, pedagogical: "this is the idea."
2) Feature-space separation via PCA(2) over the 18 derived features -- shows the
   classes as clusters WITH realistic overlap: "this is why it is 85.7%, not 100%."

Run:  python -m data.plot_separation
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from core.config import PARAMETERS, INTENT_NAMES, INTENT_COLORS, Intent, TORQUE_ENVELOPE_PCT
from core.features import FEATURE_NAMES

DATA_DIR = os.path.dirname(__file__)


def main():
    raw = np.load(os.path.join(DATA_DIR, "raw_tracks.npz"), allow_pickle=True)
    tracks, labels = raw["raw"], raw["label"]
    df = pd.read_parquet(os.path.join(DATA_DIR, "features.parquet"))
    X = df[FEATURE_NAMES].to_numpy(dtype=float)
    Xz = (X - X.mean(0)) / (X.std(0) + 1e-8)  # standardize for fair distance / PCA

    # --- (1) archetype = real track nearest to its class-mean feature vector ---
    fig1, axes = plt.subplots(len(PARAMETERS), 1, figsize=(10, 12), sharex=True)
    for intent in Intent:
        cls = np.where(labels == int(intent))[0]
        mean_vec = Xz[cls].mean(0)
        medoid = cls[np.argmin(np.linalg.norm(Xz[cls] - mean_vec, axis=1))]
        for j, p in enumerate(PARAMETERS):
            axes[j].plot(tracks[medoid, j, :], color=INTENT_COLORS[intent],
                         label=INTENT_NAMES[intent], lw=1.8)
            axes[j].set_ylabel(p)
    t_idx = PARAMETERS.index("torque_load")
    axes[t_idx].axhline(TORQUE_ENVELOPE_PCT, color="black", ls="--", lw=1,
                        label=f"envelope ({TORQUE_ENVELOPE_PCT:.0f}%)")
    axes[0].set_title("Archetype (representative) track per class")
    axes[t_idx].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("timestep")
    fig1.tight_layout()
    out1 = os.path.join(DATA_DIR, "separation_archetypes.png")
    fig1.savefig(out1, dpi=110)
    print(f"Saved {out1}")

    # --- (2) feature-space separation via PCA(2) -------------------------------
    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(Xz)
    fig2, ax = plt.subplots(figsize=(8, 7))
    for intent in Intent:
        m = labels == int(intent)
        ax.scatter(Z[m, 0], Z[m, 1], s=8, alpha=0.35,
                   color=INTENT_COLORS[intent], label=INTENT_NAMES[intent])
    ev = pca.explained_variance_ratio_
    ax.set_xlabel(f"PC1 ({ev[0]:.0%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.0%} var)")
    ax.set_title("Class separation in feature space (PCA of 18 KAN features)")
    ax.legend()
    fig2.tight_layout()
    out2 = os.path.join(DATA_DIR, "separation_feature_space.png")
    fig2.savefig(out2, dpi=110)
    print(f"Saved {out2}")


if __name__ == "__main__":
    main()
