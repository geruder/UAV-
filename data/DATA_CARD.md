# Data Card — Synthetic UAV Intent Dataset

**Status:** synthetic / simulated. No real sensor data. For research and demo use only.

## Summary
6,000 simulated low-altitude (< 330 m) micro-UAV flight tracks, balanced across
three intent classes. Each track is a set of time series for five per-drone
mechatronic parameters plus a scene-level swarm size. Generated deterministically
from a fixed seed.

## Generation
- **Script:** `data/generator.py` (`python -m data.generator`)
- **Seed:** 42 (`core.config.SEED`) — fully reproducible.
- **Tracks:** 2,000 per class × 3 classes = 6,000.
- **Track length:** 120 timesteps (`core.config.TRACK_LENGTH`).

## Classes (`core.config.Intent`)
| label | name | profile |
|---|---|---|
| 0 | benign | commercial delivery: smooth, lawful, within hardware limits |
| 1 | hobbyist | recreational: erratic/jittery, but within hardware limits |
| 2 | hostile | loiter/strike: aggressive, sustained torque exceeds the civilian envelope |

## Parameters (raw time series)
altitude (m) · pitch_angle (deg) · rotor_rpm · velocity (m/s) · torque_load (%).

## Generative model (continuous latent factors)
To avoid trivially-separable classes (which gave a misleading 100% accuracy),
tracks are NOT hand-crafted per class. Instead each track is synthesized from two
latent factors by one shared function:
- **agg (aggression)** -> pitch amplitude, rotor spikes, velocity burst, dive
  depth, and torque level (high agg breaches the civilian envelope).
- **err (erraticness)** -> jitter / variability across all parameters.

The three classes are **overlapping** Gaussians in (agg, err) space
(benign agg≈0.18, hobbyist≈0.38, hostile≈0.68; err 0.18 / 0.72 / 0.40). A calm
hostile genuinely resembles a benign drone and an aggressive hobbyist resembles a
hostile one -> realistic, adjacent-class confusion rather than perfect separation.

## Scene feature
`object_count` = swarm size (UAVs in scene). Class-conditioned but **overlapping**:
benign is usually 1 yet occasionally high (busy delivery corridor), hostile is
either lone (1) or a swarm (4–12). Deliberately correlated-not-deterministic so
the model cannot shortcut "many drones = hostile".

## Difficulty / overlap (`HARD_FRACTION = 0.25`)
~26% of tracks are "hard": their latent factors are drawn with wider spread and
pulled toward a neighboring class, making them genuinely ambiguous. A `hard` flag
(0/1) is stored per track. Measured effect (Phase 2 test set): easy-subset
accuracy ≈ 0.98, hard-subset accuracy ≈ 0.48 — the model is confident on clear
cases and appropriately uncertain on ambiguous ones.

## Actuator envelope (the spine)
Civilian torque limit = 100% (`core.config.TORQUE_ENVELOPE_PCT`). Because torque
is tied to the continuous aggression factor, breaching is now graded, not binary:
mean fraction of flight above the envelope ≈ benign 0.01, hobbyist 0.02, hostile
0.12, with overlap. Torque is a strong-but-imperfect signal, as intended.

## Files
- `raw_tracks.npz` — raw series `(6000, 5, 120)`, `object_count`, `label`, `parameters`.
- `features.parquet` — 18 derived features (via `core.features.extract_features`) + `intent`.
- `sample_tracks.png` — one track per class (verification plot).

## Known limitations
Synthetic dynamics, not flight-validated. Single-track generation with a drawn
scene count (no true multi-drone scene simulation — future work). Noise is
Gaussian/random-walk, not a calibrated sensor model.
