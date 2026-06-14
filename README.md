# UAV Intent Estimation — KAN + Unified Mechatronic Engine

Proof-of-concept that classifies low-altitude UAV **intent** (benign / hobbyist /
hostile) from mechatronic flight parameters using a **Kolmogorov–Arnold Network
(KAN)**, and visualizes geographic **risk** on an interactive map.

> ⚠ **Proof-of-concept on simulated data — not an operational system.**

## Core idea
Digital telemetry (GPS, RF) can be spoofed; **actuator physics cannot**. The model
reads five per-drone time series — altitude, pitch angle, rotor RPM, velocity, and
**torque load (%)** — plus scene swarm size. Sustained torque above the civilian
envelope (100%) is the un-spoofable signal at the heart of the thesis.

## How it works
- **Data** (`data/`): synthetic tracks from a continuous *aggression/erraticness*
  latent model, so classes overlap realistically (no trivially-perfect accuracy).
- **Model** (`model/`): a small KAN on 18 derived features. Held-out test accuracy
  **85.7%** (easy 98%, hard/ambiguous 48%), with adjacent-class confusion. Honest,
  reproducible (seed 42) — see `model/metrics.json`.
- **API** (`api/`): FastAPI service. `predict()` reuses the *same* `core` feature
  extractor as training, so model and serving can't disagree.
- **Web** (`web/`): dashboard (Leaflet + Chart.js) with three input modes —
  Simulate, Configure (live sliders), Upload CSV/JSON — plus a green→red risk
  density map over real Istanbul strategic sites, with draw-on-map trajectories.

## Run locally
```bash
pip install -r requirements.txt
python -m data.generator     # generate dataset (once)
python -m model.train        # train + save model (once)
python -m api.server         # serve → http://127.0.0.1:8000
```

## API
`GET /api/metrics` · `GET /api/sliders` · `POST /api/simulate` · `POST /api/synthesize`
· `POST /api/classify` · `POST /api/classify_file` · `POST /api/risk` · `GET /api/sample`

## Deploy
Docker image; see [DEPLOY.md](DEPLOY.md) (Render / Railway / Fly). Project plan and
status: [ROADMAP.md](ROADMAP.md). Data details: [data/DATA_CARD.md](data/DATA_CARD.md).
Figure guide: [data/FIGURES.md](data/FIGURES.md).
