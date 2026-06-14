# Deployment

The app is a FastAPI service that also serves the static dashboard (`web/index.html`).
It ships as a Docker image, so it deploys the same way anywhere.

## What the image contains
- `core/`, `api/`, `data/`, `model/` (trained KAN checkpoint + scaler + metrics), `web/`.
- Heavy artifacts (datasets, plots, intermediate checkpoints) are excluded via `.dockerignore`.
- CPU-only PyTorch (no CUDA) to keep the image small.

## Run locally (Docker)
```bash
docker build -t uav-intent .
docker run --rm -p 8000:8000 uav-intent
# open http://localhost:8000
```
The container honors `$PORT` (defaults to 8000):
```bash
docker run --rm -e PORT=9000 -p 8000:9000 uav-intent
```

## Deploy options

### Render (simplest — uses render.yaml)
1. Push this repo to GitHub.
2. Render → **New + → Blueprint** → select the repo.
3. Render reads `render.yaml`, builds the Dockerfile, and gives a public URL.
   Health check: `/api/metrics`. Free plan sleeps when idle (slow first hit).

### Railway
1. Push to GitHub → Railway → **New Project → Deploy from repo**.
2. Railway auto-detects the `Dockerfile`. No port config needed (it sets `$PORT`).

### Fly.io
```bash
flyctl launch        # detects the Dockerfile; accept defaults
flyctl deploy
```

## Notes / gotchas
- **Cold start:** the *first* classification loads the KAN checkpoint (a few seconds);
  subsequent requests are instant. On free tiers the instance also sleeps when idle.
- **Memory:** torch + a tiny KAN fits comfortably; ~512 MB is enough for CPU inference.
- **Frontend:** Leaflet + Chart.js load from CDNs, so the deployed host needs outbound
  internet (true on all the above). No build step.
- **Keep the banner:** the dashboard states "Proof-of-concept on simulated data — not an
  operational system." Leave it on for a public deployment.
