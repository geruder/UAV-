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

## Keeping it always-on (so the link works whenever someone visits)
Render's **free** plan sleeps after ~15 min idle → a cold visitor waits ~50–60s.
The app **preloads the KAN at startup** (see `lifespan` in `api/server.py`), so once
an instance is awake, responses are instant. To keep it awake:
- Create a free **UptimeRobot** (or cron-job.org) monitor that pings
  `https://<your-app>.onrender.com/api/metrics` every **5–10 minutes**.
- A single always-on service fits within Render's free 750 instance-hours/month.
- Alternatively, the **Starter plan ($7/mo)** never sleeps — zero tricks.

## Notes / gotchas
- **Startup takes ~20s** (model preload). Render waits for this on deploy; it's normal.
- **Cold start:** the *first* classification loads the KAN checkpoint (a few seconds);
  subsequent requests are instant. On free tiers the instance also sleeps when idle.
- **Memory:** torch + a tiny KAN fits comfortably; ~512 MB is enough for CPU inference.
- **Frontend:** Leaflet + Chart.js load from CDNs, so the deployed host needs outbound
  internet (true on all the above). No build step.
- **Keep the banner:** the dashboard states "Proof-of-concept on simulated data — not an
  operational system." Leave it on for a public deployment.
