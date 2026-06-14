# UAV Intent Estimation — Project Roadmap

**Project:** KAN-based UAV intent classifier with a mechatronic feature engine and a GIS risk dashboard, delivered as a web service (decision-support demo).
**Author:** Mehmet Mertcan Aksu
**Guiding principle:** *Consistent and working over fast.* Every layer is verified before the next begins. One source of truth for definitions, so the pieces never drift apart.

---

## Architecture: single source of truth

The whole project depends on three definitions agreeing everywhere. They live in `core/` and every other module imports them — never copy-pasted.

```
core/      features, classes, ACTUATOR ENVELOPE   ← single source of truth
data/      generator + saved dataset + data-card
model/     KAN training, saved weights, metrics.json
api/       predict() — imports core, loads saved model
app/       Streamlit UI — imports core, calls predict()
docs/      README, architecture diagram, demo
```

The things that must stay consistent:
1. **Raw parameters (per-drone time series)** — Altitude, Pitch Angle, Rotor RPM, Velocity, Torque Load (%). Generated as `value(t)` over each flight.
2. **Scene parameter** — `object_count` = swarm size (UAVs in the scene). Scene-level context attached to each track; correlates with but does not determine hostility.
3. **Feature-extraction function** — the ONE `core` function that turns raw series → model feature vector (mean/max/std per parameter, plus `peak_torque`, `time_above_envelope`). Training and inference both call it, so they cannot disagree.
4. **Classes** — `benign / hobbyist / hostile` — same labels, order, and colors everywhere.
5. **Actuator envelope** — the civilian-hardware torque limit (~100%). The generator uses it to label hostile tracks; it appears in the model via `peak_torque`/`time_above_envelope`; the UI explains flags with it. The conceptual spine of the "physics can't be spoofed" thesis.

**Data representation:** both. Raw time series saved to disk (for visualization + animation); the derived feature vector (via the `core` extractor) is what the KAN trains and predicts on.

---

## Phase 0 — Foundation & definitions
**Goal:** Lock the contracts the rest of the project depends on.

- [x] Scaffold repo structure above.
- [x] Write `core/`: feature list, class enum, actuator-envelope constants/function.
- [x] Pin `requirements.txt`. Set a global random seed.
- [x] Decide the demo sentence: *trajectory in → intent + confidence + why → risk on a map.*

**Verification gate:** `core/` imports cleanly; definitions reviewed and frozen.
**Done when:** every later module has one place to import features, classes, and the envelope.

---

## Phase 1 — Synthetic data generator
**Goal:** Reproducible, labeled, realistic tracks. This is where credibility is earned.

- [x] Generator emits, per track, raw time series for the 5 per-drone parameters + a scene-level `object_count`. ~2,000 tracks/class, ~6k total. Parameter behavior by class:

  | Parameter | benign | hobbyist | hostile |
  |---|---|---|---|
  | Altitude | stable, lane-like | low, variable | terrain-hug / target dive |
  | Pitch Angle | low, smooth | erratic | high, aggressive |
  | Rotor RPM | steady | fluctuating | high, sustained spikes |
  | Velocity | constant cruise | variable | burst → loiter |
  | Torque Load % | low, within envelope | spikes, within envelope | sustained, **exceeds** envelope |
  | object_count (scene) | usually 1 | 1, occasionally few | 1 (lone) or many (swarm) |

- [x] `object_count` drawn from a class-conditioned distribution (individual-track generation; full multi-drone scenes = future work).
- [x] Add realistic sensor noise (the "noise signature" idea) to the raw series.
- [x] Run every track through the `core` feature-extractor to produce the training feature vector.
- [x] Save BOTH raw series and derived features to disk (parquet) + a short **data-card** documenting how it was made.

**Verification gate:** plot one track per class — they are visibly distinct; class balance is correct.
**Done when:** a fixed-seed script regenerates the identical dataset.

---

## Phase 2 — Train & validate the KAN
**Goal:** A real, honest, saved model.

- [x] Small `pykan` model on the feature vectors (width [18,8,3], grid 3, k 3; trains in ~30s on CPU).
- [x] Proper stratified train / val / test split (4200 / 900 / 900).
- [x] Report **real** accuracy + confusion matrix + loss curve. No hardcoded numbers.
- [x] L1 regularization (lamb) + early stopping on val loss to control overfitting.
- [ ] (Optional, if time) MLP baseline to substantiate KAN's parameter-efficiency claim.
- [x] Save weights (`kan_ckpt`), `scaler.npz`, `metrics.json`, feature attribution (KAN's "why").

**Result (held-out test, seed 42):** overall **85.7%** | easy 98.2% | hard 48.2%.
Confusion is adjacent-class and symmetric; top features span object_count,
rotor_rpm, velocity, and the torque-envelope feature (no single shortcut).

**Verification gate:** PASSED — believable accuracy, physically-sensible adjacent
confusion, train/val loss converge without divergence.
**Done when:** saved model + saved metrics that anyone can reproduce. ✓

---

## Phase 3 — Inference API
**Goal:** A clean boundary between model and UI.

- [x] `core`-importing `predict(track, object_count) -> {intent, confidence, probabilities, explanation}` (`api/predict.py`).
- [x] In-process functions (pure-Streamlit path). Per-prediction "why" = top |z|×importance features + envelope note. *(Optional REST FastAPI wrapper deferred.)*
- [x] Risk scoring: `api/geo.py` — strategic value grid × proximity-to-path × hostility.

**Verification gate:** PASSED — 3/3 known tracks classified correctly with sensible
confidence; hostile explanation reports the envelope breach; risk scales with hostility.
**Done when:** prediction works independently of any UI. ✓

---

## Phase 4 — Web app: dashboard + GIS heatmap  *(built FastAPI-style, not Streamlit)*
**Goal:** The demonstrable thing.

**Architecture:** FastAPI REST backend (`api/server.py`) + static frontend
(`web/index.html`, Leaflet + Chart.js via CDN, no build step).
Endpoints: `GET /api/metrics`, `GET /api/sliders`, `POST /api/simulate`,
`POST /api/synthesize`, `POST /api/classify`, `POST /api/classify_file`,
`POST /api/risk`, `GET /api/sample`; `GET /` serves the dashboard.

**Three input modes (all funnel through one render pipeline):**
- **Simulate** — generate a benign/hobbyist/hostile track.
- **Configure (sliders)** — 7 sliders (altitude, pitch, RPM, velocity, torque %,
  erraticness, swarm) synthesize a track live via `api/synth.py`; pushing torque
  > 100% flips intent toward hostile. Verified live.
- **Upload CSV/JSON** — user supplies a track; `/api/sample` provides templates.

- [x] UI: simulate a track (benign/hobbyist/hostile, easy/hard) → trajectory on a city map (Istanbul, Leaflet).
- [x] Intent + confidence badge + probability bars; class colors from `core` via `/api/metrics`.
- [x] **"Why" panel:** top contributing features + live actuator-envelope breach note.
- [x] GIS **risk density heatmap** (smooth green→red colormap, 80×80 + Gaussian blur) + ground path.
- [x] **Real strategic sites** (Bosphorus Bridge, Selimiye Barracks, Dolmabahçe, Taksim, Karaköy Port, Vodafone Park, Topkapı) by lat/lon, each with a stated strategic meaning shown via marker popup + a site legend.
- [x] **Diversified, intent-aware trajectories:** path enters at an edge, routes THROUGH a target (hostile → favors strategic sites; benign → ordinary areas), then exits. Deterministic per track (stable), varied across tracks.
- [x] **Draw-on-map trajectory** in Configure & Upload modes: click to place waypoints; risk uses the drawn path (`POST /api/risk` accepts `path:[[lat,lon],…]`). Classification stays from parameters/file.
- [x] Prominent banner: *"Proof-of-concept on simulated data — not an operational system."*

**Verification gate:** PASSED — verified live via preview: clicking Hostile →
predicts hostile (88% conf), "why" shows envelope breach, map renders risk + path.
**Done when:** full loop works locally, end to end. ✓ (run: `python -m api.server` → http://127.0.0.1:8000)

---

## Phase 5 — Deploy
**Goal:** A public URL for the report.

**Approach:** Docker image (FastAPI + static frontend) → Render/Railway/Fly.
Artifacts: `Dockerfile` (CPU-only torch), `.dockerignore`, `render.yaml`, `DEPLOY.md`.

- [x] Port-aware startup (honors `$PORT`); Docker CMD `uvicorn api.server:app --host 0.0.0.0 --port $PORT`.
- [x] `Dockerfile` + `.dockerignore` (excludes datasets/plots/intermediate checkpoints; keeps trained model).
- [x] `render.yaml` blueprint + `DEPLOY.md` (Render / Railway / Fly steps).
- [x] Verified app serves on an arbitrary port via the Docker CMD form (port 8137: metrics, simulate, index all OK).
- [ ] **TODO (needs user):** push repo to GitHub + create the cloud service → get public URL.
- [ ] Record a 60-second demo video as a backup.

**Note:** Local `docker build` was NOT run here — Docker Desktop's daemon was offline
(only the CLI is present). The image build is therefore unverified; build it locally
(`docker build -t uav-intent .`) or let Render build it on deploy.

**Verification gate:** the live URL classifies and maps correctly from a clean browser.
**Done when:** link works for anyone, no local setup.

---

## Phase 6 — Documentation & defense prep
**Goal:** The project tells one coherent story.

- [ ] README: architecture diagram, honest metrics, data-card, how to reproduce.
- [ ] "Future work: edge deployment" — reuse the paper's latency argument (why the *operational* engine wouldn't be this web app).
- [ ] Align the report/paper narrative with the actual code and real numbers.

**Done when:** code, dashboard, and written report all state the same definitions and the same metrics.

---

## Consistency checklist (revisit at every phase)
- [ ] Features defined only in `core/` and imported everywhere.
- [ ] Classes identical (labels, order, colors) across data, model, UI.
- [ ] Actuator envelope defined once; used by generator, model, and UI explanation.
- [ ] Metrics read from the trained model — never hardcoded.
- [ ] Fixed seeds; dataset and model saved to disk; `requirements.txt` pinned.
- [ ] Each layer verified standalone before the next is built.

## What is explicitly out of scope (state as future work)
React frontend · Docker/k8s · real sensor feeds · multi-node edge grid · autonomous response.
Framing these as deliberate future work shows judgment, not incompleteness.
