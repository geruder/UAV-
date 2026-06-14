# Executable Roadmap — Standalone Desktop Build

**Goal:** A Windows executable the teacher can **double-click** to run the dashboard
locally — no Python, no `pip install`, no internet required for the app itself.
On launch it starts the bundled FastAPI server on a local port and opens the
browser to the dashboard.

**Approach:** package the existing FastAPI app + trained KAN + `web/` assets with
**PyInstaller**. No code rewrite — we add a launcher and a build spec.

---

## Key decisions (locked unless you say otherwise)
- **OS:** Windows (`.exe`). (macOS/Linux possible later with the same launcher.)
- **Packager:** PyInstaller, **one-folder** build (`--onedir`) — far more reliable
  with PyTorch than one-file, and starts faster. Ship it as a zip (or an installer).
- **UX:** double-click → a small console window shows "starting…" → the default
  browser opens at `http://127.0.0.1:<port>` → closing the window quits the app.
- **Model:** keep the real PyTorch/KAN model. ⚠ This makes the build **large
  (~1–1.5 GB)** because torch ships big DLLs. See "Size note" for a lean option.

---

## Phase 0 — Prep & decisions
- [ ] Confirm target = Windows, one-folder build, keep torch.
- [ ] Confirm offline requirement: the frontend currently loads **Leaflet, Chart.js,
      Google Fonts, and map tiles from the internet (CDN)**. For a *truly offline*
      exe these must be **vendored locally** (download into `web/vendor/`), and the
      map needs either bundled/offline tiles or a graceful "no-tiles" fallback.
      Decide: "needs internet" (simple) vs "fully offline" (extra Phase 4b work).

**Done when:** scope (online vs offline, OS) is fixed.

---

## Phase 1 — Launcher + frozen-path support
**Goal:** one entrypoint that works both from source and when frozen.

- [ ] Add `run_app.py`: pick a free localhost port, start `uvicorn` (in a thread),
      wait for `/api/metrics` to respond (model preload), then `webbrowser.open` the URL;
      keep running until the console window is closed.
- [ ] Add a `resource_path()` helper (handles PyInstaller `sys._MEIPASS`) and use it
      in `api/server.py`, `api/predict.py`, `api/geo.py` for locating `web/`, the model
      checkpoint, `scaler.npz`, `metrics.json`, and `web/img/`. (These currently use
      `__file__`-relative paths — they must resolve correctly inside the bundle.)
- [ ] Friendly console output ("Loading model… / Ready — opening browser / Close this
      window to quit").

**Verification gate:** `python run_app.py` (from source) starts the server, opens the
browser, and the dashboard fully works.
**Done when:** a single command launches the whole app.

---

## Phase 2 — PyInstaller spec & build
**Goal:** produce the `.exe`.

- [ ] `pip install pyinstaller`.
- [ ] Write `uav.spec` with:
  - `datas`: `web/` (incl. `web/img/`), `model/kan_ckpt_*`, `model/scaler.npz`,
    `model/metrics.json`, and any vendored frontend assets.
  - `--collect-all torch`, `--collect-all kan` (pykan ships data/config files),
    plus `numpy`, `pandas`; exclude `matplotlib`, `scikit-learn` (not needed at runtime).
  - `hiddenimports` for uvicorn internals (`uvicorn.lifespan.*`, `uvicorn.loops.*`,
    `uvicorn.protocols.*`) and `python-multipart` — PyInstaller often misses these.
  - App icon from the project favicon (converted to `.ico`).
- [ ] `pyinstaller uav.spec` → one-folder build in `dist/`.

**Verification gate:** `dist/UAV-Intent/UAV-Intent.exe` launches on the dev machine.
**Done when:** the exe builds and runs locally.

---

## Phase 3 — Verify on a clean machine
**Goal:** prove it works without the dev environment.

- [ ] Copy `dist/UAV-Intent/` to a machine/user **without Python or the project**.
- [ ] Double-click the exe; confirm: server starts, browser opens, and **Simulate /
      Configure / Upload / About / Tour** all work, the **map + risk heatmap** render,
      **CSV/JSON upload** classifies, and the **About figures** load.
- [ ] Confirm first classification is ready (model preloaded) and there are no missing-DLL
      or missing-file errors.

**Verification gate:** a non-developer can run it end-to-end by double-clicking.
**Done when:** it works on a clean Windows machine.

---

## Phase 4 — Packaging & handoff
**Goal:** something easy to send to the teacher.

- [ ] Zip `dist/UAV-Intent/` → `UAV-Intent-Windows.zip`, OR build a friendly installer
      with **Inno Setup** (Start-menu shortcut, uninstaller).
- [ ] `HOW_TO_RUN.txt` for the teacher: unzip, double-click `UAV-Intent.exe`, allow the
      Windows firewall prompt (local server), browser opens automatically.
- [ ] Note the download size and that Windows SmartScreen may warn (unsigned app →
      "More info → Run anyway"). Code-signing optional/out of scope.

### Phase 4b — Offline mode (only if "fully offline" was chosen in Phase 0)
- [ ] Vendor Leaflet + Chart.js + fonts into `web/vendor/`; switch `index.html` to them.
- [ ] Provide offline map tiles (bundled tile pack) or fall back to "map unavailable
      offline" while keeping classification + charts working.

**Done when:** the teacher has one file/zip that just runs.

---

## Size note — a much leaner exe (optional)
PyTorch is ~90% of the build size and the main source of PyInstaller friction.
If we **reimplement the KAN forward pass in pure NumPy** (export the trained weights
once, then drop torch/pykan), the exe shrinks from **~1.5 GB to ~100–150 MB**, builds
faster, and avoids torch DLL issues. Same idea as the "full-Vercel" option. It's extra
work + numerical validation, but it makes the executable dramatically smaller and more
robust. Recommended if the file size matters for sharing.

---

## Out of scope (future)
macOS/Linux builds · auto-update · system-tray app · code-signing · native window
(Electron/Tauri) instead of the browser.
