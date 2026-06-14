"""Geographic risk scoring for the dashboard.

Risk = (drone hostility) x (strategic value of the ground it overflies), rendered
as a smooth density field. Strategic sites are REAL locations around the Bosphorus
in Istanbul, each with a stated strategic meaning. Trajectories are diversified:
they enter from a map edge, route THROUGH a chosen target (more likely a strategic
site when the drone reads hostile), then exit -- or, in the Configure/Upload modes,
the user draws the path directly on the map.
"""

from __future__ import annotations

import numpy as np

GRID_N = 80  # higher resolution -> smoother density field

# Map framing (Bosphorus / central Istanbul). The grid covers this lat/lon box.
MAP_CENTER = [41.030, 29.000]
MAP_SPAN = 0.075  # degrees across the grid

# Real places around the region, each with a strategic rationale. weight = value,
# sigma = footprint as a fraction of the grid. (Locations are illustrative; the
# point is plausible strategic geography, not that each is e.g. an airport.)
STRATEGIC_LOCATIONS = [
    {"name": "15 July Martyrs Bridge", "lat": 41.0451, "lon": 29.0345, "weight": 1.00, "sigma": 0.050,
     "type": "bridge",   "meaning": "Critical intercontinental transport link; severing it splits the city."},
    {"name": "Selimiye Barracks",      "lat": 41.0205, "lon": 29.0155, "weight": 0.95, "sigma": 0.050,
     "type": "military", "meaning": "Historic military installation on the Asian shore."},
    {"name": "Dolmabahce Palace",      "lat": 41.0392, "lon": 29.0000, "weight": 0.90, "sigma": 0.048,
     "type": "gov",      "meaning": "State & ceremonial palace; protected government site."},
    {"name": "Taksim Square",          "lat": 41.0370, "lon": 28.9850, "weight": 0.85, "sigma": 0.050,
     "type": "civic",    "meaning": "Primary civic & political gathering hub; high crowd density."},
    {"name": "Karakoy Port",           "lat": 41.0240, "lon": 28.9770, "weight": 0.75, "sigma": 0.048,
     "type": "port",     "meaning": "Maritime passenger & freight terminal on the Golden Horn."},
    {"name": "Vodafone Park",          "lat": 41.0392, "lon": 29.0070, "weight": 0.70, "sigma": 0.045,
     "type": "stadium",  "meaning": "Large stadium; mass-gathering venue when in use."},
    {"name": "Topkapi / Sarayburnu",   "lat": 41.0130, "lon": 28.9839, "weight": 0.70, "sigma": 0.048,
     "type": "heritage", "meaning": "Historic state landmark; dense tourism and shipping lanes."},
]
BASELINE_VALUE = 0.06  # low ambient value over ordinary urban area


def grid_bounds() -> tuple[float, float, float, float]:
    """Return (north, south, west, east) lat/lon bounds of the grid."""
    s, c = MAP_SPAN, MAP_CENTER
    return c[0] + s / 2, c[0] - s / 2, c[1] - s / 2, c[1] + s / 2


def latlon_to_frac(lat: float, lon: float) -> tuple[float, float]:
    """Map a lat/lon to (x, y) fractions in [0,1] (x=east, y=south-from-north)."""
    north, south, west, east = grid_bounds()
    return (lon - west) / (east - west), (north - lat) / (north - south)


def locations_view() -> list[dict]:
    """Locations with grid fractions added, for the frontend (markers + legend)."""
    out = []
    for l in STRATEGIC_LOCATIONS:
        fx, fy = latlon_to_frac(l["lat"], l["lon"])
        out.append({"name": l["name"], "type": l["type"], "meaning": l["meaning"],
                    "weight": l["weight"], "lat": l["lat"], "lon": l["lon"], "x": fx, "y": fy})
    return out


def strategic_value_grid(n: int = GRID_N) -> np.ndarray:
    """Synthetic strategic-value field in [0,1] built from the real sites."""
    yy, xx = np.mgrid[0:n, 0:n] / n
    grid = np.full((n, n), BASELINE_VALUE)
    for l in STRATEGIC_LOCATIONS:
        fx, fy = latlon_to_frac(l["lat"], l["lon"])
        grid += l["weight"] * np.exp(-(((xx - fx) ** 2 + (yy - fy) ** 2) / (2 * l["sigma"] ** 2)))
    return np.clip(grid, 0, 1)


def _edge_point(rng, n: int) -> np.ndarray:
    side = rng.integers(0, 4)
    t = rng.uniform(0.1, 0.9) * n
    return {0: np.array([t, 0.0]), 1: np.array([t, n - 1.0]),
            2: np.array([0.0, t]), 3: np.array([n - 1.0, t])}[int(side)]


def choose_target(rng, hostile_prob: float, n: int = GRID_N) -> dict:
    """Hostile drones favor strategic sites; benign drones usually transit
    ordinary areas -> diversified, intent-aware routing."""
    p_strategic = 0.20 + 0.70 * float(hostile_prob)
    if rng.random() < p_strategic:
        weights = np.array([l["weight"] for l in STRATEGIC_LOCATIONS])
        l = STRATEGIC_LOCATIONS[rng.choice(len(STRATEGIC_LOCATIONS), p=weights / weights.sum())]
        fx, fy = latlon_to_frac(l["lat"], l["lon"])
        return {"name": l["name"], "type": l["type"], "xy": np.array([fx * n, fy * n])}
    return {"name": "ordinary area", "type": "normal",
            "xy": np.array([rng.uniform(0.15, 0.85) * n, rng.uniform(0.15, 0.85) * n])}


def synthesize_ground_path(track: dict[str, np.ndarray], n: int = GRID_N,
                           seed: int | None = None, hostile_prob: float = 0.5) -> tuple[np.ndarray, dict]:
    """Diversified 2D ground path: edge -> target -> edge, with wander driven by
    the flight dynamics. Returns (path[T,2] in grid coords, target_info)."""
    rng = np.random.default_rng(seed)
    pitch = np.asarray(track["pitch_angle"], dtype=float)
    velocity = np.asarray(track["velocity"], dtype=float)
    T = len(pitch)
    target = choose_target(rng, hostile_prob, n)
    start, end = _edge_point(rng, n), _edge_point(rng, n)
    tgt = target["xy"]

    loiter = 1.0 - np.clip(velocity.mean() / (velocity.max() + 1e-9), 0, 1)
    split = int(T * (0.5 + 0.15 * loiter))
    leg1 = np.linspace(0, 1, split)[:, None] * (tgt - start) + start
    leg2 = np.linspace(0, 1, T - split)[:, None] * (end - tgt) + tgt
    path = np.vstack([leg1, leg2])

    heading = np.arctan2(*(np.gradient(path, axis=0)[:, ::-1].T))
    perp = np.stack([-np.sin(heading), np.cos(heading)], axis=1)
    amp = (np.abs(pitch) / 35.0).clip(0, 1) * (n * 0.05)
    wander = np.cumsum(rng.normal(0, 1, T)) * 0.15
    path = np.clip(path + perp * (amp * wander)[:, None], 0, n - 1)
    return path, target


def latlon_path_to_grid(latlon: list[list[float]], n: int = GRID_N, total: int = 160) -> np.ndarray:
    """Turn user-drawn [lat,lon] waypoints into a dense grid-coord path."""
    pts = np.array([[*latlon_to_frac(lat, lon)] for lat, lon in latlon]) * n
    pts = np.clip(pts, 0, n - 1)
    if len(pts) < 2:
        return pts
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    if d[-1] == 0:
        return pts
    s = np.linspace(0, d[-1], total)
    return np.stack([np.interp(s, d, pts[:, 0]), np.interp(s, d, pts[:, 1])], axis=1)


def _gaussian_blur(a: np.ndarray, sigma: float = 1.4) -> np.ndarray:
    r = max(1, int(sigma * 3))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 0, a)
    return np.apply_along_axis(lambda m: np.convolve(m, k, mode="same"), 1, out)


def risk_heatmap(value_grid: np.ndarray, path: np.ndarray, hostile_prob: float,
                 spread: float | None = None) -> np.ndarray:
    """Smooth risk density in [0,1] on an ABSOLUTE scale (benign green, hostile red)."""
    n = value_grid.shape[0]
    spread = spread if spread is not None else n * 0.075
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    proximity = np.zeros((n, n), dtype=float)
    for px, py in path[::2]:
        proximity = np.maximum(proximity, np.exp(-(((xx - px) ** 2 + (yy - py) ** 2) / (2 * spread ** 2))))
    risk = _gaussian_blur(value_grid * proximity * float(hostile_prob), sigma=2.2)
    return np.clip(risk / (0.5 * value_grid.max() + 1e-9), 0, 1)


if __name__ == "__main__":
    g = strategic_value_grid()
    print(f"value grid {g.shape}, peak {g.max():.2f}")
    for l in locations_view():
        print(f"  {l['name']:24s} ({l['lat']:.4f},{l['lon']:.4f}) w={l['weight']} :: {l['meaning']}")
