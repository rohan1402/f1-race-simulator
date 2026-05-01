"""
Extract F1 circuit outlines from FastF1 telemetry and save as normalised JSON.

Each JSON file contains a list of {x, y} points (0-1 range) tracing one
full lap around the circuit, plus the circuit's aspect ratio so the canvas
can size itself correctly.

Usage:
    python scripts/extract_tracks.py
"""

import json
import math
from pathlib import Path

import fastf1
import numpy as np

# ── FastF1 cache ──────────────────────────────────────────────────────────────
CACHE_DIR = Path(__file__).parent.parent / "data" / "ff1_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUT_DIR = Path(__file__).parent.parent / "data" / "tracks"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# race key → (year, round_name)
RACES = {
    "2021_Abu_Dhabi":  (2021, "Abu Dhabi Grand Prix"),
    "2022_Monaco":     (2022, "Monaco Grand Prix"),
    "2023_Monza":      (2023, "Italian Grand Prix"),
    "2019_Hockenheim": (2019, "German Grand Prix"),
    "2023_Las_Vegas":  (2023, "Las Vegas Grand Prix"),
    "2022_Brazil":     (2022, "Brazilian Grand Prix"),
}

N_POINTS = 300   # points to keep after resampling


def resample_path(xs, ys, n):
    """Resample an (xs, ys) path to exactly n evenly-spaced points."""
    pts = np.column_stack([xs, ys]).astype(float)
    diffs = np.diff(pts, axis=0)
    seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
    cum = np.concatenate([[0], np.cumsum(seg_lens)])
    total = cum[-1]
    if total == 0:
        return pts[:n].tolist()
    t_new = np.linspace(0, total, n)
    xi = np.interp(t_new, cum, pts[:, 0])
    yi = np.interp(t_new, cum, pts[:, 1])
    return list(zip(xi.tolist(), yi.tolist()))


def normalise(points):
    """Scale points to fit in [0.05, 0.95]² preserving aspect ratio."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1
    span_y = max_y - min_y or 1
    # aspect ratio of the raw bounding box (width / height)
    aspect = span_x / span_y

    norm = [
        ((p[0] - min_x) / span_x * 0.90 + 0.05,
         (p[1] - min_y) / span_y * 0.90 + 0.05)
        for p in points
    ]
    return norm, aspect


for key, (year, name) in RACES.items():
    out_path = OUT_DIR / f"{key}.json"
    if out_path.exists():
        print(f"  skip  {key}  (already extracted)")
        continue

    print(f"  loading {year} {name} …", end=" ", flush=True)
    try:
        session = fastf1.get_session(year, name, "R")
        session.load(telemetry=True, laps=True, weather=False, messages=False)

        fastest = session.laps.pick_fastest()
        tel = fastest.get_telemetry()

        if "X" not in tel.columns or "Y" not in tel.columns:
            print("no X/Y in telemetry — skipping")
            continue

        xs = tel["X"].values.astype(float)
        ys = tel["Y"].values.astype(float)

        # Remove NaN
        mask = ~(np.isnan(xs) | np.isnan(ys))
        xs, ys = xs[mask], ys[mask]

        pts = resample_path(xs, ys, N_POINTS)
        norm, aspect = normalise(pts)

        payload = {
            "key":    key,
            "aspect": round(aspect, 4),
            "points": [{"x": round(p[0], 5), "y": round(p[1], 5)} for p in norm],
        }
        out_path.write_text(json.dumps(payload))
        print(f"✓  ({len(norm)} pts, aspect={aspect:.2f})")

    except Exception as exc:
        print(f"ERROR: {exc}")

print("\nDone. Track files in:", OUT_DIR)
