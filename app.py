"""
F1 Race Rewind — Streamlit App
================================
What-if race simulator: pick a demo race, choose a fork lap,
apply a strategy intervention, and watch the race diverge.

Run:
    streamlit run app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from simulator.engine import load_race, simulate, list_races

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="F1 Race Rewind",
    page_icon="🏎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark F1 theme ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background:#0e0e0e; }
  [data-testid="stSidebar"]          { background:#1a1a1a; }
  h1,h2,h3,h4,p,label,div           { color:#f0f0f0 !important; }
  .result-banner {
      background: linear-gradient(90deg,#e10600 0%,#1a1a1a 60%);
      border-radius:8px; padding:18px 24px; margin:12px 0;
  }
  .stButton button { background:#e10600 !important; color:white !important;
                     font-weight:700; border:none; border-radius:4px; }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
COMPOUND_COLOR = {
    "SOFT": "#e8002d", "MEDIUM": "#ffd700",
    "HARD": "#ffffff",  "INTER":  "#43b649", "WET": "#0067ff",
}
PLOTLY_TEMPLATE = "plotly_dark"


# ── Helpers ───────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def _load(race_key: str) -> pd.DataFrame:
    return load_race(race_key)


# ── Track map loader ──────────────────────────────────────────────────────────
_TRACK_KEY = {
    "2021 Abu Dhabi Grand Prix":  "2021_Abu_Dhabi",
    "2022 Monaco Grand Prix":     "2022_Monaco",
    "2023 Italian Grand Prix":    "2023_Monza",
    "2019 German Grand Prix":     "2019_Hockenheim",
    "2023 Las Vegas Grand Prix":  "2023_Las_Vegas",
    "2022 Brazilian Grand Prix":  "2022_Brazil",
}

@st.cache_data(show_spinner=False)
def _load_track(race_key: str) -> dict | None:
    key  = _TRACK_KEY.get(race_key, "")
    path = Path(__file__).parent / "data" / "tracks" / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _driver_strategy(race_df: pd.DataFrame, driver: str) -> str:
    drv  = race_df[race_df["Driver"] == driver].sort_values("LapNumber")
    pits = drv[drv["PitInLap"]]
    if pits.empty:
        return "No pit stops recorded"
    parts = []
    for _, row in pits.iterrows():
        lap = int(row["LapNumber"])
        nxt = drv[drv["LapNumber"] == lap + 1]
        comp = nxt["Compound"].iloc[0] if not nxt.empty else "?"
        parts.append(f"Lap {lap} → {comp}")
    return "  |  ".join(parts)


def _final_pos(df: pd.DataFrame, driver: str) -> tuple[int, float]:
    last = df[df["Driver"] == driver].sort_values("LapNumber").iloc[-1]
    return int(last["Position"]), round(float(last["GapToLeader"]), 3)


# ── Race Animation (HTML5 Canvas) ─────────────────────────────────────────────

def _build_race_animation(
    baseline:   pd.DataFrame,
    counter:    pd.DataFrame,
    sim_driver: str,
    fork_lap:   int,
    race_key:   str = "",
) -> str:
    """Return a self-contained HTML page with an animated circuit race replay."""

    total_laps = int(baseline["LapNumber"].max())

    # Per-lap car state dicts ─────────────────────────────────────────────────
    def _lap_dict(df: pd.DataFrame) -> dict:
        out: dict[str, list] = {}
        prev: list = []
        for lap in range(1, total_laps + 1):
            rows = df[df["LapNumber"] == lap].sort_values("Position")
            if rows.empty:
                out[str(lap)] = prev
            else:
                prev = [
                    {
                        "d":    str(r["Driver"]),
                        "pos":  int(r["Position"]),
                        "gap":  round(min(float(r["GapToLeader"]), 200.0), 2),
                        "comp": str(r["Compound"]),
                        "tl":   int(r["TyreLife"]),
                        "pit":  bool(r["PitInLap"]),
                    }
                    for _, r in rows.iterrows()
                ]
                out[str(lap)] = prev
        return out

    actual_data = _lap_dict(baseline)
    whatif_data = _lap_dict(counter)

    # Average lap time (for converting gap → track fraction) ──────────────────
    lt = baseline[(baseline["LapNumber"] > 3) & (baseline["LapTime"].notna())
                  & (baseline["LapTime"] > 60)]["LapTime"]
    avg_lap_time = float(lt.median()) if not lt.empty else 90.0

    # Track path ──────────────────────────────────────────────────────────────
    track_data = _load_track(race_key)   # {key, aspect, points:[{x,y}]} or None

    # Key events ──────────────────────────────────────────────────────────────
    events: list[dict] = []

    drv_cf = counter[counter["Driver"] == sim_driver].sort_values("LapNumber")

    for _, row in drv_cf[drv_cf["PitInLap"]].iterrows():
        lap  = int(row["LapNumber"])
        nxt  = drv_cf[drv_cf["LapNumber"] == lap + 1]
        comp = str(nxt["Compound"].iloc[0]) if not nxt.empty else "?"
        events.append({"lap": lap, "type": "pit",
                        "text": f"Lap {lap} — {sim_driver} pits for {comp}"})

    prev_pos = None
    for _, row in drv_cf.iterrows():
        lap = int(row["LapNumber"])
        pos = int(row["Position"])
        if lap >= fork_lap and prev_pos is not None and pos < prev_pos:
            events.append({"lap": lap, "type": "overtake",
                            "text": f"Lap {lap} — {sim_driver} moves to P{pos}"})
        prev_pos = pos

    events.append({"lap": fork_lap, "type": "fork",
                   "text": f"Lap {fork_lap} — Simulation diverges from history"})
    events.sort(key=lambda e: e["lap"])

    # Payload ─────────────────────────────────────────────────────────────────
    payload = json.dumps({
        "totalLaps":   total_laps,
        "forkLap":     fork_lap,
        "hero":        sim_driver,
        "avgLapTime":  round(avg_lap_time, 3),
        "actual":      actual_data,
        "whatif":      whatif_data,
        "events":      events,
        "track":       track_data,   # may be null if JSON not found
    })

    # HTML / CSS / JS ─────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{ box-sizing:border-box; margin:0; padding:0; }}
body {{
  background:#0a0a0a;
  font-family:'Courier New',monospace;
  color:#f0f0f0;
  padding:10px;
}}

/* ── Top bar ── */
#topbar {{
  display:flex; justify-content:space-between; align-items:center;
  margin-bottom:8px;
}}
#title {{ font-size:12px; font-weight:bold; letter-spacing:3px; color:#e10600; }}
#lap-info {{ display:flex; align-items:center; gap:8px; }}
#lap-display {{
  font-size:17px; font-weight:bold; color:#fff; letter-spacing:2px;
  background:#1a1a1a; padding:4px 12px; border-radius:4px; border:1px solid #333;
}}
#fork-badge {{
  font-size:10px; color:#00d2ff; background:#001828;
  padding:3px 9px; border-radius:4px; border:1px solid #00d2ff44;
  display:none; letter-spacing:1px;
}}

/* ── Controls ── */
#controls {{
  display:flex; align-items:center; gap:6px;
  background:#111; padding:8px 10px; border-radius:6px; margin-bottom:10px;
}}
.cbtn {{
  background:#1e1e1e; border:1px solid #3a3a3a; color:#ccc;
  padding:5px 11px; border-radius:4px; cursor:pointer; font-size:13px;
  transition:background .15s;
}}
.cbtn:hover {{ background:#2a2a2a; color:#fff; }}
#btn-play {{
  background:#e10600; border-color:#e10600; color:#fff;
  font-size:15px; padding:5px 16px; min-width:50px;
}}
#btn-play:hover {{ background:#ff2222; }}
#btn-play.paused {{ background:#333; border-color:#555; }}
#lap-slider {{ flex:1; accent-color:#e10600; cursor:pointer; }}
#speed-label {{ font-size:10px; color:#666; white-space:nowrap; }}
#speed-select {{
  background:#1e1e1e; border:1px solid #3a3a3a; color:#ccc;
  padding:4px 6px; border-radius:4px; font-size:11px; cursor:pointer;
}}

/* ── Circuit grid ── */
#race-grid {{
  display:grid; grid-template-columns:1fr 1fr; gap:8px; margin-bottom:8px;
}}
.circuit-panel {{
  background:#0d0d0d; border:1px solid #1c1c1c;
  border-radius:8px; padding:6px 8px;
}}
.circuit-header {{
  display:flex; justify-content:space-between; align-items:center;
  margin-bottom:5px;
}}
.track-label {{
  font-size:10px; font-weight:bold; letter-spacing:4px; color:#555;
  border-left:3px solid #444; padding-left:7px;
}}
.track-label.wi {{ color:#00d2ff; border-left-color:#00d2ff; }}
.track-p1 {{
  font-size:11px; font-weight:bold; color:#777;
  background:#181818; padding:2px 8px; border-radius:3px;
}}
.track-p1.wi {{ color:#00d2ff; }}
canvas {{
  display:block; width:100%; height:260px;
  border-radius:5px;
}}

/* ── Event feed ── */
#event-section {{
  background:#0f0f0f; border:1px solid #1a1a1a;
  border-radius:6px; padding:8px 12px;
}}
#event-title {{
  font-size:9px; letter-spacing:4px; color:#3a3a3a;
  margin-bottom:5px; font-weight:bold;
}}
#event-list {{ display:flex; flex-direction:column; gap:3px; max-height:80px; overflow-y:auto; }}
.ev {{ font-size:11px; padding:3px 8px; border-radius:3px; border-left:3px solid transparent; }}
.ev.pit      {{ color:#ffd700; border-left-color:#ffd700; }}
.ev.overtake {{ color:#00ff88; border-left-color:#00ff88; }}
.ev.fork     {{ color:#00d2ff; border-left-color:#00d2ff; }}
</style>
</head>
<body>

<div id="topbar">
  <span id="title">🏁 RACE REPLAY</span>
  <div id="lap-info">
    <span id="fork-badge">AFTER FORK</span>
    <span id="lap-display">LAP 1 / {total_laps}</span>
  </div>
</div>

<div id="controls">
  <button class="cbtn" id="btn-first">⏮</button>
  <button class="cbtn" id="btn-prev">◄</button>
  <button class="cbtn" id="btn-play">▶</button>
  <button class="cbtn" id="btn-next">►</button>
  <button class="cbtn" id="btn-last">⏭</button>
  <input type="range" id="lap-slider" min="1" max="{total_laps}" value="1">
  <span id="speed-label">SPEED</span>
  <select id="speed-select">
    <option value="1400">½×</option>
    <option value="750" selected>1×</option>
    <option value="350">2×</option>
    <option value="120">4×</option>
  </select>
</div>

<div id="race-grid">
  <div class="circuit-panel">
    <div class="circuit-header">
      <span class="track-label">ACTUAL RACE</span>
      <span class="track-p1" id="p1-actual">P1: —</span>
    </div>
    <canvas id="cv-actual"></canvas>
  </div>
  <div class="circuit-panel">
    <div class="circuit-header">
      <span class="track-label wi">WHAT-IF</span>
      <span class="track-p1 wi" id="p1-whatif">P1: —</span>
    </div>
    <canvas id="cv-whatif"></canvas>
  </div>
</div>

<div id="event-section">
  <div id="event-title">KEY EVENTS</div>
  <div id="event-list"></div>
</div>

<script>
const DATA = {payload};

// ── Driver / compound colours ─────────────────────────────────────────────────
const DC = {{
  VER:'#3671C6',PER:'#3671C6',
  HAM:'#27F4D2',BOT:'#27F4D2',RUS:'#27F4D2',
  LEC:'#E8002D',SAI:'#E8002D',
  NOR:'#FF8000',RIC:'#FF8000',PIA:'#FF8000',
  ALO:'#358C75',OCO:'#358C75',
  VET:'#2D826D',STR:'#2D826D',
  GAS:'#4E7A9B',TSU:'#4E7A9B',LAW:'#4E7A9B',
  ZHO:'#B02740',GIO:'#B02740',RAI:'#B02740',
  MSC:'#B6BABD',MAG:'#B6BABD',HUL:'#B6BABD',
  LAT:'#37BEDD',ALB:'#37BEDD',SAR:'#37BEDD',
}};
const CC = {{SOFT:'#e8002d',MEDIUM:'#ffd700',HARD:'#e8e8e8',INTER:'#43b649',WET:'#0067ff'}};

// ── Canvas refs ───────────────────────────────────────────────────────────────
const CVS = {{
  actual: document.getElementById('cv-actual'),
  whatif: document.getElementById('cv-whatif'),
}};
const CTX = {{
  actual: CVS.actual.getContext('2d'),
  whatif: CVS.whatif.getContext('2d'),
}};

// ── DPR / init ────────────────────────────────────────────────────────────────
const DPR = Math.min(window.devicePixelRatio || 1, 2);
const CV_H = 260;   // CSS height (matches canvas {{ height:260px }})
const cvW  = {{}};   // logical width per scenario

function initCanvas(key) {{
  const cv  = CVS[key];
  const ctx = CTX[key];
  const w   = cv.clientWidth || cv.offsetWidth || 420;
  cv.width  = Math.round(w  * DPR);
  cv.height = Math.round(CV_H * DPR);
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  cvW[key] = w;
}}

// ── State ─────────────────────────────────────────────────────────────────────
let curLap  = DATA.forkLap > 1 ? DATA.forkLap - 1 : 1;
let playing = false;
let playTmr = null;
let rafId   = null;

// smoothFrac[scenario][driverCode] = current fractional track position [0,1)
const smoothFrac = {{ actual:{{}}, whatif:{{}} }};

// ── Colour helpers ────────────────────────────────────────────────────────────
function isDark(hex) {{
  if (!hex || hex.length < 7) return true;
  const r = parseInt(hex.slice(1,3),16),
        g = parseInt(hex.slice(3,5),16),
        b = parseInt(hex.slice(5,7),16);
  return (r*.299 + g*.587 + b*.114) < 140;
}}

// ── Track-path helpers ────────────────────────────────────────────────────────
// Convert time-gap to fractional lap position.
// Leader (gap=0) → frac=0 (start/finish).  A car X sec behind → frac=(1 − X/T) mod 1
function carFrac(gap) {{
  if (!gap || gap <= 0) return 0;
  const lf = (gap / DATA.avgLapTime) % 1.0;
  return lf < 0.0001 ? 0 : 1.0 - lf;
}}

// Interpolate fractional position with wrap-around awareness
function lerpFrac(cur, tgt, k) {{
  let d = tgt - cur;
  if (d >  0.5) d -= 1.0;
  if (d < -0.5) d += 1.0;
  return ((cur + d * k) + 1.0) % 1.0;
}}

// Get canvas (x,y) for a fractional position along the circuit path
function pathPoint(frac, pts, W, H) {{
  const N = pts.length;
  const f = ((frac % 1) + 1) % 1;
  const ri = f * N;
  const i0 = Math.floor(ri) % N;
  const i1 = (i0 + 1) % N;
  const t  = ri - Math.floor(ri);
  return {{
    x: (pts[i0].x + (pts[i1].x - pts[i0].x) * t) * W,
    y: (pts[i0].y + (pts[i1].y - pts[i0].y) * t) * H,
  }};
}}

// Fallback oval points if no JSON track available
function makeOval(n) {{
  return Array.from({{length: n}}, (_, i) => {{
    const a = (i / n) * Math.PI * 2;
    return {{ x: 0.5 + 0.43 * Math.cos(a), y: 0.5 + 0.40 * Math.sin(a) }};
  }});
}}
const TRACK_PTS = (DATA.track && DATA.track.points) ? DATA.track.points : makeOval(300);

// ── Draw circuit outline ──────────────────────────────────────────────────────
function drawCircuitPath(ctx, W, H) {{
  const pts = TRACK_PTS;

  function px(p) {{ return p.x * W; }}
  function py(p) {{ return p.y * H; }}

  // Outer wall (dark, thick)
  ctx.strokeStyle = '#1a1a1a';
  ctx.lineWidth   = 18;
  ctx.lineCap     = 'round';
  ctx.lineJoin    = 'round';
  ctx.beginPath();
  ctx.moveTo(px(pts[0]), py(pts[0]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(px(pts[i]), py(pts[i]));
  ctx.closePath();
  ctx.stroke();

  // Asphalt (mid grey)
  ctx.strokeStyle = '#2c2c2c';
  ctx.lineWidth   = 11;
  ctx.beginPath();
  ctx.moveTo(px(pts[0]), py(pts[0]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(px(pts[i]), py(pts[i]));
  ctx.closePath();
  ctx.stroke();

  // Kerb edge (thin, slightly lighter)
  ctx.strokeStyle = '#404040';
  ctx.lineWidth   = 1.5;
  ctx.beginPath();
  ctx.moveTo(px(pts[0]), py(pts[0]));
  for (let i = 1; i < pts.length; i++) ctx.lineTo(px(pts[i]), py(pts[i]));
  ctx.closePath();
  ctx.stroke();

  // Start / finish line — perpendicular white tick at pts[0]
  const p0 = pts[0], p1 = pts[1];
  const dx = p1.x - p0.x, dy = p1.y - p0.y;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len, ny = dx / len;   // perpendicular unit vector
  const off = 0.022;
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth   = 2.5;
  ctx.lineCap     = 'butt';
  ctx.beginPath();
  ctx.moveTo((p0.x + nx * off) * W, (p0.y + ny * off) * H);
  ctx.lineTo((p0.x - nx * off) * W, (p0.y - ny * off) * H);
  ctx.stroke();

  // "S/F" tiny label near start/finish
  ctx.fillStyle   = 'rgba(255,255,255,0.35)';
  ctx.font        = '7px Courier New';
  ctx.textAlign   = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('S/F', p0.x * W + nx * 16, p0.y * H + ny * 16);
}}

// ── Draw one circuit panel ────────────────────────────────────────────────────
function drawCircuit(scenario, lapCars) {{
  const ctx = CTX[scenario];
  const W   = cvW[scenario] || 420;
  const H   = CV_H;

  // Background
  ctx.fillStyle = '#0c0c0c';
  ctx.fillRect(0, 0, W, H);

  drawCircuitPath(ctx, W, H);

  if (!lapCars || lapCars.length === 0) return;

  // ── Advance smooth positions ─────────────────────────────────────────
  const sm = smoothFrac[scenario];
  for (const c of lapCars) {{
    const tgt = carFrac(c.gap);
    if (sm[c.d] === undefined) sm[c.d] = tgt;
    else                       sm[c.d] = lerpFrac(sm[c.d], tgt, 0.18);
  }}

  // Draw cars back-to-front (leader on top)
  const sorted = [...lapCars].sort((a, b) => b.pos - a.pos);

  for (const c of sorted) {{
    const frac   = sm[c.d] ?? carFrac(c.gap);
    const pt     = pathPoint(frac, TRACK_PTS, W, H);
    const col    = DC[c.d] || '#666';
    const isHero = c.d === DATA.hero;
    const heroCol = scenario === 'whatif' ? '#00d2ff' : '#ffaa00';
    const r      = isHero ? 8 : 5.5;

    ctx.save();

    // Hero glow
    if (isHero) {{
      ctx.shadowColor = heroCol;
      ctx.shadowBlur  = 18;
    }}

    // Car circle
    ctx.fillStyle = col;
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, r, 0, Math.PI * 2);
    ctx.fill();

    // Tyre compound dot in centre
    ctx.shadowBlur  = 0;
    ctx.fillStyle   = CC[c.comp] || '#888';
    ctx.beginPath();
    ctx.arc(pt.x, pt.y, r * 0.38, 0, Math.PI * 2);
    ctx.fill();

    // Pit-stop flash ring
    if (c.pit) {{
      ctx.strokeStyle = 'rgba(255,255,180,0.85)';
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.arc(pt.x, pt.y, r + 3.5, 0, Math.PI * 2);
      ctx.stroke();
    }}

    // Driver code label (hero + top 8 + any pitting car)
    if (isHero || c.pos <= 8 || c.pit) {{
      const lbl  = c.d;
      const fs   = isHero ? 7.5 : 6.5;
      ctx.font   = `bold ${{fs}}px "Courier New"`;
      const tw   = ctx.measureText(lbl).width;
      const lx   = pt.x;
      const ly   = pt.y - r - 3;
      // Dark pill background
      ctx.fillStyle = 'rgba(0,0,0,0.72)';
      ctx.beginPath();
      ctx.roundRect(lx - tw/2 - 2, ly - fs - 1, tw + 4, fs + 2, 2);
      ctx.fill();
      ctx.fillStyle = isHero ? heroCol : '#e0e0e0';
      ctx.textAlign     = 'center';
      ctx.textBaseline  = 'bottom';
      ctx.fillText(lbl, lx, ly);
    }}

    ctx.restore();
  }}

  // Hero chevron indicator (above label)
  const hero = lapCars.find(c => c.d === DATA.hero);
  if (hero) {{
    const frac = sm[DATA.hero] ?? carFrac(hero.gap);
    const pt   = pathPoint(frac, TRACK_PTS, W, H);
    const heroCol = scenario === 'whatif' ? '#00d2ff' : '#ffaa00';
    ctx.fillStyle    = heroCol;
    ctx.font         = 'bold 11px sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText('▼', pt.x, pt.y - 20);
  }}
}}

// ── Render loop helpers ───────────────────────────────────────────────────────
function render(snap) {{
  const key = String(curLap);
  const aC  = DATA.actual[key] || [];
  const wC  = DATA.whatif[key] || [];

  if (snap) {{
    smoothFrac.actual = {{}};
    smoothFrac.whatif = {{}};
  }}

  drawCircuit('actual', aC);
  drawCircuit('whatif', wC);

  // HUD
  document.getElementById('lap-display').textContent =
    `LAP ${{curLap}} / ${{DATA.totalLaps}}`;
  document.getElementById('lap-slider').value = curLap;

  const fb = document.getElementById('fork-badge');
  if (curLap >= DATA.forkLap) {{
    fb.style.display = 'inline';
    const d = curLap - DATA.forkLap;
    fb.textContent = d === 0 ? 'FORK LAP' : `+${{d}} LAP${{d>1?'S':''}} AFTER FORK`;
  }} else {{
    fb.style.display = 'none';
  }}

  const aL = aC.find(c => c.pos === 1);
  const wL = wC.find(c => c.pos === 1);
  document.getElementById('p1-actual').textContent = aL ? `P1: ${{aL.d}}` : 'P1: —';
  document.getElementById('p1-whatif').textContent = wL ? `P1: ${{wL.d}}` : 'P1: —';

  const visible = DATA.events.filter(e => e.lap <= curLap).slice().reverse().slice(0, 8);
  document.getElementById('event-list').innerHTML =
    visible.map(e => `<div class="ev ${{e.type}}">${{e.text}}</div>`).join('');
}}

function animLoop() {{
  drawCircuit('actual', DATA.actual[String(curLap)] || []);
  drawCircuit('whatif', DATA.whatif[String(curLap)] || []);
  // Keep running until all fracs have converged
  let done = true;
  for (const scen of ['actual','whatif']) {{
    for (const c of (DATA[scen][String(curLap)] || [])) {{
      const tgt = carFrac(c.gap);
      const cur = smoothFrac[scen][c.d];
      if (cur !== undefined) {{
        let d = Math.abs(tgt - cur);
        if (d > 0.5) d = 1 - d;
        if (d > 0.002) {{ done = false; break; }}
      }}
    }}
    if (!done) break;
  }}
  if (!done) rafId = requestAnimationFrame(animLoop);
  else rafId = null;
}}

// ── Playback ──────────────────────────────────────────────────────────────────
function startPlay() {{
  if (playing) return;
  playing = true;
  const btn = document.getElementById('btn-play');
  btn.textContent = '⏸'; btn.classList.add('paused');
  if (curLap >= DATA.totalLaps) curLap = 1;
  tick();
}}

function tick() {{
  const delay = parseInt(document.getElementById('speed-select').value);
  playTmr = setTimeout(() => {{
    if (!playing) return;
    curLap = Math.min(curLap + 1, DATA.totalLaps);
    render(false);
    if (rafId) cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(animLoop);
    if (curLap < DATA.totalLaps) tick();
    else stopPlay();
  }}, delay);
}}

function stopPlay() {{
  playing = false;
  clearTimeout(playTmr);
  const btn = document.getElementById('btn-play');
  btn.textContent = '▶'; btn.classList.remove('paused');
}}

// ── Controls ──────────────────────────────────────────────────────────────────
document.getElementById('btn-play') .addEventListener('click', () => playing ? stopPlay() : startPlay());
document.getElementById('btn-first').addEventListener('click', () => {{ stopPlay(); curLap = 1; render(true); }});
document.getElementById('btn-last') .addEventListener('click', () => {{ stopPlay(); curLap = DATA.totalLaps; render(true); }});
document.getElementById('btn-prev') .addEventListener('click', () => {{ stopPlay(); if (curLap > 1) {{ curLap--; render(true); }} }});
document.getElementById('btn-next') .addEventListener('click', () => {{ stopPlay(); if (curLap < DATA.totalLaps) {{ curLap++; render(true); }} }});
document.getElementById('lap-slider').addEventListener('input', e => {{ stopPlay(); curLap = parseInt(e.target.value); render(true); }});

// ── Init ──────────────────────────────────────────────────────────────────────
window.addEventListener('load', () => {{
  initCanvas('actual');
  initCanvas('whatif');
  render(true);
}});
window.addEventListener('resize', () => {{
  initCanvas('actual');
  initCanvas('whatif');
  render(true);
}});
</script>
</body>
</html>"""


# ── Standings table helper ─────────────────────────────────────────────────────

def _standings_table(baseline: pd.DataFrame, counter: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for drv in baseline["Driver"].unique():
        b_pos, b_gap = _final_pos(baseline, drv)
        c_rows = counter[counter["Driver"] == drv]
        if c_rows.empty:
            continue
        c_pos, c_gap = _final_pos(counter, drv)
        rows.append({
            "Driver":      drv,
            "Actual P":    b_pos,
            "What-If P":   c_pos,
            "Δ Pos":       b_pos - c_pos,
            "Actual Gap":  f"+{b_gap}s" if b_gap > 0 else "Leader",
            "What-If Gap": f"+{c_gap}s" if c_gap > 0 else "Leader",
        })
    return pd.DataFrame(rows).sort_values("What-If P").reset_index(drop=True)


# ── Gap chart ─────────────────────────────────────────────────────────────────

def _gap_chart(baseline, counter, driver, fork_lap):
    fig = go.Figure()
    total_laps = int(baseline["LapNumber"].max())

    b = baseline[baseline["Driver"] == driver].sort_values("LapNumber")
    c = counter[counter["Driver"] == driver].sort_values("LapNumber")

    fig.add_trace(go.Scatter(
        x=b["LapNumber"], y=b["GapToLeader"], mode="lines",
        name="Actual",
        line=dict(color="#ffaa00", width=2.5, dash="dash"),
        hovertemplate="Lap %{x}  +%{y:.2f}s<extra>Actual</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=c["LapNumber"], y=c["GapToLeader"], mode="lines",
        name="What-If",
        line=dict(color="#00d2ff", width=3),
        hovertemplate="Lap %{x}  +%{y:.2f}s<extra>What-If</extra>",
    ))

    merged = pd.merge(
        b[["LapNumber","GapToLeader"]].rename(columns={"GapToLeader":"actual"}),
        c[["LapNumber","GapToLeader"]].rename(columns={"GapToLeader":"counter"}),
        on="LapNumber", how="inner",
    )
    merged = merged[merged["LapNumber"] >= fork_lap]
    if not merged.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([merged["LapNumber"], merged["LapNumber"][::-1]]),
            y=pd.concat([merged["actual"],    merged["counter"][::-1]]),
            fill="toself", fillcolor="rgba(0,210,255,0.07)",
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_vline(x=fork_lap, line_dash="dot", line_color="#555",
                  annotation_text="Fork", annotation_font_color="#888")

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        title=dict(text=f"{driver}  —  Gap to Leader", font_size=14),
        xaxis=dict(title="Lap", range=[1, total_laps]),
        yaxis=dict(title="Gap (s)"),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=40, r=20, t=50, b=40),
        paper_bgcolor="#0e0e0e", plot_bgcolor="#141414",
    )
    return fig


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("# 🏎 F1 Race Rewind")
    st.markdown("*What-if race strategy simulator*")
    st.markdown("---")

    st.markdown("### 🗓 Race")
    race_key = st.selectbox("", list_races(), label_visibility="collapsed")
    race_df  = _load(race_key)

    total_laps = int(race_df["LapNumber"].max())
    drivers    = sorted(race_df["Driver"].unique().tolist())
    winner     = (race_df.sort_values("LapNumber")
                         .groupby("Driver").last()
                         .sort_values("Position")
                         .index[0])
    st.caption(f"🏆 Winner: **{winner}**  |  {total_laps} laps  |  {len(drivers)} drivers")
    st.markdown("---")

    st.markdown("### 👤 Driver")
    driver = st.selectbox("", drivers, label_visibility="collapsed")
    st.caption(f"📋 Strategy: {_driver_strategy(race_df, driver)}")
    st.markdown("---")

    st.markdown("### ⚡ Fork Point")
    st.caption("Simulation diverges from this lap onward.")
    fork_lap = st.slider("Fork lap", min_value=2, max_value=total_laps - 5,
                         value=max(2, total_laps // 3), label_visibility="collapsed")

    fork_standings = (race_df[race_df["LapNumber"] == fork_lap - 1]
                      .sort_values("Position")[["Driver","Position","Compound","TyreLife"]]
                      .head(5))
    if not fork_standings.empty:
        st.caption(f"Standings at lap {fork_lap - 1}:")
        st.dataframe(fork_standings.set_index("Driver"), use_container_width=True, height=210)
    st.markdown("---")

    st.markdown("### 🔧 Intervention")
    intervention = st.radio("", ["Pit Earlier", "Pit Later", "Change Compound"],
                            label_visibility="collapsed")

    force_pit_lap   = None
    force_compound  = None
    block_pit_until = None

    if intervention == "Pit Earlier":
        st.caption("Force a pit stop on a specific lap.")
        force_pit_lap  = st.slider("Pit on lap", fork_lap, total_laps - 3,
                                   fork_lap, key="pit_lap")
        force_compound = st.selectbox("Compound", ["SOFT","MEDIUM","HARD"],
                                      key="comp_early")

    elif intervention == "Pit Later":
        st.caption("Prevent pitting until a later lap.")
        block_pit_until = st.slider("No pit before lap", fork_lap + 1,
                                    total_laps - 2,
                                    min(fork_lap + 8, total_laps - 2),
                                    key="block_lap")
        st.info(f"🔒 {driver} stays out until at least lap {block_pit_until}")

    elif intervention == "Change Compound":
        st.caption("Pit and fit a different tyre.")
        force_pit_lap  = st.slider("Pit on lap", fork_lap, total_laps - 3,
                                   fork_lap, key="pit_lap_comp")
        force_compound = st.selectbox("Switch to", ["SOFT","MEDIUM","HARD"],
                                      key="comp_change")
        st.info(f"🔄 {driver} fits **{force_compound}** on lap {force_pit_lap}")

    st.markdown("---")
    run_btn = st.button("🚀 Run Simulation", type="primary", use_container_width=True)


# ── Main area ─────────────────────────────────────────────────────────────────

st.markdown(f"## 🏁 {race_key}")

if "result" not in st.session_state:
    st.session_state.result = None

# Run simulation
if run_btn:
    with st.spinner("⏱ Simulating…"):
        baseline, counter = simulate(
            race_df,
            fork_lap            = fork_lap,
            intervention_driver = driver,
            force_pit_lap       = force_pit_lap,
            force_compound      = force_compound,
            block_pit_until     = block_pit_until,
        )
    st.session_state.result   = (baseline, counter)
    st.session_state.driver   = driver
    st.session_state.fork_lap = fork_lap
    st.session_state.race_key = race_key


# ── Default view ──────────────────────────────────────────────────────────────
if st.session_state.result is None:
    st.info("👈 Configure your intervention in the sidebar, then hit **Run Simulation**.")

    st.markdown("""
    <div style='background:#1a2a1a;border:1px solid #2a4a2a;border-radius:8px;
                padding:14px 20px;margin:8px 0;'>
      <b>🏆 Try this scenario — Abu Dhabi 2021 (the controversial finale):</b><br>
      Hamilton led on old HARD tyres when the Safety Car deployed on lap 52.
      Verstappen pitted for fresh SOFT and won the championship on the final lap.<br><br>
      <b>What if Hamilton had also pitted under the Safety Car?</b><br>
      → Race: <b>2021 Abu Dhabi Grand Prix</b> · Driver: <b>HAM</b> · Fork: <b>52</b>
      · Pit Earlier · Lap <b>53</b> · <b>SOFT</b>
    </div>
    """, unsafe_allow_html=True)

    # Preview chart
    st.markdown("### 📊 Actual Race — Positions")
    fig_p = go.Figure()
    top_drvs = (race_df.sort_values("LapNumber").groupby("Driver").last()
                       .sort_values("Position").head(10).index.tolist())
    pal = ["#e10600","#ff8c00","#ffd700","#00d2ff","#00ff88",
           "#cc00ff","#ff69b4","#aaaaaa","#ffffff","#888888"]
    for i, d in enumerate(top_drvs):
        dd = race_df[race_df["Driver"]==d].sort_values("LapNumber")
        fig_p.add_trace(go.Scatter(x=dd["LapNumber"], y=dd["Position"],
                                   mode="lines", name=d,
                                   line=dict(color=pal[i], width=2)))
    fig_p.update_layout(
        template=PLOTLY_TEMPLATE, xaxis_title="Lap", height=380,
        yaxis=dict(title="Position", autorange="reversed",
                   tickvals=list(range(1,21)), ticktext=[f"P{i}" for i in range(1,21)]),
        paper_bgcolor="#0e0e0e", plot_bgcolor="#141414",
        legend=dict(orientation="h"),
    )
    st.plotly_chart(fig_p, use_container_width=True)


# ── Results ───────────────────────────────────────────────────────────────────
else:
    baseline, counter = st.session_state.result
    sim_driver        = st.session_state.driver
    sim_fork          = st.session_state.fork_lap
    sim_race_key      = st.session_state.get("race_key", "")

    b_pos, b_gap = _final_pos(baseline, sim_driver)
    c_pos, c_gap = _final_pos(counter,  sim_driver)
    delta_pos    = b_pos - c_pos
    delta_gap    = b_gap - c_gap

    # ── Result banner ─────────────────────────────────────────────────────────
    if delta_pos > 0:
        emoji, verdict = "📈", f"gained {delta_pos} position{'s' if delta_pos>1 else ''}"
    elif delta_pos < 0:
        emoji, verdict = "📉", f"lost {abs(delta_pos)} position{'s' if abs(delta_pos)>1 else ''}"
    else:
        emoji, verdict = "➡️", "finished in the same position"

    gap_txt = (f"gap reduced by {delta_gap:.2f}s" if delta_gap > 0
               else f"gap increased by {abs(delta_gap):.2f}s" if delta_gap < 0
               else "same gap to leader")

    st.markdown(f"""
    <div class='result-banner'>
      <h3>{emoji} With your intervention, <b>{sim_driver} {verdict}</b></h3>
      <p>Actual: <b>P{b_pos}</b> (+{b_gap}s) &nbsp;→&nbsp;
         What-If: <b>P{c_pos}</b> (+{c_gap}s) &nbsp;|&nbsp; {gap_txt}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Metric cards ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Actual Finish",   f"P{b_pos}", f"+{b_gap}s")
    c2.metric("What-If Finish",  f"P{c_pos}", f"+{c_gap}s")
    c3.metric("Position Change", f"{delta_pos:+d}",
              delta_color="normal" if delta_pos >= 0 else "inverse")
    c4.metric("Gap Change", f"{delta_gap:+.2f}s",
              delta_color="normal" if delta_gap >= 0 else "inverse")

    st.markdown("---")

    # ── Animated race replay ──────────────────────────────────────────────────
    st.markdown("### 🏁 Race Replay")
    st.caption("Hit ▶ to watch the race play out lap by lap — or drag the slider to any lap.")
    html_anim = _build_race_animation(baseline, counter, sim_driver, sim_fork,
                                       race_key=sim_race_key)
    components.html(html_anim, height=590, scrolling=False)

    st.markdown("---")

    # ── Gap chart ─────────────────────────────────────────────────────────────
    st.plotly_chart(_gap_chart(baseline, counter, sim_driver, sim_fork),
                    use_container_width=True)

    st.markdown("---")

    # ── Final standings ───────────────────────────────────────────────────────
    st.markdown("### 🏆 Final Standings")
    standings = _standings_table(baseline, counter)

    def _cdelta(v):
        if v > 0: return "color:#00ff88;font-weight:700"
        if v < 0: return "color:#e10600;font-weight:700"
        return "color:#888"

    def _crow(row):
        s = [""] * len(row)
        if row["Driver"] == sim_driver:
            s = ["background-color:#1e3a4a;font-weight:700"] * len(row)
        return s

    st.dataframe(
        standings.style.applymap(_cdelta, subset=["Δ Pos"]).apply(_crow, axis=1),
        use_container_width=True, hide_index=True,
    )

    # ── Tyre strategy ─────────────────────────────────────────────────────────
    st.markdown(f"### 🔴 {sim_driver} — Tyre Strategy")
    tc1, tc2 = st.columns(2)

    def _pit_summary(df, drv):
        d    = df[df["Driver"]==drv].sort_values("LapNumber")
        pits = d[d["PitInLap"]]
        if pits.empty:
            return "No pit stops"
        out = []
        for _, row in pits.iterrows():
            lap  = int(row["LapNumber"])
            nxt  = d[d["LapNumber"] == lap + 1]
            comp = nxt["Compound"].iloc[0] if not nxt.empty else "?"
            out.append(f"Lap **{lap}** → {comp}")
        return "\n\n".join(out)

    with tc1:
        st.markdown("**Actual**")
        st.markdown(_pit_summary(baseline, sim_driver))
    with tc2:
        st.markdown("**What-If**")
        st.markdown(_pit_summary(counter, sim_driver))

    st.markdown("---")
    if st.button("🔄 Reset / Try Another"):
        st.session_state.result = None
        st.rerun()
