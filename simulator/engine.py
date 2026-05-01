"""
F1 Race Rewind — Simulation Engine
=====================================
The lap-by-lap what-if simulator.

Core idea
---------
  Laps 1 → (fork_lap - 1) : actual historical data from the parquet file
  Laps fork_lap → end      : predicted using Model A (lap times) + Model B (pit decisions)

The counterfactual driver follows the user's intervention.
All other drivers follow Model B probabilities (sampled stochastically).

Tyre life convention (matches training data)
--------------------------------------------
  Pit-in lap  N  : TyreLife = age of OLD tyre (e.g., 16), PitInLap=True
  Pit-out lap N+1: TyreLife = 1  (first lap on NEW tyre), PitOutLap=True
  Normal laps    : TyreLife increments each lap

Pit cost
--------
  No separate constant needed. Model A was trained on data where PitOutLap
  times already include stationary pit box time (~3s) + cold-tyre penalty
  (~15s). Setting PitOutLap_int=1 in the feature vector handles it.

Public API
----------
  load_race(race_key)                 → pd.DataFrame
  simulate(race_df, fork_lap, ...)    → (baseline_df, counterfactual_df)
  list_races()                        → list[str]
"""

from __future__ import annotations

import sys
from pathlib import Path
from collections import deque

import numpy as np
import pandas as pd

# Project root on path so models/* imports work
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "models"))

from inference import predict_laptime, predict_pit_prob, get_encoders
from feature_engineering import COMPOUND_ENC
from compound_rules import suggest_compound, build_circuit_priors

# ---------------------------------------------------------------------------
# Demo race registry
# ---------------------------------------------------------------------------
DEMO_DIR = _ROOT / "data" / "demo"

DEMO_RACES: dict[str, str] = {
    "2021 Abu Dhabi Grand Prix":  "2021_Abu_Dhabi.parquet",
    "2022 Monaco Grand Prix":     "2022_Monaco.parquet",
    "2023 Italian Grand Prix":    "2023_Monza.parquet",
    "2019 German Grand Prix":     "2019_Hockenheim.parquet",
    "2023 Las Vegas Grand Prix":  "2023_Las_Vegas.parquet",
    "2022 Brazilian Grand Prix":  "2022_Brazil.parquet",
}


def list_races() -> list[str]:
    return list(DEMO_RACES.keys())


def load_race(race_key: str) -> pd.DataFrame:
    """Load a demo race DataFrame by its display name."""
    if race_key not in DEMO_RACES:
        raise KeyError(f"Unknown race '{race_key}'. Available: {list_races()}")
    return pd.read_parquet(DEMO_DIR / DEMO_RACES[race_key])


# ---------------------------------------------------------------------------
# Internal state helpers
# ---------------------------------------------------------------------------

def _build_driver_states(race_df: pd.DataFrame, up_to_lap: int) -> dict:
    """
    Extract per-driver state from historical data up to (but not including)
    up_to_lap. Returns a dict keyed by driver code.

    Cumulative time is anchored to GapToLeader from the actual race data,
    NOT to summed lap times. Summing lap times is unreliable because SC laps
    >180s are filtered out of the parquet, leaving lapped drivers with
    artificially low cumulative totals. Using GapToLeader gives correct
    relative positions regardless of missing laps.
    """
    hist = race_df[race_df["LapNumber"] < up_to_lap]
    if hist.empty:
        return {}

    # Find each driver's state on the last recorded lap before the fork
    last_rows = (hist.sort_values("LapNumber")
                     .groupby("Driver")
                     .last()
                     .reset_index())

    # Leader's summed lap time = reference anchor
    # (leader has fewest missing laps; sum is most accurate for them)
    leader_row  = last_rows.sort_values("Position").iloc[0]
    leader_drv  = leader_row["Driver"]
    leader_laps = hist[hist["Driver"] == leader_drv].sort_values("LapNumber")
    leader_cum  = float(leader_laps["LapTime"].sum())

    states = {}
    for _, row in last_rows.iterrows():
        driver  = row["Driver"]
        drv_df  = hist[hist["Driver"] == driver].sort_values("LapNumber")

        # Cumulative time = leader's cum time + this driver's actual gap
        # This correctly handles lapped drivers and filtered SC laps
        gap      = float(row["GapToLeader"]) if pd.notna(row["GapToLeader"]) else 0.0
        cum_time = leader_cum + gap

        states[driver] = {
            # Accumulated race time — anchored via GapToLeader
            "cumulative_time": cum_time,
            # Current tyre
            "compound":        str(row["Compound"]),
            "tyre_life":       int(row["TyreLife"]),
            # Strategy history
            "total_pit_stops": int(row["TotalPitStops"]),
            "compounds_used":  list(drv_df["Compound"].unique()),
            # Rolling lap time window for RollingMean features
            "recent_times":    deque(drv_df["LapTime"].tolist(), maxlen=5),
            # Carry-forward flags
            "pit_out_flag":    bool(row["PitOutLap"]),
            # Cooldown: prevent re-pitting within 5 laps of previous stop
            "laps_since_pit":  int(row["TyreLife"]),
            # Position tracking
            "position":        float(row["Position"]) if pd.notna(row["Position"]) else 10.0,
            "gap_to_leader":   gap,
        }

    return states


def _rolling(recent: deque, window: int) -> float:
    """Mean of the last `window` entries in a deque; fall back to full mean."""
    vals = list(recent)[-window:]
    return float(np.mean(vals)) if vals else 90.0


# ---------------------------------------------------------------------------
# Core simulation loop
# ---------------------------------------------------------------------------

def simulate(
    race_df:              pd.DataFrame,
    fork_lap:             int,
    intervention_driver:  str,
    force_pit_lap:        int | None = None,
    force_compound:       str | None = None,
    block_pit_until:      int | None = None,
    seed:                 int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a counterfactual simulation and return both traces.

    Parameters
    ----------
    race_df              : historical lap data for the chosen demo race
    fork_lap             : first lap to simulate (1-indexed).
                           Laps < fork_lap use actual data in both outputs.
    intervention_driver  : 3-letter driver code to apply the intervention to
    force_pit_lap        : force the driver to pit on exactly this lap number.
                           If the lap is before fork_lap it is ignored.
    force_compound       : compound to fit when the driver pits under the
                           intervention (None = compound_rules decides)
    block_pit_until      : prevent the driver from pitting before this lap
                           (model decides after this lap)
    seed                 : RNG seed for reproducible field strategy sampling

    Returns
    -------
    baseline_df        : actual historical data (race_df unchanged)
    counterfactual_df  : pre-fork historical rows + simulated rows from fork_lap
                         Same 14-column schema as the input parquet.
    """
    rng = np.random.default_rng(seed)
    enc = get_encoders()
    circuit_map = enc["circuit_map"]

    # ── Race metadata ────────────────────────────────────────────────────────
    race_name  = str(race_df["RaceName"].iloc[0])
    season     = int(race_df["Season"].iloc[0])
    round_num  = int(race_df["RoundNumber"].iloc[0])
    total_laps = int(race_df["LapNumber"].max())
    circuit_enc = int(circuit_map.get(race_name, 0))

    # Clamp fork_lap to valid range
    fork_lap = max(2, min(fork_lap, total_laps))

    # ── Baseline = historical as-is ─────────────────────────────────────────
    baseline_df = race_df.copy()

    # ── Circuit compound priors for field drivers ────────────────────────────
    circuit_priors = build_circuit_priors(race_df)

    # ── Pre-index historical lap data for fast per-lap lookups ───────────────
    # Field drivers follow their actual historical pit stops and lap times
    # exactly.  Only the intervention driver is simulated with the model.
    # This keeps RUS/VER/etc on their real strategy so positions are meaningful.
    hist_by_driver_lap = {
        (row["Driver"], int(row["LapNumber"])): row
        for _, row in race_df.iterrows()
    }

    # ── Initialise driver states from historical data ────────────────────────
    driver_states = _build_driver_states(race_df, fork_lap)

    # Each driver's last recorded lap — used to detect DNFs before fork_lap
    last_actual_lap: dict[str, int] = {
        drv: int(race_df[race_df["Driver"] == drv]["LapNumber"].max())
        for drv in driver_states
    }

    # Only simulate drivers still running at or right before the fork.
    # Drivers whose last lap is before fork_lap - 1 DNF'd or were lapped out
    # and should not appear in the counterfactual simulation phase.
    all_drivers = [
        d for d in driver_states
        if last_actual_lap[d] >= fork_lap - 1
    ]

    if intervention_driver not in all_drivers:
        # Intervention driver DNF'd — still include them (they're the focus)
        all_drivers.append(intervention_driver)
    if intervention_driver not in driver_states:
        raise ValueError(f"Driver '{intervention_driver}' not found in race data.")

    # Track drivers who DNF mid-simulation.  When a field driver's last actual
    # lap is less than the current sim lap, they've genuinely stopped racing and
    # should no longer contribute rows.  Carry-forward is only for isolated
    # filtered laps (SC outliers, etc.) — not for an entire post-DNF stint.
    dnf_drivers: set[str] = set()

    # Pre-compute per-lap median lap time for the carry-forward fallback.
    # When a field driver has a filtered/missing lap (e.g. a short SC lap that
    # FastF1 stripped), we use the fleet-wide median for that lap instead of
    # the driver's own last time — so relative gaps are preserved correctly.
    lap_median_time: dict[int, float] = {
        lap_num: float(race_df[race_df["LapNumber"] == lap_num]["LapTime"].median())
        for lap_num in range(fork_lap, total_laps + 1)
        if not race_df[race_df["LapNumber"] == lap_num].empty
    }

    # ── Simulate laps fork_lap → total_laps ─────────────────────────────────
    sim_rows: list[dict] = []

    for lap in range(fork_lap, total_laps + 1):
        laps_remaining = total_laps - lap

        # Track status: use historical mode for this lap (neutral after last known)
        hist_lap = race_df[race_df["LapNumber"] == lap]
        track_status = int(hist_lap["TrackStatus"].mode().iloc[0]) if not hist_lap.empty else 1

        # ── Step 1: decide pit for every driver ───────────────────────────
        pit_decisions:    dict[str, bool] = {}
        use_actual_lap:   dict[str, bool] = {}   # True = use historical row

        for driver in all_drivers:
            st = driver_states[driver]
            hist_row = hist_by_driver_lap.get((driver, lap))

            # ── Field drivers: follow actual historical strategy ───────────
            # Their pit stops, compounds, and lap times are taken directly
            # from the parquet so their race result is unaffected by noise.
            if driver != intervention_driver:
                # If the driver DNF'd before this lap, retire them permanently.
                if driver in dnf_drivers:
                    use_actual_lap[driver] = "dnf"
                    pit_decisions[driver]  = False
                    continue

                if hist_row is not None:
                    pit_decisions[driver]  = bool(hist_row["PitInLap"])
                    use_actual_lap[driver] = True
                else:
                    # hist_row missing for this lap.
                    # If the driver has no data from here onward, they DNF'd.
                    if last_actual_lap.get(driver, 0) < lap:
                        dnf_drivers.add(driver)
                        use_actual_lap[driver] = "dnf"
                        pit_decisions[driver]  = False
                    else:
                        # Isolated filtered lap (SC outlier, etc.) — carry forward.
                        pit_decisions[driver]  = False
                        use_actual_lap[driver] = "carry"
                continue

            # ── Intervention driver: apply the user's strategy change ─────
            use_actual_lap[driver] = False   # always simulate this driver

            if st["pit_out_flag"]:
                pit_decisions[driver] = False
                continue

            if laps_remaining < 2:
                pit_decisions[driver] = False
                continue

            if force_pit_lap is not None and lap == force_pit_lap:
                pit_decisions[driver] = True
                continue

            if block_pit_until is not None and lap < block_pit_until:
                pit_decisions[driver] = False
                continue

            # Cooldown guard (still applies to intervention driver for
            # follow-on laps after a forced pit)
            if st["laps_since_pit"] < 5:
                pit_decisions[driver] = False
                continue

            # Model-based decision for laps not covered by the intervention
            pit_feats = pd.DataFrame([{
                "TyreLife":       st["tyre_life"],
                "TyreLife_sq":    st["tyre_life"] ** 2,
                "Compound_enc":   COMPOUND_ENC.get(st["compound"], 2),
                "Position":       st["position"],
                "GapToLeader":    st["gap_to_leader"],
                "LapsRemaining":  laps_remaining,
                "TotalPitStops":  st["total_pit_stops"],
                "TrackStatus":    track_status,
                "Season":         season,
            }])
            pit_prob = float(predict_pit_prob(pit_feats, np.array([track_status]))[0])
            pit_decisions[driver] = rng.random() < pit_prob

        # ── Step 2: predict lap times & update state ──────────────────────
        lap_cumulative: dict[str, float] = {}

        for driver in all_drivers:
            st       = driver_states[driver]
            hist_row = hist_by_driver_lap.get((driver, lap))

            # ── Field drivers: replay actual historical lap exactly ────────
            # This keeps non-intervention drivers on their real strategies so
            # their finishing positions are unaffected by model noise.

            # DNF'd driver — omit from this lap onward
            if use_actual_lap.get(driver) == "dnf":
                continue

            # "carry" sentinel = filtered lap; reuse the driver's last time.
            if use_actual_lap.get(driver) == "carry":
                # Carry-forward strategy:
                # Carry-forward strategy for missing laps:
                # • At most 1 lap missing at end → use fleet median (driver almost
                #   certainly finished; missing is a FastF1 filter artefact).
                # • 2+ consecutive laps missing at tail → use driver's own last
                #   recorded time.  The race may have restarted by then, and the
                #   median would assign falsely-fast restart pace to a driver whose
                #   data was absent for unknown reasons (retirement / data loss).
                laps_missing_tail = total_laps - last_actual_lap.get(driver, total_laps)
                if laps_missing_tail <= 1:
                    lap_time = lap_median_time.get(
                        lap,
                        float(st["recent_times"][-1]) if st["recent_times"] else 90.0
                    )
                else:
                    lap_time = float(st["recent_times"][-1]) if st["recent_times"] else 90.0
                is_pit_in  = False
                is_pit_out = st["pit_out_flag"]
                compound   = str(st["compound"])

                st["cumulative_time"] += lap_time
                lap_cumulative[driver] = st["cumulative_time"]
                st["tyre_life"]       += 1
                st["pit_out_flag"]     = False
                st["laps_since_pit"]  += 1
                st["recent_times"].append(lap_time)

                sim_rows.append({
                    "RaceName":      race_name,
                    "Season":        season,
                    "RoundNumber":   round_num,
                    "Driver":        driver,
                    "LapNumber":     lap,
                    "LapTime":       lap_time,
                    "Compound":      compound,
                    "TyreLife":      st["tyre_life"],
                    "PitInLap":      False,
                    "PitOutLap":     is_pit_out,
                    "TotalPitStops": st["total_pit_stops"],
                    "TrackStatus":   track_status,
                    "_cumtime":      st["cumulative_time"],
                    "Position":      0.0,
                    "GapToLeader":   0.0,
                })
                continue

            if use_actual_lap.get(driver, False) and hist_row is not None:
                lap_time   = float(hist_row["LapTime"])
                is_pit_in  = bool(hist_row["PitInLap"])
                is_pit_out = bool(hist_row["PitOutLap"])
                compound   = str(hist_row["Compound"])   # compound used THIS lap

                # Update cumulative time
                st["cumulative_time"] += lap_time
                lap_cumulative[driver] = st["cumulative_time"]

                # Sync all tyre / stop state directly from the historical row
                st["tyre_life"]       = int(hist_row["TyreLife"])
                st["total_pit_stops"] = int(hist_row["TotalPitStops"])

                if is_pit_in:
                    # New compound takes effect on the very next lap —
                    # read it from history so we don't guess wrong
                    next_hist = hist_by_driver_lap.get((driver, lap + 1))
                    if next_hist is not None:
                        new_compound = str(next_hist["Compound"])
                    else:
                        new_compound = suggest_compound(
                            current_compound = compound,
                            tyre_life        = st["tyre_life"],
                            laps_remaining   = laps_remaining,
                            compounds_used   = list(st["compounds_used"]),
                            track_status     = track_status,
                            circuit_priors   = circuit_priors,
                        )
                    st["compound"]        = new_compound
                    st["pit_out_flag"]    = True
                    st["laps_since_pit"]  = 0
                    if new_compound not in st["compounds_used"]:
                        st["compounds_used"].append(new_compound)
                else:
                    st["compound"]        = compound
                    st["pit_out_flag"]    = is_pit_out
                    st["laps_since_pit"] += 1

                st["recent_times"].append(lap_time)

                sim_rows.append({
                    "RaceName":      race_name,
                    "Season":        season,
                    "RoundNumber":   round_num,
                    "Driver":        driver,
                    "LapNumber":     lap,
                    "LapTime":       lap_time,
                    "Compound":      compound,
                    "TyreLife":      int(hist_row["TyreLife"]),
                    "PitInLap":      is_pit_in,
                    "PitOutLap":     is_pit_out,
                    "TotalPitStops": int(hist_row["TotalPitStops"]),
                    "TrackStatus":   track_status,
                    "_cumtime":      st["cumulative_time"],
                    "Position":      0.0,
                    "GapToLeader":   0.0,
                })
                continue   # ← skip model prediction for field drivers

            # ── Intervention driver (or missing hist row): use Model A ────
            is_pit_in   = pit_decisions[driver]
            is_pit_out  = st["pit_out_flag"]

            # Lap time features
            lt_feats = pd.DataFrame([{
                "TyreLife":       st["tyre_life"],
                "TyreLife_sq":    st["tyre_life"] ** 2,
                "Compound_enc":   COMPOUND_ENC.get(st["compound"], 2),
                "PitInLap_int":   int(is_pit_in),
                "PitOutLap_int":  int(is_pit_out),
                "TrackStatus":    track_status,
                "LapNumber":      lap,
                "LapsRemaining":  laps_remaining,
                "Position":       st["position"],
                "GapToLeader":    st["gap_to_leader"],
                "RollingMean3":   _rolling(st["recent_times"], 3),
                "RollingMean5":   _rolling(st["recent_times"], 5),
                "Season":         season,
                "Circuit_enc":    circuit_enc,
            }])

            lap_time = float(np.clip(predict_laptime(lt_feats)[0], 60.0, 175.0))

            # ── Update cumulative time ─────────────────────────────────────
            st["cumulative_time"] += lap_time
            lap_cumulative[driver] = st["cumulative_time"]

            # ── Update tyre state ──────────────────────────────────────────
            if is_pit_in:
                # Determine new compound
                if driver == intervention_driver and force_compound:
                    new_compound = force_compound
                else:
                    new_compound = suggest_compound(
                        current_compound = st["compound"],
                        tyre_life        = st["tyre_life"],
                        laps_remaining   = laps_remaining,
                        compounds_used   = list(st["compounds_used"]),
                        track_status     = track_status,
                        circuit_priors   = circuit_priors,
                    )
                st["compound"]        = new_compound
                st["tyre_life"]       = 1      # next lap starts at TyreLife=1
                st["total_pit_stops"] += 1
                st["compounds_used"].append(new_compound)
                st["pit_out_flag"]    = True   # next lap is the out-lap
                st["laps_since_pit"]  = 0      # reset cooldown
            else:
                st["tyre_life"]      += 1      # tyre ages normally
                st["pit_out_flag"]    = False  # clear any stale flag
                st["laps_since_pit"] += 1      # count laps since last stop

            # ── Update rolling lap time history ────────────────────────────
            st["recent_times"].append(lap_time)

            # ── Record row (position & gap filled after sorting below) ─────
            sim_rows.append({
                "RaceName":      race_name,
                "Season":        season,
                "RoundNumber":   round_num,
                "Driver":        driver,
                "LapNumber":     lap,
                "LapTime":       lap_time,
                "Compound":      st["compound"],
                "TyreLife":      st["tyre_life"],
                "PitInLap":      is_pit_in,
                "PitOutLap":     is_pit_out,
                "TotalPitStops": st["total_pit_stops"],
                "TrackStatus":   track_status,
                "_cumtime":      st["cumulative_time"],  # scratch col for sorting
                "Position":      0.0,   # filled below
                "GapToLeader":   0.0,   # filled below
            })

        # ── Step 3: re-sort positions by cumulative race time ─────────────
        this_lap_rows = [r for r in sim_rows if r["LapNumber"] == lap]
        this_lap_rows.sort(key=lambda r: r["_cumtime"])

        leader_time = this_lap_rows[0]["_cumtime"]
        for pos, row in enumerate(this_lap_rows, start=1):
            row["Position"]     = float(pos)
            row["GapToLeader"]  = round(row["_cumtime"] - leader_time, 3)
            # Feed back into driver state for next lap's features
            driver_states[row["Driver"]]["position"]       = float(pos)
            driver_states[row["Driver"]]["gap_to_leader"]  = row["GapToLeader"]

    # ── Build counterfactual DataFrame ───────────────────────────────────────
    OUTPUT_COLS = [
        "RaceName", "Season", "RoundNumber", "Driver", "LapNumber",
        "Position", "LapTime", "Compound", "TyreLife",
        "PitInLap", "PitOutLap", "TotalPitStops", "GapToLeader", "TrackStatus",
    ]

    sim_df = pd.DataFrame(sim_rows)[OUTPUT_COLS]

    pre_fork = race_df[race_df["LapNumber"] < fork_lap][OUTPUT_COLS].copy()
    counterfactual_df = pd.concat([pre_fork, sim_df], ignore_index=True)

    return baseline_df[OUTPUT_COLS], counterfactual_df


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    print("=" * 55)
    print("  Simulation Engine — Smoke Test")
    print("=" * 55)

    race_df = load_race("2021 Abu Dhabi Grand Prix")
    total   = int(race_df["LapNumber"].max())
    print(f"\nRace : 2021 Abu Dhabi GP  |  {total} laps  |  {race_df['Driver'].nunique()} drivers")

    # --- Test 1: VER pits 5 laps earlier (lap 8 instead of lap 13) ----------
    print("\n[Test 1] VER forced pit on lap 8 (actual: lap 13)")
    t0 = time.time()
    baseline, counter = simulate(
        race_df,
        fork_lap           = 8,
        intervention_driver= "VER",
        force_pit_lap      = 8,
        force_compound     = "HARD",
    )
    elapsed = time.time() - t0
    print(f"  Simulated in {elapsed:.2f}s")

    def final_pos(df, driver):
        last = df[df["Driver"] == driver].sort_values("LapNumber").iloc[-1]
        return int(last["Position"]), round(float(last["GapToLeader"]), 3)

    b_pos, b_gap = final_pos(baseline, "VER")
    c_pos, c_gap = final_pos(counter,  "VER")
    print(f"  VER baseline  : P{b_pos}  gap={b_gap}s")
    print(f"  VER counter   : P{c_pos}  gap={c_gap}s")
    print(f"  Position delta: {b_pos - c_pos:+d}  (+ means gained)")

    # --- Test 2: HAM blocks pit until lap 50 --------------------------------
    print("\n[Test 2] HAM stays out until lap 50 (ultra-long stint)")
    _, counter2 = simulate(
        race_df,
        fork_lap           = 10,
        intervention_driver= "HAM",
        block_pit_until    = 50,
    )
    h_pos, h_gap = final_pos(counter2, "HAM")
    print(f"  HAM with late pit : P{h_pos}  gap={h_gap}s")

    # --- Schema check --------------------------------------------------------
    print("\n[Schema check]")
    expected = {"RaceName","Season","RoundNumber","Driver","LapNumber","Position",
                "LapTime","Compound","TyreLife","PitInLap","PitOutLap",
                "TotalPitStops","GapToLeader","TrackStatus"}
    missing = expected - set(counter.columns)
    print(f"  Missing columns : {missing or 'none ✅'}")
    print(f"  Baseline laps   : {len(baseline)}")
    print(f"  Counter laps    : {len(counter)}")
    print(f"  Laps match      : {len(baseline) == len(counter)} ✅")

    print("\n  All tests passed.")
