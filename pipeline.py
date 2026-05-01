"""
F1 Race Rewind — Data Pipeline
================================
Pulls lap-by-lap race data via FastF1, cleans it, engineers features,
and saves parquet files for training (2018–2024) and demo races.

Usage:
    python3 pipeline.py

Re-runnable at the individual race level:
  - Already-saved races inside a year parquet are skipped.
  - Rate-limit errors save progress and exit cleanly; the next run resumes.

Rate limit: FastF1 uses ~7 API calls per session load.
  500 calls/hr ÷ 7 = ~71 sessions/hr max.
  SLEEP_BETWEEN_SESSIONS is set to stay safely under that budget.
  Cached sessions (data/cache/) cost 0 API calls on re-runs.
"""

import time
import warnings
from pathlib import Path

import fastf1
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent / "data"
TRAINING_DIR = BASE_DIR / "training"
DEMO_DIR = BASE_DIR / "demo"
CACHE_DIR = BASE_DIR / "cache"

for _d in [TRAINING_DIR, DEMO_DIR, CACHE_DIR]:
    _d.mkdir(parents=True, exist_ok=True)

fastf1.Cache.enable_cache(str(CACHE_DIR))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
TRAINING_YEARS = range(2018, 2025)

# (year, FastF1 event identifier, output filename)
DEMO_RACES = [
    (2021, "Abu Dhabi Grand Prix",   "2021_Abu_Dhabi.parquet"),
    (2022, "Monaco Grand Prix",      "2022_Monaco.parquet"),
    (2023, "Italian Grand Prix",     "2023_Monza.parquet"),
    (2019, "German Grand Prix",      "2019_Hockenheim.parquet"),
    (2023, "Las Vegas Grand Prix",   "2023_Las_Vegas.parquet"),
    (2022, "Brazilian Grand Prix",   "2022_Brazil.parquet"),
]

REQUIRED_COLS = [
    "RaceName", "Season", "RoundNumber", "Driver", "LapNumber", "Position",
    "LapTime", "Compound", "TyreLife", "PitInLap", "PitOutLap",
    "TotalPitStops", "GapToLeader", "TrackStatus",
]

MAX_LAP_TIME_SEC = 180.0  # discard laps > 3 min (SC laps / outliers)

# Sleep between session loads to stay under 500 API calls/hr.
# Budget: 500 calls/hr ÷ 7 calls/session = 71 sessions/hr max
# Each session loads in ~15 s → sleep 38 s → cycle ~53 s → ~68 sessions/hr ✓
SLEEP_BETWEEN_SESSIONS = 38  # seconds

# TrackStatus string codes → int
TRACK_STATUS_MAP = {"1": 1, "2": 2, "4": 4, "5": 5, "6": 6}

# Compound aliases → canonical name
COMPOUND_MAP = {
    "SOFT": "SOFT", "MEDIUM": "MEDIUM", "HARD": "HARD",
    "INTERMEDIATE": "INTER", "INTER": "INTER", "WET": "WET",
    "SUPERSOFT": "SOFT", "ULTRASOFT": "SOFT", "HYPERSOFT": "SOFT",
    "SUPERHARD": "HARD", "TEST_UNKNOWN": "UNKNOWN", "UNKNOWN": "UNKNOWN",
}

summary_records: list[dict] = []


# ---------------------------------------------------------------------------
# Rate-limit handling
# ---------------------------------------------------------------------------

class RateLimitError(Exception):
    pass


def _is_rate_limited(exc: Exception) -> bool:
    msg = str(exc)
    return "500 calls/h" in msg or "rate limit" in msg.lower() or "too many requests" in msg.lower()


# ---------------------------------------------------------------------------
# Session loading
# ---------------------------------------------------------------------------

def load_session(year: int, identifier) -> fastf1.core.Session:
    """
    Load a FastF1 race session.
    Raises RateLimitError if the API rate limit is hit.
    Returns None for any other failure (bad round number, no data, etc.).
    """
    try:
        session = fastf1.get_session(year, identifier, "R")
        try:
            session.load(telemetry=False, weather=False, messages=False)
        except TypeError:
            session.load()
        return session
    except RateLimitError:
        raise
    except Exception as exc:
        if _is_rate_limited(exc):
            raise RateLimitError(str(exc))
        print(f"    WARNING: Failed to load {year} '{identifier}' — {exc}")
        return None


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def _parse_track_status(val) -> int:
    """
    FastF1 TrackStatus can be a single char ('1') or composite ('24').
    Extract the highest-priority code: red(5) > SC(4) > VSC(6) > yellow(2) > clear(1).
    """
    s = str(val).strip()
    for code in [5, 4, 6, 2, 1]:
        if str(code) in s:
            return code
    return 1


def _compute_tyre_life(group: pd.DataFrame) -> pd.DataFrame:
    """Fallback: recompute TyreLife, resetting to 1 after each pit stop."""
    group = group.sort_values("LapNumber").copy()
    life, counter = [], 1
    for _, row in group.iterrows():
        life.append(counter)
        counter = 1 if row["PitInLap"] else counter + 1
    group["TyreLife"] = life
    return group


def _gap_to_leader(laps: pd.DataFrame) -> np.ndarray:
    """
    Compute GapToLeader (seconds) per lap.
    Uses the `Time` column (real session-elapsed timedelta) when available;
    falls back to cumulative LapTime sums.
    """
    if "Time" in laps.columns and laps["Time"].notna().any():
        leader_time = laps.groupby("LapNumber")["Time"].min().rename("LeaderTime")
        tmp = laps.join(leader_time, on="LapNumber")
        gap = (tmp["Time"] - tmp["LeaderTime"]).dt.total_seconds().clip(lower=0)
    else:
        tmp = laps.copy()
        tmp["_CumLap"] = tmp.groupby("Driver")["LapTimeSec"].cumsum()
        leader_cum = tmp.groupby("LapNumber")["_CumLap"].min().rename("_LeaderCum")
        tmp = tmp.join(leader_cum, on="LapNumber")
        gap = (tmp["_CumLap"] - tmp["_LeaderCum"]).clip(lower=0)
    return gap.values


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_session(session: fastf1.core.Session) -> pd.DataFrame | None:
    """
    Extract, clean, and engineer features from a loaded FastF1 session.
    Returns a DataFrame with REQUIRED_COLS, or None if data is unusable.
    """
    laps = session.laps.copy()
    if laps.empty:
        return None

    laps["LapTimeSec"] = laps["LapTime"].dt.total_seconds()
    laps = laps[laps["LapTimeSec"].notna()]
    laps = laps[(laps["LapTimeSec"] > 0) & (laps["LapTimeSec"] <= MAX_LAP_TIME_SEC)]
    if laps.empty:
        return None

    laps = laps.sort_values(["Driver", "LapNumber"]).reset_index(drop=True)

    laps["PitInLap"]  = laps["PitInTime"].notna()
    laps["PitOutLap"] = laps["PitOutTime"].notna()
    laps["TotalPitStops"] = laps.groupby("Driver")["PitInLap"].cumsum().astype(int)

    if "TyreLife" not in laps.columns or laps["TyreLife"].isna().all():
        laps = laps.groupby("Driver", group_keys=False).apply(_compute_tyre_life)
    else:
        bad = laps.loc[laps["TyreLife"].isna(), "Driver"].unique()
        if len(bad):
            fixed = (laps[laps["Driver"].isin(bad)]
                     .groupby("Driver", group_keys=False)
                     .apply(_compute_tyre_life))
            laps.loc[laps["Driver"].isin(bad), "TyreLife"] = fixed["TyreLife"].values

    laps["TrackStatus"] = (laps["TrackStatus"].apply(_parse_track_status)
                           if "TrackStatus" in laps.columns else 1)
    laps["GapToLeader"] = _gap_to_leader(laps)
    laps["Compound"] = (laps["Compound"].fillna("UNKNOWN").str.upper()
                        .map(lambda x: COMPOUND_MAP.get(x, x)))

    event     = session.event
    race_name = str(event.get("EventName", event.get("OfficialEventName", "Unknown")))
    season    = int(event.get("Year", event.get("EventDate", pd.Timestamp("1900")).year))
    round_num = int(event.get("RoundNumber", 0))

    if "Position" not in laps.columns or laps["Position"].isna().all():
        laps["Position"] = np.nan

    out = pd.DataFrame({
        "RaceName":      race_name,
        "Season":        season,
        "RoundNumber":   round_num,
        "Driver":        laps["Driver"].values,
        "LapNumber":     laps["LapNumber"].values,
        "Position":      laps["Position"].values,
        "LapTime":       laps["LapTimeSec"].values,
        "Compound":      laps["Compound"].values,
        "TyreLife":      laps["TyreLife"].values,
        "PitInLap":      laps["PitInLap"].values,
        "PitOutLap":     laps["PitOutLap"].values,
        "TotalPitStops": laps["TotalPitStops"].values,
        "GapToLeader":   laps["GapToLeader"].values,
        "TrackStatus":   laps["TrackStatus"].values,
    })

    out["LapNumber"]     = out["LapNumber"].astype(int)
    out["TyreLife"]      = pd.to_numeric(out["TyreLife"], errors="coerce").fillna(1).astype(int)
    out["TotalPitStops"] = out["TotalPitStops"].astype(int)
    out["TrackStatus"]   = out["TrackStatus"].astype(int)
    out["PitInLap"]      = out["PitInLap"].astype(bool)
    out["PitOutLap"]     = out["PitOutLap"].astype(bool)
    out["LapTime"]       = out["LapTime"].astype(float)
    out["GapToLeader"]   = out["GapToLeader"].astype(float)

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _record(df: pd.DataFrame | None, year: int, race_name: str,
            round_num: int, status: str) -> None:
    if df is not None and not df.empty:
        missing = [c for c in REQUIRED_COLS if c not in df.columns]
        summary_records.append({
            "Season": year, "RaceName": race_name, "RoundNumber": round_num,
            "NumLaps": len(df), "NumDrivers": df["Driver"].nunique(),
            "MissingCols": ", ".join(missing) if missing else "",
            "Status": status,
        })
    else:
        summary_records.append({
            "Season": year, "RaceName": race_name, "RoundNumber": round_num,
            "NumLaps": 0, "NumDrivers": 0, "MissingCols": "", "Status": status,
        })


def _flush_summary() -> None:
    """Write current summary_records to data_summary.csv."""
    if not summary_records:
        return
    summary_df = pd.DataFrame(summary_records)
    summary_path = BASE_DIR / "data_summary.csv"
    summary_df.to_csv(summary_path, index=False)


# ---------------------------------------------------------------------------
# Year processing (training data)
# ---------------------------------------------------------------------------

def get_schedule(year: int) -> pd.DataFrame | None:
    try:
        sched = fastf1.get_event_schedule(year, include_testing=False)
        return sched[sched["RoundNumber"] > 0]
    except Exception as exc:
        print(f"  WARNING: Could not fetch schedule for {year} — {exc}")
        return None


def _save_year(output_path: Path, existing_df: pd.DataFrame | None,
               new_dfs: list[pd.DataFrame]) -> None:
    """Append new_dfs to existing_df and write parquet."""
    if not new_dfs:
        return
    parts = ([existing_df] if existing_df is not None else []) + new_dfs
    combined = pd.concat(parts, ignore_index=True)
    combined.to_parquet(output_path, index=False)
    print(f"  => Saved {output_path.name}  ({len(combined):,} laps total, "
          f"{combined['RaceName'].nunique()} races)")


def process_year(year: int) -> bool:
    """
    Process all races for a year, appending only new races to the parquet.
    Returns False if a rate limit was hit (caller should stop the run).
    """
    output_path = TRAINING_DIR / f"{year}_all_races.parquet"

    # Load already-processed races so we can skip them
    existing_df: pd.DataFrame | None = None
    existing_races: set[str] = set()
    if output_path.exists():
        try:
            existing_df = pd.read_parquet(output_path)
            existing_races = set(existing_df["RaceName"].unique())
            for rn, grp in existing_df.groupby("RaceName"):
                _record(grp, year, rn, int(grp["RoundNumber"].iloc[0]),
                        "skipped (already saved)")
        except Exception:
            pass

    schedule = get_schedule(year)
    if schedule is None:
        return True

    pending = schedule[~schedule["EventName"].isin(existing_races)]
    if pending.empty:
        print(f"  [DONE] {year} — all {len(existing_races)} races already saved.")
        return True

    print(f"  {year}: {len(existing_races)} done, {len(pending)} pending.")
    new_dfs: list[pd.DataFrame] = []

    for _, event in pending.iterrows():
        race_name = event["EventName"]
        round_num = int(event["RoundNumber"])
        print(f"    {year} | R{round_num:02d} | {race_name}  "
              f"[sleeping {SLEEP_BETWEEN_SESSIONS}s]")
        time.sleep(SLEEP_BETWEEN_SESSIONS)

        try:
            session = load_session(year, round_num)
        except RateLimitError:
            print("    Rate limit hit — saving progress and stopping.")
            _save_year(output_path, existing_df, new_dfs)
            _flush_summary()
            return False

        if session is None:
            _record(None, year, race_name, round_num, "FAILED (load error)")
            continue

        df = process_session(session)
        if df is None or df.empty:
            _record(None, year, race_name, round_num, "FAILED (no usable laps)")
            continue

        _record(df, year, race_name, round_num, "ok")
        new_dfs.append(df)

    _save_year(output_path, existing_df, new_dfs)
    return True


# ---------------------------------------------------------------------------
# Demo race processing
# ---------------------------------------------------------------------------

def process_demo_race(year: int, race_name: str, filename: str) -> bool:
    """
    Process a single demo race.
    Returns False if a rate limit was hit.
    """
    output_path = DEMO_DIR / filename

    if output_path.exists():
        print(f"    [SKIP] {year} {race_name} — already saved.")
        try:
            existing = pd.read_parquet(output_path)
            _record(existing, year, race_name,
                    int(existing["RoundNumber"].iloc[0]),
                    "skipped/demo (already saved)")
        except Exception:
            pass
        return True

    print(f"    {year} | {race_name}  [sleeping {SLEEP_BETWEEN_SESSIONS}s]")
    time.sleep(SLEEP_BETWEEN_SESSIONS)

    try:
        session = load_session(year, race_name)
    except RateLimitError:
        print("    Rate limit hit — stopping.")
        _flush_summary()
        return False

    if session is None:
        _record(None, year, race_name, 0, "FAILED/demo (load error)")
        return True

    df = process_session(session)
    if df is None or df.empty:
        _record(None, year, race_name, 0, "FAILED/demo (no usable laps)")
        return True

    df.to_parquet(output_path, index=False)
    print(f"    => Saved {output_path.name}  ({len(df):,} laps)")
    _record(df, year, race_name, int(df["RoundNumber"].iloc[0]), "ok (demo)")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("  F1 Race Rewind — Data Pipeline")
    print("=" * 60)

    # ── Training data ────────────────────────────────────────────────────────
    print("\n[1/2] Training data (2018–2024)\n")
    for year in TRAINING_YEARS:
        print(f"  ── {year} ──────────────────────────────")
        if not process_year(year):
            print("\n  Stopped early due to rate limit.")
            print("  Progress saved. Re-run to continue from here.")
            _print_summary()
            return

    # ── Demo races ───────────────────────────────────────────────────────────
    print("\n[2/2] Demo races\n")
    for year, race_name, filename in DEMO_RACES:
        if not process_demo_race(year, race_name, filename):
            print("\n  Stopped early due to rate limit.")
            print("  Progress saved. Re-run to continue from here.")
            _print_summary()
            return

    _print_summary()


def _print_summary() -> None:
    _flush_summary()
    summary_df = pd.DataFrame(summary_records)
    summary_path = BASE_DIR / "data_summary.csv"

    print("\n" + "=" * 60)
    print(f"  Summary saved → {summary_path}")
    total   = len(summary_df)
    ok      = summary_df["Status"].str.startswith("ok").sum()
    failed  = summary_df["Status"].str.startswith("FAILED").sum()
    skipped = total - ok - failed
    print(f"  Races: {total} total | {ok} ok | {failed} failed | {skipped} skipped")
    print("=" * 60)


if __name__ == "__main__":
    main()
