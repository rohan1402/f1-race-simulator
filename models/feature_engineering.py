"""
Feature Engineering for F1 Race Rewind
========================================
Builds the feature matrix used by both ML models from the cleaned parquet data.

Key design decisions:
- All rolling features use .shift(1) to prevent data leakage (lap N only sees laps < N)
- TyreLife² captures the non-linear degradation cliff (SOFT ~lap 20, HARD ~lap 35)
- Circuit and Compound are label-encoded; mappings saved alongside models
- TrackStatus left as raw int — tree models handle non-ordinal splits naturally
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR  = Path(__file__).parent.parent / "data"
TRAIN_DIR = BASE_DIR / "training"
DEMO_DIR  = BASE_DIR / "demo"

# ---------------------------------------------------------------------------
# Encodings (built once, reused at inference)
# ---------------------------------------------------------------------------
COMPOUND_ENC = {
    "SOFT": 0, "MEDIUM": 1, "HARD": 2,
    "INTER": 3, "WET": 4,
    "NONE": 2, "UNKNOWN": 2,        # treat as HARD (neutral)
}

# Features used by Model A (lap time predictor)
LAPTIME_FEATURES = [
    "TyreLife", "TyreLife_sq",
    "Compound_enc",
    "PitInLap_int", "PitOutLap_int",
    "TrackStatus",
    "LapNumber", "LapsRemaining",
    "Position",
    "GapToLeader",
    "RollingMean3", "RollingMean5",
    "Season",
    "Circuit_enc",
]

# Features used by Model B (pit decision classifier)
PIT_FEATURES = [
    "TyreLife", "TyreLife_sq",
    "Compound_enc",
    "Position",
    "GapToLeader",
    "LapsRemaining",
    "TotalPitStops",
    "TrackStatus",
    "Season",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_training_data(years: range | list = range(2018, 2024)) -> pd.DataFrame:
    """Load and concatenate parquet files for the given years."""
    dfs = []
    for year in years:
        p = TRAIN_DIR / f"{year}_all_races.parquet"
        if p.exists():
            dfs.append(pd.read_parquet(p))
        else:
            print(f"  WARNING: {p.name} not found, skipping.")
    if not dfs:
        raise FileNotFoundError("No training parquet files found.")
    df = pd.concat(dfs, ignore_index=True)
    print(f"  Loaded {len(df):,} laps across {df['RaceName'].nunique()} races "
          f"({df['Season'].min()}–{df['Season'].max()})")
    return df


def load_demo_data(filename: str) -> pd.DataFrame:
    """Load a single demo race parquet by filename."""
    return pd.read_parquet(DEMO_DIR / filename)


# ---------------------------------------------------------------------------
# Core feature builder
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame,
                   circuit_map: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """
    Add all engineered features to the DataFrame.

    Parameters
    ----------
    df          : raw lap DataFrame (output of pipeline.py)
    circuit_map : existing circuit→int mapping (pass None to build fresh)

    Returns
    -------
    df          : enriched DataFrame
    circuit_map : the circuit encoding dict (save this alongside the model)
    """
    df = df.copy()

    # Ensure correct sort order for rolling features
    df = df.sort_values(["Season", "RaceName", "Driver", "LapNumber"]).reset_index(drop=True)

    # ── Total laps per race (for LapsRemaining) ──────────────────────────────
    total_laps = (df.groupby(["Season", "RaceName"])["LapNumber"]
                    .transform("max"))
    df["LapsRemaining"] = total_laps - df["LapNumber"]

    # ── TyreLife² ────────────────────────────────────────────────────────────
    df["TyreLife_sq"] = df["TyreLife"] ** 2

    # ── Compound encoding ────────────────────────────────────────────────────
    df["Compound_enc"] = df["Compound"].map(COMPOUND_ENC).fillna(2).astype(int)

    # ── Pit flags as ints ────────────────────────────────────────────────────
    df["PitInLap_int"]  = df["PitInLap"].astype(int)
    df["PitOutLap_int"] = df["PitOutLap"].astype(int)

    # ── Position: fill rare NaNs ─────────────────────────────────────────────
    df["Position"] = df["Position"].fillna(10.0)

    # ── Rolling mean lap times (shift-1 to prevent leakage) ─────────────────
    # Groups: same driver, same race — rolling window over lap sequence
    grp = df.groupby(["Season", "RaceName", "Driver"])["LapTime"]

    df["RollingMean3"] = (grp.transform(lambda x: x.shift(1).rolling(3,  min_periods=1).mean())
                            .fillna(df["LapTime"]))
    df["RollingMean5"] = (grp.transform(lambda x: x.shift(1).rolling(5,  min_periods=1).mean())
                            .fillna(df["LapTime"]))

    # ── Circuit label encoding ────────────────────────────────────────────────
    if circuit_map is None:
        circuits = sorted(df["RaceName"].unique())
        circuit_map = {c: i for i, c in enumerate(circuits)}
    df["Circuit_enc"] = df["RaceName"].map(circuit_map).fillna(0).astype(int)

    return df, circuit_map


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

def train_val_split(df: pd.DataFrame,
                    val_year: int = 2024
                    ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split by season: everything before val_year is train, val_year is val."""
    train = df[df["Season"] < val_year].reset_index(drop=True)
    val   = df[df["Season"] == val_year].reset_index(drop=True)
    print(f"  Train: {len(train):,} laps ({train['Season'].min()}–{train['Season'].max()-1 if len(train) else '?'})")
    print(f"  Val:   {len(val):,}  laps ({val_year})")
    return train, val


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("Loading data...")
    df_raw = load_training_data(range(2018, 2025))

    print("\nBuilding features...")
    df_feat, cmap = build_features(df_raw)

    print(f"\nFeature columns present:")
    for col in LAPTIME_FEATURES + PIT_FEATURES:
        ok = "✅" if col in df_feat.columns else "❌"
        print(f"  {ok}  {col}")

    print(f"\nCircuit map has {len(cmap)} circuits")
    print(f"NaN check (should all be 0):")
    check_cols = LAPTIME_FEATURES + ["LapTime", "PitInLap"]
    for c in check_cols:
        n = df_feat[c].isna().sum() if c in df_feat.columns else "MISSING"
        if n != 0:
            print(f"  ⚠️  {c}: {n} NaNs")
    print("  Done.")
