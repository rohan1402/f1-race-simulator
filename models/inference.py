"""
Model Inference Wrappers
==========================
Single import point for both models. Applies two post-hoc corrections to the
pit probability model:

  1. Global calibration  — scale_pos_weight=27 inflates all predictions.
                           Correct by the ratio of actual vs predicted mean rate.
                           Actual: 2.99%  |  Model mean: 5.80%  →  factor = 0.516

  2. TrackStatus boosts  — after calibration, SC and VSC laps are still under-
                           predicted relative to the true rates in the data.
                           Multipliers are derived from actual pit rates per status:

     Status   True rate   Calibrated model   Multiplier
     ------   ---------   ----------------   ----------
     Clear       2.47%          2.89%            1.0×
     Yellow      3.47%          ~2.5%            1.2×
     SC         11.01%          4.85%            2.3×
     Red           n/a            —              0.0×  (race stopped, no pits)
     VSC         9.18%          3.61%            2.6×

All simulator code imports from here — never directly from the .pkl files.
"""

from __future__ import annotations
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Correction constants (data-derived, see module docstring)
# ---------------------------------------------------------------------------
_CALIBRATION_FACTOR = 0.516   # overall probability deflator

_STATUS_BOOST: dict[int, float] = {
    1: 1.0,   # Clear
    2: 1.2,   # Yellow flag
    4: 2.3,   # Safety Car
    5: 0.0,   # Red flag — race stopped, no pitting
    6: 2.6,   # Virtual Safety Car
}


# ---------------------------------------------------------------------------
# Model loader (cached on first call)
# ---------------------------------------------------------------------------
_cache: dict = {}

def _load():
    if not _cache:
        _cache["laptime"] = joblib.load(MODELS_DIR / "laptime_model.pkl")
        _cache["pit"]     = joblib.load(MODELS_DIR / "pit_model.pkl")
        _cache["enc"]     = joblib.load(MODELS_DIR / "encoders.pkl")
    return _cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_encoders() -> dict:
    """Return the circuit_map and feature column lists."""
    return _load()["enc"]


def predict_laptime(features: pd.DataFrame) -> np.ndarray:
    """
    Predict lap time in seconds.

    Parameters
    ----------
    features : DataFrame with exactly the columns in encoders['laptime_features']

    Returns
    -------
    np.ndarray of shape (n,) — predicted lap times in seconds
    """
    m = _load()
    cols = m["enc"]["laptime_features"]
    return m["laptime"].predict(features[cols])


def predict_pit_prob(features: pd.DataFrame,
                     track_statuses: pd.Series | np.ndarray | None = None
                     ) -> np.ndarray:
    """
    Predict pit-stop probability with calibration + SC/VSC boost applied.

    Parameters
    ----------
    features       : DataFrame with columns in encoders['pit_features']
    track_statuses : 1-D array of TrackStatus ints, one per row.
                     If None, reads from features['TrackStatus'].

    Returns
    -------
    np.ndarray of shape (n,) — calibrated pit probabilities in [0, 1]
    """
    m = _load()
    cols = m["enc"]["pit_features"]

    # Raw model probability
    raw = m["pit"].predict_proba(features[cols])[:, 1]

    # Step 1 — global calibration
    calibrated = raw * _CALIBRATION_FACTOR

    # Step 2 — per-lap TrackStatus boost
    if track_statuses is None:
        track_statuses = features["TrackStatus"].values
    track_statuses = np.asarray(track_statuses, dtype=int)

    boosts = np.vectorize(lambda ts: _STATUS_BOOST.get(ts, 1.0))(track_statuses)
    adjusted = calibrated * boosts

    return np.clip(adjusted, 0.0, 0.95)


# ---------------------------------------------------------------------------
# Sanity check (run this file directly to verify corrections)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(MODELS_DIR))
    from feature_engineering import load_training_data, build_features

    print("Loading 2024 validation data for sanity check...")
    df_raw = load_training_data([2024])
    enc     = get_encoders()
    df, _   = build_features(df_raw, circuit_map=enc["circuit_map"])

    feats = df[enc["pit_features"]]
    ts    = df["TrackStatus"].values

    raw_probs  = _load()["pit"].predict_proba(feats)[:, 1]
    adj_probs  = predict_pit_prob(feats, ts)

    actual_rate = df["PitInLap"].mean() * 100

    print(f"\n{'Status':<8} {'True rate':>10} {'Raw model':>10} {'Adjusted':>10} {'Boost':>6}")
    print("─" * 48)
    status_labels = {1: "Clear", 2: "Yellow", 4: "SC", 5: "Red", 6: "VSC"}
    for code, label in status_labels.items():
        mask = ts == code
        if mask.sum() == 0:
            continue
        true_r = df.loc[mask, "PitInLap"].mean() * 100
        raw_r  = raw_probs[mask].mean() * 100
        adj_r  = adj_probs[mask].mean() * 100
        boost  = _STATUS_BOOST.get(code, 1.0)
        print(f"{label:<8} {true_r:>9.2f}%  {raw_r:>9.2f}%  {adj_r:>9.2f}%  {boost:>5.1f}×")

    print(f"\nOverall actual pit rate : {actual_rate:.2f}%")
    print(f"Overall adjusted mean   : {adj_probs.mean()*100:.2f}%")

    sc_mask  = ts == 4
    clr_mask = ts == 1
    if sc_mask.sum() and clr_mask.sum():
        sc_adj  = adj_probs[sc_mask].mean()
        clr_adj = adj_probs[clr_mask].mean()
        print(f"\nAdjusted SC/Clear ratio : {sc_adj/clr_adj:.1f}×  (target: 4.5×)")
