"""
Rule-Based Tyre Compound Recommender
======================================
Determines which compound a driver should switch to when pitting.

Why rules instead of ML:
  - Compound choice is heavily constrained by FIA regulations
    (must use 2 different dry compounds per race)
  - Training data is dominated by team allocation rules, not driver/strategy signal
  - A logistic model would learn circuit→compound correlations that don't generalise
    to counterfactual scenarios where a driver is pitting at an unusual lap

Logic:
  1. WET/INTER conditions → stay on appropriate wet tyre
  2. Already used both required compounds → pick fastest (SOFT)
  3. First stop from SOFT → switch to MEDIUM or HARD (depends on laps remaining)
  4. First stop from MEDIUM/HARD → switch to SOFT if <20 laps remain, else MEDIUM
  5. Fallback → use circuit historical prior from training data
"""

from __future__ import annotations
import pandas as pd

# Default stint lengths (laps) per compound — used to judge if a compound
# can realistically last to the end of the race
COMPOUND_LIFE = {"SOFT": 20, "MEDIUM": 32, "HARD": 45, "INTER": 30, "WET": 30}

# Dry compounds in preference order (fastest → most durable)
DRY_ORDER = ["SOFT", "MEDIUM", "HARD"]


def suggest_compound(
    current_compound: str,
    tyre_life: int,
    laps_remaining: int,
    compounds_used: list[str],
    track_status: int = 1,
    circuit_priors: dict | None = None,
) -> str:
    """
    Recommend the next tyre compound for a pitting driver.

    Parameters
    ----------
    current_compound : compound being removed
    tyre_life        : how many laps it has been used
    laps_remaining   : laps left in the race after this stop
    compounds_used   : list of all dry compounds this driver has used so far
                       (including current_compound)
    track_status     : 1=clear, 2=yellow, 4=SC, 5=red, 6=VSC
    circuit_priors   : {(outgoing_compound, stop_number): most_common_incoming}
                       pre-computed from training data for this circuit (optional)

    Returns
    -------
    str : recommended compound (e.g. "SOFT")
    """
    # ── Wet / Inter conditions ─────────────────────────────────────────────
    if current_compound in ("INTER", "WET"):
        # Switching back to dry — pick MEDIUM as safe default
        return "MEDIUM"

    # ── If track status suggests rain → go INTER ───────────────────────────
    # (Edge case: user triggers a rain counterfactual in the app)
    # Not modelled here — frontend should prevent wet-tyre interventions.

    # ── Normalise compounds_used set ──────────────────────────────────────
    dry_used = set(c for c in compounds_used if c in DRY_ORDER)

    # ── Must-use rule: FIA requires 2 different dry compounds ─────────────
    # If only one dry compound has been used so far, force a different one
    unused_dry = [c for c in DRY_ORDER if c not in dry_used]

    # ── Laps-remaining based logic ─────────────────────────────────────────
    if laps_remaining <= 0:
        return "SOFT"  # sprint to finish

    if laps_remaining <= 15:
        # Short stint remaining — pick SOFT regardless
        # (unless SOFT already used and must use something new)
        if "SOFT" in dry_used and len(dry_used) < 2:
            return unused_dry[0] if unused_dry else "SOFT"
        return "SOFT"

    if laps_remaining <= COMPOUND_LIFE["MEDIUM"]:
        # MEDIUM can make it to the end
        if current_compound == "SOFT":
            return "MEDIUM"
        if current_compound == "MEDIUM":
            return "SOFT" if laps_remaining <= 20 else "HARD"
        if current_compound == "HARD":
            return "SOFT" if laps_remaining <= 20 else "MEDIUM"

    # Long stint remaining — prefer HARD or MEDIUM
    if current_compound == "SOFT":
        # Must use different compound; HARD for long stints
        return "HARD" if laps_remaining > COMPOUND_LIFE["MEDIUM"] else "MEDIUM"
    if current_compound in ("MEDIUM", "HARD"):
        return "SOFT" if laps_remaining <= 20 else "MEDIUM"

    # ── Fallback: circuit historical prior ────────────────────────────────
    if circuit_priors and current_compound in circuit_priors:
        return circuit_priors[current_compound]

    return "MEDIUM"  # safe neutral default


def build_circuit_priors(race_df: pd.DataFrame) -> dict[str, str]:
    """
    From historical lap data for a single race, build a mapping:
        outgoing_compound → most common incoming compound

    Used as the circuit_priors argument to suggest_compound().
    """
    pit_laps = race_df[race_df["PitInLap"]].copy()
    if pit_laps.empty:
        return {}

    # The "incoming" compound is on the lap after the pit
    race_df_sorted = race_df.sort_values(["Driver", "LapNumber"])
    race_df_sorted["NextCompound"] = (
        race_df_sorted.groupby("Driver")["Compound"].shift(-1)
    )

    pit_rows = race_df_sorted[race_df_sorted["PitInLap"] & race_df_sorted["NextCompound"].notna()]

    priors = {}
    for out_comp, grp in pit_rows.groupby("Compound"):
        most_common = grp["NextCompound"].mode()
        if not most_common.empty:
            priors[out_comp] = most_common.iloc[0]

    return priors


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    tests = [
        # (current, life, laps_rem, used, expected_ish)
        ("SOFT",   18, 35, ["SOFT"],             "HARD or MEDIUM"),
        ("MEDIUM", 28, 15, ["SOFT", "MEDIUM"],   "SOFT"),
        ("HARD",   40,  8, ["MEDIUM", "HARD"],   "SOFT"),
        ("SOFT",   10, 50, ["SOFT"],             "HARD"),
        ("INTER",  12, 25, ["INTER"],            "MEDIUM"),
    ]
    print("Compound recommender tests:")
    for cur, life, rem, used, hint in tests:
        result = suggest_compound(cur, life, rem, used)
        print(f"  {cur} (life={life}, rem={rem}) → {result}  (expect ~{hint})")
