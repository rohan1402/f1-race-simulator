# 🏎 F1 Race Rewind

A what-if F1 race strategy simulator. Pick a historical race, choose a driver, apply a strategy intervention (pit earlier, pit later, or swap compounds), and watch how the race would have unfolded differently — lap by lap on the actual circuit map.

---

## Demo Scenario

**Abu Dhabi 2021 — the controversial finale.**
Hamilton led on old Hard tyres when the Safety Car deployed on lap 52. Verstappen pitted for fresh Softs and won the championship on the last lap.

> Race: **2021 Abu Dhabi Grand Prix** · Driver: **HAM** · Fork: **52** · Pit Earlier · Lap **53** · **SOFT**

---

## Features

- **Animated circuit replay** — cars move around the real GPS-traced circuit layout for each race
- **Side-by-side comparison** — Actual race and What-If scenario rendered simultaneously
- **Strategy interventions** — Pit Earlier, Pit Later, or Change Compound
- **ML-powered simulation** — lap times predicted by a gradient-boosted model trained on historical F1 telemetry; field drivers follow their real historical strategies
- **6 demo races** — Abu Dhabi 2021, Monaco 2022, Monza 2023, Hockenheim 2019, Las Vegas 2023, Brazil 2022

---

## Setup

```bash
pip install streamlit fastf1 pandas numpy scikit-learn plotly pyarrow
streamlit run app.py
```

---

## Project Structure

```
app.py                  # Streamlit UI + animation builder
simulator/engine.py     # Simulation logic
models/
  train.py              # Model training pipeline
  inference.py          # Lap time + pit probability predictions
  feature_engineering.py
  compound_rules.py
data/
  demo/                 # Pre-built race parquet files
  tracks/               # Circuit outlines (300-pt GPS paths, JSON)
scripts/
  extract_tracks.py     # One-time FastF1 telemetry → track JSON
pipeline.py             # End-to-end data + training pipeline
```

---

## How the Simulation Works

1. **Before the fork lap** — all drivers follow exact historical data
2. **At the fork lap** — the chosen driver's strategy diverges per the intervention
3. **After the fork lap** — the hero driver's lap times are predicted by the ML model; all other drivers continue on their actual historical strategies
4. **Positions** are recalculated each lap based on cumulative time

---

## Races Available

| Race | Season | Circuit |
|------|--------|---------|
| Abu Dhabi Grand Prix | 2021 | Yas Marina |
| Monaco Grand Prix | 2022 | Monte Carlo |
| Italian Grand Prix | 2023 | Monza |
| German Grand Prix | 2019 | Hockenheim |
| Las Vegas Grand Prix | 2023 | Las Vegas Strip |
| Brazilian Grand Prix | 2022 | Interlagos |
