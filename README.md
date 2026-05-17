# 🏎 F1 Race Rewind

A what-if F1 race strategy simulator. Pick a historical race, choose a driver, apply a strategy intervention (pit earlier, pit later, or swap compounds), and watch how the race would have unfolded differently — lap by lap on the actual circuit map.

Live Demo: https://f1-race-simulator-adv-programming.streamlit.app/
---

## What Problem Does It Solve?

F1 race strategy is one of the most debated topics in motorsport. After every race, fans and analysts ask: *"What if Hamilton had pitted under the Safety Car?"* or *"What if Leclerc had stayed out on harder tyres?"*

There is no tool that lets you actually simulate those decisions and see a lap-by-lap outcome. F1 Race Rewind fills that gap — it takes real historical race data, lets you fork the strategy at any lap, and simulates the rest of the race using a machine learning model trained on real F1 telemetry.

---

## Who Is It For?

- F1 fans who want to explore "what-if" scenarios from famous races
- Students or analysts studying race strategy
- Anyone curious about how a single pit stop decision can swing a championship

---

## Demo Scenario

**Abu Dhabi 2021 — the controversial finale.**
Hamilton led on old Hard tyres when the Safety Car deployed on lap 52. Verstappen pitted for fresh Softs and won the championship on the last lap.

> Race: **2021 Abu Dhabi Grand Prix** · Driver: **HAM** · Fork: **52** · Pit Earlier · Lap **53** · **SOFT**

---

## Setup & Installation

**Requirements:** Python 3.10+

### 1. Clone the repo

```bash
git clone https://github.com/rohan1402/f1-race-simulator.git
cd f1-race-simulator
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the app

```bash
streamlit run app.py
```

The app opens automatically at **http://localhost:8501** in your browser. No login, no API key, no internet connection required after the first run.

> **Note:** The first time you run a simulation, FastF1 may download a small cache file for that race session (~5 MB). Subsequent runs use the local cache and are instant.

---

## How to Use

### Step 1 — Pick a Race
In the left sidebar, select one of the 6 available demo races from the dropdown.

### Step 2 — Choose a Driver
Select the driver whose strategy you want to change.

### Step 3 — Set the Fork Lap
Use the slider to choose which lap the simulation diverges from history. Everything before this lap uses real data. Everything after is simulated.

### Step 4 — Choose an Intervention
Pick one of three strategy interventions:
| Intervention | What It Does |
|---|---|
| **Pit Earlier** | Forces a pit stop on a specific lap with a chosen compound |
| **Pit Later** | Blocks the driver from pitting until a later lap |
| **Change Compound** | Pits the driver and fits a different tyre type |

### Step 5 — Run the Simulation
Click **🚀 Run Simulation**. The app will:
- Re-simulate the race from the fork lap onward
- Show a side-by-side animated circuit replay (Actual vs What-If)
- Display position changes, a gap-to-leader chart, and a final standings table

### Inputs & Outputs

| Input | Description |
|---|---|
| Race | Historical race from the 6 available options |
| Driver | Any driver who competed in that race |
| Fork lap | Lap number where the strategy change takes effect |
| Intervention type | Pit Earlier / Pit Later / Change Compound |

| Output | Description |
|---|---|
| Animated replay | Cars moving around the real circuit map, lap by lap |
| Result banner | Summary of position and gap change |
| Gap chart | Driver's gap to leader over the race (actual vs what-if) |
| Standings table | Full final standings comparison |

---

## Features

### Animated Circuit Replay
Cars are positioned on the real GPS-traced circuit layout for each race (extracted from FastF1 telemetry). Position on track is calculated from each car's time gap to the leader, so a car 45 seconds behind on a 90-second lap appears halfway around the circuit. Both the actual race and the what-if scenario play side by side.

### ML-Powered Lap Time Prediction
After the fork lap, the simulated driver's lap times are predicted by a gradient-boosted regression model trained on historical F1 telemetry. Features include tyre compound, tyre age, lap number, and pit status. All other drivers follow their exact historical strategies — only the chosen driver's path changes.

### Strategy Interventions
Three intervention types cover the most common strategic decisions: pitting early under a safety car, extending a stint to gain track position, or fitting a different compound to target faster lap times late in the race.

### 6 Demo Races
Curated races that each showcase a different strategic scenario — from safety car pile-ups (Abu Dhabi 2021, Brazil 2022) to tight street circuits (Monaco 2022) to high-speed tracks where tyre degradation dominates (Monza 2023).

---

## Limitations & Assumptions

- **Field drivers follow history.** Only the selected driver's strategy changes. All other drivers continue their actual historical strategies regardless of how the simulated driver moves around them. Real-world reactive strategy (e.g., a team pitting in reaction to the simulated driver) is not modelled.
- **Lap times are predicted, not exact.** The ML model approximates performance but cannot account for safety cars, mechanical issues, or traffic in the simulated scenario.
- **Gap-to-leader is cumulative.** Positions are determined by total elapsed time from lap 1. Sector-level passing is not simulated — if the model gives the driver a faster lap, they gain time; if slower, they lose it.
- **6 races available.** The simulator is built around pre-loaded demo races. Adding new races requires running the data pipeline.
- **No multiplayer / authentication.** The app runs locally for a single user. There are no user accounts, roles, or login requirements.

---

## Project Structure

```
app.py                    # Streamlit UI + animated circuit builder
simulator/
  engine.py               # Core simulation logic
models/
  train.py                # Model training pipeline
  inference.py            # Lap time + pit probability predictions
  feature_engineering.py  # Feature construction
  compound_rules.py       # Tyre compound strategy rules
  laptime_model.pkl        # Trained lap time model
  pit_model.pkl            # Trained pit probability model
  encoders.pkl             # Label encoders
data/
  demo/                   # Pre-built race parquet files (6 races)
  tracks/                 # Circuit outlines — 300-point GPS paths (JSON)
scripts/
  extract_tracks.py       # One-time: FastF1 telemetry → track JSON
pipeline.py               # End-to-end data collection + training pipeline
requirements.txt          # Python dependencies
```

---

## Races Available

| Race | Season | Circuit | Key Scenario |
|------|--------|---------|--------------|
| Abu Dhabi Grand Prix | 2021 | Yas Marina | Safety car strategy swing |
| Monaco Grand Prix | 2022 | Monte Carlo | Undercut on a street circuit |
| Italian Grand Prix | 2023 | Monza | Tyre degradation at high speed |
| German Grand Prix | 2019 | Hockenheim | Wet weather compound choice |
| Las Vegas Grand Prix | 2023 | Las Vegas Strip | Aggressive early pit strategy |
| Brazilian Grand Prix | 2022 | Interlagos | Safety car pit window |
