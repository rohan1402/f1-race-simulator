"""
F1 Race Rewind — Model Training
=================================
Trains two LightGBM models and saves them alongside their encoders.

Models
------
  Model A  laptime_model.pkl    LightGBM regressor  → predicted lap time (seconds)
  Model B  pit_model.pkl        LightGBM classifier → probability driver pits this lap

Outputs (all saved to models/)
-------------------------------
  laptime_model.pkl
  pit_model.pkl
  encoders.pkl          circuit_map + feature column lists (needed at inference time)
  training_report.txt   human-readable evaluation summary

Usage
-----
  python3 models/train.py
"""

import joblib
import time
import warnings
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                              roc_auc_score, average_precision_score,
                              classification_report)

warnings.filterwarnings("ignore")

# Add project root to path so feature_engineering imports cleanly
import sys
sys.path.insert(0, str(Path(__file__).parent))

from feature_engineering import (
    load_training_data, build_features, train_val_split,
    LAPTIME_FEATURES, PIT_FEATURES,
)

MODELS_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# LightGBM hyperparameters — intentionally conservative (no overfitting risk)
# ---------------------------------------------------------------------------
LAPTIME_PARAMS = {
    "objective":        "regression",
    "metric":           "mae",
    "n_estimators":     800,
    "learning_rate":    0.05,
    "num_leaves":       63,
    "min_child_samples": 30,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "reg_alpha":        0.1,
    "reg_lambda":       1.0,
    "n_jobs":           -1,
    "verbose":          -1,
    "random_state":     42,
}

PIT_PARAMS = {
    "objective":         "binary",
    "metric":            "binary_logloss",
    "n_estimators":      600,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "min_child_samples": 50,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "scale_pos_weight":  27,   # ~3.5% pit rate → weight minority class
    "n_jobs":            -1,
    "verbose":           -1,
    "random_state":      42,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    print(f"\n{'─' * 55}")
    print(f"  {title}")
    print(f"{'─' * 55}")


def _feature_importance_top(model, feature_names: list, n: int = 10) -> str:
    imp = pd.Series(model.feature_importances_, index=feature_names)
    top = imp.sort_values(ascending=False).head(n)
    lines = [f"    {f:<22} {v:>8.0f}" for f, v in top.items()]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Model A — Lap Time Regressor
# ---------------------------------------------------------------------------

def train_laptime_model(train: pd.DataFrame,
                        val: pd.DataFrame,
                        report_lines: list) -> lgb.LGBMRegressor:
    _print_section("Model A — Lap Time Predictor (LightGBM Regressor)")

    # Remove pit-in / pit-out laps from training target
    # They are ~34s slower and would bias the "normal pace" prediction.
    # PitInLap_int / PitOutLap_int are still FEATURES — the model learns the penalty.
    # But we don't want outlier SC laps (already filtered in pipeline) contaminating.
    train_clean = train[train["TrackStatus"] != 5].copy()  # exclude red flags
    val_clean   = val[val["TrackStatus"]   != 5].copy()

    X_train = train_clean[LAPTIME_FEATURES]
    y_train = train_clean["LapTime"]
    X_val   = val_clean[LAPTIME_FEATURES]
    y_val   = val_clean["LapTime"]

    print(f"  Train size : {len(X_train):,} laps")
    print(f"  Val size   : {len(X_val):,}  laps")
    print(f"  Features   : {len(LAPTIME_FEATURES)}")
    print(f"  Training...")

    t0 = time.time()
    model = lgb.LGBMRegressor(**LAPTIME_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    elapsed = time.time() - t0

    preds = model.predict(X_val)
    mae  = mean_absolute_error(y_val, preds)
    rmse = np.sqrt(mean_squared_error(y_val, preds))

    # MAE by tyre life bucket (key diagnostic)
    val_clean = val_clean.copy()
    val_clean["pred"] = preds
    val_clean["abs_err"] = (val_clean["pred"] - val_clean["LapTime"]).abs()
    bins = [0, 5, 15, 25, 35, 100]
    labels = ["1-5", "6-15", "16-25", "26-35", "35+"]
    val_clean["TL_bucket"] = pd.cut(val_clean["TyreLife"], bins=bins, labels=labels)
    mae_by_tl = val_clean.groupby("TL_bucket", observed=True)["abs_err"].mean().round(3)

    # MAE by compound
    mae_by_comp = val_clean.groupby("Compound")["abs_err"].mean().round(3).sort_values()

    print(f"\n  ✅ Trained in {elapsed:.1f}s  |  Best iteration: {model.best_iteration_}")
    print(f"  MAE  : {mae:.3f}s")
    print(f"  RMSE : {rmse:.3f}s")
    print(f"\n  MAE by TyreLife:")
    for bucket, v in mae_by_tl.items():
        print(f"    Laps {bucket:<5}: {v:.3f}s")
    print(f"\n  MAE by Compound:")
    for comp, v in mae_by_comp.items():
        print(f"    {comp:<8}: {v:.3f}s")
    print(f"\n  Top features:")
    print(_feature_importance_top(model, LAPTIME_FEATURES))

    # Save to report
    report_lines += [
        "\n=== Model A: Lap Time Predictor ===",
        f"Train laps : {len(X_train):,}",
        f"Val laps   : {len(X_val):,}",
        f"Train time : {elapsed:.1f}s",
        f"MAE        : {mae:.3f}s",
        f"RMSE       : {rmse:.3f}s",
        "",
        "MAE by TyreLife bucket:",
        mae_by_tl.to_string(),
        "",
        "MAE by Compound:",
        mae_by_comp.to_string(),
        "",
        "Top feature importances:",
        _feature_importance_top(model, LAPTIME_FEATURES),
    ]

    return model


# ---------------------------------------------------------------------------
# Model B — Pit Decision Classifier
# ---------------------------------------------------------------------------

def train_pit_model(train: pd.DataFrame,
                    val: pd.DataFrame,
                    report_lines: list) -> lgb.LGBMClassifier:
    _print_section("Model B — Pit Decision Classifier (LightGBM Binary)")

    X_train = train[PIT_FEATURES]
    y_train = train["PitInLap"].astype(int)
    X_val   = val[PIT_FEATURES]
    y_val   = val["PitInLap"].astype(int)

    print(f"  Train size     : {len(X_train):,} laps  (pit rate: {y_train.mean()*100:.1f}%)")
    print(f"  Val size       : {len(X_val):,}  laps  (pit rate: {y_val.mean()*100:.1f}%)")
    print(f"  Features       : {len(PIT_FEATURES)}")
    print(f"  Training...")

    t0 = time.time()
    model = lgb.LGBMClassifier(**PIT_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )
    elapsed = time.time() - t0

    probs = model.predict_proba(X_val)[:, 1]
    preds_binary = (probs >= 0.5).astype(int)

    auc  = roc_auc_score(y_val, probs)
    ap   = average_precision_score(y_val, probs)

    # Calibration check: mean predicted probability vs actual rate
    mean_pred_prob = probs.mean()
    actual_rate    = y_val.mean()

    print(f"\n  ✅ Trained in {elapsed:.1f}s  |  Best iteration: {model.best_iteration_}")
    print(f"  ROC-AUC           : {auc:.4f}")
    print(f"  Avg Precision     : {ap:.4f}")
    print(f"  Mean pred prob    : {mean_pred_prob*100:.2f}%  (actual rate: {actual_rate*100:.2f}%)")
    print(f"\n  Classification report (threshold=0.5):")
    print(classification_report(y_val, preds_binary,
                                 target_names=["No Pit", "Pit"],
                                 digits=3))
    print(f"\n  Top features:")
    print(_feature_importance_top(model, PIT_FEATURES))

    # SC lap uplift check — model should assign high pit prob during SC
    val_sc  = val[val["TrackStatus"] == 4]
    val_clr = val[val["TrackStatus"] == 1]
    if len(val_sc) > 0 and len(val_clr) > 0:
        sc_prob  = model.predict_proba(val_sc[PIT_FEATURES])[:, 1].mean()
        clr_prob = model.predict_proba(val_clr[PIT_FEATURES])[:, 1].mean()
        print(f"\n  Mean pit prob SC laps  : {sc_prob*100:.1f}%")
        print(f"  Mean pit prob clear laps: {clr_prob*100:.1f}%")
        print(f"  SC uplift ratio         : {sc_prob/clr_prob:.1f}x  (expect >3x)")

    report_lines += [
        "\n=== Model B: Pit Decision Classifier ===",
        f"Train laps : {len(X_train):,}",
        f"Val laps   : {len(X_val):,}",
        f"Train time : {elapsed:.1f}s",
        f"ROC-AUC    : {auc:.4f}",
        f"Avg Prec   : {ap:.4f}",
        f"Mean pred pit prob : {mean_pred_prob*100:.2f}% (actual: {actual_rate*100:.2f}%)",
        "",
        "Top feature importances:",
        _feature_importance_top(model, PIT_FEATURES),
    ]

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 55)
    print("  F1 Race Rewind — Model Training")
    print("=" * 55)

    report_lines = ["F1 Race Rewind — Training Report", "=" * 40]

    # ── 1. Load data ─────────────────────────────────────────────────────────
    _print_section("Loading & Engineering Features")
    print("  Loading 2018–2024 training data...")
    df_raw = load_training_data(range(2018, 2025))

    print("  Building features...")
    df, circuit_map = build_features(df_raw)

    print("  Splitting train / val (2024 held out)...")
    train, val = train_val_split(df, val_year=2024)

    report_lines += [
        f"\nTotal laps  : {len(df):,}",
        f"Train laps  : {len(train):,}  (2018–2023)",
        f"Val laps    : {len(val):,}   (2024)",
        f"Circuits    : {len(circuit_map)}",
    ]

    # ── 2. Train models ───────────────────────────────────────────────────────
    laptime_model = train_laptime_model(train, val, report_lines)
    pit_model     = train_pit_model(train, val, report_lines)

    # ── 3. Save everything ────────────────────────────────────────────────────
    _print_section("Saving Models")

    encoders = {
        "circuit_map":       circuit_map,
        "laptime_features":  LAPTIME_FEATURES,
        "pit_features":      PIT_FEATURES,
    }

    joblib.dump(laptime_model, MODELS_DIR / "laptime_model.pkl")
    joblib.dump(pit_model,     MODELS_DIR / "pit_model.pkl")
    joblib.dump(encoders,      MODELS_DIR / "encoders.pkl")

    report_path = MODELS_DIR / "training_report.txt"
    report_path.write_text("\n".join(report_lines))

    print(f"  ✅ laptime_model.pkl  →  {(MODELS_DIR/'laptime_model.pkl').stat().st_size // 1024} KB")
    print(f"  ✅ pit_model.pkl      →  {(MODELS_DIR/'pit_model.pkl').stat().st_size // 1024} KB")
    print(f"  ✅ encoders.pkl")
    print(f"  ✅ training_report.txt")

    print("\n" + "=" * 55)
    print("  Training complete. Models ready for the simulator.")
    print("=" * 55)


if __name__ == "__main__":
    main()
