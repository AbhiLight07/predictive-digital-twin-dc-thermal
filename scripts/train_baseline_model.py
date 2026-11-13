#!/usr/bin/env python3
"""Train and persist a verified baseline LightGBM model for offline inference in Research Studio.

Follows the verified v2/v3 protocol:
- Dataset 1 (Cluster 1: Dell PowerEdge R840)
- Split A (Chronological: 70% Train, 10% Validation, 6h Embargo, 20% Test)
- Feature set: S1 (Temperature lags & current state, 36 features)
- Target: 1-hour delta target Delta_T = T(t+6) - T(t)
- Model: LightGBM regressor with early stopping on validation partition
- Output: models/dataset1_split_a_s1_lightgbm_1h.joblib and companion metadata JSON.
"""

import sys
from pathlib import Path
from datetime import datetime, timezone
import json
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import split_chronological
from ttwin.metrics import mae, rmse, skill_score_vs_persistence

class LightGBMPredictor:
    """Lightweight pickleable predictor wrapper."""
    def __init__(self, booster, feature_names):
        self.booster = booster
        self.feature_names = feature_names

    def predict(self, X):
        return self.booster.predict(X)


OUT_DIR = ROOT / "models"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def train_s1_model(horizon_steps: int = 6):
    print("=" * 70)
    print("TRAINING GENUINE LIGHTGBM S1 MODEL CHECKPOINT")
    print(f"Target: CPU_Temperature_delta_{horizon_steps} (Horizon: {horizon_steps * 10} min)")
    print("=" * 70)

    # 1. Load data
    data_path = ROOT / "data" / "interim" / "dataset1" / "_combined.parquet"
    print(f"Loading {data_path}...")
    df = load_parquet(data_path)
    df["Series"] = df["Series"].astype(str).str.strip()

    # 2. Build features
    print("Building S1 features...")
    df_feat = build_features(df, horizons=[horizon_steps], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])

    target_delta = f"CPU_Temperature_delta_{horizon_steps}"
    target_abs = f"CPU_Temperature_target_{horizon_steps}"
    feature_cols = get_feature_set("S1", df_feat.columns.tolist())
    print(f"S1 Feature count: {len(feature_cols)}")

    # 3. Chronological split (70% train, 10% val, 20% test with 36 step embargo)
    print("Splitting chronologically (Split A)...")
    train_df, val_df, test_df = split_chronological(df_feat, train_frac=0.70, val_frac=0.10, embargo_steps=36)

    # Clean missing values
    needed_cols = list(dict.fromkeys(feature_cols + [target_delta, target_abs, "CPU_Temperature", "Series", "Time"]))
    train_clean = train_df[needed_cols].dropna().reset_index(drop=True)
    val_clean = val_df[needed_cols].dropna().reset_index(drop=True)
    test_clean = test_df[needed_cols].dropna().reset_index(drop=True)

    print(f"Train rows: {len(train_clean):,}")
    print(f"Validation rows: {len(val_clean):,}")
    print(f"Test rows: {len(test_clean):,}")

    X_train = train_clean[feature_cols].values
    y_train = train_clean[target_delta].values
    X_val = val_clean[feature_cols].values
    y_val = val_clean[target_delta].values
    X_test = test_clean[feature_cols].values
    y_test_abs = test_clean[target_abs].values
    y_test_curr = test_clean["CPU_Temperature"].values

    # 4. Train LightGBM booster
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=feature_cols)

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "verbosity": -1,
        "seed": 42,
    }

    print("Fitting LightGBM booster...")
    booster = lgb.train(
        params,
        dtrain,
        num_boost_round=200,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)],
    )

    # 5. Evaluate on held-out test partition
    pred_delta = booster.predict(X_test)
    pred_abs = y_test_curr + pred_delta

    test_mae = mae(y_test_abs, pred_abs)
    test_rmse = rmse(y_test_abs, pred_abs)
    pers_mae = mae(y_test_abs, y_test_curr)
    skill = skill_score_vs_persistence(test_mae, pers_mae)

    print("\n--- TEST EVALUATION ON HELD-OUT SPLIT A TEST WINDOW ---")
    print(f"Model Test MAE: {test_mae:.4f} °C")
    print(f"Persistence Test MAE: {pers_mae:.4f} °C")
    print(f"Skill Score: {skill * 100:+.2f}%")

    # 6. Save checkpoint and metadata
    model_name = f"dataset1_split_a_s1_lightgbm_{horizon_steps * 10}min"
    model_path = OUT_DIR / f"{model_name}.joblib"
    meta_path = OUT_DIR / f"{model_name}.json"

    joblib.dump(booster, model_path)
    # Also save native lightgbm txt model
    booster.save_model(str(model_path.with_suffix(".txt")))
    print(f"\nSaved model checkpoint to: {model_path}")

    metadata = {
        "name": model_name,
        "dataset": "dataset1",
        "split": "Split A (Chronological)",
        "feature_set": "S1",
        "target_type": "delta",
        "horizon_steps": horizon_steps,
        "horizon_label": f"{horizon_steps * 10}min",
        "model_class": "LightGBM (Booster)",
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "persistence_mae": float(pers_mae),
        "skill_pct": float(skill * 100),
        "feature_names": feature_cols,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": "verified_offline_training",
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved checkpoint metadata to: {meta_path}")

    return model_path


if __name__ == "__main__":
    train_s1_model()
