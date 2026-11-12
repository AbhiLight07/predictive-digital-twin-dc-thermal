#!/usr/bin/env python3
"""v2 Experiment E1: Unified Ablation Across Datasets, Splits, and Horizons.

Corrected protocol:
- Target: Delta target Delta_T = T(t+h) - T(t) for S1 and S4 (evaluating T(t) + Delta_T).
- Features: Corrected features including current-state T(t) in S1 and S4.
- Splits:
    - Split A: Chronological 70% train / 10% val / 20% test with 6h embargo.
    - Split B: 5-Fold GroupKFold by server (ALL servers tested across folds).
    - Split C: Strict holdout (unseen servers + later time).
    - Random 5-fold CV "reference" replication (to measure leakage inflation).
- Models: Persistence baseline, Ridge (delta target), LightGBM (delta target).
- Both Dataset 1 (Dell R840) and Dataset 2 (Dell C6320).
- Identical rows evaluated within each horizon.
- SHAP analysis on best model.

Outputs:
- experiments/v2_ablation/results_split_a.csv
- experiments/v2_ablation/results_split_b_groupkfold.csv
- experiments/v2_ablation/results_split_c.csv
- experiments/v2_ablation/results_random_kfold_reference.csv
- experiments/v2_ablation/shap_summary.csv
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

import os
os.environ["MPLCONFIGDIR"] = os.environ.get("TEMP", "C:/tmp")

from pathlib import Path
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold
import shap

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import (
    split_chronological,
    get_group_kfold_splits,
    split_strict,
)
from ttwin.metrics import mae, rmse, skill_score_vs_persistence

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--out-prefix", default="v2", help="Prefix for output experiment directory, e.g. v2 or v3")
args, _ = parser.parse_known_args()

OUT_DIR = ROOT / "experiments" / f"{args.out_prefix}_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_DIR = ROOT / "data" / "interim"
HORIZONS = [1, 6, 36]  # 10min, 1h, 6h
HORIZON_LABELS = {1: "10min", 6: "1h", 36: "6h"}


def per_server_metrics(y_true, y_pred, servers):
    df_err = pd.DataFrame({"true": y_true, "pred": y_pred, "server": servers})
    df_err["err"] = np.abs(df_err["true"] - df_err["pred"])
    server_maes = df_err.groupby("server")["err"].mean()
    med_mae = float(server_maes.median())
    worst_decile = float(server_maes.quantile(0.90))
    return med_mae, worst_decile


def prepare_data(dataset_name: str):
    path = DATA_DIR / dataset_name / "_combined.parquet"
    print(f"Loading {dataset_name} from {path}...")
    df = load_parquet(path)
    df["Series"] = df["Series"].astype(str).str.strip()

    # Filter servers for Dataset 2 according to gate (< 20% missing on used columns)
    if dataset_name == "dataset2":
        # Disqualify servers with extreme missingness
        disqual = {"qh2-rcc263", "qh2-rcc264", "qh2-rcc226", "qh2-rcc205"}
        df = df[~df["Series"].isin(disqual)].copy()

    print(f"Building features for {dataset_name} ({len(df):,} rows, {df['Series'].nunique()} servers)...")
    df = build_features(
        df,
        horizons=HORIZONS,
        lags=[1, 2, 3, 6, 12],
        rolling_windows=[6, 12, 36],
    )
    return df


def run_split_a(df: pd.DataFrame, dataset_name: str, feature_sets: list):
    print(f"\n{'='*70}\nRUNNING SPLIT A — {dataset_name}\n{'='*70}")
    train_df, val_df, test_df = split_chronological(df, train_frac=0.70, val_frac=0.10, embargo_steps=36)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    results = []

    for h in HORIZONS:
        h_label = HORIZON_LABELS[h]
        target_abs = f"CPU_Temperature_target_{h}"
        target_delta = f"CPU_Temperature_delta_{h}"

        # Clean test set on identical rows for all feature sets
        all_req_cols = set(["CPU_Temperature", target_abs, target_delta, "Series", "Time"])
        for fs in feature_sets:
            all_req_cols.update(get_feature_set(fs, df.columns.tolist()))
        avail_req = [c for c in all_req_cols if c in df.columns]

        test_clean = test_df[avail_req].dropna()
        n_rows = len(test_clean)
        print(f"\n--- Horizon {h_label} ({n_rows:,} identical test rows) ---")

        y_test_abs = test_clean[target_abs].values
        y_test_curr = test_clean["CPU_Temperature"].values
        test_servers = test_clean["Series"].values

        # Persistence baseline S0
        mae_pers = mae(y_test_abs, y_test_curr)
        rmse_pers = rmse(y_test_abs, y_test_curr)
        med_pers, wd_pers = per_server_metrics(y_test_abs, y_test_curr, test_servers)
        print(f"  S0 Persistence: MAE = {mae_pers:.4f}°C | RMSE = {rmse_pers:.4f}°C")

        results.append({
            "dataset": dataset_name, "split": "Split A", "horizon": h_label,
            "feature_set": "S0", "model": "persistence", "target_type": "current_T",
            "mae": mae_pers, "rmse": rmse_pers, "skill_score": 0.0,
            "median_server_mae": med_pers, "worst_decile_server_mae": wd_pers,
            "n_test_rows": n_rows, "n_test_servers": len(np.unique(test_servers)),
        })

        for fs in feature_sets:
            feats = get_feature_set(fs, df.columns.tolist())
            is_monitoring = fs in ["S1", "S4"]

            # Train and val clean (avoid duplicate columns)
            cols_to_use = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature"]))
            train_clean = train_df[cols_to_use].dropna()
            val_clean = val_df[cols_to_use].dropna()

            X_train = train_clean[feats].values
            X_val = val_clean[feats].values
            X_test = test_clean[feats].values

            # Target: Delta for monitoring (S1, S4); Absolute for planning (S2, S3)
            y_train_target = train_clean[target_delta].values if is_monitoring else train_clean[target_abs].values
            y_val_target = val_clean[target_delta].values if is_monitoring else val_clean[target_abs].values

            # 1. Ridge
            ridge = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100.0))])
            ridge.fit(X_train, y_train_target)
            pred_ridge_target = ridge.predict(X_test)
            pred_ridge_abs = (y_test_curr + pred_ridge_target) if is_monitoring else pred_ridge_target

            mae_ridge = mae(y_test_abs, pred_ridge_abs)
            rmse_ridge = rmse(y_test_abs, pred_ridge_abs)
            skill_ridge = skill_score_vs_persistence(mae_ridge, mae_pers)
            med_ridge, wd_ridge = per_server_metrics(y_test_abs, pred_ridge_abs, test_servers)

            results.append({
                "dataset": dataset_name, "split": "Split A", "horizon": h_label,
                "feature_set": fs, "model": "ridge",
                "target_type": "delta" if is_monitoring else "absolute",
                "mae": mae_ridge, "rmse": rmse_ridge, "skill_score": skill_ridge,
                "median_server_mae": med_ridge, "worst_decile_server_mae": wd_ridge,
                "n_test_rows": n_rows, "n_test_servers": len(np.unique(test_servers)),
            })

            # 2. LightGBM
            dtrain = lgb.Dataset(X_train, label=y_train_target, feature_name=feats)
            dval = lgb.Dataset(X_val, label=y_val_target, reference=dtrain, feature_name=feats)
            lgb_params = {
                "objective": "regression", "metric": "mae",
                "learning_rate": 0.05, "num_leaves": 63,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "verbosity": -1, "seed": 42,
            }
            booster = lgb.train(
                lgb_params, dtrain, num_boost_round=250, valid_sets=[dval],
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
            )
            pred_lgb_target = booster.predict(X_test)
            pred_lgb_abs = (y_test_curr + pred_lgb_target) if is_monitoring else pred_lgb_target

            mae_lgb = mae(y_test_abs, pred_lgb_abs)
            rmse_lgb = rmse(y_test_abs, pred_lgb_abs)
            skill_lgb = skill_score_vs_persistence(mae_lgb, mae_pers)
            med_lgb, wd_lgb = per_server_metrics(y_test_abs, pred_lgb_abs, test_servers)

            print(f"  {fs} LightGBM: MAE = {mae_lgb:.4f}°C | RMSE = {rmse_lgb:.4f}°C | Skill vs Pers = {skill_lgb:+.2%}")

            results.append({
                "dataset": dataset_name, "split": "Split A", "horizon": h_label,
                "feature_set": fs, "model": "lightgbm",
                "target_type": "delta" if is_monitoring else "absolute",
                "mae": mae_lgb, "rmse": rmse_lgb, "skill_score": skill_lgb,
                "median_server_mae": med_lgb, "worst_decile_server_mae": wd_lgb,
                "n_test_rows": n_rows, "n_test_servers": len(np.unique(test_servers)),
            })

    return pd.DataFrame(results)


def run_split_b_groupkfold(df: pd.DataFrame, dataset_name: str, feature_sets: list, horizon_step: int = 6):
    """Run Split B: 5-Fold GroupKFold by server, so 100% of servers are tested."""
    h_label = HORIZON_LABELS[horizon_step]
    print(f"\n{'='*70}\nRUNNING SPLIT B (5-FOLD GROUP-K-FOLD) — {dataset_name} ({h_label})\n{'='*70}")

    target_abs = f"CPU_Temperature_target_{horizon_step}"
    target_delta = f"CPU_Temperature_delta_{horizon_step}"

    splits = get_group_kfold_splits(df, n_splits=5, seed=42)
    fold_results = []

    for fold_idx, (train_srvs, test_srvs) in enumerate(splits):
        print(f"\n--- Fold {fold_idx + 1}/5: {len(test_srvs)} test servers ({len(train_srvs)} train) ---")

        train_part = df[df["Series"].isin(train_srvs)]
        test_part = df[df["Series"].isin(test_srvs)]

        all_req_cols = set(["CPU_Temperature", target_abs, target_delta, "Series", "Time"])
        for fs in feature_sets:
            all_req_cols.update(get_feature_set(fs, df.columns.tolist()))
        avail_req = [c for c in all_req_cols if c in df.columns]

        test_clean = test_part[avail_req].dropna()
        y_test_abs = test_clean[target_abs].values
        y_test_curr = test_clean["CPU_Temperature"].values
        test_servers = test_clean["Series"].values

        mae_pers = mae(y_test_abs, y_test_curr)
        rmse_pers = rmse(y_test_abs, y_test_curr)
        med_pers, wd_pers = per_server_metrics(y_test_abs, y_test_curr, test_servers)

        fold_results.append({
            "fold": fold_idx + 1, "dataset": dataset_name, "feature_set": "S0", "model": "persistence",
            "mae": mae_pers, "rmse": rmse_pers, "skill": 0.0,
            "median_server_mae": med_pers, "worst_decile_server_mae": wd_pers,
            "n_test_rows": len(test_clean),
        })

        for fs in feature_sets:
            feats = get_feature_set(fs, df.columns.tolist())
            is_monitoring = fs in ["S1", "S4"]

            cols_to_use = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature"]))
            train_clean = train_part[cols_to_use].dropna()
            X_train = train_clean[feats].values
            X_test = test_clean[feats].values
            y_train_target = train_clean[target_delta].values if is_monitoring else train_clean[target_abs].values

            # LightGBM
            dtrain = lgb.Dataset(X_train, label=y_train_target, feature_name=feats)
            lgb_params = {
                "objective": "regression", "metric": "mae",
                "learning_rate": 0.05, "num_leaves": 63,
                "verbosity": -1, "seed": 42,
            }
            booster = lgb.train(lgb_params, dtrain, num_boost_round=150)
            pred_target = booster.predict(X_test)
            pred_abs = (y_test_curr + pred_target) if is_monitoring else pred_target

            mae_m = mae(y_test_abs, pred_abs)
            rmse_m = rmse(y_test_abs, pred_abs)
            skill_m = skill_score_vs_persistence(mae_m, mae_pers)
            med_m, wd_m = per_server_metrics(y_test_abs, pred_abs, test_servers)

            print(f"  Fold {fold_idx + 1} {fs} LightGBM: MAE = {mae_m:.4f}°C | Skill = {skill_m:+.2%}")

            fold_results.append({
                "fold": fold_idx + 1, "dataset": dataset_name, "feature_set": fs, "model": "lightgbm",
                "mae": mae_m, "rmse": rmse_m, "skill": skill_m,
                "median_server_mae": med_m, "worst_decile_server_mae": wd_m,
                "n_test_rows": len(test_clean),
            })

    df_folds = pd.DataFrame(fold_results)
    return df_folds


def run_split_c(df: pd.DataFrame, dataset_name: str, feature_sets: list, horizon_step: int = 6):
    """Run Split C: Strict Holdout (unseen servers AND later time)."""
    h_label = HORIZON_LABELS[horizon_step]
    print(f"\n{'='*70}\nRUNNING SPLIT C — {dataset_name} ({h_label})\n{'='*70}")

    train_df, val_df, test_df = split_strict(df, seed=42)
    print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")

    target_abs = f"CPU_Temperature_target_{horizon_step}"
    target_delta = f"CPU_Temperature_delta_{horizon_step}"

    all_req_cols = set(["CPU_Temperature", target_abs, target_delta, "Series", "Time"])
    for fs in feature_sets:
        all_req_cols.update(get_feature_set(fs, df.columns.tolist()))
    avail_req = [c for c in all_req_cols if c in df.columns]

    test_clean = test_df[avail_req].dropna()
    y_test_abs = test_clean[target_abs].values
    y_test_curr = test_clean["CPU_Temperature"].values
    test_servers = test_clean["Series"].values

    mae_pers = mae(y_test_abs, y_test_curr)
    rmse_pers = rmse(y_test_abs, y_test_curr)
    med_pers, wd_pers = per_server_metrics(y_test_abs, y_test_curr, test_servers)
    print(f"  S0 Persistence: MAE = {mae_pers:.4f}°C | RMSE = {rmse_pers:.4f}°C")

    results = [{
        "dataset": dataset_name, "split": "Split C", "horizon": h_label,
        "feature_set": "S0", "model": "persistence",
        "mae": mae_pers, "rmse": rmse_pers, "skill": 0.0,
        "median_server_mae": med_pers, "worst_decile_server_mae": wd_pers,
        "n_test_rows": len(test_clean), "n_test_servers": len(np.unique(test_servers)),
    }]

    for fs in feature_sets:
        feats = get_feature_set(fs, df.columns.tolist())
        is_monitoring = fs in ["S1", "S4"]

        cols_to_use = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature"]))
        train_clean = train_df[cols_to_use].dropna()
        val_clean = val_df[cols_to_use].dropna()

        X_train = train_clean[feats].values
        X_val = val_clean[feats].values
        X_test = test_clean[feats].values
        y_train_target = train_clean[target_delta].values if is_monitoring else train_clean[target_abs].values
        y_val_target = val_clean[target_delta].values if is_monitoring else val_clean[target_abs].values

        dtrain = lgb.Dataset(X_train, label=y_train_target, feature_name=feats)
        dval = lgb.Dataset(X_val, label=y_val_target, reference=dtrain, feature_name=feats)
        lgb_params = {
            "objective": "regression", "metric": "mae",
            "learning_rate": 0.05, "num_leaves": 63,
            "verbosity": -1, "seed": 42,
        }
        booster = lgb.train(
            lgb_params, dtrain, num_boost_round=250, valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        pred_target = booster.predict(X_test)
        pred_abs = (y_test_curr + pred_target) if is_monitoring else pred_target

        mae_m = mae(y_test_abs, pred_abs)
        rmse_m = rmse(y_test_abs, pred_abs)
        skill_m = skill_score_vs_persistence(mae_m, mae_pers)
        med_m, wd_m = per_server_metrics(y_test_abs, pred_abs, test_servers)

        print(f"  {fs} LightGBM: MAE = {mae_m:.4f}°C | Skill = {skill_m:+.2%}")

        results.append({
            "dataset": dataset_name, "split": "Split C", "horizon": h_label,
            "feature_set": fs, "model": "lightgbm",
            "mae": mae_m, "rmse": rmse_m, "skill": skill_m,
            "median_server_mae": med_m, "worst_decile_server_mae": wd_m,
            "n_test_rows": len(test_clean), "n_test_servers": len(np.unique(test_servers)),
        })

    return pd.DataFrame(results)


def run_random_kfold_reference(df: pd.DataFrame, horizon_step: int = 6):
    """Run random 5-fold CV to demonstrate leakage inflation."""
    h_label = HORIZON_LABELS[horizon_step]
    print(f"\n{'='*70}\nRUNNING RANDOM 5-FOLD CV REFERENCE (LEAKAGE DEMONSTRATION)\n{'='*70}")

    target_abs = f"CPU_Temperature_target_{horizon_step}"
    target_delta = f"CPU_Temperature_delta_{horizon_step}"
    feats = get_feature_set("S4", df.columns.tolist())

    cols_to_use = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature", "Series"]))
    clean_df = df[cols_to_use].dropna().sample(n=50000, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    fold_maes = []
    for f, (tr_idx, te_idx) in enumerate(kf.split(clean_df)):
        tr = clean_df.iloc[tr_idx]
        te = clean_df.iloc[te_idx]

        X_tr = tr[feats].values
        y_tr = tr[target_delta].values
        X_te = te[feats].values
        y_te_abs = te[target_abs].values
        y_te_curr = te["CPU_Temperature"].values

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feats)
        booster = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1}, dtrain, num_boost_round=150)
        pred_delta = booster.predict(X_te)
        pred_abs = y_te_curr + pred_delta
        err = mae(y_te_abs, pred_abs)
        fold_maes.append(err)
        print(f"  Random Fold {f+1} MAE: {err:.4f}°C")

    mean_rand_mae = np.mean(fold_maes)
    print(f"Mean Random CV MAE: {mean_rand_mae:.4f}°C (compare with chronological Split A ~0.83°C)")
    res = pd.DataFrame([{"split": "Random 5-Fold CV", "mean_mae": mean_rand_mae, "std_mae": np.std(fold_maes)}])
    res.to_csv(OUT_DIR / "results_random_kfold_reference.csv", index=False)
    return res


def run_shap_analysis(df: pd.DataFrame, horizon_step: int = 6):
    """Compute TreeSHAP feature importances on sample."""
    print(f"\n{'='*70}\nCOMPUTING SHAP FEATURE IMPORTANCE (S4 LightGBM)\n{'='*70}")
    target_delta = f"CPU_Temperature_delta_{horizon_step}"
    feats = get_feature_set("S4", df.columns.tolist())

    cols_to_use = list(dict.fromkeys(feats + [target_delta, "CPU_Temperature"]))
    sample_df = df[cols_to_use].dropna().sample(n=10000, random_state=42)
    X = sample_df[feats].values
    y = sample_df[target_delta].values

    dtrain = lgb.Dataset(X, label=y, feature_name=feats)
    model = lgb.train({"objective": "regression", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1}, dtrain, num_boost_round=100)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X[:1000])

    mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
    shap_df = pd.DataFrame({"feature": feats, "mean_abs_shap": mean_abs_shap}).sort_values("mean_abs_shap", ascending=False)
    print("\nTop 15 Most Important Features by Mean |SHAP|:")
    print(shap_df.head(15).to_string(index=False))
    shap_df.to_csv(OUT_DIR / "shap_summary.csv", index=False)
    return shap_df


def main():
    # 1. Dataset 1
    df1 = prepare_data("dataset1")
    res_a1 = run_split_a(df1, "dataset1", ["S1", "S2", "S3", "S4"])
    res_b1 = run_split_b_groupkfold(df1, "dataset1", ["S1", "S4"], horizon_step=6)
    res_c1 = run_split_c(df1, "dataset1", ["S1", "S4"], horizon_step=6)

    # 2. Dataset 2
    df2 = prepare_data("dataset2")
    res_a2 = run_split_a(df2, "dataset2", ["S1", "S2", "S3"])
    res_b2 = run_split_b_groupkfold(df2, "dataset2", ["S1", "S2"], horizon_step=6)
    res_c2 = run_split_c(df2, "dataset2", ["S1", "S2"], horizon_step=6)

    # Combine and save
    all_split_a = pd.concat([res_a1, res_a2], ignore_index=True)
    all_split_a.to_csv(OUT_DIR / "results_split_a.csv", index=False)

    all_split_b = pd.concat([res_b1, res_b2], ignore_index=True)
    all_split_b.to_csv(OUT_DIR / "results_split_b_groupkfold.csv", index=False)

    all_split_c = pd.concat([res_c1, res_c2], ignore_index=True)
    all_split_c.to_csv(OUT_DIR / "results_split_c.csv", index=False)

    # 3. Random CV reference and SHAP
    run_random_kfold_reference(df1, horizon_step=6)
    run_shap_analysis(df1, horizon_step=6)
    print(f"\nAll v2 ablation experiments complete! Results saved in {OUT_DIR}")


if __name__ == "__main__":
    main()
