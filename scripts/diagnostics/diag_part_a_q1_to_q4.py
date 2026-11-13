#!/usr/bin/env python3
"""Diagnostic script for Part A: Questions 1, 2, 3, and 4."""

import sys
sys.stdout.reconfigure(line_buffering=True)
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import get_group_kfold_splits
from ttwin.metrics import mae, rmse, skill_score_vs_persistence

def main():
    print("=" * 80)
    print("PART A DIAGNOSTICS: QUESTIONS 1, 2, 3, 4")
    print("=" * 80)

    path = ROOT / "data" / "interim" / "dataset1" / "_combined.parquet"
    print(f"Loading data from {path}...")
    df = load_parquet(path)
    df["Series"] = df["Series"].astype(str).str.strip()

    # Drop 5 servers without S4 columns
    disqual = {"qh2-rcc196", "qh2-rcc197", "qh2-rcc198", "qh2-rcc199", "qh2-rcc200"}
    df = df[~df["Series"].isin(disqual)].copy()

    horizon_step = 6
    target_abs_1h = f"CPU_Temperature_target_{horizon_step}"
    target_delta_1h = f"CPU_Temperature_delta_{horizon_step}"
    target_contemp = "CPU_Temperature"  # Contemporaneous T(t)

    print("Building features...")
    df = build_features(df, horizons=[horizon_step], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])

    s2_feats = list(dict.fromkeys(get_feature_set("S2", df.columns.tolist())))
    s4_feats = list(dict.fromkeys(get_feature_set("S4", df.columns.tolist())))

    splits = get_group_kfold_splits(df, n_splits=5, seed=42)
    source_servers, target_servers = splits[0]
    print(f"Source servers ({len(source_servers)}): {source_servers[:5]} ...")
    print(f"Target servers ({len(target_servers)}): {target_servers}")

    df_source = df[df["Series"].isin(source_servers)].copy()
    split_time = df_source["Time"].min() + (df_source["Time"].max() - df_source["Time"].min()) * 0.70
    df_source_train = df_source[df_source["Time"] < split_time].copy()
    df_source_val = df_source[(df_source["Time"] >= split_time) & (df_source["Time"] < split_time + pd.Timedelta(days=25))].copy()

    print(f"\nTraining set date range: {df_source_train['Time'].min()} to {df_source_train['Time'].max()}")
    print(f"Source train rows: {len(df_source_train):,}")

    clean_s2_src_1h = df_source_train[s2_feats + [target_abs_1h]].dropna()
    clean_s2_src_contemp = df_source_train[s2_feats + [target_contemp]].dropna()
    clean_s4_src_delta = df_source_train[s4_feats + [target_delta_1h]].dropna()

    # 1. Fit Pretrained Planning Models (1h ahead vs Contemporaneous)
    print("\nTraining Planning Models...")
    dtrain_s2_1h = lgb.Dataset(clean_s2_src_1h[s2_feats].values, label=clean_s2_src_1h[target_abs_1h].values, feature_name=s2_feats)
    lgb_plan_1h = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtrain_s2_1h, num_boost_round=200,
    )

    dtrain_s2_contemp = lgb.Dataset(clean_s2_src_contemp[s2_feats].values, label=clean_s2_src_contemp[target_contemp].values, feature_name=s2_feats)
    lgb_plan_contemp = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtrain_s2_contemp, num_boost_round=200,
    )

    # Stronger Baselines on df_source_train:
    # 1. Cluster Mean
    cluster_mean_1h = float(clean_s2_src_1h[target_abs_1h].mean())
    cluster_mean_contemp = float(clean_s2_src_contemp[target_contemp].mean())

    # 2. Ridge on CPU_Load only
    ridge_cpu_1h = Ridge(alpha=1.0)
    ridge_cpu_1h.fit(clean_s2_src_1h[["CPU_Load"]].values, clean_s2_src_1h[target_abs_1h].values)

    ridge_cpu_contemp = Ridge(alpha=1.0)
    ridge_cpu_contemp.fit(clean_s2_src_contemp[["CPU_Load"]].values, clean_s2_src_contemp[target_contemp].values)

    # 3. Load-Conditional Mean (10 bins: 0-10, 10-20, ..., 90-100)
    clean_s2_src_1h["load_bin"] = pd.cut(clean_s2_src_1h["CPU_Load"], bins=np.linspace(0, 100, 11), labels=False, include_lowest=True)
    load_bin_means_1h = clean_s2_src_1h.groupby("load_bin")[target_abs_1h].mean().to_dict()

    clean_s2_src_contemp["load_bin"] = pd.cut(clean_s2_src_contemp["CPU_Load"], bins=np.linspace(0, 100, 11), labels=False, include_lowest=True)
    load_bin_means_contemp = clean_s2_src_contemp.groupby("load_bin")[target_contemp].mean().to_dict()

    # Pretrain Monitoring Model (S4, delta)
    print("Training Monitoring Model...")
    dtrain_s4 = lgb.Dataset(clean_s4_src_delta[s4_feats].values, label=clean_s4_src_delta[target_delta_1h].values, feature_name=s4_feats)
    lgb_mon = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtrain_s4, num_boost_round=200,
    )

    # Validation set for shrinkage parameter tuning (lambda)
    val_cols = list(dict.fromkeys(s4_feats + [target_abs_1h, target_delta_1h, "CPU_Temperature"]))
    clean_s4_val = df_source_val[val_cols].dropna()
    y_val_abs = clean_s4_val[target_abs_1h].values
    y_val_curr = clean_s4_val["CPU_Temperature"].values
    pred_val_delta = lgb_mon.predict(clean_s4_val[s4_feats].values)
    best_lambda = 1.0
    best_val_mae = float("inf")
    for lam in np.linspace(0.0, 1.0, 21):
        pred_val = y_val_curr + lam * pred_val_delta
        m = mae(y_val_abs, pred_val)
        if m < best_val_mae:
            best_val_mae = m
            best_lambda = lam
    print(f"Best validation shrinkage lambda: {best_lambda:.2f} (Val MAE = {best_val_mae:.4f} vs pers = {mae(y_val_abs, y_val_curr):.4f})")

    # Evaluate across 16 target servers
    max_budget_h = 168
    embargo_h = 6
    records_q1 = []
    records_q2 = []
    offsets_comparison = []
    records_q4 = []

    for srv in target_servers:
        sdf = df[df["Series"] == srv].sort_values("Time").reset_index(drop=True)
        t_min = sdf["Time"].min()
        test_start = t_min + pd.Timedelta(hours=max_budget_h + embargo_h)
        sdf_test = sdf[sdf["Time"] >= test_start].copy()

        req_cols = set(["CPU_Temperature", target_abs_1h, target_delta_1h, "Series", "Time"] + s2_feats + s4_feats)
        clean_test = sdf_test[[c for c in req_cols if c in sdf_test.columns]].dropna()

        if len(clean_test) < 100:
            continue

        y_te_1h = clean_test[target_abs_1h].values
        y_te_contemp = clean_test[target_contemp].values
        y_te_curr = clean_test["CPU_Temperature"].values
        X_te_s2 = clean_test[s2_feats].values
        X_te_s4 = clean_test[s4_feats].values
        cpu_load_te = clean_test["CPU_Load"].values

        # ---------------- Q1: BASelines Comparison (1h target) ----------------
        # 1. Cluster mean
        mae_cmean_1h = mae(y_te_1h, np.full_like(y_te_1h, cluster_mean_1h))
        # 2. Ridge on CPU load only
        pred_ridge_cpu_1h = ridge_cpu_1h.predict(clean_test[["CPU_Load"]].values)
        mae_ridge_cpu_1h = mae(y_te_1h, pred_ridge_cpu_1h)
        # 3. Load conditional mean
        te_bins = pd.cut(clean_test["CPU_Load"], bins=np.linspace(0, 100, 11), labels=False, include_lowest=True)
        pred_lc_mean_1h = te_bins.map(load_bin_means_1h).fillna(cluster_mean_1h).values
        mae_lc_mean_1h = mae(y_te_1h, pred_lc_mean_1h)
        # 4. Pretrained Twin (Zero-shot)
        pred_zs_1h = lgb_plan_1h.predict(X_te_s2)
        mae_zs_1h = mae(y_te_1h, pred_zs_1h)

        # ---------------- Q2: Contemporaneous Form ----------------
        mae_cmean_contemp = mae(y_te_contemp, np.full_like(y_te_contemp, cluster_mean_contemp))
        pred_ridge_cpu_contemp = ridge_cpu_contemp.predict(clean_test[["CPU_Load"]].values)
        mae_ridge_cpu_contemp = mae(y_te_contemp, pred_ridge_cpu_contemp)
        te_bins_c = pd.cut(clean_test["CPU_Load"], bins=np.linspace(0, 100, 11), labels=False, include_lowest=True)
        pred_lc_mean_contemp = te_bins_c.map(load_bin_means_contemp).fillna(cluster_mean_contemp).values
        mae_lc_mean_contemp = mae(y_te_contemp, pred_lc_mean_contemp)
        pred_zs_contemp = lgb_plan_contemp.predict(X_te_s2)
        mae_zs_contemp = mae(y_te_contemp, pred_zs_contemp)

        # ---------------- Q1 / Q3: Local Server Mean & Offset Stability ----------------
        # For each budget N in [6, 24, 72, 168]:
        for bh in [6, 24, 72, 168]:
            t_end = t_min + pd.Timedelta(hours=bh)
            sdf_ad = sdf[sdf["Time"] < t_end].dropna(subset=[target_abs_1h, target_contemp] + s2_feats)
            if len(sdf_ad) < 5:
                continue

            # Local server mean from N hours
            mean_local_ad_1h = float(sdf_ad[target_abs_1h].mean())
            mae_local_mean_1h = mae(y_te_1h, np.full_like(y_te_1h, mean_local_ad_1h))

            mean_local_ad_contemp = float(sdf_ad[target_contemp].mean())
            mae_local_mean_contemp = mae(y_te_contemp, np.full_like(y_te_contemp, mean_local_ad_contemp))

            # Actual mean of test window
            true_test_mean_1h = float(np.mean(y_te_1h))

            # Offset on adaptation window vs test window
            ad_pred_1h = lgb_plan_1h.predict(sdf_ad[s2_feats].values)
            offset_ad = float(np.mean(sdf_ad[target_abs_1h].values - ad_pred_1h))
            offset_test = float(np.mean(y_te_1h - pred_zs_1h))

            pred_offset_applied = pred_zs_1h + offset_ad
            mae_offset_applied = mae(y_te_1h, pred_offset_applied)

            if bh == 168:
                records_q1.append({
                    "server": srv,
                    "cluster_mean_mae": mae_cmean_1h,
                    "ridge_cpu_mae": mae_ridge_cpu_1h,
                    "load_conditional_mean_mae": mae_lc_mean_1h,
                    "server_mean_168h_mae": mae_local_mean_1h,
                    "zero_shot_mae": mae_zs_1h,
                    "adapt_168h_mean_temp": mean_local_ad_1h,
                    "test_window_mean_temp": true_test_mean_1h,
                    "temp_drift_delta": true_test_mean_1h - mean_local_ad_1h,
                })
                records_q2.append({
                    "server": srv,
                    "cluster_mean_contemp": mae_cmean_contemp,
                    "ridge_cpu_contemp": mae_ridge_cpu_contemp,
                    "load_conditional_mean_contemp": mae_lc_mean_contemp,
                    "server_mean_168h_contemp": mae_local_mean_contemp,
                    "zero_shot_contemp": mae_zs_contemp,
                })
                offsets_comparison.append({
                    "server": srv,
                    "offset_168h": offset_ad,
                    "offset_test": offset_test,
                    "offset_diff": offset_test - offset_ad,
                    "zero_shot_mae": mae_zs_1h,
                    "offset_corrected_mae": mae_offset_applied,
                })

        # ---------------- Q4: Monitoring Zero-Shot & Shrinkage ----------------
        mae_pers = mae(y_te_1h, y_te_curr)
        pred_mon_delta = lgb_mon.predict(X_te_s4)
        pred_mon_raw = y_te_curr + pred_mon_delta
        mae_mon_raw = mae(y_te_1h, pred_mon_raw)

        # Shrinkage model: T(t) + lambda * delta
        pred_mon_shrink = y_te_curr + best_lambda * pred_mon_delta
        mae_mon_shrink = mae(y_te_1h, pred_mon_shrink)

        records_q4.append({
            "server": srv,
            "persistence_mae": mae_pers,
            "raw_twin_mae": mae_mon_raw,
            "raw_twin_skill": skill_score_vs_persistence(mae_mon_raw, mae_pers),
            "shrinkage_twin_mae": mae_mon_shrink,
            "shrinkage_twin_skill": skill_score_vs_persistence(mae_mon_shrink, mae_pers),
        })

    # Output results
    df_q1 = pd.DataFrame(records_q1)
    df_q2 = pd.DataFrame(records_q2)
    df_offsets = pd.DataFrame(offsets_comparison)
    df_q4 = pd.DataFrame(records_q4)

    df_q1.to_csv(ROOT / "experiments" / "diag_part_a_q1_baselines.csv", index=False)
    df_q2.to_csv(ROOT / "experiments" / "diag_part_a_q2_contemporaneous.csv", index=False)
    df_offsets.to_csv(ROOT / "experiments" / "diag_part_a_q3_offsets.csv", index=False)
    df_q4.to_csv(ROOT / "experiments" / "diag_part_a_q4_shrinkage.csv", index=False)

    print("\n" + "=" * 60)
    print("QUESTION 1: PLANNING BASELINES (1-hour ahead target)")
    print("=" * 60)
    print(f"Cluster Mean Baseline:         MAE = {df_q1['cluster_mean_mae'].mean():.4f} ± {df_q1['cluster_mean_mae'].std():.4f}°C")
    print(f"Ridge (CPU Load Only):         MAE = {df_q1['ridge_cpu_mae'].mean():.4f} ± {df_q1['ridge_cpu_mae'].std():.4f}°C")
    print(f"Load-Conditional Mean:         MAE = {df_q1['load_conditional_mean_mae'].mean():.4f} ± {df_q1['load_conditional_mean_mae'].std():.4f}°C")
    print(f"Local Server Mean (168h):      MAE = {df_q1['server_mean_168h_mae'].mean():.4f} ± {df_q1['server_mean_168h_mae'].std():.4f}°C")
    print(f"Pretrained Twin (Zero-Shot S2): MAE = {df_q1['zero_shot_mae'].mean():.4f} ± {df_q1['zero_shot_mae'].std():.4f}°C")
    print(f"\nGain of Twin over Cluster Mean:        {(1 - df_q1['zero_shot_mae'].mean()/df_q1['cluster_mean_mae'].mean()):+.2%}")
    print(f"Gain of Twin over Ridge (CPU Load):    {(1 - df_q1['zero_shot_mae'].mean()/df_q1['ridge_cpu_mae'].mean()):+.2%}")
    print(f"Gain of Twin over Load-Cond. Mean:     {(1 - df_q1['zero_shot_mae'].mean()/df_q1['load_conditional_mean_mae'].mean()):+.2%}")
    print(f"Gain of Twin over Local Server Mean:   {(1 - df_q1['zero_shot_mae'].mean()/df_q1['server_mean_168h_mae'].mean()):+.2%}")
    print(f"\nTemperature Drift (Test Window Mean - 168h Mean):")
    print(f"  Mean drift across servers: {df_q1['temp_drift_delta'].mean():+.2f}°C (Min: {df_q1['temp_drift_delta'].min():+.2f}°C, Max: {df_q1['temp_drift_delta'].max():+.2f}°C)")

    print("\n" + "=" * 60)
    print("QUESTION 2: CONTEMPORANEOUS PLANNING TARGET (T(t))")
    print("=" * 60)
    print(f"Cluster Mean:                  MAE = {df_q2['cluster_mean_contemp'].mean():.4f} ± {df_q2['cluster_mean_contemp'].std():.4f}°C")
    print(f"Ridge (CPU Load Only):         MAE = {df_q2['ridge_cpu_contemp'].mean():.4f} ± {df_q2['ridge_cpu_contemp'].std():.4f}°C")
    print(f"Load-Conditional Mean:         MAE = {df_q2['load_conditional_mean_contemp'].mean():.4f} ± {df_q2['load_conditional_mean_contemp'].std():.4f}°C")
    print(f"Local Server Mean (168h):      MAE = {df_q2['server_mean_168h_contemp'].mean():.4f} ± {df_q2['server_mean_168h_contemp'].std():.4f}°C")
    print(f"Contemporaneous Twin (S2):     MAE = {df_q2['zero_shot_contemp'].mean():.4f} ± {df_q2['zero_shot_contemp'].std():.4f}°C")
    print(f"Gain of Contemp Twin over Cluster Mean:        {(1 - df_q2['zero_shot_contemp'].mean()/df_q2['cluster_mean_contemp'].mean()):+.2%}")
    print(f"Gain of Contemp Twin over Ridge (CPU Load):    {(1 - df_q2['zero_shot_contemp'].mean()/df_q2['ridge_cpu_contemp'].mean()):+.2%}")
    print(f"Gain of Contemp Twin over Load-Cond. Mean:     {(1 - df_q2['zero_shot_contemp'].mean()/df_q2['load_conditional_mean_contemp'].mean()):+.2%}")

    print("\n" + "=" * 60)
    print("QUESTION 3: RESIDUAL OFFSET ANALYSIS")
    print("=" * 60)
    r_pearson, p_val = pearsonr(df_offsets["offset_168h"], df_offsets["offset_test"])
    r_spearman, _ = spearmanr(df_offsets["offset_168h"], df_offsets["offset_test"])
    print(f"Correlation between 168h offset and Test Window offset:")
    print(f"  Pearson r = {r_pearson:+.4f} (p = {p_val:.4f})")
    print(f"  Spearman rho = {r_spearman:+.4f}")
    print(f"Mean Absolute Change in Server Bias (|offset_test - offset_168h|): {np.mean(np.abs(df_offsets['offset_diff'])):.4f}°C")
    print("Per-server offset table:")
    print(df_offsets[["server", "offset_168h", "offset_test", "offset_diff", "zero_shot_mae", "offset_corrected_mae"]].to_string(index=False))

    print("\n" + "=" * 60)
    print("QUESTION 4: MONITORING ZERO-SHOT & SHRINKAGE")
    print("=" * 60)
    print(f"Persistence Baseline:          MAE = {df_q4['persistence_mae'].mean():.4f} ± {df_q4['persistence_mae'].std():.4f}°C")
    print(f"Raw Pretrained Twin (Delta):   MAE = {df_q4['raw_twin_mae'].mean():.4f} ± {df_q4['raw_twin_mae'].std():.4f}°C (Skill: {df_q4['raw_twin_skill'].mean():+.2%})")
    print(f"Shrinkage Twin (lambda={best_lambda:.2f}):  MAE = {df_q4['shrinkage_twin_mae'].mean():.4f} ± {df_q4['shrinkage_twin_mae'].std():.4f}°C (Skill: {df_q4['shrinkage_twin_skill'].mean():+.2%})")

if __name__ == "__main__":
    main()
