#!/usr/bin/env python3
"""v3 Comprehensive Experiments Script.
Implements the missing Tier B experiments:
1. Paired bootstrap over servers and weeks for skill vs persistence in E1 (both datasets).
2. Sensor ablation: S4 without fans, without power, without both, and S1.
3. Split B (GroupKFold) and Split C at 10min, 1h, 6h; Split C with 5 random server draws.
4. Planning adaptation with budgets [0, 1, 6, 24, 72, 168] h across all 5 folds.
5. Cross-cluster planning adaptation (C1 -> C2) with the same budgets.
6. Random k-fold vs GroupKFold on identical rows for S1, S2, S3, S4.
7. S1 LightGBM learning curves with lr in [0.05, 0.01, 0.005].
"""

import sys
from pathlib import Path
import json
import argparse
import subprocess
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.generate_results_v3_doc import build_markdown_content

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import split_chronological, get_group_kfold_splits, split_strict
from ttwin.metrics import mae, rmse, skill_score_vs_persistence

OUT_DIR = ROOT / "experiments" / "v3_experiments"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "interim"


def bootstrap_ci(values, n_boot=2000, ci=0.95, seed=42):
    values = np.asarray(values)
    if len(values) <= 1:
        return float(values[0]), float(values[0])
    rng = np.random.RandomState(seed)
    boot_means = [np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(n_boot)]
    alpha = (1.0 - ci) / 2.0
    return float(np.percentile(boot_means, 100 * alpha)), float(np.percentile(boot_means, 100 * (1 - alpha)))


def load_data(name: str):
    path = DATA_DIR / name / "_combined.parquet"
    df = load_parquet(path)
    df["Series"] = df["Series"].astype(str).str.strip()
    if name == "dataset2":
        disqual = {"qh2-rcc263", "qh2-rcc264", "qh2-rcc226", "qh2-rcc205"}
        df = df[~df["Series"].isin(disqual)].copy()
    print(f"Loaded {name}: {len(df):,} rows, {df['Series'].nunique()} servers.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# 1. Paired Bootstrap Analysis
# ─────────────────────────────────────────────────────────────────────────────
def run_exp1_bootstrap(d1_feat, d2_feat):
    print("\n--- 1. Paired Bootstrap Analysis (Servers & Weeks) ---")
    results = []

    for name, df in [("Dataset 1", d1_feat), ("Dataset 2", d2_feat)]:
        train_df, val_df, test_df = split_chronological(df, train_frac=0.70, val_frac=0.10, embargo_steps=36)
        target_abs = "CPU_Temperature_target_6"
        target_delta = "CPU_Temperature_delta_6"
        feats = get_feature_set("S1", df.columns.tolist())
        cols = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature", "Series", "Time"]))

        test_clean = test_df[cols].dropna().copy()
        train_clean = train_df[cols].dropna()
        val_clean = val_df[cols].dropna()

        dtrain = lgb.Dataset(train_clean[feats].values, label=train_clean[target_delta].values, feature_name=feats)
        dval = lgb.Dataset(val_clean[feats].values, label=val_clean[target_delta].values, reference=dtrain, feature_name=feats)
        params = {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42}
        booster = lgb.train(params, dtrain, num_boost_round=100, valid_sets=[dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        y_test_abs = test_clean[target_abs].values
        y_test_curr = test_clean["CPU_Temperature"].values
        pred_lgb = y_test_curr + booster.predict(test_clean[feats].values)

        test_clean["err_pers"] = np.abs(y_test_abs - y_test_curr)
        test_clean["err_lgb"] = np.abs(y_test_abs - pred_lgb)

        # Per-server paired bootstrap
        server_stats = test_clean.groupby("Series").agg(mae_pers=("err_pers", "mean"), mae_lgb=("err_lgb", "mean"))
        server_stats["skill"] = 1.0 - (server_stats["mae_lgb"] / server_stats["mae_pers"])
        ci_srv_low, ci_srv_high = bootstrap_ci(server_stats["skill"].values)

        # Per-week paired bootstrap (using year-week to prevent cross-year collision)
        iso_cal = test_clean["Time"].dt.isocalendar()
        test_clean["year_week"] = iso_cal.year.astype(str) + "-W" + iso_cal.week.astype(str).str.zfill(2)
        week_stats = test_clean.groupby("year_week").agg(mae_pers=("err_pers", "mean"), mae_lgb=("err_lgb", "mean"))
        week_stats["skill"] = 1.0 - (week_stats["mae_lgb"] / week_stats["mae_pers"])
        ci_wk_low, ci_wk_high = bootstrap_ci(week_stats["skill"].values)

        overall_mae_pers = mae(y_test_abs, y_test_curr)
        overall_mae_lgb = mae(y_test_abs, pred_lgb)
        overall_skill = skill_score_vs_persistence(overall_mae_lgb, overall_mae_pers)

        results.append({
            "dataset": name,
            "overall_mae_persistence": overall_mae_pers,
            "overall_mae_s1_lgbm": overall_mae_lgb,
            "overall_skill_pct": overall_skill * 100.0,
            "mean_server_skill_pct": server_stats["skill"].mean() * 100.0,
            "server_ci_95_low_pct": ci_srv_low * 100.0,
            "server_ci_95_high_pct": ci_srv_high * 100.0,
            "mean_week_skill_pct": week_stats["skill"].mean() * 100.0,
            "week_ci_95_low_pct": ci_wk_low * 100.0,
            "week_ci_95_high_pct": ci_wk_high * 100.0,
            "n_servers": len(server_stats),
            "n_weeks": len(week_stats),
        })

    res_df = pd.DataFrame(results)
    res_df.to_csv(OUT_DIR / "e1_bootstrap_summary.csv", index=False)
    print(res_df.to_string())
    return res_df


# ─────────────────────────────────────────────────────────────────────────────
# 2. Sensor Ablation (S4 variations vs S1)
# ─────────────────────────────────────────────────────────────────────────────
def run_exp2_sensor_ablation(d1_feat):
    print("\n--- 2. Sensor Ablation on Dataset 1 (1h Horizon) ---")
    train_df, val_df, test_df = split_chronological(d1_feat, train_frac=0.70, val_frac=0.10, embargo_steps=36)
    target_abs = "CPU_Temperature_target_6"
    target_delta = "CPU_Temperature_delta_6"

    fan_cols = ["Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4"]
    power_cols = ["Power"]

    s4_all = get_feature_set("S4", d1_feat.columns.tolist())
    s1_all = get_feature_set("S1", d1_feat.columns.tolist())

    configs = {
        "S4 Full (Fans + Power + Inlet + Workload + Lags)": s4_all,
        "S4 without Fans (Power + Inlet + Workload + Lags)": [c for c in s4_all if c not in fan_cols],
        "S4 without Power (Fans + Inlet + Workload + Lags)": [c for c in s4_all if c not in power_cols],
        "S4 without Fans & Power (Inlet + Workload + Lags)": [c for c in s4_all if c not in fan_cols and c not in power_cols],
        "S1 (Temperature Lags & Current State Only)": s1_all,
    }

    # Clean rows strictly on identical evaluation subset
    all_needed = set(s4_all + [target_abs, target_delta, "CPU_Temperature", "Series"])
    test_clean = test_df[[c for c in all_needed if c in test_df.columns]].dropna().copy()
    y_test_abs = test_clean[target_abs].values
    y_test_curr = test_clean["CPU_Temperature"].values
    pers_mae = mae(y_test_abs, y_test_curr)

    rows = []
    rows.append({
        "configuration": "S0 Persistence Baseline",
        "n_features": 0, "test_mae": pers_mae, "test_rmse": rmse(y_test_abs, y_test_curr),
        "skill_vs_pers_pct": 0.0, "n_test_rows": len(test_clean)
    })

    for label, feats in configs.items():
        cols = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature"]))
        train_clean = train_df[cols].dropna()
        val_clean = val_df[cols].dropna()

        dtrain = lgb.Dataset(train_clean[feats].values, label=train_clean[target_delta].values, feature_name=feats)
        dval = lgb.Dataset(val_clean[feats].values, label=val_clean[target_delta].values, reference=dtrain, feature_name=feats)
        params = {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42}
        booster = lgb.train(params, dtrain, num_boost_round=100, valid_sets=[dval], callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        pred_delta = booster.predict(test_clean[feats].values)
        pred_abs = y_test_curr + pred_delta
        m = mae(y_test_abs, pred_abs)
        r = rmse(y_test_abs, pred_abs)
        sk = skill_score_vs_persistence(m, pers_mae)

        rows.append({
            "configuration": label,
            "n_features": len(feats), "test_mae": m, "test_rmse": r,
            "skill_vs_pers_pct": sk * 100.0, "n_test_rows": len(test_clean)
        })

    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(OUT_DIR / "sensor_ablation.csv", index=False)
    print(ablation_df.to_string())
    return ablation_df


# ─────────────────────────────────────────────────────────────────────────────
# 3. Split B & Split C across Horizons (and Split C with 5 random seeds)
# ─────────────────────────────────────────────────────────────────────────────
def run_exp3_splits_bc_horizons(d1_feat):
    print("\n--- 3. Split B and Split C across Horizons ---")
    results = []

    for h, h_label in [(1, "10min"), (6, "1h"), (36, "6h")]:
        target_abs = f"CPU_Temperature_target_{h}"
        target_delta = f"CPU_Temperature_delta_{h}"
        s1_feats = get_feature_set("S1", d1_feat.columns.tolist())
        s4_feats = get_feature_set("S4", d1_feat.columns.tolist())

        # Split B: GroupKFold 5-fold cross-validation
        splits_b = get_group_kfold_splits(d1_feat, n_splits=5, seed=42)
        b_pers_maes, b_s1_maes, b_s4_maes = [], [], []

        for fold_idx, (tr_srv, te_srv) in enumerate(splits_b, 1):
            train_df = d1_feat[d1_feat["Series"].isin(tr_srv)]
            test_df = d1_feat[d1_feat["Series"].isin(te_srv)]
            cols_req = list(dict.fromkeys(s4_feats + [target_abs, target_delta, "CPU_Temperature"]))
            test_c = test_df[cols_req].dropna()
            train_c = train_df[cols_req].dropna()

            y_true = test_c[target_abs].values
            y_curr = test_c["CPU_Temperature"].values
            pers_m = mae(y_true, y_curr)
            b_pers_maes.append(pers_m)

            # S1
            dtrain_s1 = lgb.Dataset(train_c[s1_feats].values, label=train_c[target_delta].values)
            b_s1 = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "verbosity": -1, "seed": 42}, dtrain_s1, num_boost_round=40)
            pred_s1 = y_curr + b_s1.predict(test_c[s1_feats].values)
            b_s1_maes.append(mae(y_true, pred_s1))

            # S4
            dtrain_s4 = lgb.Dataset(train_c[s4_feats].values, label=train_c[target_delta].values)
            b_s4 = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "verbosity": -1, "seed": 42}, dtrain_s4, num_boost_round=40)
            pred_s4 = y_curr + b_s4.predict(test_c[s4_feats].values)
            b_s4_maes.append(mae(y_true, pred_s4))

        results.append({
            "split": "Split B (GroupKFold)", "horizon": h_label,
            "persistence_mae_mean": np.mean(b_pers_maes), "persistence_mae_std": np.std(b_pers_maes),
            "s1_mae_mean": np.mean(b_s1_maes), "s1_mae_std": np.std(b_s1_maes),
            "s4_mae_mean": np.mean(b_s4_maes), "s4_mae_std": np.std(b_s4_maes),
        })

    # Split C with 5 random server draws (seeds: 42, 123, 456, 789, 1024)
    seeds = [42, 123, 456, 789, 1024]
    c_results = []
    for h, h_label in [(1, "10min"), (6, "1h"), (36, "6h")]:
        target_abs = f"CPU_Temperature_target_{h}"
        target_delta = f"CPU_Temperature_delta_{h}"
        s1_feats = get_feature_set("S1", d1_feat.columns.tolist())
        s4_feats = get_feature_set("S4", d1_feat.columns.tolist())

        c_pers_maes, c_s1_maes, c_s4_maes = [], [], []
        for s in seeds:
            train_df, val_df, test_df = split_strict(d1_feat, train_frac=0.70, holdout_frac=0.20, embargo_steps=36, seed=s)
            cols_req = list(dict.fromkeys(s4_feats + [target_abs, target_delta, "CPU_Temperature"]))
            test_c = test_df[cols_req].dropna()
            train_c = train_df[cols_req].dropna()
            val_c = val_df[cols_req].dropna()

            y_true = test_c[target_abs].values
            y_curr = test_c["CPU_Temperature"].values
            c_pers_maes.append(mae(y_true, y_curr))

            dtrain = lgb.Dataset(train_c[s1_feats].values, label=train_c[target_delta].values)
            dval = lgb.Dataset(val_c[s1_feats].values, label=val_c[target_delta].values, reference=dtrain)
            b = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "verbosity": -1, "seed": 42}, dtrain, num_boost_round=60, valid_sets=[dval], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            c_s1_maes.append(mae(y_true, y_curr + b.predict(test_c[s1_feats].values)))

            dtrain_s4 = lgb.Dataset(train_c[s4_feats].values, label=train_c[target_delta].values)
            dval_s4 = lgb.Dataset(val_c[s4_feats].values, label=val_c[target_delta].values, reference=dtrain_s4)
            b4 = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "verbosity": -1, "seed": 42}, dtrain_s4, num_boost_round=60, valid_sets=[dval_s4], callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])
            c_s4_maes.append(mae(y_true, y_curr + b4.predict(test_c[s4_feats].values)))

        results.append({
            "split": "Split C (5 Random Seeds)", "horizon": h_label,
            "persistence_mae_mean": np.mean(c_pers_maes), "persistence_mae_std": np.std(c_pers_maes),
            "s1_mae_mean": np.mean(c_s1_maes), "s1_mae_std": np.std(c_s1_maes),
            "s4_mae_mean": np.mean(c_s4_maes), "s4_mae_std": np.std(c_s4_maes),
        })

    df_splits = pd.DataFrame(results)
    df_splits.to_csv(OUT_DIR / "split_b_and_c_horizons.csv", index=False)
    print(df_splits.to_string())
    return df_splits


# ─────────────────────────────────────────────────────────────────────────────
# 4. Planning Adaptation across All 5 Folds
# ─────────────────────────────────────────────────────────────────────────────
def run_exp4_planning_adaptation_all_folds(d1_feat):
    print("\n--- 4. Planning Adaptation Across All 5 Folds (76 Target Servers) ---")
    splits = get_group_kfold_splits(d1_feat, n_splits=5, seed=42)
    s2_feats = get_feature_set("S2", d1_feat.columns.tolist())
    target_abs = "CPU_Temperature_target_6"
    cols = list(dict.fromkeys(s2_feats + [target_abs, "Series", "Time"]))

    all_server_results = []

    for fold_idx, (tr_srv, te_srv) in enumerate(splits, 1):
        train_df = d1_feat[d1_feat["Series"].isin(tr_srv)]
        test_df = d1_feat[d1_feat["Series"].isin(te_srv)]
        train_clean = train_df[cols].dropna()
        test_clean = test_df[cols].dropna()

        # Train global source model on training servers
        dtrain = lgb.Dataset(train_clean[s2_feats].values, label=train_clean[target_abs].values, feature_name=s2_feats)
        source_model = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42}, dtrain, num_boost_round=60)

        # For each target server in fold
        target_servers = test_clean["Series"].unique()
        for server in target_servers:
            srv_df = test_clean[test_clean["Series"] == server].sort_values("Time")
            if len(srv_df) < 200:
                continue

            t_min = srv_df["Time"].min()
            t_eval_start = t_min + pd.Timedelta(hours=174)  # 168h max adapt + 6h embargo
            eval_df = srv_df[srv_df["Time"] >= t_eval_start]
            if len(eval_df) < 50:
                continue

            y_eval = eval_df[target_abs].values
            X_eval = eval_df[s2_feats].values

            # Zero-shot
            pred_zero = source_model.predict(X_eval)
            mae_zero = mae(y_eval, pred_zero)

            # Fleet mean
            fleet_mean_val = train_clean[target_abs].mean()
            mae_fleet = mae(y_eval, np.full_like(y_eval, fleet_mean_val))

            for budget in [0, 1, 6, 24, 72, 168]:
                if budget == 0:
                    all_server_results.append({"server": server, "fold": fold_idx, "budget": 0, "strategy": "zero_shot", "mae": mae_zero})
                    all_server_results.append({"server": server, "fold": fold_idx, "budget": 0, "strategy": "fleet_mean", "mae": mae_fleet})
                else:
                    adapt_df = srv_df[(srv_df["Time"] >= t_min) & (srv_df["Time"] < t_min + pd.Timedelta(hours=budget))]
                    if len(adapt_df) < 5:
                        continue
                    # Residual offset
                    adapt_preds = source_model.predict(adapt_df[s2_feats].values)
                    offset = np.mean(adapt_df[target_abs].values - adapt_preds)
                    mae_offset = mae(y_eval, pred_zero + offset)
                    all_server_results.append({"server": server, "fold": fold_idx, "budget": budget, "strategy": "residual_offset", "mae": mae_offset})

                    # Continued boosting
                    dadapt = lgb.Dataset(adapt_df[s2_feats].values, label=adapt_df[target_abs].values, feature_name=s2_feats)
                    finetuned = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.01, "num_leaves": 31, "verbosity": -1, "seed": 42}, dadapt, num_boost_round=15, init_model=source_model)
                    mae_boost = mae(y_eval, finetuned.predict(X_eval))
                    all_server_results.append({"server": server, "fold": fold_idx, "budget": budget, "strategy": "continued_boosting", "mae": mae_boost})

    df_adapt_raw = pd.DataFrame(all_server_results)
    summary = []
    for (strategy, budget), grp in df_adapt_raw.groupby(["strategy", "budget"]):
        vals = grp["mae"].values
        ci_l, ci_h = bootstrap_ci(vals)
        summary.append({
            "strategy": strategy, "budget_hours": budget, "n_servers": len(grp),
            "mean_mae": float(np.mean(vals)), "std_mae": float(np.std(vals)),
            "ci_95_low": ci_l, "ci_95_high": ci_h
        })

    sum_df = pd.DataFrame(summary).sort_values(["strategy", "budget_hours"])
    sum_df.to_csv(OUT_DIR / "planning_adaptation_all_folds.csv", index=False)
    print(sum_df.to_string())
    return sum_df


# ─────────────────────────────────────────────────────────────────────────────
# 5. Cross-Cluster Planning Adaptation (C1 -> C2)
# ─────────────────────────────────────────────────────────────────────────────
def run_exp5_cross_cluster_adaptation(d1_feat, d2_feat):
    print("\n--- 5. Cross-Cluster Planning Adaptation (C1 -> C2) ---")
    shared_feats = ["CPU_Load", "Ram_Used", "No_Of_Running_vms", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    target_abs = "CPU_Temperature_target_6"

    # Train C1 source model on all clean D1 data
    cols1 = list(dict.fromkeys(shared_feats + [target_abs]))
    d1_clean = d1_feat[cols1].dropna()
    dtrain = lgb.Dataset(d1_clean[shared_feats].values, label=d1_clean[target_abs].values, feature_name=shared_feats)
    c1_model = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42}, dtrain, num_boost_round=80)

    # Split C2 strictly
    train_c2, val_c2, test_c2 = split_strict(d2_feat, train_frac=0.70, holdout_frac=0.20, embargo_steps=36, seed=42)
    cols2 = list(dict.fromkeys(shared_feats + [target_abs, "Series", "Time"]))
    test_clean2 = test_c2[cols2].dropna()

    results = []
    target_servers = test_clean2["Series"].unique()

    for budget in [0, 1, 6, 24, 72, 168]:
        budget_maes_zero, budget_maes_offset, budget_maes_boost = [], [], []
        for server in target_servers:
            srv_df = test_clean2[test_clean2["Series"] == server].sort_values("Time")
            if len(srv_df) < 150:
                continue
            t_min = srv_df["Time"].min()
            t_eval = t_min + pd.Timedelta(hours=174)
            eval_df = srv_df[srv_df["Time"] >= t_eval]
            if len(eval_df) < 30:
                continue

            y_eval = eval_df[target_abs].values
            X_eval = eval_df[shared_feats].values
            pred_zero = c1_model.predict(X_eval)
            budget_maes_zero.append(mae(y_eval, pred_zero))

            if budget > 0:
                adapt_df = srv_df[(srv_df["Time"] >= t_min) & (srv_df["Time"] < t_min + pd.Timedelta(hours=budget))]
                if len(adapt_df) >= 3:
                    off = np.mean(adapt_df[target_abs].values - c1_model.predict(adapt_df[shared_feats].values))
                    budget_maes_offset.append(mae(y_eval, pred_zero + off))

                    dadapt = lgb.Dataset(adapt_df[shared_feats].values, label=adapt_df[target_abs].values, feature_name=shared_feats)
                    finetuned = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.01, "num_leaves": 31, "verbosity": -1, "seed": 42}, dadapt, num_boost_round=15, init_model=c1_model)
                    budget_maes_boost.append(mae(y_eval, finetuned.predict(X_eval)))

        if budget == 0:
            ci_l, ci_h = bootstrap_ci(budget_maes_zero)
            results.append({"budget_hours": 0, "strategy": "zero_shot", "mean_mae": np.mean(budget_maes_zero), "ci_95_low": ci_l, "ci_95_high": ci_h})
        else:
            if budget_maes_offset:
                ci_l, ci_h = bootstrap_ci(budget_maes_offset)
                results.append({"budget_hours": budget, "strategy": "residual_offset", "mean_mae": np.mean(budget_maes_offset), "ci_95_low": ci_l, "ci_95_high": ci_h})
            if budget_maes_boost:
                ci_l, ci_h = bootstrap_ci(budget_maes_boost)
                results.append({"budget_hours": budget, "strategy": "continued_boosting", "mean_mae": np.mean(budget_maes_boost), "ci_95_low": ci_l, "ci_95_high": ci_h})

    c2_adapt_df = pd.DataFrame(results)
    c2_adapt_df.to_csv(OUT_DIR / "cross_cluster_adaptation.csv", index=False)
    print(c2_adapt_df.to_string())
    return c2_adapt_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Random KFold vs GroupKFold on Identical Rows
# ─────────────────────────────────────────────────────────────────────────────
def run_exp6_random_vs_group(d1_feat):
    print("\n--- 6. Random KFold vs GroupKFold on Identical Rows ---")
    target_abs = "CPU_Temperature_target_6"
    target_delta = "CPU_Temperature_delta_6"

    # Common identical evaluation rows across S1, S2, S3, S4
    all_needed = set()
    for fs in ["S1", "S2", "S3", "S4"]:
        all_needed.update(get_feature_set(fs, d1_feat.columns.tolist()))
    all_needed.update([target_abs, target_delta, "CPU_Temperature", "Series"])
    df_clean = d1_feat[[c for c in all_needed if c in d1_feat.columns]].dropna().reset_index(drop=True)

    results = []

    # 5-fold Random KFold
    rkf = KFold(n_splits=5, shuffle=True, random_state=42)
    # 5-fold GroupKFold
    gkf_splits = get_group_kfold_splits(df_clean, n_splits=5, seed=42)

    for fs in ["S1", "S2", "S3", "S4"]:
        feats = get_feature_set(fs, df_clean.columns.tolist())
        is_monitoring = fs in ["S1", "S4"]
        target = target_delta if is_monitoring else target_abs

        # Random KFold
        rand_maes = []
        for train_idx, test_idx in rkf.split(df_clean):
            tr, te = df_clean.iloc[train_idx], df_clean.iloc[test_idx]
            dtr = lgb.Dataset(tr[feats].values, label=tr[target].values)
            m = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "verbosity": -1, "seed": 42}, dtr, num_boost_round=40)
            pred = m.predict(te[feats].values)
            pred_final = (te["CPU_Temperature"].values + pred) if is_monitoring else pred
            rand_maes.append(mae(te[target_abs].values, pred_final))

        # GroupKFold
        group_maes = []
        for tr_srv, te_srv in gkf_splits:
            tr = df_clean[df_clean["Series"].isin(tr_srv)]
            te = df_clean[df_clean["Series"].isin(te_srv)]
            dtr = lgb.Dataset(tr[feats].values, label=tr[target].values)
            m = lgb.train({"objective": "regression", "metric": "mae", "learning_rate": 0.05, "verbosity": -1, "seed": 42}, dtr, num_boost_round=40)
            pred = m.predict(te[feats].values)
            pred_final = (te["CPU_Temperature"].values + pred) if is_monitoring else pred
            group_maes.append(mae(te[target_abs].values, pred_final))

        results.append({
            "feature_set": fs,
            "regime": "monitoring" if is_monitoring else "planning",
            "random_kfold_mae": float(np.mean(rand_maes)),
            "group_kfold_mae": float(np.mean(group_maes)),
            "leakage_underestimation_pct": float((np.mean(group_maes) - np.mean(rand_maes)) / np.mean(group_maes) * 100.0)
        })

    df_leakage = pd.DataFrame(results)
    df_leakage.to_csv(OUT_DIR / "random_vs_group_identical_rows.csv", index=False)
    print(df_leakage.to_string())
    return df_leakage


# ─────────────────────────────────────────────────────────────────────────────
# 7. S1 LightGBM Learning Rate & Tree Depth Study
# ─────────────────────────────────────────────────────────────────────────────
def run_exp7_learning_rates(d1_feat):
    print("\n--- 7. S1 LightGBM Learning Rate & Early Stopping Study ---")
    train_df, val_df, test_df = split_chronological(d1_feat, train_frac=0.70, val_frac=0.10, embargo_steps=36)
    target_abs = "CPU_Temperature_target_6"
    target_delta = "CPU_Temperature_delta_6"
    feats = get_feature_set("S1", d1_feat.columns.tolist())
    cols = list(dict.fromkeys(feats + [target_abs, target_delta, "CPU_Temperature"]))

    train_c = train_df[cols].dropna()
    val_c = val_df[cols].dropna()
    test_c = test_df[cols].dropna()

    y_test_abs = test_c[target_abs].values
    y_test_curr = test_c["CPU_Temperature"].values
    pers_mae = mae(y_test_abs, y_test_curr)

    dtrain = lgb.Dataset(train_c[feats].values, label=train_c[target_delta].values, feature_name=feats)
    dval = lgb.Dataset(val_c[feats].values, label=val_c[target_delta].values, reference=dtrain, feature_name=feats)

    results = []
    learning_rates = [0.05, 0.01, 0.005]

    for lr in learning_rates:
        evals_result = {}
        params = {"objective": "regression", "metric": "mae", "learning_rate": lr, "num_leaves": 63, "verbosity": -1, "seed": 42}
        booster = lgb.train(
            params, dtrain, num_boost_round=300, valid_sets=[dval],
            callbacks=[lgb.early_stopping(30), lgb.record_evaluation(evals_result)]
        )

        best_iter = booster.best_iteration
        best_val_mae = evals_result["valid_0"]["l1"][best_iter - 1]

        pred_delta = booster.predict(test_c[feats].values)
        pred_abs = y_test_curr + pred_delta
        test_m = mae(y_test_abs, pred_abs)
        test_r = rmse(y_test_abs, pred_abs)
        sk = skill_score_vs_persistence(test_m, pers_mae)

        results.append({
            "learning_rate": lr,
            "best_iteration": best_iter,
            "best_val_mae": best_val_mae,
            "test_mae": test_m,
            "test_rmse": test_r,
            "skill_score_pct": sk * 100.0,
            "persistence_test_mae": pers_mae,
        })

    df_lr = pd.DataFrame(results)
    df_lr.to_csv(OUT_DIR / "s1_learning_rate_study.csv", index=False)
    print(df_lr.to_string())
    return df_lr


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility Controls & Main Execution
# ─────────────────────────────────────────────────────────────────────────────
def get_git_info():
    """Retrieve git commit and dirty status safely."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        is_dirty = len(status) > 0
        return commit, is_dirty
    except Exception:
        return "unknown", False


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description="Run or synthesize v3 comprehensive experiments with explicit reproducibility controls."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Explicitly reuse existing CSV result files if present. Missing outputs will be computed.",
    )
    group.add_argument(
        "--force",
        action="store_true",
        help="Force recomputation of all experiments from scratch, overwriting existing CSVs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check existing experiment outputs and report planned actions without modifying files.",
    )
    return parser.parse_args(args)


EXPERIMENT_CATALOG = [
    {
        "id": "e1_bootstrap",
        "name": "Paired Bootstrap (E1)",
        "file": "e1_bootstrap_summary.csv",
        "runner": lambda d1, d2: run_exp1_bootstrap(d1, d2),
        "needs_d2": True,
    },
    {
        "id": "e2_sensor_ablation",
        "name": "Sensor Hardware Feature Ablation",
        "file": "sensor_ablation.csv",
        "runner": lambda d1, d2: run_exp2_sensor_ablation(d1),
        "needs_d2": False,
    },
    {
        "id": "e3_splits_bc",
        "name": "Splits B & C Horizons",
        "file": "split_b_and_c_horizons.csv",
        "runner": lambda d1, d2: run_exp3_splits_bc_horizons(d1),
        "needs_d2": False,
    },
    {
        "id": "e4_planning_adaptation",
        "name": "Planning Adaptation Across All 5 Folds",
        "file": "planning_adaptation_all_folds.csv",
        "runner": lambda d1, d2: run_exp4_planning_adaptation_all_folds(d1),
        "needs_d2": False,
    },
    {
        "id": "e5_cross_cluster",
        "name": "Cross-Cluster Hardware Adaptation (C1 -> C2)",
        "file": "cross_cluster_adaptation.csv",
        "runner": lambda d1, d2: run_exp5_cross_cluster_adaptation(d1, d2),
        "needs_d2": True,
    },
    {
        "id": "e6_random_vs_group",
        "name": "Random KFold vs GroupKFold on Identical Rows",
        "file": "random_vs_group_identical_rows.csv",
        "runner": lambda d1, d2: run_exp6_random_vs_group(d1),
        "needs_d2": False,
    },
    {
        "id": "e7_learning_rates",
        "name": "S1 LightGBM Learning Rate & Early Stopping Study",
        "file": "s1_learning_rate_study.csv",
        "runner": lambda d1, d2: run_exp7_learning_rates(d1),
        "needs_d2": False,
    },
]


def main(argv=None):
    args = parse_args(argv)
    git_commit, git_dirty = get_git_info()

    print(f"\n{'='*70}\nV3 EXPERIMENTS RUNNER")
    print(f"Git commit: {git_commit} {'(dirty)' if git_dirty else '(clean)'}")
    print(f"Output directory: {OUT_DIR}")
    print(f"{'='*70}")

    existing_files = {exp["file"]: (OUT_DIR / exp["file"]).exists() for exp in EXPERIMENT_CATALOG}
    num_existing = sum(existing_files.values())

    # Safe default: prevent silent overwrite or silent reuse
    if not args.reuse_existing and not args.force and not args.dry_run:
        if num_existing > 0:
            print("\n[SAFEGUARD] Existing experiment results detected in output directory:")
            for exp in EXPERIMENT_CATALOG:
                fn = exp["file"]
                st = "FOUND" if existing_files[fn] else "MISSING"
                print(f"  - {fn:35s}: {st}")
            print("\nTo ensure reproducibility and prevent unintended recomputations, choose an explicit mode:")
            print("  python scripts/run_v3_experiments.py --reuse-existing")
            print("      Reuse valid existing CSVs and synthesize docs/results_v3.md.")
            print("  python scripts/run_v3_experiments.py --force")
            print("      Recompute all experiments from scratch and overwrite existing CSVs.")
            print("  python scripts/run_v3_experiments.py --dry-run")
            print("      Inspect execution plan without computing or writing files.\n")
            return 1

    if args.dry_run:
        print("\n[DRY RUN] Inspection of experiment outputs:")
        for exp in EXPERIMENT_CATALOG:
            fn = exp["file"]
            p = OUT_DIR / fn
            status = f"EXISTS ({p.stat().st_size} bytes)" if p.exists() else "NOT FOUND"
            action = "WOULD REUSE" if (p.exists() and not args.force) else "WOULD COMPUTE"
            print(f"  - {exp['name']:48s} -> {fn:35s}: {status} | {action}")
        return 0

    mode_label = "FORCE RECOMPUTE" if args.force else "REUSE EXISTING"
    print(f"\nExecution Mode: {mode_label}")

    # Determine which experiments need computation
    needs_compute = []
    for exp in EXPERIMENT_CATALOG:
        fn = exp["file"]
        p = OUT_DIR / fn
        if args.force or not p.exists():
            needs_compute.append(exp)
            status_desc = "WILL RECOMPUTE (forced)" if args.force else "WILL COMPUTE (missing file)"
        else:
            status_desc = f"REUSING existing CSV (modified: {datetime.fromtimestamp(p.stat().st_mtime, timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})"
        print(f"  * {exp['file']:35s}: {status_desc}")

    # Lazy-load parquet and feature engineering ONLY if at least one experiment needs computation
    d1_feat, d2_feat = None, None
    if needs_compute:
        print(f"\nLoading raw datasets to execute {len(needs_compute)} experiment(s)...")
        needs_d1 = True
        needs_d2 = any(exp["needs_d2"] for exp in needs_compute)

        if needs_d1:
            print("Building features for Dataset 1...")
            d1 = load_data("dataset1")
            d1_feat = build_features(d1, horizons=[1, 6, 36], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])
        if needs_d2:
            print("Building features for Dataset 2...")
            d2 = load_data("dataset2")
            d2_feat = build_features(d2, horizons=[1, 6, 36], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])

    experiment_status = {}
    results = {}

    for exp in EXPERIMENT_CATALOG:
        fn = exp["file"]
        p = OUT_DIR / fn
        if args.force or not p.exists():
            print(f"\n>>> Running {exp['name']}...")
            res_df = exp["runner"](d1_feat, d2_feat)
            results[exp["id"]] = res_df
            experiment_status[fn] = "COMPUTED"
        else:
            print(f"Reading existing {fn}...")
            results[exp["id"]] = pd.read_csv(p)
            experiment_status[fn] = "REUSED"

    # Save execution manifest for complete audit trail
    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "cli_mode": "force" if args.force else "reuse-existing",
        "protocol_frozen": True,
        "random_seeds": {
            "bootstrap_seed": 42,
            "split_b_seed": 42,
            "split_c_seeds": [42, 123, 456, 789, 1024],
            "lightgbm_seed": 42,
        },
        "dataset_info": {
            "dataset1": "Melbourne Cloud Cluster 1 (Dell PowerEdge R840)",
            "dataset2": "Melbourne Cloud Cluster 2 (Dell PowerEdge C6320)",
            "frequency": "10min",
            "horizons_steps": [1, 6, 36],
            "lags": [1, 2, 3, 6, 12],
            "rolling_windows": [6, 12, 36],
            "embargo_steps": 36,
        },
        "experiment_status": experiment_status,
    }
    manifest_path = OUT_DIR / "execution_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nExecution metadata recorded in {manifest_path}")

    # Synthesize docs/results_v3.md using shared verified narrative builder
    md_content = build_markdown_content(
        results["e1_bootstrap"],
        results["e2_sensor_ablation"],
        results["e3_splits_bc"],
        results["e4_planning_adaptation"],
        results["e5_cross_cluster"],
        results["e6_random_vs_group"],
        results["e7_learning_rates"],
    )
    with open(ROOT / "docs" / "results_v3.md", "w", encoding="utf-8") as f:
        f.write(md_content)
    print("Saved docs/results_v3.md successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
