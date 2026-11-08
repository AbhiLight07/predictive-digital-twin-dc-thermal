#!/usr/bin/env python3
"""P1 Diagnostics: target formulation, ablation on identical rows, delta target, Split B/C analysis.

Outputs to experiments/diag_* and prints findings to stdout.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from ttwin.io import load_parquet
from ttwin.features import build_features, get_feature_set
from ttwin.splits import (
    split_chronological, split_leave_servers_out, split_strict,
    get_split_b_seeds,
)
from ttwin.metrics import mae, rmse, skill_score_vs_persistence, per_server_metrics, summarize_per_server

DATA_DIR = ROOT / "data" / "interim"
EXP_DIR = ROOT / "experiments"

LGB_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "verbosity": -1,
    "seed": 42,
    "n_jobs": -1,
}


def load_and_prepare(dataset="dataset1"):
    combined = DATA_DIR / dataset / "_combined.parquet"
    print(f"Loading {combined}...")
    df = load_parquet(combined)
    print(f"Loaded {len(df):,} rows, {df['Series'].nunique()} servers")
    df = build_features(df, target_col="CPU_Temperature", horizons=[1, 6, 36],
                        lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])
    return df


def diag_1_1(df):
    """1.1: Show that the target is T(t+h), not delta."""
    print("\n" + "=" * 70)
    print("1.1 TARGET FORMULATION")
    print("=" * 70)
    print()
    print("Target construction code (features.py lines 61-71):")
    print("  def create_target(df, target_col, horizon_steps, server_col='Series'):")
    print("    target_name = f'{target_col}_target_{horizon_steps}'")
    print("    df[target_name] = df.groupby(server_col)[target_col].shift(-horizon_steps)")
    print()
    print("FINDING: Target is the ABSOLUTE value T(t+h), NOT the change T(t+h) - T(t).")
    print()

    # Verify with actual data
    sample = df[df["Series"] == df["Series"].iloc[0]].head(5)
    t_col = "CPU_Temperature"
    for h_name, h_steps in [("10min", 1), ("1h", 6), ("6h", 36)]:
        tgt = f"{t_col}_target_{h_steps}"
        print(f"  Horizon {h_name} (steps={h_steps}):")
        for i in range(min(3, len(sample))):
            row = sample.iloc[i]
            print(f"    row {i}: T(t)={row[t_col]:.2f}, target T(t+{h_steps})={row.get(tgt, 'NaN')}")
    print()

    # Features for S1
    s1_feats = get_feature_set("S1", df.columns.tolist())
    print(f"  S1 features ({len(s1_feats)}): {s1_feats}")
    print()
    s4_feats = get_feature_set("S4", df.columns.tolist())
    print(f"  S4 features ({len(s4_feats)}): {s4_feats}")
    print()

    # KEY CHECK: Does S1/S4 include lag_0 or the current temperature?
    has_current = any("lag_0" in f or f == "CPU_Temperature" for f in s1_feats + s4_feats)
    print(f"  CRITICAL CHECK: Do S1/S4 include current temperature T(t) as a feature? {has_current}")
    if not has_current:
        print("  NOTE: T(t) is NOT explicitly in the feature set. The closest is lag_1 = T(t-1).")
        print("  Persistence uses T(t). The ML models use T(t-1) as their most recent info.")
        print("  This means persistence has a STRUCTURAL ADVANTAGE over S1/S4 at short horizons!")
    print()
    return s1_feats, s4_feats


def diag_1_2(df, s1_feats, s4_feats):
    """1.2: Unified ablation on identical rows, Split A."""
    print("\n" + "=" * 70)
    print("1.2 UNIFIED ABLATION TABLE — SPLIT A, IDENTICAL ROWS")
    print("=" * 70)

    horizons = {"10min": 1, "1h": 6, "6h": 36}
    target_col = "CPU_Temperature"

    all_results = []

    for h_name, h_steps in horizons.items():
        target_name = f"{target_col}_target_{h_steps}"

        # Get all feature sets
        feature_sets = {
            "S1": get_feature_set("S1", df.columns.tolist()),
            "S2": get_feature_set("S2", df.columns.tolist()),
            "S3": get_feature_set("S3", df.columns.tolist()),
            "S4": get_feature_set("S4", df.columns.tolist()),
        }

        # Widest feature set determines the common rows (after dropna)
        all_feature_cols = sorted(set(sum(feature_sets.values(), [])))
        common_cols = all_feature_cols + [target_name, target_col, "Series", "Time"]
        common_cols = [c for c in common_cols if c in df.columns]

        # Split first
        train_df, val_df, test_df = split_chronological(df, embargo_steps=h_steps)

        # Find intersection of non-NaN rows across ALL feature sets in the test set
        test_subset = test_df[common_cols].copy()
        mask = test_subset[target_name].notna()
        for fs_name, fs_cols in feature_sets.items():
            for c in fs_cols:
                if c in test_subset.columns:
                    mask = mask & test_subset[c].notna()
        mask = mask & test_subset[target_col].notna()

        test_common = test_subset[mask].copy()
        y_test = test_common[target_name].values
        y_current = test_common[target_col].values
        n_rows = len(test_common)

        print(f"\n--- Horizon: {h_name} ({h_steps} steps) | Common test rows: {n_rows:,} ---")

        # Persistence on IDENTICAL rows
        mae_pers = mae(y_test, y_current)
        rmse_pers = rmse(y_test, y_current)

        # Per-server persistence
        test_common["_pers_pred"] = y_current
        ps_pers = per_server_metrics(test_common, target_name, "_pers_pred")

        all_results.append({
            "horizon": h_name, "feature_set": "S0", "model": "persistence",
            "mae": mae_pers, "rmse": rmse_pers, "skill": 0.0,
            "ps_median_mae": ps_pers["mae"].median(),
            "ps_worst_decile": ps_pers["mae"].quantile(0.9),
            "n_rows": n_rows,
        })

        # Train all feature sets
        for fs_name, fs_cols in feature_sets.items():
            # Train set: also need common column intersection
            train_clean = train_df[fs_cols + [target_name]].dropna()
            val_clean = val_df[fs_cols + [target_name]].dropna()

            X_train = train_clean[fs_cols].values
            y_train = train_clean[target_name].values
            X_val = val_clean[fs_cols].values
            y_val = val_clean[target_name].values
            X_test_fs = test_common[fs_cols].values

            # LightGBM
            dtrain = lgb.Dataset(X_train, label=y_train, feature_name=fs_cols)
            dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=fs_cols)
            model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500,
                              valid_sets=[dtrain, dval], valid_names=["train", "valid"],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            y_pred = model.predict(X_test_fs)

            mae_v = mae(y_test, y_pred)
            rmse_v = rmse(y_test, y_pred)
            skill = skill_score_vs_persistence(mae_v, mae_pers)

            test_common[f"_pred_{fs_name}"] = y_pred
            ps = per_server_metrics(test_common, target_name, f"_pred_{fs_name}")

            all_results.append({
                "horizon": h_name, "feature_set": fs_name, "model": "lightgbm",
                "mae": mae_v, "rmse": rmse_v, "skill": skill,
                "ps_median_mae": ps["mae"].median(),
                "ps_worst_decile": ps["mae"].quantile(0.9),
                "n_rows": n_rows,
            })

            # Ridge for S1 and S4
            if fs_name in ("S1", "S4"):
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_train)
                ridge = Ridge(alpha=1.0, random_state=42)
                ridge.fit(X_tr_s, y_train)
                y_pred_r = ridge.predict(scaler.transform(X_test_fs))

                mae_r = mae(y_test, y_pred_r)
                rmse_r = rmse(y_test, y_pred_r)
                skill_r = skill_score_vs_persistence(mae_r, mae_pers)

                test_common[f"_pred_ridge_{fs_name}"] = y_pred_r
                ps_r = per_server_metrics(test_common, target_name, f"_pred_ridge_{fs_name}")

                all_results.append({
                    "horizon": h_name, "feature_set": fs_name, "model": "ridge",
                    "mae": mae_r, "rmse": rmse_r, "skill": skill_r,
                    "ps_median_mae": ps_r["mae"].median(),
                    "ps_worst_decile": ps_r["mae"].quantile(0.9),
                    "n_rows": n_rows,
                })

    results_df = pd.DataFrame(all_results)
    print("\n" + "-" * 100)
    print("UNIFIED ABLATION TABLE (Split A, Dataset 1, identical rows per horizon)")
    print("-" * 100)
    print(results_df.to_string(index=False, float_format="%.4f"))

    out_dir = EXP_DIR / "diag_unified_ablation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "results.csv", index=False)
    return results_df


def diag_1_3(df):
    """1.3: Delta target formulation."""
    print("\n" + "=" * 70)
    print("1.3 DELTA TARGET: T(t+h) - T(t)")
    print("=" * 70)

    horizons = {"10min": 1, "1h": 6, "6h": 36}
    target_col = "CPU_Temperature"
    all_results = []

    for h_name, h_steps in horizons.items():
        target_name = f"{target_col}_target_{h_steps}"
        delta_target = f"delta_{h_steps}"

        # Create delta target
        df_local = df.copy()
        df_local[delta_target] = df_local[target_name] - df_local[target_col]

        for fs_name in ["S1", "S4"]:
            fs_cols = get_feature_set(fs_name, df_local.columns.tolist())

            train_df, val_df, test_df = split_chronological(df_local, embargo_steps=h_steps)

            train_clean = train_df[fs_cols + [delta_target, target_name, target_col]].dropna()
            val_clean = val_df[fs_cols + [delta_target, target_name, target_col]].dropna()
            test_clean = test_df[fs_cols + [delta_target, target_name, target_col]].dropna()

            X_train = train_clean[fs_cols].values
            y_train_delta = train_clean[delta_target].values
            X_val = val_clean[fs_cols].values
            y_val_delta = val_clean[delta_target].values
            X_test = test_clean[fs_cols].values
            y_test_abs = test_clean[target_name].values
            y_current_test = test_clean[target_col].values

            # LightGBM on delta target
            dtrain = lgb.Dataset(X_train, label=y_train_delta, feature_name=fs_cols)
            dval = lgb.Dataset(X_val, label=y_val_delta, reference=dtrain, feature_name=fs_cols)
            model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500,
                              valid_sets=[dtrain, dval], valid_names=["train", "valid"],
                              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
            delta_pred = model.predict(X_test)
            # Convert back to absolute: pred_T(t+h) = T(t) + predicted_delta
            y_pred_abs = y_current_test + delta_pred

            mae_v = mae(y_test_abs, y_pred_abs)
            rmse_v = rmse(y_test_abs, y_pred_abs)
            mae_pers = mae(y_test_abs, y_current_test)
            skill = skill_score_vs_persistence(mae_v, mae_pers)

            all_results.append({
                "horizon": h_name, "feature_set": fs_name, "model": "lgbm_delta",
                "mae": mae_v, "rmse": rmse_v, "skill_vs_pers": skill,
                "mae_persistence": mae_pers, "n_rows": len(X_test),
            })

            # Ridge on delta target
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_train)
            ridge = Ridge(alpha=1.0, random_state=42)
            ridge.fit(X_tr_s, y_train_delta)
            delta_pred_r = ridge.predict(scaler.transform(X_test))
            y_pred_abs_r = y_current_test + delta_pred_r

            mae_r = mae(y_test_abs, y_pred_abs_r)
            rmse_r = rmse(y_test_abs, y_pred_abs_r)
            skill_r = skill_score_vs_persistence(mae_r, mae_pers)

            all_results.append({
                "horizon": h_name, "feature_set": fs_name, "model": "ridge_delta",
                "mae": mae_r, "rmse": rmse_r, "skill_vs_pers": skill_r,
                "mae_persistence": mae_pers, "n_rows": len(X_test),
            })

    results_df = pd.DataFrame(all_results)
    print("\nDELTA TARGET RESULTS:")
    print(results_df.to_string(index=False, float_format="%.4f"))

    out_dir = EXP_DIR / "diag_delta_target"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "results.csv", index=False)
    return results_df


def diag_1_4(df):
    """1.4: Split B — all 5 seeds, per-seed persistence."""
    print("\n" + "=" * 70)
    print("1.4 SPLIT B — ALL 5 SEEDS, S4 LightGBM, 1h horizon")
    print("=" * 70)

    target_col = "CPU_Temperature"
    h_steps = 6
    target_name = f"{target_col}_target_{h_steps}"
    fs_cols = get_feature_set("S4", df.columns.tolist())
    seeds = get_split_b_seeds()
    results = []

    for seed in seeds:
        train_full, test_df, train_servers, test_servers = split_leave_servers_out(df, seed=seed)
        train_df, val_df, _ = split_chronological(train_full, val_frac=0.1, embargo_steps=h_steps)

        train_clean = train_df[fs_cols + [target_name]].dropna()
        val_clean = val_df[fs_cols + [target_name]].dropna()
        test_clean = test_df[fs_cols + [target_name, target_col]].dropna()

        X_train = train_clean[fs_cols].values
        y_train = train_clean[target_name].values
        X_val = val_clean[fs_cols].values
        y_val = val_clean[target_name].values
        X_test = test_clean[fs_cols].values
        y_test = test_clean[target_name].values
        y_current = test_clean[target_col].values

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=fs_cols)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain, feature_name=fs_cols)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=500,
                          valid_sets=[dtrain, dval], valid_names=["train", "valid"],
                          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        y_pred = model.predict(X_test)

        mae_model = mae(y_test, y_pred)
        rmse_model = rmse(y_test, y_pred)
        mae_pers = mae(y_test, y_current)
        skill = skill_score_vs_persistence(mae_model, mae_pers)

        results.append({
            "seed": seed, "n_test_servers": len(test_servers),
            "n_test_rows": len(X_test),
            "mae_model": mae_model, "rmse_model": rmse_model,
            "mae_persistence": mae_pers, "skill_vs_pers": skill,
            "test_servers": str(test_servers[:3]) + "...",
        })
        print(f"  seed={seed}: MAE={mae_model:.4f}, RMSE={rmse_model:.4f}, "
              f"Pers MAE={mae_pers:.4f}, Skill={skill:.4f}, "
              f"n_servers={len(test_servers)}, n_rows={len(X_test)}")

    results_df = pd.DataFrame(results)
    mae_vals = results_df["mae_model"].values
    print(f"\n  Mean MAE: {mae_vals.mean():.4f} +/- {mae_vals.std():.4f}")
    print(f"  Was the reported 0.527 a single seed? The original experiment used --seeds 42 only.")

    out_dir = EXP_DIR / "diag_split_b_5seeds"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(out_dir / "results.csv", index=False)
    return results_df


def diag_1_5(df):
    """1.5: Split C analysis — why is C < B?"""
    print("\n" + "=" * 70)
    print("1.5 SPLIT C vs B vs A — TEST WINDOW ANALYSIS")
    print("=" * 70)

    target_col = "CPU_Temperature"
    h_steps = 6
    target_name = f"{target_col}_target_{h_steps}"
    df["Time"] = pd.to_datetime(df["Time"])

    # Split A
    train_a, val_a, test_a = split_chronological(df, embargo_steps=h_steps)
    test_a_clean = test_a[[target_name, target_col, "Time", "Series"]].dropna()
    y_test_a = test_a_clean[target_name].values
    y_curr_a = test_a_clean[target_col].values

    # Split B (seed=42)
    train_b, test_b, _, test_servers_b = split_leave_servers_out(df, seed=42)
    test_b_clean = test_b[[target_name, target_col, "Time", "Series"]].dropna()
    y_test_b = test_b_clean[target_name].values
    y_curr_b = test_b_clean[target_col].values

    # Split C (seed=42)
    train_c, val_c, test_c = split_strict(df, embargo_steps=h_steps, seed=42)
    test_c_clean = test_c[[target_name, target_col, "Time", "Series"]].dropna()
    y_test_c = test_c_clean[target_name].values
    y_curr_c = test_c_clean[target_col].values

    for label, test_clean, y_test, y_curr in [
        ("A", test_a_clean, y_test_a, y_curr_a),
        ("B", test_b_clean, y_test_b, y_curr_b),
        ("C", test_c_clean, y_test_c, y_curr_c),
    ]:
        n_srv = test_clean["Series"].nunique()
        t_min, t_max = test_clean["Time"].min(), test_clean["Time"].max()
        delta_t = np.abs(y_test - y_curr)
        print(f"\n  Split {label}:")
        print(f"    Time range: {t_min} to {t_max}")
        print(f"    Servers: {n_srv}")
        print(f"    Rows: {len(test_clean):,}")
        print(f"    Persistence MAE: {mae(y_test, y_curr):.4f}")
        print(f"    T(t) mean={np.mean(y_curr):.2f}, std={np.std(y_curr):.2f}")
        print(f"    |delta T| mean={np.mean(delta_t):.4f}, std={np.std(delta_t):.4f}, max={np.max(delta_t):.4f}")

    print()
    print("  EXPLANATION: Split C uses UNSEEN servers (20%) in the LATER time window.")
    print("  If those servers happen to be more stable (lower |delta T|), the model")
    print("  and persistence will both show lower MAE. This does NOT mean C is 'easier'")
    print("  — it reflects the temperature dynamics of the specific held-out servers.")


def diag_1_6():
    """1.6: Which feature sets/horizons were run under B and C?"""
    print("\n" + "=" * 70)
    print("1.6 EXPERIMENTS RUN UNDER SPLITS B AND C")
    print("=" * 70)

    exp_dirs = sorted(list(EXP_DIR.glob("dataset*")))
    for d in exp_dirs:
        parts = d.name.split("_")
        if len(parts) >= 5:
            ds = parts[0]
            split = parts[1]
            fs = parts[2]
            model = parts[3]
            horizon = parts[4]
            if split in ("B", "C"):
                print(f"  {d.name}: {ds} | Split {split} | {fs} | {model} | {horizon}")

    # Check D2
    d2_dirs = list(EXP_DIR.glob("dataset2*"))
    print(f"\n  Dataset 2 E1 experiments: {len(d2_dirs)}")
    if d2_dirs:
        for d in d2_dirs:
            print(f"    {d.name}")
    else:
        print("  NOT DONE — Dataset 2 was NOT run in E1.")


def main():
    df = load_and_prepare("dataset1")

    print("\n" + "#" * 80)
    print("# P1 DIAGNOSTICS")
    print("#" * 80)

    s1_feats, s4_feats = diag_1_1(df)
    ablation_df = diag_1_2(df, s1_feats, s4_feats)
    delta_df = diag_1_3(df)
    splitb_df = diag_1_4(df)
    diag_1_5(df)
    diag_1_6()

    print("\n" + "#" * 80)
    print("# P1 DIAGNOSTICS COMPLETE")
    print("#" * 80)


if __name__ == "__main__":
    main()
