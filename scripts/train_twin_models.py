#!/usr/bin/env python3
"""Train and save twin models (monitoring + planning) for the predictive digital twin.

Usage:
    python scripts/train_twin_models.py --cluster d1 --horizons 1 6 36 --out models/twin
    python scripts/train_twin_models.py --cluster d2 --horizons 1 6 36 --out models/twin
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.io import load_all_servers
from ttwin.features import (
    build_features,
    compute_effective_cpu_temp,
    add_calendar_features,
    add_lag_features,
    add_rolling_features,
    get_feature_set,
)
from ttwin.clean import fill_missing_timestamps
from ttwin.metrics import mae, rmse, skill_score_vs_persistence
from ttwin.uncertainty import compute_conformal_quantile, coverage, average_width

SEED = 42
LR = 0.05
EARLY_STOP = 50
EMBARGO = 36  # 36 steps = 6 hours

CLUSTER_MAP = {
    "d1": {"dataset": "dataset1", "interim_dir": ROOT / "data" / "interim" / "dataset1",
           "server_csv": ROOT / "docs" / "snapshot" / "server_sets" / "dataset1_e1_split_a_servers.csv"},
    "d2": {"dataset": "dataset2", "interim_dir": ROOT / "data" / "interim" / "dataset2",
           "server_csv": ROOT / "docs" / "snapshot" / "server_sets" / "dataset2_e1_split_a_servers.csv"},
}

# Planning feature building columns
WORKLOAD_COLS = [
    "CPU_Load", "Ram", "Ram_Used", "CPU_cores", "CPU_cores_used",
    "Network_RX", "Network_TX", "No_Of_Running_vms",
]
ROLLING_PLAN_COLS = ["CPU_Load", "Ram_Used", "No_Of_Running_vms", "Network_RX", "Network_TX"]
PLAN_ROLLING_WINDOWS = [6, 36]  # 1h, 6h in 10-min steps


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def chronological_split(df, time_col="Time"):
    """Split globally by time: train 0-70%, embargo, val_sel 70-75%, embargo,
    val_cal 75-80%, embargo, test 80-100%."""
    times = df[time_col].sort_values().unique()
    n = len(times)
    t_train_end = times[int(n * 0.70) - 1]
    t_val_sel_start = times[min(int(n * 0.70) + EMBARGO, n - 1)]
    t_val_sel_end = times[int(n * 0.75) - 1]
    t_val_cal_start = times[min(int(n * 0.75) + EMBARGO, n - 1)]
    t_val_cal_end = times[int(n * 0.80) - 1]
    t_test_start = times[min(int(n * 0.80) + EMBARGO, n - 1)]

    train = df[df[time_col] <= t_train_end].copy()
    val_sel = df[(df[time_col] >= t_val_sel_start) & (df[time_col] <= t_val_sel_end)].copy()
    val_cal = df[(df[time_col] >= t_val_cal_start) & (df[time_col] <= t_val_cal_end)].copy()
    test = df[df[time_col] >= t_test_start].copy()

    # Assert disjoint timestamps
    sets = [set(train[time_col]), set(val_sel[time_col]), set(val_cal[time_col]), set(test[time_col])]
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            overlap = sets[i] & sets[j]
            assert len(overlap) == 0, f"Partition overlap between {i} and {j}: {len(overlap)} timestamps"

    boundaries = {
        "train_end": str(t_train_end),
        "val_sel_start": str(t_val_sel_start),
        "val_sel_end": str(t_val_sel_end),
        "val_cal_start": str(t_val_cal_start),
        "val_cal_end": str(t_val_cal_end),
        "test_start": str(t_test_start),
        "test_end": str(times[-1]),
    }
    counts = {
        "train": len(train), "val_sel": len(val_sel),
        "val_cal": len(val_cal), "test": len(test),
    }
    return train, val_sel, val_cal, test, boundaries, counts


def train_lgbm(X_train, y_train, X_val, y_val, objective="regression", alpha=None):
    params = {
        "objective": objective,
        "learning_rate": LR,
        "verbosity": -1,
        "seed": SEED,
        "n_jobs": -1,
        "metric": "mae",
    }
    if objective == "quantile":
        params["alpha"] = alpha
        params["metric"] = "quantile"
    ds_train = lgb.Dataset(X_train, y_train)
    ds_val = lgb.Dataset(X_val, y_val, reference=ds_train)
    model = lgb.train(
        params, ds_train,
        num_boost_round=2000,
        valid_sets=[ds_val],
        callbacks=[lgb.early_stopping(EARLY_STOP, verbose=False)],
    )
    return model


def build_monitoring_features(df):
    """Build features for monitoring regime (S1 + existing pipeline)."""
    df = fill_missing_timestamps(df, host_col="Series", time_col="Time", freq="10min")
    df = compute_effective_cpu_temp(df)
    df = add_calendar_features(df, "Time")
    temp_cols = [c for c in ["CPU_Temperature", "CPU1_Temperature", "CPU2_Temperature"] if c in df.columns]
    df = add_lag_features(df, temp_cols, [1, 2, 3, 6, 12], "Series")
    df = add_rolling_features(df, temp_cols, [6, 12, 36], "Series")
    return df


def build_planning_features(df):
    """Build features for planning regime: workload + rolling means + static + calendar."""
    df = fill_missing_timestamps(df, host_col="Series", time_col="Time", freq="10min")
    df = compute_effective_cpu_temp(df)
    df = add_calendar_features(df, "Time")
    # Rolling means of workload signals
    for col in ROLLING_PLAN_COLS:
        if col in df.columns:
            for w in PLAN_ROLLING_WINDOWS:
                rolling = df.groupby("Series")[col].rolling(window=w, min_periods=1)
                df[f"{col}_rmean_{w}"] = rolling.mean().reset_index(level=0, drop=True)
    return df


def get_monitoring_features(df_columns):
    """Return S1 feature list from available columns."""
    return get_feature_set("S1", list(df_columns))


def get_planning_features(df_columns):
    """Return planning feature list from available columns."""
    base = WORKLOAD_COLS + ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "CPU_cores", "Ram"]
    rolling = [f"{c}_rmean_{w}" for c in ROLLING_PLAN_COLS for w in PLAN_ROLLING_WINDOWS]
    all_feats = base + rolling
    return [c for c in all_feats if c in df_columns]


def compute_coverage_by_regime(test_df, y_true_col, lower_col, upper_col, load_col="CPU_Load"):
    """Coverage by CPU load regime: idle (<10%), active (10-50%), heavy (>50%)."""
    result = {}
    if load_col not in test_df.columns:
        return {"idle": None, "active": None, "heavy": None}
    valid = test_df[[y_true_col, lower_col, upper_col, load_col]].dropna()
    if len(valid) == 0:
        return {"idle": None, "active": None, "heavy": None}
    for name, mask in [
        ("idle", valid[load_col] < 10),
        ("active", (valid[load_col] >= 10) & (valid[load_col] <= 50)),
        ("heavy", valid[load_col] > 50),
    ]:
        sub = valid[mask]
        if len(sub) > 0:
            result[name] = coverage(sub[y_true_col].values, sub[lower_col].values, sub[upper_col].values)
        else:
            result[name] = None
    return result


def train_monitoring(cluster, horizons, out_dir, df, servers):
    """Train monitoring models (delta target, S1 features) for each horizon."""
    print(f"  Building monitoring features...")
    df_feat = build_monitoring_features(df)
    feat_cols = get_monitoring_features(df_feat.columns)
    print(f"  Monitoring features ({len(feat_cols)}): {feat_cols[:5]}...")

    results = []
    models_info = []

    for h in horizons:
        t0 = time.time()
        target_col = f"CPU_Temperature_target_{h}"
        delta_col = f"CPU_Temperature_delta_{h}"

        # Create targets
        from ttwin.features import create_target
        df_h = create_target(df_feat, "CPU_Temperature", h, "Series")

        # Drop rows with NaN target or features
        keep_cols = ["Series", "Time", "CPU_Temperature", "CPU_Load"] + feat_cols + [target_col, delta_col]
        keep_cols = list(dict.fromkeys(c for c in keep_cols if c in df_h.columns))
        df_h = df_h[keep_cols].dropna()

        # Split
        train, val_sel, val_cal, test, boundaries, counts = chronological_split(df_h)

        X_tr, y_tr = train[feat_cols].values, train[delta_col].values
        X_vs, y_vs = val_sel[feat_cols].values, val_sel[delta_col].values
        X_vc, y_vc = val_cal[feat_cols].values, val_cal[delta_col].values
        X_te, y_te_delta = test[feat_cols].values, test[delta_col].values
        y_te_abs = test[target_col].values
        t_current_test = test["CPU_Temperature"].values

        # Train point model (delta)
        m_point = train_lgbm(X_tr, y_tr, X_vs, y_vs)
        # Train quantile models (delta)
        m_lower = train_lgbm(X_tr, y_tr, X_vs, y_vs, "quantile", 0.05)
        m_upper = train_lgbm(X_tr, y_tr, X_vs, y_vs, "quantile", 0.95)

        # CQR on val_cal
        pred_delta_vc = m_point.predict(X_vc)
        pred_lower_vc = m_lower.predict(X_vc)
        pred_upper_vc = m_upper.predict(X_vc)

        # CQR scores on val_cal (in absolute temperature space)
        abs_lower_vc = val_cal["CPU_Temperature"].values + pred_lower_vc
        abs_upper_vc = val_cal["CPU_Temperature"].values + pred_upper_vc
        abs_true_vc = val_cal[target_col].values

        scores_lo = abs_lower_vc - abs_true_vc  # positive if lower bound is too high
        scores_hi = abs_true_vc - abs_upper_vc  # positive if upper bound is too low
        cqr_scores = np.maximum(scores_lo, scores_hi)
        q_hat = compute_conformal_quantile(cqr_scores, alpha=0.10)

        # Test predictions
        pred_delta_te = m_point.predict(X_te)
        pred_abs_te = t_current_test + pred_delta_te
        pred_lower_te = t_current_test + m_lower.predict(X_te) - q_hat
        pred_upper_te = t_current_test + m_upper.predict(X_te) + q_hat

        # Persistence baseline (absolute temperature)
        persistence_abs = t_current_test  # T(t+h) = T(t)

        # Metrics
        test_mae = mae(y_te_abs, pred_abs_te)
        persist_mae = mae(y_te_abs, persistence_abs)
        skill = skill_score_vs_persistence(test_mae, persist_mae)
        emp_cov = coverage(y_te_abs, pred_lower_te, pred_upper_te)
        avg_width = average_width(pred_lower_te, pred_upper_te)
        elapsed = time.time() - t0

        # Coverage by regime
        test_eval = test.copy()
        test_eval["pred_lower"] = pred_lower_te
        test_eval["pred_upper"] = pred_upper_te
        cov_regime = compute_coverage_by_regime(test_eval, target_col, "pred_lower", "pred_upper")

        # Save models
        prefix = f"{cluster}_monitoring_{h}"
        m_point.save_model(str(out_dir / f"{prefix}_point.txt"))
        m_lower.save_model(str(out_dir / f"{prefix}_lower.txt"))
        m_upper.save_model(str(out_dir / f"{prefix}_upper.txt"))

        results.append({
            "cluster": cluster, "kind": "monitoring", "horizon": h,
            "test_mae": round(test_mae, 4), "persist_mae": round(persist_mae, 4),
            "skill": round(skill, 4), "coverage": round(emp_cov, 4),
            "width": round(avg_width, 2), "time_s": round(elapsed, 1),
        })
        models_info.append({
            "kind": "monitoring", "horizon": h,
            "feature_names": feat_cols,
            "target": delta_col,
            "boundaries": boundaries,
            "counts": counts,
            "q_hat": float(q_hat),
            "test_mae": test_mae, "persistence_mae": persist_mae,
            "skill_score": skill, "empirical_coverage": emp_cov,
            "mean_interval_width": avg_width,
            "coverage_by_regime": cov_regime,
            "n_test_rows": int(counts["test"]),
        })
        print(f"    h={h}: MAE={test_mae:.4f}, persist={persist_mae:.4f}, cov={emp_cov:.4f}, width={avg_width:.2f} ({elapsed:.1f}s)")

    return results, models_info


def train_planning(cluster, out_dir, df, servers):
    """Train planning model: workload → absolute CPU temp (contemporaneous)."""
    t0 = time.time()
    print(f"  Building planning features...")
    df_feat = build_planning_features(df)
    feat_cols = get_planning_features(df_feat.columns)
    print(f"  Planning features ({len(feat_cols)}): {feat_cols[:5]}...")

    target_col = "CPU_Temperature"
    keep_cols = ["Series", "Time"] + feat_cols + [target_col]
    keep_cols = list(dict.fromkeys(c for c in keep_cols if c in df_feat.columns))
    df_feat = df_feat[keep_cols].dropna()

    train, val_sel, val_cal, test, boundaries, counts = chronological_split(df_feat)

    X_tr, y_tr = train[feat_cols].values, train[target_col].values
    X_vs, y_vs = val_sel[feat_cols].values, val_sel[target_col].values
    X_vc, y_vc = val_cal[feat_cols].values, val_cal[target_col].values
    X_te, y_te = test[feat_cols].values, test[target_col].values

    # Train
    m_point = train_lgbm(X_tr, y_tr, X_vs, y_vs)
    m_lower = train_lgbm(X_tr, y_tr, X_vs, y_vs, "quantile", 0.05)
    m_upper = train_lgbm(X_tr, y_tr, X_vs, y_vs, "quantile", 0.95)

    # CQR on val_cal
    pred_vc = m_point.predict(X_vc)
    pred_lower_vc = m_lower.predict(X_vc)
    pred_upper_vc = m_upper.predict(X_vc)
    scores_lo = pred_lower_vc - y_vc
    scores_hi = y_vc - pred_upper_vc
    cqr_scores = np.maximum(scores_lo, scores_hi)
    q_hat = compute_conformal_quantile(cqr_scores, alpha=0.10)

    # Test
    pred_te = m_point.predict(X_te)
    pred_lower_te = m_lower.predict(X_te) - q_hat
    pred_upper_te = m_upper.predict(X_te) + q_hat

    test_mae_val = mae(y_te, pred_te)
    emp_cov = coverage(y_te, pred_lower_te, pred_upper_te)
    avg_width = average_width(pred_lower_te, pred_upper_te)
    elapsed = time.time() - t0

    # Coverage by regime
    test_eval = test.copy()
    test_eval["pred_lower"] = pred_lower_te
    test_eval["pred_upper"] = pred_upper_te
    cov_regime = compute_coverage_by_regime(test_eval, target_col, "pred_lower", "pred_upper")

    # Save
    prefix = f"{cluster}_planning_0"
    m_point.save_model(str(out_dir / f"{prefix}_point.txt"))
    m_lower.save_model(str(out_dir / f"{prefix}_lower.txt"))
    m_upper.save_model(str(out_dir / f"{prefix}_upper.txt"))

    print(f"    planning: MAE={test_mae_val:.4f}, cov={emp_cov:.4f}, width={avg_width:.2f} ({elapsed:.1f}s)")
    result = {
        "cluster": cluster, "kind": "planning", "horizon": 0,
        "test_mae": round(test_mae_val, 4), "persist_mae": None,
        "skill": None, "coverage": round(emp_cov, 4),
        "width": round(avg_width, 2), "time_s": round(elapsed, 1),
    }
    info = {
        "kind": "planning", "horizon": 0,
        "feature_names": feat_cols,
        "target": target_col,
        "boundaries": boundaries,
        "counts": counts,
        "q_hat": float(q_hat),
        "test_mae": test_mae_val,
        "persistence_mae": None,
        "skill_score": None,
        "empirical_coverage": emp_cov,
        "mean_interval_width": avg_width,
        "coverage_by_regime": cov_regime,
        "n_test_rows": int(counts["test"]),
    }
    return result, info


# --- Sample predictions for reproducibility test ---
def save_sample_predictions(cluster, out_dir, df, horizons):
    """Save a small set of deterministic predictions for reproducibility testing."""
    samples = {}
    df_feat = build_monitoring_features(df)
    feat_cols = get_monitoring_features(df_feat.columns)

    for h in horizons:
        from ttwin.features import create_target
        df_h = create_target(df_feat, "CPU_Temperature", h, "Series")
        delta_col = f"CPU_Temperature_delta_{h}"
        keep = ["Series", "Time"] + feat_cols + [delta_col]
        keep = list(dict.fromkeys(c for c in keep if c in df_h.columns))
        df_h = df_h[keep].dropna()
        # Take first 5 test-partition rows
        _, _, _, test, _, _ = chronological_split(df_h)
        if len(test) >= 5:
            X_sample = test[feat_cols].iloc[:5].values
            model_path = out_dir / f"{cluster}_monitoring_{h}_point.txt"
            m = lgb.Booster(model_file=str(model_path))
            preds = m.predict(X_sample).tolist()
            samples[f"monitoring_{h}"] = {
                "input_rows": test[feat_cols].iloc[:5].to_dict(orient="list"),
                "predictions": preds,
            }

    # Planning
    df_plan = build_planning_features(df)
    plan_feats = list(dict.fromkeys(get_planning_features(df_plan.columns)))
    keep = ["Series", "Time"] + plan_feats + ["CPU_Temperature"]
    keep = list(dict.fromkeys(c for c in keep if c in df_plan.columns))
    df_plan = df_plan[keep].dropna()
    _, _, _, test, _, _ = chronological_split(df_plan)
    if len(test) >= 5:
        X_sample = test[plan_feats].iloc[:5].values
        model_path = out_dir / f"{cluster}_planning_0_point.txt"
        m = lgb.Booster(model_file=str(model_path))
        preds = m.predict(X_sample).tolist()
        samples["planning_0"] = {
            "input_rows": test[plan_feats].iloc[:5].to_dict(orient="list"),
            "predictions": preds,
        }

    with open(out_dir / f"{cluster}_sample_predictions.json", "w") as f:
        json.dump(samples, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Train twin models")
    parser.add_argument("--cluster", required=True, choices=["d1", "d2"])
    parser.add_argument("--horizons", nargs="+", type=int, default=[1, 6, 36])
    parser.add_argument("--out", type=str, default="models/twin")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = CLUSTER_MAP[args.cluster]
    servers = pd.read_csv(cfg["server_csv"])["server"].tolist()
    print(f"Cluster {args.cluster}: {len(servers)} servers from {cfg['server_csv'].name}")

    # Load data
    print("Loading interim data...")
    df = load_all_servers(cfg["interim_dir"])
    df = df[df["Series"].str.strip().isin([s.strip() for s in servers])].copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["Series", "Time"]).reset_index(drop=True)
    print(f"  Loaded {len(df):,} rows, {df['Series'].nunique()} servers")

    # Train monitoring
    print(f"\n--- Monitoring models (horizons {args.horizons}) ---")
    mon_results, mon_info = train_monitoring(args.cluster, args.horizons, out_dir, df, servers)

    # Train planning
    print(f"\n--- Planning model ---")
    plan_result, plan_info = train_planning(args.cluster, out_dir, df, servers)

    all_results = mon_results + [plan_result]
    all_info = mon_info + [plan_info]

    # Save sample predictions for reproducibility test
    print("\nSaving sample predictions...")
    save_sample_predictions(args.cluster, out_dir, df, args.horizons)

    # Save metadata
    metadata = {
        "cluster": args.cluster,
        "dataset": cfg["dataset"],
        "n_servers": len(servers),
        "host_list": servers,
        "horizons": args.horizons,
        "hyperparameters": {"seed": SEED, "learning_rate": LR, "early_stopping": EARLY_STOP, "num_boost_round_max": 2000},
        "git_commit": get_git_commit(),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "models": all_info,
    }
    meta_path = out_dir / f"{args.cluster}_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\nMetadata saved to {meta_path}")

    # Print results table
    print(f"\n{'cluster':<8} {'kind':<12} {'h':>3} {'MAE':>7} {'pMAE':>7} {'skill':>7} {'cov':>7} {'width':>7} {'time':>6}")
    print("-" * 70)
    for r in all_results:
        pmae = f"{r['persist_mae']:.4f}" if r['persist_mae'] is not None else "   -"
        sk = f"{r['skill']:.4f}" if r['skill'] is not None else "   -"
        print(f"{r['cluster']:<8} {r['kind']:<12} {r['horizon']:>3} {r['test_mae']:>7.4f} {pmae:>7} {sk:>7} {r['coverage']:>7.4f} {r['width']:>7.2f} {r['time_s']:>5.1f}s")

    # Check model sizes and add large ones to .gitignore
    large_files = []
    for f in out_dir.glob("*.txt"):
        if f.stat().st_size > 2 * 1024 * 1024:
            large_files.append(f.name)
    if large_files:
        gitignore = ROOT / ".gitignore"
        existing = gitignore.read_text() if gitignore.exists() else ""
        additions = []
        for lf in large_files:
            entry = f"{args.out}/{lf}"
            if entry not in existing:
                additions.append(entry)
        if additions:
            with open(gitignore, "a") as gi:
                gi.write("\n# Large twin model files (>2MB)\n")
                for a in additions:
                    gi.write(a + "\n")
            print(f"\nAdded {len(additions)} large model files to .gitignore")


if __name__ == "__main__":
    main()
