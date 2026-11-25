#!/usr/bin/env python3
"""v4 Planning Addendum (NEXT-2 T4):
1. All-folds planning adaptation on all 81 D1 servers with two load-aware baselines
   (Ridge on CPU_Load; 10%-width load-bin mean).
2. Cross-cluster planning adaptation (C1 -> C2) exact target server count and
   reconciliation of zero-shot MAE (6.85 C in v3 vs 4.88 C in v2).

Outputs to experiments/v4_planning_addendum/ and docs/results_v4_addendum.md.
Never touches docs/results_v3.md or experiments/v3_experiments/.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from scripts.run_v3_experiments import bootstrap_ci, load_data  # noqa: E402
from ttwin.features import build_features, get_feature_set  # noqa: E402
from ttwin.metrics import mae  # noqa: E402
from ttwin.splits import get_group_kfold_splits, split_chronological, split_strict  # noqa: E402

_OUT_DIR = _REPO / "experiments" / "v4_planning_addendum"
_DOC_PATH = _REPO / "docs" / "results_v4_addendum.md"


def _fit_load_bin_means(loads: np.ndarray, targets: np.ndarray) -> tuple[Dict[int, float], float]:
    bins = np.clip((np.asarray(loads, dtype=np.float64) // 10.0).astype(int), 0, 9)
    global_mean = float(np.mean(targets))
    bin_means: Dict[int, float] = {}
    for b in range(10):
        mask = bins == b
        if np.any(mask):
            bin_means[b] = float(np.mean(targets[mask]))
        else:
            bin_means[b] = global_mean
    return bin_means, global_mean


def _predict_load_bin_means(loads: np.ndarray, bin_means: Dict[int, float], fallback: float) -> np.ndarray:
    bins = np.clip((np.asarray(loads, dtype=np.float64) // 10.0).astype(int), 0, 9)
    return np.array([bin_means.get(int(b), fallback) for b in bins], dtype=np.float64)


def _paired_gain_ci(base_maes: np.ndarray, model_maes: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict:
    """Compute overall ratio gain and per-server paired gain with 95% bootstrap CIs."""
    overall_pct = (1.0 - np.mean(model_maes) / np.mean(base_maes)) * 100.0
    per_srv_pct = (1.0 - model_maes / base_maes) * 100.0

    rng = np.random.RandomState(seed)
    n = len(base_maes)
    boot_overall = []
    boot_srv = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        boot_overall.append((1.0 - np.mean(model_maes[idx]) / np.mean(base_maes[idx])) * 100.0)
        boot_srv.append(float(np.mean(per_srv_pct[idx])))

    return {
        "overall_gain_pct": round(float(overall_pct), 2),
        "overall_ci_lo": round(float(np.percentile(boot_overall, 2.5)), 2),
        "overall_ci_hi": round(float(np.percentile(boot_overall, 97.5)), 2),
        "mean_server_gain_pct": round(float(np.mean(per_srv_pct)), 2),
        "server_ci_lo": round(float(np.percentile(boot_srv, 2.5)), 2),
        "server_ci_hi": round(float(np.percentile(boot_srv, 97.5)), 2),
    }


def run_part1_81_servers(d1_feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n--- Part 1: Planning Baselines Across All 5 Folds (N=81) ---")
    splits = get_group_kfold_splits(d1_feat, n_splits=5, seed=42)
    s2_feats = get_feature_set("S2", d1_feat.columns.tolist())
    target_abs = "CPU_Temperature_target_6"
    cols = list(dict.fromkeys(s2_feats + [target_abs, "Series", "Time"]))

    raw_rows: List[dict] = []
    for fold_idx, (tr_srv, te_srv) in enumerate(splits, 1):
        train_df = d1_feat[d1_feat["Series"].isin(tr_srv)]
        test_df = d1_feat[d1_feat["Series"].isin(te_srv)]
        train_clean = train_df[cols].dropna()
        test_clean = test_df[cols].dropna()

        # Global LightGBM S2 source model (exact v3 hyperparameters)
        dtrain = lgb.Dataset(train_clean[s2_feats].values, label=train_clean[target_abs].values, feature_name=s2_feats)
        source_model = lgb.train(
            {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
            dtrain,
            num_boost_round=60,
        )

        # Baselines fit on train_clean
        fleet_mean_val = float(train_clean[target_abs].mean())
        ridge = Ridge(alpha=1.0).fit(train_clean[["CPU_Load"]].values, train_clean[target_abs].values)
        bin_means, bin_fallback = _fit_load_bin_means(train_clean["CPU_Load"].values, train_clean[target_abs].values)

        for server in test_clean["Series"].unique():
            srv_df = test_clean[test_clean["Series"] == server].sort_values("Time")
            if len(srv_df) < 200:
                continue
            t_min = srv_df["Time"].min()
            t_eval_start = t_min + pd.Timedelta(hours=174)
            eval_df = srv_df[srv_df["Time"] >= t_eval_start]
            if len(eval_df) < 50:
                continue

            y_eval = eval_df[target_abs].values
            X_eval = eval_df[s2_feats].values
            loads_eval = eval_df["CPU_Load"].values

            pred_zero = source_model.predict(X_eval)
            pred_ridge = ridge.predict(eval_df[["CPU_Load"]].values)
            pred_bin = _predict_load_bin_means(loads_eval, bin_means, bin_fallback)

            raw_rows.extend([
                {"server": server, "fold": fold_idx, "budget": 0, "strategy": "fleet_mean", "mae": mae(y_eval, np.full_like(y_eval, fleet_mean_val))},
                {"server": server, "fold": fold_idx, "budget": 0, "strategy": "ridge_cpu_load", "mae": mae(y_eval, pred_ridge)},
                {"server": server, "fold": fold_idx, "budget": 0, "strategy": "load_bin_mean", "mae": mae(y_eval, pred_bin)},
                {"server": server, "fold": fold_idx, "budget": 0, "strategy": "zero_shot", "mae": mae(y_eval, pred_zero)},
            ])

            for budget in [1, 6, 24, 72, 168]:
                adapt_df = srv_df[(srv_df["Time"] >= t_min) & (srv_df["Time"] < t_min + pd.Timedelta(hours=budget))]
                if len(adapt_df) < 5:
                    continue
                adapt_preds = source_model.predict(adapt_df[s2_feats].values)
                offset = float(np.mean(adapt_df[target_abs].values - adapt_preds))
                raw_rows.append({"server": server, "fold": fold_idx, "budget": budget, "strategy": "residual_offset", "mae": mae(y_eval, pred_zero + offset)})

                dadapt = lgb.Dataset(adapt_df[s2_feats].values, label=adapt_df[target_abs].values, feature_name=s2_feats)
                finetuned = lgb.train(
                    {"objective": "regression", "metric": "mae", "learning_rate": 0.01, "num_leaves": 31, "verbosity": -1, "seed": 42},
                    dadapt,
                    num_boost_round=15,
                    init_model=source_model,
                )
                raw_rows.append({"server": server, "fold": fold_idx, "budget": budget, "strategy": "continued_boosting", "mae": mae(y_eval, finetuned.predict(X_eval))})

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(_OUT_DIR / "planning_baselines_per_server.csv", index=False)

    summary = []
    for (strategy, budget), grp in raw_df.groupby(["strategy", "budget"]):
        vals = grp["mae"].values
        ci_l, ci_h = bootstrap_ci(vals)
        summary.append({
            "strategy": strategy,
            "budget_hours": budget,
            "n_servers": len(grp),
            "mean_mae": float(np.mean(vals)),
            "std_mae": float(np.std(vals)),
            "ci_95_low": ci_l,
            "ci_95_high": ci_h,
        })
    sum_df = pd.DataFrame(summary).sort_values(["budget_hours", "mean_mae"]).reset_index(drop=True)
    sum_df.to_csv(_OUT_DIR / "planning_baselines_summary.csv", index=False)

    # Paired gains of zero_shot vs baselines across the 81 servers
    b0 = raw_df[raw_df["budget"] == 0].pivot(index="server", columns="strategy", values="mae")
    gains = []
    for base_col in ("fleet_mean", "ridge_cpu_load", "load_bin_mean"):
        g = _paired_gain_ci(b0[base_col].values, b0["zero_shot"].values)
        gains.append({"baseline": base_col, "n_servers": len(b0), **g})
    gains_df = pd.DataFrame(gains)
    gains_df.to_csv(_OUT_DIR / "planning_paired_gains.csv", index=False)

    print(sum_df.to_string(index=False))
    print("\nPaired zero-shot gains vs baselines:")
    print(gains_df.to_string(index=False))
    return sum_df, gains_df


def run_part2_cross_cluster_reconciliation(d1_feat: pd.DataFrame, d2_feat: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n--- Part 2: Cross-Cluster Server Count & v2 vs v3 Reconciliation ---")
    shared_feats = ["CPU_Load", "Ram_Used", "No_Of_Running_vms", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    target_abs = "CPU_Temperature_target_6"

    # A. Exact v3 protocol (run_v3_experiments.py::run_exp5_cross_cluster_adaptation)
    cols1 = list(dict.fromkeys(shared_feats + [target_abs]))
    d1_clean = d1_feat[cols1].dropna()
    dtrain = lgb.Dataset(d1_clean[shared_feats].values, label=d1_clean[target_abs].values, feature_name=shared_feats)
    c1_v3 = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtrain,
        num_boost_round=80,
    )

    _, _, test_c2 = split_strict(d2_feat, train_frac=0.70, holdout_frac=0.20, embargo_steps=36, seed=42)
    cols2 = list(dict.fromkeys(shared_feats + [target_abs, "Series", "Time"]))
    test_clean2 = test_c2[cols2].dropna()
    target_servers_all = sorted(test_clean2["Series"].unique())

    v3_rows = []
    evaluated_servers: List[str] = []
    for budget in [0, 1, 6, 24, 72, 168]:
        maes_zero, maes_off, maes_boost = [], [], []
        srv_used = []
        for server in target_servers_all:
            srv_df = test_clean2[test_clean2["Series"] == server].sort_values("Time")
            if len(srv_df) < 150:
                continue
            t_min = srv_df["Time"].min()
            eval_df = srv_df[srv_df["Time"] >= t_min + pd.Timedelta(hours=174)]
            if len(eval_df) < 30:
                continue

            srv_used.append(server)
            y_eval = eval_df[target_abs].values
            X_eval = eval_df[shared_feats].values
            pred_zero = c1_v3.predict(X_eval)
            maes_zero.append(mae(y_eval, pred_zero))

            if budget > 0:
                adapt_df = srv_df[(srv_df["Time"] >= t_min) & (srv_df["Time"] < t_min + pd.Timedelta(hours=budget))]
                if len(adapt_df) >= 3:
                    off = float(np.mean(adapt_df[target_abs].values - c1_v3.predict(adapt_df[shared_feats].values)))
                    maes_off.append(mae(y_eval, pred_zero + off))
                    dadapt = lgb.Dataset(adapt_df[shared_feats].values, label=adapt_df[target_abs].values, feature_name=shared_feats)
                    finetuned = lgb.train(
                        {"objective": "regression", "metric": "mae", "learning_rate": 0.01, "num_leaves": 31, "verbosity": -1, "seed": 42},
                        dadapt,
                        num_boost_round=15,
                        init_model=c1_v3,
                    )
                    maes_boost.append(mae(y_eval, finetuned.predict(X_eval)))

        if budget == 0:
            evaluated_servers = srv_used
            ci_l, ci_h = bootstrap_ci(maes_zero)
            v3_rows.append({"budget_hours": 0, "strategy": "zero_shot", "n_servers": len(maes_zero), "mean_mae": float(np.mean(maes_zero)), "ci_95_low": ci_l, "ci_95_high": ci_h})
        else:
            if maes_off:
                ci_l, ci_h = bootstrap_ci(maes_off)
                v3_rows.append({"budget_hours": budget, "strategy": "residual_offset", "n_servers": len(maes_off), "mean_mae": float(np.mean(maes_off)), "ci_95_low": ci_l, "ci_95_high": ci_h})
            if maes_boost:
                ci_l, ci_h = bootstrap_ci(maes_boost)
                v3_rows.append({"budget_hours": budget, "strategy": "continued_boosting", "n_servers": len(maes_boost), "mean_mae": float(np.mean(maes_boost)), "ci_95_low": ci_l, "ci_95_high": ci_h})

    v3_df = pd.DataFrame(v3_rows)
    v3_df.to_csv(_OUT_DIR / "cross_cluster_server_counts.csv", index=False)

    # B. Exact v2 protocol (run_v2_cross_cluster.py) + intermediate decomposition
    tr1, va1, _ = split_chronological(d1_feat, train_frac=0.70, val_frac=0.10, embargo_steps=36)
    _, _, te2_chron = split_chronological(d2_feat, train_frac=0.70, val_frac=0.10, embargo_steps=36)

    tr1_c = tr1[cols1].dropna()
    va1_c = va1[cols1].dropna()
    dtr1 = lgb.Dataset(tr1_c[shared_feats].values, label=tr1_c[target_abs].values, feature_name=shared_feats)
    dva1 = lgb.Dataset(va1_c[shared_feats].values, label=va1_c[target_abs].values, reference=dtr1, feature_name=shared_feats)
    c1_v2 = lgb.train(
        {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
        dtr1,
        num_boost_round=250,
        valid_sets=[dva1],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
    )

    te2_chron_c = te2_chron[cols2].dropna()
    pred_v2_chron = c1_v2.predict(te2_chron_c[shared_feats].values)
    v2_pooled_mae = mae(te2_chron_c[target_abs].values, pred_v2_chron)
    v2_srv_means = [
        mae(g[target_abs].values, c1_v2.predict(g[shared_feats].values))
        for _, g in te2_chron_c.groupby("Series")
    ]

    # v3 model evaluated on v2 chronological test vs v2 model on v3 strict test
    pred_v3_on_chron = c1_v3.predict(te2_chron_c[shared_feats].values)
    v3_model_on_chron_pooled = mae(te2_chron_c[target_abs].values, pred_v3_on_chron)

    eval_strict_frames = []
    for server in evaluated_servers:
        srv_df = test_clean2[test_clean2["Series"] == server].sort_values("Time")
        t_min = srv_df["Time"].min()
        eval_strict_frames.append(srv_df[srv_df["Time"] >= t_min + pd.Timedelta(hours=174)])
    strict_eval_all = pd.concat(eval_strict_frames, ignore_index=True)
    v3_strict_pooled_mae = mae(strict_eval_all[target_abs].values, c1_v3.predict(strict_eval_all[shared_feats].values))
    v2_model_on_strict_unweighted = float(np.mean([
        mae(g[target_abs].values, c1_v2.predict(g[shared_feats].values))
        for _, g in strict_eval_all.groupby("Series")
    ]))

    reconcile_rows = [
        {
            "protocol": "v2 exact (Split A chron, 70% C1 train + early stop, pooled rows)",
            "n_servers": int(te2_chron_c["Series"].nunique()),
            "n_rows": len(te2_chron_c),
            "c1_best_iter": int(c1_v2.best_iteration),
            "mae_c": round(v2_pooled_mae, 4),
            "notes": f"unweighted server mean = {np.mean(v2_srv_means):.4f} C",
        },
        {
            "protocol": "v3 model (100% C1, 80 trees) on v2 Split A chron (pooled rows)",
            "n_servers": int(te2_chron_c["Series"].nunique()),
            "n_rows": len(te2_chron_c),
            "c1_best_iter": 80,
            "mae_c": round(v3_model_on_chron_pooled, 4),
            "notes": "Isolates C1 training set/rounds difference on identical C2 Split A rows",
        },
        {
            "protocol": "v2 model on v3 Split C strict held-out servers (unweighted server mean)",
            "n_servers": len(evaluated_servers),
            "n_rows": len(strict_eval_all),
            "c1_best_iter": int(c1_v2.best_iteration),
            "mae_c": round(v2_model_on_strict_unweighted, 4),
            "notes": "Isolates C2 held-out server subset + post-174h time window",
        },
        {
            "protocol": "v3 exact (Split C strict, unweighted server mean)",
            "n_servers": len(evaluated_servers),
            "n_rows": len(strict_eval_all),
            "c1_best_iter": 80,
            "mae_c": round(float(v3_df[v3_df["budget_hours"] == 0]["mean_mae"].iloc[0]), 4),
            "notes": f"pooled row-weighted MAE on same Split C rows = {v3_strict_pooled_mae:.4f} C; hosts = {', '.join(evaluated_servers)}",
        },
    ]
    rec_df = pd.DataFrame(reconcile_rows)
    rec_df.to_csv(_OUT_DIR / "cross_cluster_reconciliation.csv", index=False)

    print(v3_df.to_string(index=False))
    print("\nReconciliation:")
    print(rec_df[["protocol", "n_servers", "n_rows", "mae_c"]].to_string(index=False))
    return v3_df, rec_df


def _write_doc(sum_df: pd.DataFrame, gains_df: pd.DataFrame, v3_df: pd.DataFrame, rec_df: pd.DataFrame) -> None:
    zs = sum_df[sum_df["strategy"] == "zero_shot"].iloc[0]
    fm = sum_df[sum_df["strategy"] == "fleet_mean"].iloc[0]
    rd = sum_df[sum_df["strategy"] == "ridge_cpu_load"].iloc[0]
    lb = sum_df[sum_df["strategy"] == "load_bin_mean"].iloc[0]

    g_fm = gains_df[gains_df["baseline"] == "fleet_mean"].iloc[0]
    g_rd = gains_df[gains_df["baseline"] == "ridge_cpu_load"].iloc[0]
    g_lb = gains_df[gains_df["baseline"] == "load_bin_mean"].iloc[0]

    n_cc = int(v3_df[v3_df["budget_hours"] == 0]["n_servers"].iloc[0])
    v2_mae = float(rec_df.iloc[0]["mae_c"])
    v3_mae = float(rec_df.iloc[-1]["mae_c"])

    lines = [
        "# Results v4 Addendum — Planning Baselines & Cross-Cluster Reconciliation",
        "",
        "> Addendum to `docs/results_v3.md` (`scripts/run_v4_planning_addendum.py`). Existing v3 files and `experiments/v3_experiments/` are untouched.",
        "",
        "## 1. Fleet-Wide Planning Adaptation with Load-Aware Baselines ($N=81$ Servers)",
        "",
        "Evaluated on the exact same 81 Dataset 1 servers, 5 GroupKFold folds (`seed=42`), S2 features, and post-174h evaluation windows as `docs/results_v3.md` §4, adding two zero-shot load-aware baselines fit on each fold's training servers:",
        "- `ridge_cpu_load`: Ridge regression (`alpha=1.0`) on `CPU_Load` alone.",
        "- `load_bin_mean`: mean training temperature per 10%-wide `CPU_Load` bin (`[0,10), ..., [90,100]`).",
        "",
        "| Strategy | Budget (h) | n_servers | Mean MAE (°C) | Std (°C) | 95% Bootstrap CI (°C) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for _, r in sum_df.iterrows():
        lines.append(
            f"| `{r['strategy']}` | {int(r['budget_hours'])} | {int(r['n_servers'])} | "
            f"{r['mean_mae']:.4f} | {r['std_mae']:.4f} | [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}] |"
        )

    lines.extend([
        "",
        "### Paired Zero-Shot (`S2` LightGBM) Gain Over Baselines ($N=81$ Servers)",
        "",
        "| Baseline | Overall Pooled Reduction [95% CI] | Mean Per-Server Reduction [95% CI] |",
        "|---|---|---|",
        f"| `fleet_mean` ({fm['mean_mae']:.2f}°C) | **{g_fm['overall_gain_pct']:+.2f}%** [{g_fm['overall_ci_lo']:+.2f}%, {g_fm['overall_ci_hi']:+.2f}%] | {g_fm['mean_server_gain_pct']:+.2f}% [{g_fm['server_ci_lo']:+.2f}%, {g_fm['server_ci_hi']:+.2f}%] |",
        f"| `ridge_cpu_load` ({rd['mean_mae']:.2f}°C) | **{g_rd['overall_gain_pct']:+.2f}%** [{g_rd['overall_ci_lo']:+.2f}%, {g_rd['overall_ci_hi']:+.2f}%] | {g_rd['mean_server_gain_pct']:+.2f}% [{g_rd['server_ci_lo']:+.2f}%, {g_rd['server_ci_hi']:+.2f}%] |",
        f"| `load_bin_mean` ({lb['mean_mae']:.2f}°C) | **{g_lb['overall_gain_pct']:+.2f}%** [{g_lb['overall_ci_lo']:+.2f}%, {g_lb['overall_ci_hi']:+.2f}%] | {g_lb['mean_server_gain_pct']:+.2f}% [{g_lb['server_ci_lo']:+.2f}%, {g_lb['server_ci_hi']:+.2f}%] |",
        "",
        f"**Impact on the \"43.2% vs fleet mean\" claim:** Against an unconditioned `fleet_mean` ({fm['mean_mae']:.2f}°C), zero-shot `S2` LightGBM ({zs['mean_mae']:.2f}°C) reduces error by **{g_fm['overall_gain_pct']:.1f}%** (95% CI `[{g_fm['overall_ci_lo']:.1f}%, {g_fm['overall_ci_hi']:.1f}%]`). However, simple load-aware baselines already capture most of that reduction (`ridge_cpu_load`: **{rd['mean_mae']:.2f}°C**; `load_bin_mean`: **{lb['mean_mae']:.2f}°C**), leaving S2 LightGBM with a modest **{g_rd['overall_gain_pct']:+.2f}%** gain over Ridge on CPU load and **{g_lb['overall_gain_pct']:+.2f}%** over 10%-binned load mean.",
        "",
        "---",
        "",
        "## 2. Cross-Cluster Adaptation (`C1 -> C2`): Server Count & 6.85°C vs 4.88°C Reconciliation",
        "",
        f"In the v3 cross-cluster adaptation experiment (`docs/results_v3.md` §5), **`N = {n_cc}` target servers** in Cluster 2 met the Split C test holdout and post-174h evaluation criteria (`>= 150` rows and `>= 30` post-174h rows):",
        "",
        "| Budget (h) | Strategy | n_servers | Mean MAE (°C) | 95% CI (°C) |",
        "|---:|---|---:|---:|---|",
    ])
    for _, r in v3_df.iterrows():
        lines.append(
            f"| {int(r['budget_hours'])} | `{r['strategy']}` | {int(r['n_servers'])} | "
            f"{r['mean_mae']:.4f} | [{r['ci_95_low']:.4f}, {r['ci_95_high']:.4f}] |"
        )

    lines.extend([
        "",
        f"### Reconciling Zero-Shot MAE: `{v3_mae:.2f}°C` (v3) vs `{v2_mae:.2f}°C` (v2)",
        "",
        "| Protocol Variant | C2 Servers | Eval Rows | Zero-Shot MAE (°C) | Notes |",
        "|---|---:|---:|---:|---|",
    ])
    for _, r in rec_df.iterrows():
        lines.append(f"| {r['protocol']} | {int(r['n_servers'])} | {int(r['n_rows']):,} | **{r['mae_c']:.4f}** | {r['notes']} |")

    lines.extend([
        "",
        "**Why v3 (`6.85°C`) differs from v2 (`4.88°C`):**",
        "1. **Evaluation Split & Server Subset:** v2 evaluated chronological Split A across all `53` qualifying C2 servers (`4.88°C` pooled), whereas v3 used Split C (`seed=42`) evaluating strictly held-out C2 servers (`N = " + str(n_cc) + "`) after a 174-hour adaptation buffer.",
        "2. **C1 Training Configuration & Aggregation:** v2 trained on 70% of C1 with early stopping (`best_iteration` trees), whereas v3 trained on 100% of C1 for 80 rounds and averaged unweighted per-server MAEs (giving high-variance held-out C2 hosts equal weight).",
        "",
        "Source artifacts: `experiments/v4_planning_addendum/` (`planning_baselines_summary.csv`, `planning_paired_gains.csv`, `cross_cluster_server_counts.csv`, `cross_cluster_reconciliation.csv`).",
    ])
    _DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {_DOC_PATH}")


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    d1 = load_data("dataset1")
    d2 = load_data("dataset2")
    d1_feat = build_features(d1, horizons=[6], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])
    d2_feat = build_features(d2, horizons=[6], lags=[1, 2, 3, 6, 12], rolling_windows=[6, 12, 36])

    sum_df, gains_df = run_part1_81_servers(d1_feat)
    v3_df, rec_df = run_part2_cross_cluster_reconciliation(d1_feat, d2_feat)
    _write_doc(sum_df, gains_df, v3_df, rec_df)


if __name__ == "__main__":
    main()
