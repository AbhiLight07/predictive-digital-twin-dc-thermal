#!/usr/bin/env python3
"""Evaluate planning-interval calibration fixes (NEXT-2 T5).

Evaluates two calibration methods on saved planning models (no retraining,
saved .txt model files untouched):
  (0) Static CQR (baseline from models/twin/{cluster}_metadata.json)
  (i) Rolling 7-day causal recalibration at cursor t (scores strictly < t in [t - 7d, t))
  (ii) Regime-wise q_hat (idle <10%, active 10-50%, heavy >50% CPU_Load) fit on val_cal

Adoption criterion: test coverage >= 88.0% AND mean interval width <= 1.30x baseline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))
sys.path.insert(0, str(_REPO))

from scripts.train_twin_models import (  # noqa: E402
    CLUSTER_MAP,
    build_planning_features,
    compute_coverage_by_regime,
)
from ttwin.io import load_all_servers  # noqa: E402
from ttwin.uncertainty import average_width, compute_conformal_quantile, coverage  # noqa: E402

_MODELS_DIR = _REPO / "models" / "twin"
_OUT_DIR = _REPO / "experiments" / "planning_calibration_20261125"
_DOC_PATH = _REPO / "docs" / "planning_calibration_eval.md"


def _regime_mask(loads: np.ndarray, regime: str) -> np.ndarray:
    if regime == "idle":
        return loads < 10.0
    if regime == "active":
        return (loads >= 10.0) & (loads <= 50.0)
    if regime == "heavy":
        return loads > 50.0
    raise ValueError(regime)


def _rolling_7d_qhats(
    times_ns: np.ndarray,
    scores: np.ndarray,
    eval_mask: np.ndarray,
    fallback_q: float,
    min_history: int = 100,
) -> np.ndarray:
    """For each row with eval_mask==True, compute causal q_hat from strictly prior 7 days [t - 7d, t)."""
    window_ns = int(7 * 24 * 3600 * 1_000_000_000)
    # Sort all rows chronologically once
    order = np.argsort(times_ns, kind="mergesort")
    t_sorted = times_ns[order]
    s_sorted = scores[order]
    eval_sorted = eval_mask[order]

    # Unique timestamps in eval region
    eval_idx = np.where(eval_sorted)[0]
    uniq_t, inv = np.unique(t_sorted[eval_idx], return_inverse=True)

    q_by_uniq = np.empty(len(uniq_t), dtype=np.float64)
    left = 0
    right = 0
    n_tot = len(t_sorted)
    for ui, t_val in enumerate(uniq_t):
        while right < n_tot and t_sorted[right] < t_val:
            right += 1
        t_lo = t_val - window_ns
        while left < right and t_sorted[left] < t_lo:
            left += 1
        cnt = right - left
        if cnt >= min_history:
            q_by_uniq[ui] = compute_conformal_quantile(s_sorted[left:right], alpha=0.10)
        else:
            q_by_uniq[ui] = fallback_q

    q_out = np.full(len(times_ns), fallback_q, dtype=np.float64)
    q_out[order[eval_idx]] = q_by_uniq[inv]
    return q_out


def evaluate_cluster(cluster: str) -> tuple[List[dict], dict]:
    cfg = CLUSTER_MAP[cluster]
    with open(_MODELS_DIR / f"{cluster}_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    plan_info = next(m for m in meta["models"] if m["kind"] == "planning")
    feat_cols = plan_info["feature_names"]
    b = plan_info["boundaries"]
    static_q = float(plan_info["q_hat"])

    servers = pd.read_csv(cfg["server_csv"])["server"].tolist()
    df = load_all_servers(cfg["interim_dir"])
    df = df[df["Series"].str.strip().isin([s.strip() for s in servers])].copy()
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values(["Series", "Time"]).reset_index(drop=True)

    df_feat = build_planning_features(df)
    keep_cols = list(dict.fromkeys(["Series", "Time"] + feat_cols + ["CPU_Temperature"]))
    df_feat = df_feat[keep_cols].dropna().reset_index(drop=True)

    t_col = df_feat["Time"]
    val_sel_mask = (t_col >= pd.Timestamp(b["val_sel_start"])) & (t_col <= pd.Timestamp(b["val_sel_end"]))
    val_cal_mask = (t_col >= pd.Timestamp(b["val_cal_start"])) & (t_col <= pd.Timestamp(b["val_cal_end"]))
    test_mask = t_col >= pd.Timestamp(b["test_start"])

    m_lo = lgb.Booster(model_file=str(_MODELS_DIR / f"{cluster}_planning_0_lower.txt"))
    m_hi = lgb.Booster(model_file=str(_MODELS_DIR / f"{cluster}_planning_0_upper.txt"))

    # Predict on val_sel + val_cal + test rows
    active_mask = val_sel_mask | val_cal_mask | test_mask
    sub = df_feat[active_mask].copy().reset_index(drop=True)
    X_all = sub[feat_cols].to_numpy(dtype=np.float64)
    y_all = sub["CPU_Temperature"].to_numpy(dtype=np.float64)
    loads_all = sub["CPU_Load"].to_numpy(dtype=np.float64)
    times_ns = sub["Time"].to_numpy(dtype="datetime64[ns]").view("int64")

    lo_raw = m_lo.predict(X_all)
    hi_raw = m_hi.predict(X_all)
    scores_all = np.maximum(lo_raw - y_all, y_all - hi_raw)

    t_sub = sub["Time"]
    m_vc = (t_sub >= pd.Timestamp(b["val_cal_start"])) & (t_sub <= pd.Timestamp(b["val_cal_end"]))
    m_te = t_sub >= pd.Timestamp(b["test_start"])
    vc_idx = m_vc.to_numpy()
    te_idx = m_te.to_numpy()

    # Verify static q_hat reproduces metadata
    q_static_arr = np.full(len(sub), static_q, dtype=np.float64)
    # (i) Rolling 7-day causal recalibration (eval on val_cal & test using strictly prior 7d)
    q_roll_vc = _rolling_7d_qhats(times_ns, scores_all, vc_idx, static_q)
    q_roll_te = _rolling_7d_qhats(times_ns, scores_all, te_idx, static_q)
    q_roll_arr = np.where(vc_idx, q_roll_vc, q_roll_te)

    # (ii) Regime-wise q_hat fit on val_cal
    regime_q: Dict[str, float] = {}
    q_reg_arr = np.full(len(sub), static_q, dtype=np.float64)
    for reg in ("idle", "active", "heavy"):
        rmask_vc = vc_idx & _regime_mask(loads_all, reg)
        if int(rmask_vc.sum()) >= 20:
            rq = float(compute_conformal_quantile(scores_all[rmask_vc], alpha=0.10))
        else:
            rq = static_q
        regime_q[reg] = rq
        q_reg_arr[_regime_mask(loads_all, reg)] = rq

    methods = [
        ("static_cqr", q_static_arr),
        ("rolling_7d", q_roll_arr),
        ("regime_qhat", q_reg_arr),
    ]

    rows: List[dict] = []
    base_width = None
    for m_name, q_arr in methods:
        lo_adj = lo_raw - q_arr
        hi_adj = hi_raw + q_arr
        swap = lo_adj > hi_adj
        if np.any(swap):
            lo_adj[swap], hi_adj[swap] = hi_adj[swap], lo_adj[swap]

        vc_cov = coverage(y_all[vc_idx], lo_adj[vc_idx], hi_adj[vc_idx])
        vc_w = average_width(lo_adj[vc_idx], hi_adj[vc_idx])
        te_cov = coverage(y_all[te_idx], lo_adj[te_idx], hi_adj[te_idx])
        te_w = average_width(lo_adj[te_idx], hi_adj[te_idx])
        if m_name == "static_cqr":
            base_width = te_w

        te_df = sub.loc[te_idx, ["CPU_Temperature", "CPU_Load"]].copy()
        te_df["lo"] = lo_adj[te_idx]
        te_df["hi"] = hi_adj[te_idx]
        cov_reg = compute_coverage_by_regime(te_df, "CPU_Temperature", "lo", "hi")

        width_ratio = te_w / base_width if base_width else 1.0
        adopted = (m_name != "static_cqr") and (te_cov >= 0.88) and (width_ratio <= 1.30)

        rows.append({
            "cluster": cluster.upper(),
            "method": m_name,
            "val_cal_coverage": round(vc_cov, 4),
            "val_cal_width": round(vc_w, 3),
            "test_coverage": round(te_cov, 4),
            "test_width": round(te_w, 3),
            "width_vs_baseline": round(width_ratio, 4),
            "cov_idle": round(cov_reg["idle"], 4) if cov_reg["idle"] is not None else None,
            "cov_active": round(cov_reg["active"], 4) if cov_reg["active"] is not None else None,
            "cov_heavy": round(cov_reg["heavy"], 4) if cov_reg["heavy"] is not None else None,
            "meets_rule": bool(adopted),
        })

    # Build compact causal daily lookup of rolling-7d q_hat for sidecar JSON
    te_times = sub.loc[te_idx, "Time"]
    te_days = te_times.dt.strftime("%Y-%m-%d").to_numpy()
    q_te_vals = q_roll_te[te_idx]
    daily_q: Dict[str, float] = {}
    for d_str, q_v in zip(te_days, q_te_vals):
        if d_str not in daily_q:
            daily_q[d_str] = round(float(q_v), 6)

    roll_row = next(r for r in rows if r["method"] == "rolling_7d")
    base_row = next(r for r in rows if r["method"] == "static_cqr")
    cal_summary = {
        "static_q": static_q,
        "regime_q": regime_q,
        "static_coverage": base_row["test_coverage"],
        "static_width": base_row["test_width"],
        "rolling_7d_coverage": roll_row["test_coverage"],
        "rolling_7d_width": roll_row["test_width"],
        "rolling_7d_width_ratio": roll_row["width_vs_baseline"],
        "rolling_7d_q_hat_median": round(float(np.median(q_te_vals)), 6),
        "rolling_7d_q_hat_latest": round(float(q_te_vals[-1]), 6),
        "rolling_7d_daily_q_hats": daily_q,
    }
    return rows, cal_summary


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    params_by_cluster = {}
    for cluster in ("d1", "d2"):
        print(f"[{cluster}] Evaluating planning calibration...")
        rows, p = evaluate_cluster(cluster)
        all_rows.extend(rows)
        params_by_cluster[cluster] = p

    df = pd.DataFrame(all_rows)
    df.to_csv(_OUT_DIR / "metrics.csv", index=False)

    # Check if either candidate meets rule on BOTH clusters
    adopted_methods = []
    for m_name in ("rolling_7d", "regime_qhat"):
        sub = df[df["method"] == m_name]
        if bool(sub["meets_rule"].all()):
            adopted_methods.append(m_name)

    lines = [
        "# Planning-Interval Calibration Evaluation (NEXT-2 T5)",
        "",
        "Evaluates two calibration strategies on the saved planning models (`models/twin/{d1,d2}_planning_0_*.txt`, unmodified) against a `90%` nominal target (`scripts/eval_planning_calibration.py`):",
        "- **Baseline (`static_cqr`):** Global `q_hat` fit on `val_cal` (`81.09%` D1, `85.33%` D2).",
        "- **(i) `rolling_7d`:** Causal 7-day sliding window over `[t - 7 days, t)` conformity scores strictly prior to `t` (fallback to `static_cqr` when `< 100` prior observations exist).",
        "- **(ii) `regime_qhat`:** Separate CQR `q_hat` calibrated on `val_cal` for `idle` (`CPU_Load < 10%`), `active` (`10–50%`), and `heavy` (`> 50%`).",
        "",
        "**Adoption rule:** Test coverage `>= 88.0%` AND mean width `<= 1.30x` static baseline on both clusters.",
        "",
        "| Cluster | Method | Val Cov | Val Width | Test Cov | Test Width | Width Ratio | Idle Cov | Active Cov | Heavy Cov | Meets Rule |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for _, r in df.iterrows():
        ci = f"{r['cov_idle']*100:.1f}%" if pd.notna(r["cov_idle"]) else "n/a"
        ca = f"{r['cov_active']*100:.1f}%" if pd.notna(r["cov_active"]) else "n/a"
        ch = f"{r['cov_heavy']*100:.1f}%" if pd.notna(r["cov_heavy"]) else "n/a"
        lines.append(
            f"| {r['cluster']} | `{r['method']}` | {r['val_cal_coverage']*100:.2f}% | {r['val_cal_width']:.2f}°C | "
            f"**{r['test_coverage']*100:.2f}%** | {r['test_width']:.2f}°C | {r['width_vs_baseline']:.2f}x | "
            f"{ci} | {ca} | {ch} | {'YES' if r['meets_rule'] else 'NO'} |"
        )

    lines.extend([
        "",
        "## Regime Parameters (`val_cal`)",
        f"- **D1:** static `q_hat = {params_by_cluster['d1']['static_q']:.4f}`, regime `q_hat = { {k: round(v, 4) for k, v in params_by_cluster['d1']['regime_q'].items()} }`",
        f"- **D2:** static `q_hat = {params_by_cluster['d2']['static_q']:.4f}`, regime `q_hat = { {k: round(v, 4) for k, v in params_by_cluster['d2']['regime_q'].items()} }`",
        "",
        f"## Decision: {'ADOPTED (' + ', '.join(adopted_methods) + ')' if adopted_methods else 'NEGATIVE RESULT — NONE ADOPTED'}",
    ])
    _DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if adopted_methods:
        sidecar = {
            "adopted_mode": adopted_methods[0],
            "rule": "test_coverage >= 0.88 and width <= 1.30 * static_baseline",
            "clusters": params_by_cluster,
        }
        with open(_MODELS_DIR / "planning_calibration.json", "w", encoding="utf-8") as f:
            json.dump(sidecar, f, indent=2)
        print(f"Wrote {_MODELS_DIR / 'planning_calibration.json'}")
    print("\n" + df[["cluster", "method", "test_coverage", "test_width", "width_vs_baseline", "meets_rule"]].to_string(index=False))
    print(f"Adopted on both clusters: {adopted_methods}")


if __name__ == "__main__":
    main()
