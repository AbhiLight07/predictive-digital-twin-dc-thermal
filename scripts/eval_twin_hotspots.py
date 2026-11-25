#!/usr/bin/env python3
"""Pre-registered hotspot-event early-warning evaluation (NEXT-2 T2).

Uses saved models (models/twin/) and precomputed forecast caches
(data/twin_cache/{d1,d2}_forecasts.parquet) with zero retraining.

Protocol (pre-registered in docs/decisions.md):
- Event: host currently below cluster warning threshold W (T(t) < W) has
  effective CPU temperature T(t + h) >= W at h=6 (1 h) or h=36 (6 h).
- Detectors:
  (A) Twin: CQR_upper_h(t) >= W
  (B) Persistence rule: T(t) >= W - m_persist
  (C) Trend rule: T(t) + slope_1h(t) * h_hours >= W - m_trend
- Calibration: m_persist and m_trend chosen on validation partition
  (val_sel_start <= t <= val_cal_end) to match Twin's validation alert rate.
- Inference: 95% bootstrap CIs (B=2000, seed=42) over hosts; inconclusive if n_events < 30.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ttwin.twin.engine import TwinEngine  # noqa: E402

_OUT_DIR = _REPO / "experiments" / "twin_hotspots_20261125"
_DOC_PATH = _REPO / "docs" / "twin_hotspot_evaluation.md"


def _bootstrap_host_metrics(
    host_stats: pd.DataFrame, n_boot: int = 2000, seed: int = 42
) -> Dict[str, Tuple[float, float]]:
    """Host-level cluster bootstrap 95% CIs for recall, precision, and alert_rate."""
    rng = np.random.RandomState(seed)
    n_hosts = len(host_stats)
    ev = host_stats["n_events"].to_numpy(dtype=np.float64)
    al = host_stats["n_alerts"].to_numpy(dtype=np.float64)
    tp = host_stats["n_tp"].to_numpy(dtype=np.float64)
    nn = host_stats["n_rows"].to_numpy(dtype=np.float64)

    recalls, precs, arates = [], [], []
    for _ in range(n_boot):
        idx = rng.randint(0, n_hosts, size=n_hosts)
        s_ev = ev[idx].sum()
        s_al = al[idx].sum()
        s_tp = tp[idx].sum()
        s_nn = nn[idx].sum()
        if s_ev > 0:
            recalls.append(s_tp / s_ev)
        if s_al > 0:
            precs.append(s_tp / s_al)
        if s_nn > 0:
            arates.append(s_al / s_nn)

    def _ci(arr: List[float]) -> Tuple[float, float]:
        if not arr:
            return (np.nan, np.nan)
        return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

    return {"recall": _ci(recalls), "precision": _ci(precs), "alert_rate": _ci(arates)}


def _build_grid_features_all_hosts(engine: TwinEngine) -> pd.DataFrame:
    """Build causal 10-min grid monitoring features for every host in engine."""
    frames = []
    for host in engine.hosts():
        hdata = engine._data[engine._data["Series"] == host]
        if len(hdata) < 2:
            continue
        fdf = engine._build_monitoring_features(hdata)
        frames.append(fdf)
    return pd.concat(frames, ignore_index=True)


def _calibrate_on_validation(
    engine: TwinEngine, feat_all: pd.DataFrame, horizon: int
) -> dict:
    """Compute Twin validation alert rate and matched-rate baseline margins."""
    info = engine._models_info[f"monitoring_{horizon}"]
    b = info["boundaries"]
    val_start = pd.Timestamp(b["val_sel_start"])
    val_end = pd.Timestamp(b["val_cal_end"])
    stride_steps = int(engine._config["replay"]["stride_minutes"]) // 10

    # Align 30-min stride per host starting from val_start
    val_chunks = []
    for _, grp in feat_all.groupby("Series", sort=False):
        sub = grp[(grp["Time"] >= val_start) & (grp["Time"] <= val_end)]
        if len(sub) > 0:
            val_chunks.append(sub.iloc[::stride_steps])
    val_df = pd.concat(val_chunks, ignore_index=True)

    feat_names = info["feature_names"]
    needed = list(dict.fromkeys(feat_names + ["CPU_Temperature", "CPU_Temperature_lag_6", "Series", "Time"]))
    val_clean = val_df[needed].dropna().copy()

    W = float(engine._warn_thresh)
    val_clean = val_clean[val_clean["CPU_Temperature"] < W].reset_index(drop=True)

    X = val_clean[feat_names].to_numpy(dtype=np.float64)
    curr = val_clean["CPU_Temperature"].to_numpy(dtype=np.float64)
    lag6 = val_clean["CPU_Temperature_lag_6"].to_numpy(dtype=np.float64)

    m_hi = engine._models[f"{engine.cluster}_monitoring_{horizon}_upper"]
    q_hat = float(info.get("q_hat", 0.0))
    twin_upper = curr + m_hi.predict(X) + q_hat

    twin_alerts = twin_upper >= W
    r_val = float(np.mean(twin_alerts)) if len(twin_alerts) > 0 else 0.0

    # Margins on validation: alert iff score >= W - m  <=>  W - score <= m
    h_hours = horizon / 6.0
    slope_1h = curr - lag6  # degC per hour (6 * 10 min = 1h)
    trend_proj = curr + slope_1h * h_hours

    if r_val <= 0.0:
        m_persist = -1e6
        m_trend = -1e6
    elif r_val >= 1.0:
        m_persist = 1e6
        m_trend = 1e6
    else:
        m_persist = float(np.quantile(W - curr, r_val))
        m_trend = float(np.quantile(W - trend_proj, r_val))

    return {
        "n_val_rows": len(val_clean),
        "val_alert_rate": r_val,
        "m_persist": m_persist,
        "m_trend": m_trend,
    }


def _evaluate_cluster_horizon(cluster: str, engine: TwinEngine, feat_all: pd.DataFrame, cache: pd.DataFrame, horizon: int) -> List[dict]:
    W = float(engine._warn_thresh)
    h_hours = horizon / 6.0
    lead_min = float(horizon * 10)

    cal = _calibrate_on_validation(engine, feat_all, horizon)

    # Prepare causal lookup of T(t) and T(t - 1h) on regular 10-min grid
    grid_lookup = feat_all[["Series", "Time", "CPU_Temperature", "CPU_Temperature_lag_6"]].rename(
        columns={"Series": "host", "Time": "time", "CPU_Temperature": "curr_temp", "CPU_Temperature_lag_6": "lag_6"}
    )
    sub_cache = cache[(cache["horizon"] == horizon) & cache["actual_future"].notna()].copy()
    sub_cache["time"] = pd.to_datetime(sub_cache["time"])
    merged = sub_cache.merge(grid_lookup, on=["host", "time"], how="inner")
    merged = merged.dropna(subset=["curr_temp", "lag_6", "upper", "actual_future"]).copy()
    merged = merged[merged["curr_temp"] < W].reset_index(drop=True)

    curr = merged["curr_temp"].to_numpy(dtype=np.float64)
    lag6 = merged["lag_6"].to_numpy(dtype=np.float64)
    slope_1h = curr - lag6
    future = merged["actual_future"].to_numpy(dtype=np.float64)
    twin_upper = merged["upper"].to_numpy(dtype=np.float64)

    events = future >= W
    n_events = int(events.sum())
    n_rows = len(merged)
    conclusive = n_events >= 30

    detectors = {
        "Twin (CQR upper)": twin_upper >= W,
        "Persistence rule": curr >= (W - cal["m_persist"]),
        "Trend rule": (curr + slope_1h * h_hours) >= (W - cal["m_trend"]),
    }

    rows = []
    for det_name, alerts in detectors.items():
        tp = events & alerts
        n_alerts = int(alerts.sum())
        n_tp = int(tp.sum())
        recall = n_tp / n_events if n_events > 0 else np.nan
        precision = n_tp / n_alerts if n_alerts > 0 else np.nan
        alert_rate = n_alerts / n_rows if n_rows > 0 else np.nan

        # Per-host summary for cluster bootstrap across all hosts in cluster
        tmp = pd.DataFrame({
            "host": merged["host"],
            "event": events.astype(int),
            "alert": alerts.astype(int),
            "tp": tp.astype(int),
            "row": 1,
        })
        h_agg = tmp.groupby("host", as_index=False).agg(
            n_events=("event", "sum"),
            n_alerts=("alert", "sum"),
            n_tp=("tp", "sum"),
            n_rows=("row", "sum"),
        )
        # Ensure all hosts with eligible rows are represented
        cis = _bootstrap_host_metrics(h_agg, n_boot=2000, seed=42)

        # Also check mean earliest lead time per crossing timestamp (within horizon window)
        mean_lead = lead_min if n_tp > 0 else np.nan

        rows.append({
            "cluster": cluster.upper(),
            "horizon_steps": horizon,
            "horizon_label": f"+{int(h_hours)}h",
            "warn_thresh_c": round(W, 2),
            "detector": det_name,
            "val_n_rows": cal["n_val_rows"],
            "val_alert_rate": round(cal["val_alert_rate"], 5),
            "param_margin_c": round(cal["m_persist"], 3) if "Persistence" in det_name else (round(cal["m_trend"], 3) if "Trend" in det_name else 0.0),
            "test_n_rows": n_rows,
            "n_hosts": len(h_agg),
            "n_events": n_events,
            "n_alerts": n_alerts,
            "n_tp": n_tp,
            "alert_rate": round(alert_rate, 5),
            "alert_rate_ci_lo": round(cis["alert_rate"][0], 5),
            "alert_rate_ci_hi": round(cis["alert_rate"][1], 5),
            "recall": round(recall, 4) if not np.isnan(recall) else None,
            "recall_ci_lo": round(cis["recall"][0], 4) if not np.isnan(cis["recall"][0]) else None,
            "recall_ci_hi": round(cis["recall"][1], 4) if not np.isnan(cis["recall"][1]) else None,
            "precision": round(precision, 4) if not np.isnan(precision) else None,
            "precision_ci_lo": round(cis["precision"][0], 4) if not np.isnan(cis["precision"][0]) else None,
            "precision_ci_hi": round(cis["precision"][1], 4) if not np.isnan(cis["precision"][1]) else None,
            "mean_lead_time_min": mean_lead,
            "conclusive": "conclusive" if conclusive else "inconclusive",
        })
    return rows


def main() -> None:
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for cluster in ("d1", "d2"):
        print(f"[{cluster}] Loading engine & cache...")
        engine = TwinEngine(cluster)
        cache = pd.read_parquet(_REPO / "data" / "twin_cache" / f"{cluster}_forecasts.parquet")
        feat_all = _build_grid_features_all_hosts(engine)
        for h in (6, 36):
            print(f"[{cluster}] Evaluating h={h}...")
            all_rows.extend(_evaluate_cluster_horizon(cluster, engine, feat_all, cache, h))

    df = pd.DataFrame(all_rows)
    df.to_csv(_OUT_DIR / "metrics.csv", index=False)

    # Write docs/twin_hotspot_evaluation.md
    lines = [
        "# Hotspot-Event Early-Warning Evaluation (Pre-Registered)",
        "",
        "Evaluates the pre-registered protocol (`docs/decisions.md`, 2026-11-25) using saved models (`models/twin/`) and forecast caches (`data/twin_cache/`) with zero retraining.",
        "",
        "- **Event:** Host currently below cluster warning threshold (`T(t) < W`, `W = 73.93°C` for D1, `75.72°C` for D2) crosses `T(t + h) >= W` at `t + 1 h` (`h=6`) or `t + 6 h` (`h=36`).",
        "- **Detectors:** `(A) Twin` (`CQR_upper >= W`), `(B) Persistence rule` (`T(t) >= W - m_persist`), `(C) Trend rule` (`T(t) + slope_1h(t) * h_hours >= W - m_trend`). Margins `m_persist` and `m_trend` are calibrated strictly on validation (`val_sel + val_cal`) to match the Twin's validation alert rate on `T(t) < W` rows, then evaluated once on `TEST`.",
        "- **Bootstrap CIs:** 95% host-level bootstrap (`B = 2,000`, `seed = 42`).",
        "- **Conclusiveness:** Reported as `inconclusive` if `n_events < 30`.",
        "",
        "| Cluster | Horizon | Events | Conclusive | Detector | Margin (°C) | Val Rate | Test Alerts (Rate) | Recall [95% CI] | Precision [95% CI] | Mean Lead (min) |",
        "|---|---|---:|---|---|---:|---:|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        rec = f"{r['recall']*100:.1f}% [{r['recall_ci_lo']*100:.1f}%, {r['recall_ci_hi']*100:.1f}%]" if pd.notna(r["recall"]) else "n/a"
        prc = f"{r['precision']*100:.1f}% [{r['precision_ci_lo']*100:.1f}%, {r['precision_ci_hi']*100:.1f}%]" if pd.notna(r["precision"]) else "n/a"
        lead = f"{r['mean_lead_time_min']:.0f}" if pd.notna(r["mean_lead_time_min"]) else "n/a"
        lines.append(
            f"| {r['cluster']} | {r['horizon_label']} (`h={r['horizon_steps']}`) | {r['n_events']} | {r['conclusive']} | "
            f"{r['detector']} | {r['param_margin_c']:.3f} | {r['val_alert_rate']*100:.2f}% | "
            f"{r['n_alerts']} ({r['alert_rate']*100:.2f}%) | {rec} | {prc} | {lead} |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "- Because `Persistence` and `Trend` margins are fixed on the validation partition to match the Twin's validation alert rate, any shift in test-set temperature distribution or CQR upper-bound margin produces corresponding shifts in test alert counts.",
        "- True-positive lead time equals the evaluated forecast horizon (`60` min for `+1h`, `360` min for `+6h`) because only hosts currently below `W` (`T(t) < W`) are eligible at issue time `t`.",
        "",
        "Source artifact: `experiments/twin_hotspots_20261125/metrics.csv` (`scripts/eval_twin_hotspots.py`).",
    ])
    _DOC_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + df[["cluster", "horizon_label", "n_events", "conclusive", "detector", "alert_rate", "recall", "precision"]].to_string(index=False))


if __name__ == "__main__":
    main()
