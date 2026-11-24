"""Synthetic-fault evaluation of the digital twin alert rule (WP4).

SYNTHETIC FAULTS - not real-world detection accuracy.
Evaluates the twin's exact alert rule (conformal upper bound at h=6 or h=36
reaching cluster warning threshold from configs/twin.yaml) on deterministic,
seeded synthetic thermal faults injected into TEST-partition telemetry copies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from ttwin.features import compute_effective_cpu_temp  # noqa: E402
from ttwin.twin.engine import TwinEngine  # noqa: E402

SCENARIOS: Dict[str, dict] = {
    "clean_control": {"kind": "clean", "mag": 0.0, "duration_steps": 0},
    "ramp_3C_2h": {"kind": "ramp", "mag": 3.0, "duration_steps": 12},
    "ramp_6C_2h": {"kind": "ramp", "mag": 6.0, "duration_steps": 12},
    "ramp_10C_6h": {"kind": "ramp", "mag": 10.0, "duration_steps": 36},
    "step_5C": {"kind": "step", "mag": 5.0, "duration_steps": 1},
    "step_10C": {"kind": "step", "mag": 10.0, "duration_steps": 1},
    "drift_0.5C_per_h": {"kind": "drift", "mag": 0.5, "duration_steps": 72},
}


def inject_synthetic_fault(df: pd.DataFrame, scenario: str, onset_idx: int) -> pd.DataFrame:
    """Return a copy of df with a deterministic synthetic thermal fault injected from onset_idx.

    Rows strictly before onset_idx remain untouched (no future leakage).
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown synthetic fault scenario: {scenario}")
    spec = SCENARIOS[scenario]
    out = df.copy().reset_index(drop=True)
    n = len(out)
    deltas = np.zeros(n, dtype=np.float64)

    kind = spec["kind"]
    mag = float(spec["mag"])
    dur = int(spec["duration_steps"])

    for k in range(onset_idx, n):
        elapsed = k - onset_idx
        if kind == "clean":
            d = 0.0
        elif kind == "step":
            d = mag
        elif kind == "ramp":
            d = mag * min(1.0, (elapsed + 1) / max(1, dur))
        elif kind == "drift":
            d = mag * ((elapsed + 1) / 6.0)  # mag °C per hour (6 steps/h)
        else:
            raise ValueError(f"Unknown fault kind: {kind}")
        deltas[k] = d

    for col in ("CPU1_Temperature", "CPU2_Temperature", "CPU_Temperature"):
        if col in out.columns:
            out[col] = (out[col].astype(np.float64) + deltas).astype(out[col].dtype)
    out = compute_effective_cpu_temp(out)
    return out


def _select_test_windows(engine: TwinEngine, n_windows: int = 8, seed: int = 42) -> List[pd.DataFrame]:
    """Select continuous 30h (180-step) TEST windows with clean baseline (no prior alerts)."""
    b = engine._models_info["monitoring_1"]["boundaries"]
    test_start = pd.Timestamp(b["test_start"])
    warn_thresh = float(engine._warn_thresh)

    # Prefer warm-but-safe windows [warn - 10, warn - 2.5] so threshold-crossing faults are informative
    candidates = []
    pre_steps = 72   # 12h lookback before onset (covers max rolling 36 + lag 12)
    post_steps = 72  # 12h evaluation window starting at onset
    win_len = pre_steps + post_steps

    for host, grp in engine._data.groupby("Series", sort=False):
        g = grp.sort_values("Time").reset_index(drop=True)
        g_test = g[g["Time"] >= test_start].reset_index(drop=True)
        if len(g_test) < win_len + 10:
            continue
        # Step through non-overlapping 12h blocks
        for start in range(0, len(g_test) - win_len, post_steps):
            sub = g_test.iloc[start : start + win_len]
            if sub["CPU_Temperature"].isna().any():
                continue
            dt_ok = (sub["Time"].diff().dt.total_seconds().dropna() == 600.0).all()
            if not dt_ok:
                continue
            tmax = float(sub["CPU_Temperature"].max())
            tmean = float(sub["CPU_Temperature"].mean())
            if tmax <= warn_thresh - 2.5 and tmean >= warn_thresh - 14.0:
                candidates.append(sub.copy())

    if len(candidates) < n_windows:
        # Fallback: relax lower mean bound if needed
        for host, grp in engine._data.groupby("Series", sort=False):
            g = grp.sort_values("Time").reset_index(drop=True)
            g_test = g[g["Time"] >= test_start].reset_index(drop=True)
            for start in range(0, len(g_test) - win_len, post_steps):
                sub = g_test.iloc[start : start + win_len]
                if sub["CPU_Temperature"].isna().any():
                    continue
                if float(sub["CPU_Temperature"].max()) <= warn_thresh - 2.5:
                    candidates.append(sub.copy())
                if len(candidates) >= n_windows * 4:
                    break
            if len(candidates) >= n_windows * 4:
                break

    rng = np.random.RandomState(seed)
    idx = rng.choice(len(candidates), size=min(n_windows, len(candidates)), replace=False)
    return [candidates[int(i)].reset_index(drop=True) for i in idx]


def _evaluate_window(engine: TwinEngine, win_df: pd.DataFrame, scenario: str, onset_idx: int) -> dict:
    inj = inject_synthetic_fault(win_df, scenario, onset_idx)
    feat_df = engine._build_monitoring_features(inj)
    post = feat_df.iloc[onset_idx:].copy()
    times = post["Time"].reset_index(drop=True)
    temps = post["CPU_Temperature"].to_numpy(dtype=np.float64)
    warn_thresh = float(engine._warn_thresh)

    # Check if twin triggers alert at ANY post-onset timestamp (upper >= warn at h=6 or h=36)
    alert_mask = np.zeros(len(post), dtype=bool)
    for h in (6, 36):
        info = engine._models_info[f"monitoring_{h}"]
        feats = info["feature_names"]
        X = post[feats].to_numpy(dtype=np.float64)
        valid = ~np.isnan(X).any(axis=1) & ~np.isnan(temps)
        if not valid.any():
            continue
        m_hi = engine._models.get(f"{engine.cluster}_monitoring_{h}_upper")
        m_pt = engine._models[f"{engine.cluster}_monitoring_{h}_point"]
        q_hat = float(info.get("q_hat", 0.0))
        if m_hi is not None:
            upper = temps[valid] + m_hi.predict(X[valid]) + q_hat
        else:
            upper = temps[valid] + m_pt.predict(X[valid]) + abs(q_hat)
        hits = upper >= warn_thresh
        alert_mask[np.where(valid)[0][hits]] = True

    crossed_idx = np.where(temps >= warn_thresh)[0]
    crossed = len(crossed_idx) > 0
    detected = bool(alert_mask.any())
    lead_min = None
    if detected and crossed:
        t_first_alert = times.iloc[int(np.where(alert_mask)[0][0])]
        t_first_cross = times.iloc[int(crossed_idx[0])]
        lead_min = (t_first_cross - t_first_alert).total_seconds() / 60.0

    return {
        "detected": detected,
        "crossed_warn": crossed,
        "lead_time_min": lead_min,
    }


def main() -> None:
    seed = 42
    n_windows = 12
    onset_idx = 72  # 12h pre-onset history

    rows = []
    for cluster in ("d1", "d2"):
        engine = TwinEngine(cluster)
        windows = _select_test_windows(engine, n_windows=n_windows, seed=seed)
        print(f"[{cluster}] Evaluating {len(windows)} synthetic windows across {len(SCENARIOS)} scenarios...")
        for sc in SCENARIOS:
            evals = [_evaluate_window(engine, w, sc, onset_idx) for w in windows]
            n_det = sum(1 for e in evals if e["detected"])
            n_cross = sum(1 for e in evals if e["crossed_warn"])
            det_on_cross = sum(1 for e in evals if e["detected"] and e["crossed_warn"])
            leads = [e["lead_time_min"] for e in evals if e["lead_time_min"] is not None]
            rows.append({
                "cluster": cluster.upper(),
                "scenario": sc,
                "n_windows": len(windows),
                "n_crossed_warn": n_cross,
                "n_alerts": n_det,
                "alert_rate_all": round(n_det / len(windows), 4),
                "detection_rate_crossed": round(det_on_cross / n_cross, 4) if n_cross > 0 else None,
                "median_lead_time_min": round(float(np.median(leads)), 1) if leads else None,
                "warn_thresh_c": round(float(engine._warn_thresh), 2),
                "label": "SYNTHETIC FAULTS - not real-world detection accuracy",
            })

    df = pd.DataFrame(rows)
    exp_dir = _REPO / "experiments" / "twin_alerts_synthetic_20261124"
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"seed": seed, "n_windows_per_cluster": n_windows, "onset_idx": onset_idx, "scenarios": SCENARIOS}, f)
    df.to_csv(exp_dir / "metrics.csv", index=False)

    # Notes & doc summary
    table_lines = [
        "> **SYNTHETIC FAULTS - not real-world detection accuracy.** All alerts below are evaluated on synthetically injected perturbations of TEST-partition windows using the exact twin rule (`CQR upper >= warn_thresh` at `h=6` or `h=36`).",
        "",
        "| Cluster | Scenario | n_windows | n_crossed_warn | n_alerts | Alert Rate (All) | Detection Rate (Crossed) | Median Lead Time (min) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        dc = f"{r['detection_rate_crossed']*100:.1f}%" if pd.notna(r["detection_rate_crossed"]) else "n/a"
        lt = f"{r['median_lead_time_min']:+.1f}" if pd.notna(r["median_lead_time_min"]) else "n/a"
        table_lines.append(
            f"| {r['cluster']} | `{r['scenario']}` | {r['n_windows']} | {r['n_crossed_warn']} | {r['n_alerts']} | {r['alert_rate_all']*100:.1f}% | {dc} | {lt} |"
        )

    notes_txt = "\n".join([
        "# Twin Synthetic Fault Alert Evaluation (2026-11-24)",
        "",
        "## What Was Run",
        f"Seeded ({seed}) synthetic ramps, steps, and slow drift faults injected into {n_windows} clean warm-normal TEST windows per cluster (`h=6` and `h=36` conformal upper bounds vs `configs/twin.yaml` warning thresholds).",
        "",
        "## Results (SYNTHETIC FAULTS - not real-world detection accuracy)",
        *table_lines,
        "",
        "## What This Shows & Does NOT Show",
        "- **Shows:** How the CQR upper-bound alert rule behaves under synthetic temperature offsets/ramps when no unlabelled real fault ground truth exists.",
        "- **Does NOT show:** Real-world anomaly detection precision/recall, hardware failure lead times, or causal fault attribution.",
    ]) + "\n"
    (exp_dir / "notes.md").write_text(notes_txt, encoding="utf-8")

    doc_txt = "\n".join([
        "# Twin Alert Rule — Synthetic Fault Evaluation",
        "",
        "SYNTHETIC FAULTS - not real-world detection accuracy.",
        "",
        *table_lines,
        "",
        "Source artifact: `experiments/twin_alerts_synthetic_20261124/metrics.csv` (`scripts/eval_twin_alerts.py`).",
    ]) + "\n"
    (_REPO / "docs" / "twin_alert_evaluation.md").write_text(doc_txt, encoding="utf-8")
    print(df[["cluster", "scenario", "n_crossed_warn", "n_alerts", "median_lead_time_min"]].to_string(index=False))


if __name__ == "__main__":
    main()
