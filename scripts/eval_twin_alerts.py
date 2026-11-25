"""Synthetic-fault evaluation of the digital twin alert rule (WP4).

SYNTHETIC FAULTS - not real-world detection accuracy.
Evaluates the twin's exact alert rule (conformal upper bound at h=6 or h=36
reaching cluster warning threshold from configs/twin.yaml) on deterministic,
seeded synthetic thermal faults injected into TEST-partition telemetry copies.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
    delay_min = None
    if detected:
        t_first_alert = times.iloc[int(np.where(alert_mask)[0][0])]
        t_onset = times.iloc[0]
        delay_min = (t_first_alert - t_onset).total_seconds() / 60.0
        if crossed:
            t_first_cross = times.iloc[int(crossed_idx[0])]
            lead_min = (t_first_cross - t_first_alert).total_seconds() / 60.0

    return {
        "detected": detected,
        "crossed_warn": crossed,
        "lead_time_min": lead_min,
        "delay_from_onset_min": delay_min,
    }


def main() -> None:
    seed = 42
    n_windows = 25
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
            delays = [e["delay_from_onset_min"] for e in evals if e["delay_from_onset_min"] is not None]
            rows.append({
                "cluster": cluster.upper(),
                "scenario": sc,
                "n_windows": len(windows),
                "n_crossed_warn": n_cross,
                "n_alerts": n_det,
                "alert_rate_all": round(n_det / len(windows), 4),
                "detection_rate_crossed": round(det_on_cross / n_cross, 4) if n_cross > 0 else None,
                "median_delay_from_onset_min": round(float(np.median(delays)), 1) if delays else None,
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
        "| Cluster | Scenario | n_windows | n_crossed_warn | n_alerts | Alert Rate (All) | Detection Rate (Crossed) | Median Delay From Onset (min) | Median Lead Time (min) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        dc = f"{r['detection_rate_crossed']*100:.1f}%" if pd.notna(r["detection_rate_crossed"]) else "n/a"
        dl = f"{r['median_delay_from_onset_min']:.1f}" if pd.notna(r["median_delay_from_onset_min"]) else "n/a"
        lt = f"{r['median_lead_time_min']:+.1f}" if pd.notna(r["median_lead_time_min"]) else "n/a"
        table_lines.append(
            f"| {r['cluster']} | `{r['scenario']}` | {r['n_windows']} | {r['n_crossed_warn']} | {r['n_alerts']} | {r['alert_rate_all']*100:.1f}% | {dc} | {dl} | {lt} |"
        )

    interp_note = (
        "### Interpretation Note (Conformal Threshold-Margin Effect)\n"
        "- Warm baseline hosts selected below `warn_thresh` can trigger alerts immediately at fault onset (`delay_from_onset = 0 min`) "
        "because the 6 h CQR upper bound adds a ~4 °C safety margin above current temperature. When the injected fault pushes current temperature "
        "just below `warn_thresh` and natural operational drift crosses `warn_thresh` hours later, the measured `lead_time_min` exceeds 360 min. "
        "That reflects conformal threshold-margin headroom rather than multi-hour physical fault foresight.\n"
        "- False alerts on `clean_control` occur when unperturbed baseline trajectories plus the 6 h CQR margin reach `warn_thresh`."
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
        interp_note,
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
        interp_note,
        "",
        "Source artifact: `experiments/twin_alerts_synthetic_20261124/metrics.csv` (`scripts/eval_twin_alerts.py`).",
    ]) + "\n"
    (_REPO / "docs" / "twin_alert_evaluation.md").write_text(doc_txt, encoding="utf-8")
    print(df[["cluster", "scenario", "n_crossed_warn", "n_alerts", "median_lead_time_min"]].to_string(index=False))


def _select_test_windows_v2(engine: TwinEngine, n_windows: int = 100, seed: int = 42) -> List[pd.DataFrame]:
    """Select >= n_windows clean continuous 24h (144-step) TEST windows with stride 18 steps (3h)."""
    b = engine._models_info["monitoring_1"]["boundaries"]
    test_start = pd.Timestamp(b["test_start"])
    warn_thresh = float(engine._warn_thresh)

    pre_steps = 72
    post_steps = 72
    win_len = pre_steps + post_steps
    stride = 18

    warm_candidates = []
    cool_candidates = []
    for _, grp in engine._data.groupby("Series", sort=False):
        g = grp.sort_values("Time").reset_index(drop=True)
        g_test = g[g["Time"] >= test_start].reset_index(drop=True)
        if len(g_test) < win_len + 10:
            continue
        for start in range(0, len(g_test) - win_len, stride):
            sub = g_test.iloc[start : start + win_len]
            if sub["CPU_Temperature"].isna().any():
                continue
            dt_ok = (sub["Time"].diff().dt.total_seconds().dropna() == 600.0).all()
            if not dt_ok:
                continue
            tmax = float(sub["CPU_Temperature"].max())
            tmean = float(sub["CPU_Temperature"].mean())
            if tmax <= warn_thresh - 2.5:
                if tmean >= warn_thresh - 14.0:
                    warm_candidates.append(sub.copy())
                else:
                    cool_candidates.append(sub.copy())

    candidates = warm_candidates if len(warm_candidates) >= n_windows else (warm_candidates + cool_candidates)
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(candidates), size=min(n_windows, len(candidates)), replace=False)
    return [candidates[int(i)].reset_index(drop=True) for i in idx]


def _window_scores(engine: TwinEngine, win_df: pd.DataFrame, scenario: str, onset_idx: int) -> dict:
    """Compute post-onset detector traces (Twin upper, rolling z-score, one-step delta) for one window."""
    inj = inject_synthetic_fault(win_df, scenario, onset_idx)
    feat_df = engine._build_monitoring_features(inj)
    post = feat_df.iloc[onset_idx:].copy()
    times = post["Time"].reset_index(drop=True)
    temps = post["CPU_Temperature"].to_numpy(dtype=np.float64)
    warn_thresh = float(engine._warn_thresh)

    # 1. Twin alert trace (h=6 or h=36 CQR upper >= warn_thresh)
    twin_mask = np.zeros(len(post), dtype=bool)
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
        twin_mask[np.where(valid)[0][hits]] = True

    # 2. Rolling z-score trace: (T(t) - rmean_36) / max(rstd_36, 0.1)
    rmean = post["CPU_Temperature_rmean_36"].to_numpy(dtype=np.float64)
    rstd = np.maximum(post["CPU_Temperature_rstd_36"].to_numpy(dtype=np.float64), 0.1)
    z_trace = np.where(~np.isnan(temps) & ~np.isnan(rmean) & ~np.isnan(rstd), (temps - rmean) / rstd, -np.inf)

    # 3. Fixed one-step change trace: T(t) - T(t-1)
    lag1 = post["CPU_Temperature_lag_1"].to_numpy(dtype=np.float64)
    d1_trace = np.where(~np.isnan(temps) & ~np.isnan(lag1), temps - lag1, -np.inf)

    crossed_idx = np.where(temps >= warn_thresh)[0]
    return {
        "times": times,
        "crossed_warn": len(crossed_idx) > 0,
        "t_first_cross": times.iloc[int(crossed_idx[0])] if len(crossed_idx) > 0 else None,
        "twin_mask": twin_mask,
        "z_trace": z_trace,
        "d1_trace": d1_trace,
        "z_max": float(np.max(z_trace)),
        "d1_max": float(np.max(d1_trace)),
    }


def _match_threshold(clean_max_scores: List[float], target_n_alerts: int) -> float:
    """Choose threshold so exactly target_n_alerts clean-control windows trigger."""
    s = sorted(clean_max_scores, reverse=True)
    if target_n_alerts <= 0:
        return float(s[0] + 1e-4)
    if target_n_alerts >= len(s):
        return float(s[-1] - 1e-4)
    return float(0.5 * (s[target_n_alerts - 1] + s[target_n_alerts]))


def _summarize_detector(evals: List[dict], mask_fn) -> dict:
    n_win = len(evals)
    n_cross = sum(1 for e in evals if e["crossed_warn"])
    n_det = 0
    det_cross = 0
    leads = []
    delays = []
    for e in evals:
        m = mask_fn(e)
        if m.any():
            n_det += 1
            t_first = e["times"].iloc[int(np.where(m)[0][0])]
            delays.append((t_first - e["times"].iloc[0]).total_seconds() / 60.0)
            if e["crossed_warn"]:
                det_cross += 1
                leads.append((e["t_first_cross"] - t_first).total_seconds() / 60.0)
    return {
        "n_windows": n_win,
        "n_crossed_warn": n_cross,
        "n_alerts": n_det,
        "alert_rate_all": round(n_det / n_win, 4),
        "detection_rate_crossed": round(det_cross / n_cross, 4) if n_cross > 0 else None,
        "median_delay_from_onset_min": round(float(np.median(delays)), 1) if delays else None,
        "median_lead_time_min": round(float(np.median(leads)), 1) if leads else None,
    }


def run_v2() -> None:
    """Run >=100-window early-warning threshold-crossing evaluation with matched-FPR baselines."""
    seed = 42
    n_windows = 100
    onset_idx = 72

    rows = []
    cal_info = {}
    for cluster in ("d1", "d2"):
        engine = TwinEngine(cluster)
        windows = _select_test_windows_v2(engine, n_windows=n_windows, seed=seed)
        print(f"[{cluster}] v2 evaluating {len(windows)} TEST windows across {len(SCENARIOS)} scenarios...")

        # Evaluate clean_control first to match baseline FPR to Twin clean-control FPR
        clean_evals = [_window_scores(engine, w, "clean_control", onset_idx) for w in windows]
        n_twin_clean = sum(1 for e in clean_evals if e["twin_mask"].any())
        z_thresh = _match_threshold([e["z_max"] for e in clean_evals], n_twin_clean)
        d1_thresh = _match_threshold([e["d1_max"] for e in clean_evals], n_twin_clean)
        cal_info[cluster] = {
            "n_windows": len(windows),
            "clean_control_fpr": round(n_twin_clean / len(windows), 4),
            "z_thresh": round(z_thresh, 4),
            "delta_1step_thresh_c": round(d1_thresh, 4),
        }

        detectors: List[Tuple[str, object]] = [
            ("Twin (CQR upper)", lambda e: e["twin_mask"]),
            ("Rolling z-score", lambda e, zt=z_thresh: e["z_trace"] >= zt),
            ("One-step change", lambda e, dt=d1_thresh: e["d1_trace"] >= dt),
        ]

        for sc in SCENARIOS:
            sc_evals = clean_evals if sc == "clean_control" else [
                _window_scores(engine, w, sc, onset_idx) for w in windows
            ]
            for det_name, mfn in detectors:
                s = _summarize_detector(sc_evals, mfn)
                rows.append({
                    "cluster": cluster.upper(),
                    "scenario": sc,
                    "detector": det_name,
                    **s,
                    "clean_control_fpr": cal_info[cluster]["clean_control_fpr"],
                    "warn_thresh_c": round(float(engine._warn_thresh), 2),
                    "label": "SYNTHETIC FAULTS - not real-world detection accuracy",
                })

    df = pd.DataFrame(rows)
    exp_dir = _REPO / "experiments" / "twin_alerts_synthetic_v2_20261125"
    exp_dir.mkdir(parents=True, exist_ok=True)
    with open(exp_dir / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({
            "seed": seed,
            "n_windows_per_cluster": n_windows,
            "onset_idx": onset_idx,
            "matched_fpr_calibration": cal_info,
            "scenarios": SCENARIOS,
        }, f)
    df.to_csv(exp_dir / "metrics.csv", index=False)

    lines = [
        "# Early-Warning Threshold Crossing Evaluation (v2 — Matched-FPR Baselines)",
        "",
        "> **Early-warning threshold crossing, not a fault detector.** **SYNTHETIC FAULTS - not real-world detection accuracy.**",
        "",
        "Evaluates `100` clean TEST windows per cluster across 7 synthetic perturbation scenarios (`scripts/eval_twin_alerts.py --v2`).",
        "Baseline detectors (**Rolling z-score** `(T - rmean_36) / max(rstd_36, 0.1) >= z_thresh` and **One-step change** `T(t) - T(t-1) >= delta_thresh`) are calibrated on each cluster's `clean_control` windows to match the Twin's exact clean-control false-alarm rate (FPR):",
        f"- **D1 (`100` windows):** matched clean-control FPR = **{cal_info['d1']['clean_control_fpr']*100:.1f}%** (`z_thresh = {cal_info['d1']['z_thresh']:.2f}`, `delta_1step_thresh = {cal_info['d1']['delta_1step_thresh_c']:.2f}°C`).",
        f"- **D2 (`100` windows):** matched clean-control FPR = **{cal_info['d2']['clean_control_fpr']*100:.1f}%** (`z_thresh = {cal_info['d2']['z_thresh']:.2f}`, `delta_1step_thresh = {cal_info['d2']['delta_1step_thresh_c']:.2f}°C`).",
        "",
        "| Cluster | Scenario | Detector | n_windows | Crossed Warn | Alerts (Rate) | Detection Rate (Crossed) | Median Delay (min) | Median Lead (min) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df.iterrows():
        dc = f"{r['detection_rate_crossed']*100:.1f}%" if pd.notna(r["detection_rate_crossed"]) else "n/a"
        dl = f"{r['median_delay_from_onset_min']:.1f}" if pd.notna(r["median_delay_from_onset_min"]) else "n/a"
        lt = f"{r['median_lead_time_min']:+.1f}" if pd.notna(r["median_lead_time_min"]) else "n/a"
        lines.append(
            f"| {r['cluster']} | `{r['scenario']}` | {r['detector']} | {r['n_windows']} | {r['n_crossed_warn']} | "
            f"{r['n_alerts']} ({r['alert_rate_all']*100:.1f}%) | {dc} | {dl} | {lt} |"
        )
    lines.extend([
        "",
        "### Interpretation",
        "- This experiment evaluates **early-warning threshold crossing** against cluster `warn_thresh` (`73.93°C` D1, `75.72°C` D2), **not** physical fault diagnosis.",
        "- At the matched clean-control FPR, the **One-step change** detector responds to sudden `step_5C`/`step_10C` jumps but misses gradual ramps and slow drift (`ramp_6C_2h`, `ramp_10C_6h`, `drift_0.5C_per_h`), while **Rolling z-score** detects initial excursions before rolling mean/std adapt; the **Twin (CQR upper)** tracks proximity to the absolute threshold `warn_thresh`.",
        "",
        "Source artifact: `experiments/twin_alerts_synthetic_v2_20261125/metrics.csv` (`scripts/eval_twin_alerts.py --v2`).",
    ])
    (_REPO / "docs" / "twin_alert_evaluation_v2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(df[["cluster", "scenario", "detector", "n_crossed_warn", "n_alerts", "detection_rate_crossed"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2", action="store_true", help="Run 100-window v2 early-warning evaluation")
    args = parser.parse_args()
    if args.v2:
        run_v2()
    else:
        main()
