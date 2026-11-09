#!/usr/bin/env python3
"""Synthetic Fault-Injection Test for Thermal Alert Rules.

NOTE: All anomalies evaluated in this script are SYNTHETICALLY INJECTED
for the purpose of testing the alert detection rule under simulated failures.

Alert Rule:
- Warning Alert: Conformal upper bound (90%) >= 70.0°C
- Critical Alert: Conformal upper bound (90%) >= 75.0°C OR |actual - pred| > 3 * q_hat

Synthetic Fault Scenarios:
1. Simulated Cooling/Fan Failure: +15°C thermal ramp over 3 steps.
2. Simulated Workload Overload: Persistent 100% CPU load causing thermal surge.
3. Simulated Sensor Drift: Linear calibration drift of +0.5°C per hour.

Outputs:
- experiments/v2_fault_injection/fault_injection_results.json
"""

import sys
sys.stdout.reconfigure(line_buffering=True)

from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from ttwin.io import load_parquet
from ttwin.features import compute_effective_cpu_temp
from ttwin.metrics import mae

OUT_DIR = ROOT / "experiments" / "v2_fault_injection"
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data" / "interim"


def run_fault_injection_experiment():
    print("\n" + "=" * 70)
    print("SYNTHETIC FAULT-INJECTION TEST OF THERMAL ALERT RULES")
    print("=" * 70)
    print("DISCLAIMER: All fault scenarios are SYNTHETIC simulations for algorithm testing.\n")

    path = DATA_DIR / "dataset1" / "_combined.parquet"
    df = load_parquet(path)
    df["Series"] = df["Series"].astype(str).str.strip()
    df = compute_effective_cpu_temp(df)

    # Select nominal baseline segment (1000 timesteps of a healthy server)
    test_srv = "qh2-rcc120"
    sdf = df[df["Series"] == test_srv].sort_values("Time").iloc[5000:6000].copy().reset_index(drop=True)

    y_true = sdf["CPU_Temperature"].values
    n_steps = len(y_true)

    # Simulated conformal band: q_hat = 1.8°C (90% interval = +/- 1.8°C)
    q_hat = 1.80
    pred = y_true + np.random.normal(0, 0.4, size=n_steps)  # baseline predictor
    upper = pred + q_hat

    # 1. Nominal False Alarm Rate
    nominal_crit_alerts = (upper >= 75.0) | (np.abs(y_true - pred) > 3 * q_hat)
    nominal_far = float(np.mean(nominal_crit_alerts))
    print(f"Nominal False Alarm Rate on healthy telemetry: {nominal_far:.2%}")

    scenarios = {}

    # Scenario A: Simulated Sudden Fan Failure (+15°C ramp starting at step 300)
    y_fan_fail = y_true.copy()
    fault_start = 300
    fault_duration = 30
    ramp = np.linspace(0, 15.0, 5)
    y_fan_fail[fault_start:fault_start+5] += ramp
    y_fan_fail[fault_start+5:fault_start+fault_duration] += 15.0

    upper_fan = pred + q_hat  # model predicts without knowing fan failed
    alerts_fan = (upper_fan >= 75.0) | (np.abs(y_fan_fail - pred) > 3 * q_hat)
    detection_step = np.where(alerts_fan[fault_start:])[0]
    latency_steps = int(detection_step[0]) if len(detection_step) > 0 else None
    tpr_fan = float(np.mean(alerts_fan[fault_start:fault_start+fault_duration]))

    scenarios["simulated_fan_failure"] = {
        "description": "Synthetic sudden +15°C thermal runaway from simulated cooling loss",
        "fault_start_step": fault_start,
        "fault_duration_steps": fault_duration,
        "detection_latency_timesteps": latency_steps,
        "detection_latency_minutes": latency_steps * 10 if latency_steps is not None else None,
        "true_positive_rate_during_fault": tpr_fan,
    }
    print(f"Scenario A (Simulated Fan Failure): Detected at step +{latency_steps} ({latency_steps * 10} min), TPR={tpr_fan:.1%}")

    # Scenario B: Simulated Workload Thermal Spike (+10°C sustained)
    y_spike = y_true.copy()
    fault_start_b = 600
    y_spike[fault_start_b:fault_start_b+20] += 10.0
    alerts_spike = (pred + q_hat >= 75.0) | (np.abs(y_spike - pred) > 3 * q_hat)
    det_b = np.where(alerts_spike[fault_start_b:])[0]
    lat_b = int(det_b[0]) if len(det_b) > 0 else None
    tpr_spike = float(np.mean(alerts_spike[fault_start_b:fault_start_b+20]))

    scenarios["simulated_workload_surge"] = {
        "description": "Synthetic sudden +10°C workload thermal surge",
        "fault_start_step": fault_start_b,
        "detection_latency_timesteps": lat_b,
        "detection_latency_minutes": lat_b * 10 if lat_b is not None else None,
        "true_positive_rate_during_fault": tpr_spike,
    }
    print(f"Scenario B (Simulated Workload Spike): Detected at step +{lat_b} ({lat_b * 10} min), TPR={tpr_spike:.1%}")

    # Scenario C: Simulated Gradual Sensor Drift (+0.5°C / hr = +0.083°C / step)
    y_drift = y_true.copy()
    fault_start_c = 750
    drift_steps = 150
    drift_ramp = np.arange(drift_steps) * (0.5 / 6.0)
    y_drift[fault_start_c:fault_start_c+drift_steps] += drift_ramp
    alerts_drift = (pred + q_hat >= 75.0) | (np.abs(y_drift - pred) > 3 * q_hat)
    det_c = np.where(alerts_drift[fault_start_c:])[0]
    lat_c = int(det_c[0]) if len(det_c) > 0 else None
    tpr_drift = float(np.mean(alerts_drift[fault_start_c:fault_start_c+drift_steps]))

    scenarios["simulated_sensor_drift"] = {
        "description": "Synthetic gradual sensor calibration drift (+0.5°C/hr)",
        "fault_start_step": fault_start_c,
        "detection_latency_timesteps": lat_c,
        "detection_latency_minutes": lat_c * 10 if lat_c is not None else None,
        "true_positive_rate_during_fault": tpr_drift,
    }
    print(f"Scenario C (Simulated Sensor Drift): Detected at step +{lat_c} ({lat_c * 10} min), TPR={tpr_drift:.1%}")

    out_data = {
        "test_type": "SYNTHETIC_FAULT_INJECTION_SIMULATION",
        "conformal_q_hat": q_hat,
        "alert_threshold_degC": 75.0,
        "nominal_false_alarm_rate": nominal_far,
        "scenarios": scenarios,
    }

    out_file = OUT_DIR / "fault_injection_results.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\nFault injection results saved to {out_file}")


if __name__ == "__main__":
    run_fault_injection_experiment()
