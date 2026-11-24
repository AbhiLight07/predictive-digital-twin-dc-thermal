# Twin Alert Rule — Synthetic Fault Evaluation

SYNTHETIC FAULTS - not real-world detection accuracy.

> **SYNTHETIC FAULTS - not real-world detection accuracy.** All alerts below are evaluated on synthetically injected perturbations of TEST-partition windows using the exact twin rule (`CQR upper >= warn_thresh` at `h=6` or `h=36`).

| Cluster | Scenario | n_windows | n_crossed_warn | n_alerts | Alert Rate (All) | Detection Rate (Crossed) | Median Lead Time (min) |
|---|---|---|---|---|---|---|---|
| D1 | `clean_control` | 12 | 0 | 2 | 16.7% | n/a | n/a |
| D1 | `ramp_3C_2h` | 12 | 0 | 5 | 41.7% | n/a | n/a |
| D1 | `ramp_6C_2h` | 12 | 2 | 10 | 83.3% | 100.0% | +545.0 |
| D1 | `ramp_10C_6h` | 12 | 4 | 12 | 100.0% | 100.0% | +135.0 |
| D1 | `step_5C` | 12 | 1 | 10 | 83.3% | 100.0% | +650.0 |
| D1 | `step_10C` | 12 | 5 | 12 | 100.0% | 100.0% | +240.0 |
| D1 | `drift_0.5C_per_h` | 12 | 2 | 6 | 50.0% | 100.0% | +70.0 |
| D2 | `clean_control` | 12 | 0 | 0 | 0.0% | n/a | n/a |
| D2 | `ramp_3C_2h` | 12 | 1 | 4 | 33.3% | 100.0% | +100.0 |
| D2 | `ramp_6C_2h` | 12 | 5 | 9 | 75.0% | 100.0% | +50.0 |
| D2 | `ramp_10C_6h` | 12 | 9 | 11 | 91.7% | 100.0% | +70.0 |
| D2 | `step_5C` | 12 | 4 | 8 | 66.7% | 100.0% | +125.0 |
| D2 | `step_10C` | 12 | 9 | 12 | 100.0% | 100.0% | +0.0 |
| D2 | `drift_0.5C_per_h` | 12 | 4 | 8 | 66.7% | 100.0% | +230.0 |

Source artifact: `experiments/twin_alerts_synthetic_20261124/metrics.csv` (`scripts/eval_twin_alerts.py`).
