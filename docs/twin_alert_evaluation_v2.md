# Early-Warning Threshold Crossing Evaluation (v2 — Matched-FPR Baselines)

> **Early-warning threshold crossing, not a fault detector.** **SYNTHETIC FAULTS - not real-world detection accuracy.**

Evaluates `100` clean TEST windows per cluster across 7 synthetic perturbation scenarios (`scripts/eval_twin_alerts.py --v2`).
Baseline detectors (**Rolling z-score** `(T - rmean_36) / max(rstd_36, 0.1) >= z_thresh` and **One-step change** `T(t) - T(t-1) >= delta_thresh`) are calibrated on each cluster's `clean_control` windows to match the Twin's exact clean-control false-alarm rate (FPR):
- **D1 (`100` windows):** matched clean-control FPR = **15.0%** (`z_thresh = 4.51`, `delta_1step_thresh = 3.41°C`).
- **D2 (`100` windows):** matched clean-control FPR = **0.0%** (`z_thresh = 5.00`, `delta_1step_thresh = 6.77°C`).

| Cluster | Scenario | Detector | n_windows | Crossed Warn | Alerts (Rate) | Detection Rate (Crossed) | Median Delay (min) | Median Lead (min) |
|---|---|---|---:|---:|---:|---:|---:|---:|
| D1 | `clean_control` | Twin (CQR upper) | 100 | 0 | 15 (15.0%) | n/a | 310.0 | n/a |
| D1 | `clean_control` | Rolling z-score | 100 | 0 | 15 (15.0%) | n/a | 280.0 | n/a |
| D1 | `clean_control` | One-step change | 100 | 0 | 15 (15.0%) | n/a | 290.0 | n/a |
| D1 | `ramp_3C_2h` | Twin (CQR upper) | 100 | 2 | 32 (32.0%) | 100.0% | 155.0 | +165.0 |
| D1 | `ramp_3C_2h` | Rolling z-score | 100 | 2 | 7 (7.0%) | 0.0% | 520.0 | n/a |
| D1 | `ramp_3C_2h` | One-step change | 100 | 2 | 17 (17.0%) | 0.0% | 280.0 | n/a |
| D1 | `ramp_6C_2h` | Twin (CQR upper) | 100 | 13 | 86 (86.0%) | 100.0% | 100.0 | +130.0 |
| D1 | `ramp_6C_2h` | Rolling z-score | 100 | 13 | 10 (10.0%) | 30.8% | 215.0 | +0.0 |
| D1 | `ramp_6C_2h` | One-step change | 100 | 13 | 18 (18.0%) | 61.5% | 220.0 | +0.0 |
| D1 | `ramp_10C_6h` | Twin (CQR upper) | 100 | 39 | 100 (100.0%) | 100.0% | 190.0 | +200.0 |
| D1 | `ramp_10C_6h` | Rolling z-score | 100 | 39 | 3 (3.0%) | 0.0% | 20.0 | n/a |
| D1 | `ramp_10C_6h` | One-step change | 100 | 39 | 17 (17.0%) | 33.3% | 280.0 | +10.0 |
| D1 | `step_5C` | Twin (CQR upper) | 100 | 10 | 68 (68.0%) | 100.0% | 15.0 | +285.0 |
| D1 | `step_5C` | Rolling z-score | 100 | 10 | 63 (63.0%) | 60.0% | 0.0 | +75.0 |
| D1 | `step_5C` | One-step change | 100 | 10 | 100 (100.0%) | 100.0% | 0.0 | +395.0 |
| D1 | `step_10C` | Twin (CQR upper) | 100 | 45 | 100 (100.0%) | 100.0% | 0.0 | +30.0 |
| D1 | `step_10C` | Rolling z-score | 100 | 45 | 91 (91.0%) | 82.2% | 0.0 | +0.0 |
| D1 | `step_10C` | One-step change | 100 | 45 | 100 (100.0%) | 100.0% | 0.0 | +30.0 |
| D1 | `drift_0.5C_per_h` | Twin (CQR upper) | 100 | 5 | 39 (39.0%) | 100.0% | 390.0 | +100.0 |
| D1 | `drift_0.5C_per_h` | Rolling z-score | 100 | 5 | 5 (5.0%) | 20.0% | 70.0 | +50.0 |
| D1 | `drift_0.5C_per_h` | One-step change | 100 | 5 | 17 (17.0%) | 80.0% | 280.0 | +75.0 |
| D2 | `clean_control` | Twin (CQR upper) | 100 | 0 | 0 (0.0%) | n/a | n/a | n/a |
| D2 | `clean_control` | Rolling z-score | 100 | 0 | 0 (0.0%) | n/a | n/a | n/a |
| D2 | `clean_control` | One-step change | 100 | 0 | 0 (0.0%) | n/a | n/a | n/a |
| D2 | `ramp_3C_2h` | Twin (CQR upper) | 100 | 7 | 36 (36.0%) | 100.0% | 120.0 | +130.0 |
| D2 | `ramp_3C_2h` | Rolling z-score | 100 | 7 | 0 (0.0%) | 0.0% | n/a | n/a |
| D2 | `ramp_3C_2h` | One-step change | 100 | 7 | 0 (0.0%) | 0.0% | n/a | n/a |
| D2 | `ramp_6C_2h` | Twin (CQR upper) | 100 | 48 | 69 (69.0%) | 100.0% | 80.0 | +50.0 |
| D2 | `ramp_6C_2h` | Rolling z-score | 100 | 48 | 0 (0.0%) | 0.0% | n/a | n/a |
| D2 | `ramp_6C_2h` | One-step change | 100 | 48 | 0 (0.0%) | 0.0% | n/a | n/a |
| D2 | `ramp_10C_6h` | Twin (CQR upper) | 100 | 79 | 94 (94.0%) | 100.0% | 180.0 | +90.0 |
| D2 | `ramp_10C_6h` | Rolling z-score | 100 | 79 | 0 (0.0%) | 0.0% | n/a | n/a |
| D2 | `ramp_10C_6h` | One-step change | 100 | 79 | 0 (0.0%) | 0.0% | n/a | n/a |
| D2 | `step_5C` | Twin (CQR upper) | 100 | 36 | 66 (66.0%) | 100.0% | 0.0 | +30.0 |
| D2 | `step_5C` | Rolling z-score | 100 | 36 | 34 (34.0%) | 36.1% | 0.0 | +40.0 |
| D2 | `step_5C` | One-step change | 100 | 36 | 0 (0.0%) | 0.0% | n/a | n/a |
| D2 | `step_10C` | Twin (CQR upper) | 100 | 82 | 99 (99.0%) | 100.0% | 0.0 | +0.0 |
| D2 | `step_10C` | Rolling z-score | 100 | 82 | 79 (79.0%) | 78.0% | 0.0 | +0.0 |
| D2 | `step_10C` | One-step change | 100 | 82 | 100 (100.0%) | 100.0% | 0.0 | +0.0 |
| D2 | `drift_0.5C_per_h` | Twin (CQR upper) | 100 | 29 | 53 (53.0%) | 100.0% | 460.0 | +170.0 |
| D2 | `drift_0.5C_per_h` | Rolling z-score | 100 | 29 | 0 (0.0%) | 0.0% | n/a | n/a |
| D2 | `drift_0.5C_per_h` | One-step change | 100 | 29 | 1 (1.0%) | 0.0% | 620.0 | n/a |

### Interpretation
- This experiment evaluates **early-warning threshold crossing** against cluster `warn_thresh` (`73.93°C` D1, `75.72°C` D2), **not** physical fault diagnosis.
- At the matched clean-control FPR, the **One-step change** detector responds to sudden `step_5C`/`step_10C` jumps but misses gradual ramps and slow drift (`ramp_6C_2h`, `ramp_10C_6h`, `drift_0.5C_per_h`), while **Rolling z-score** detects initial excursions before rolling mean/std adapt; the **Twin (CQR upper)** tracks proximity to the absolute threshold `warn_thresh`.

Source artifact: `experiments/twin_alerts_synthetic_v2_20261125/metrics.csv` (`scripts/eval_twin_alerts.py --v2`).
