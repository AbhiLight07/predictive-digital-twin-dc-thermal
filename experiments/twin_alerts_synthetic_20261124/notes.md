# Twin Synthetic Fault Alert Evaluation (2026-11-24)

## What Was Run
Seeded (42) synthetic ramps, steps, and slow drift faults injected into 25 clean warm-normal TEST windows per cluster (`h=6` and `h=36` conformal upper bounds vs `configs/twin.yaml` warning thresholds).

## Results (SYNTHETIC FAULTS - not real-world detection accuracy)
> **SYNTHETIC FAULTS - not real-world detection accuracy.** All alerts below are evaluated on synthetically injected perturbations of TEST-partition windows using the exact twin rule (`CQR upper >= warn_thresh` at `h=6` or `h=36`).

| Cluster | Scenario | n_windows | n_crossed_warn | n_alerts | Alert Rate (All) | Detection Rate (Crossed) | Median Delay From Onset (min) | Median Lead Time (min) |
|---|---|---|---|---|---|---|---|---|
| D1 | `clean_control` | 25 | 0 | 4 | 16.0% | n/a | 330.0 | n/a |
| D1 | `ramp_3C_2h` | 25 | 0 | 10 | 40.0% | n/a | 265.0 | n/a |
| D1 | `ramp_6C_2h` | 25 | 4 | 21 | 84.0% | 100.0% | 100.0 | +420.0 |
| D1 | `ramp_10C_6h` | 25 | 8 | 25 | 100.0% | 100.0% | 190.0 | +175.0 |
| D1 | `step_5C` | 25 | 4 | 19 | 76.0% | 100.0% | 0.0 | +650.0 |
| D1 | `step_10C` | 25 | 12 | 25 | 100.0% | 100.0% | 0.0 | +0.0 |
| D1 | `drift_0.5C_per_h` | 25 | 4 | 11 | 44.0% | 100.0% | 520.0 | +205.0 |
| D2 | `clean_control` | 25 | 0 | 0 | 0.0% | n/a | n/a | n/a |
| D2 | `ramp_3C_2h` | 25 | 2 | 9 | 36.0% | 100.0% | 120.0 | +215.0 |
| D2 | `ramp_6C_2h` | 25 | 14 | 20 | 80.0% | 100.0% | 70.0 | +55.0 |
| D2 | `ramp_10C_6h` | 25 | 22 | 24 | 96.0% | 100.0% | 160.0 | +75.0 |
| D2 | `step_5C` | 25 | 9 | 18 | 72.0% | 100.0% | 0.0 | +60.0 |
| D2 | `step_10C` | 25 | 22 | 25 | 100.0% | 100.0% | 0.0 | +0.0 |
| D2 | `drift_0.5C_per_h` | 25 | 9 | 18 | 72.0% | 100.0% | 510.0 | +100.0 |

### Interpretation Note (Conformal Threshold-Margin Effect)
- Warm baseline hosts selected below `warn_thresh` can trigger alerts immediately at fault onset (`delay_from_onset = 0 min`) because the 6 h CQR upper bound adds a ~4 °C safety margin above current temperature. When the injected fault pushes current temperature just below `warn_thresh` and natural operational drift crosses `warn_thresh` hours later, the measured `lead_time_min` exceeds 360 min. That reflects conformal threshold-margin headroom rather than multi-hour physical fault foresight.
- False alerts on `clean_control` occur when unperturbed baseline trajectories plus the 6 h CQR margin reach `warn_thresh`.

## What This Shows & Does NOT Show
- **Shows:** How the CQR upper-bound alert rule behaves under synthetic temperature offsets/ramps when no unlabelled real fault ground truth exists.
- **Does NOT show:** Real-world anomaly detection precision/recall, hardware failure lead times, or causal fault attribution.
