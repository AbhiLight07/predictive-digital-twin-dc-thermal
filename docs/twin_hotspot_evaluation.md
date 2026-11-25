# Hotspot-Event Early-Warning Evaluation (Pre-Registered)

Evaluates the pre-registered protocol (`docs/decisions.md`, 2026-11-25) using saved models (`models/twin/`) and forecast caches (`data/twin_cache/`) with zero retraining.

- **Event:** Host currently below cluster warning threshold (`T(t) < W`, `W = 73.93°C` for D1, `75.72°C` for D2) crosses `T(t + h) >= W` at `t + 1 h` (`h=6`) or `t + 6 h` (`h=36`).
- **Detectors:** `(A) Twin` (`CQR_upper >= W`), `(B) Persistence rule` (`T(t) >= W - m_persist`), `(C) Trend rule` (`T(t) + slope_1h(t) * h_hours >= W - m_trend`). Margins `m_persist` and `m_trend` are calibrated strictly on validation (`val_sel + val_cal`) to match the Twin's validation alert rate on `T(t) < W` rows, then evaluated once on `TEST`.
- **Bootstrap CIs:** 95% host-level bootstrap (`B = 2,000`, `seed = 42`).
- **Conclusiveness:** Reported as `inconclusive` if `n_events < 30`.

| Cluster | Horizon | Events | Conclusive | Detector | Margin (°C) | Val Rate | Test Alerts (Rate) | Recall [95% CI] | Precision [95% CI] | Mean Lead (min) |
|---|---|---:|---|---|---:|---:|---|---|---|---|
| D1 | +1h (`h=6`) | 758 | conclusive | Twin (CQR upper) | 0.000 | 1.65% | 2784 (1.54%) | 63.5% [32.7%, 78.6%] | 17.3% [7.9%, 21.6%] | 60 |
| D1 | +1h (`h=6`) | 758 | conclusive | Persistence rule | 5.246 | 1.65% | 3887 (2.14%) | 47.1% [30.4%, 54.2%] | 9.2% [2.9%, 16.6%] | 60 |
| D1 | +1h (`h=6`) | 758 | conclusive | Trend rule | 3.780 | 1.65% | 3777 (2.08%) | 34.8% [25.4%, 43.8%] | 7.0% [2.5%, 12.0%] | 60 |
| D1 | +6h (`h=36`) | 1328 | conclusive | Twin (CQR upper) | 0.000 | 3.20% | 5771 (3.20%) | 50.4% [19.4%, 79.3%] | 11.6% [3.8%, 18.2%] | 360 |
| D1 | +6h (`h=36`) | 1328 | conclusive | Persistence rule | 7.797 | 3.20% | 6144 (3.40%) | 45.4% [26.4%, 59.4%] | 9.8% [3.9%, 16.1%] | 360 |
| D1 | +6h (`h=36`) | 1328 | conclusive | Trend rule | -2.471 | 3.20% | 5621 (3.11%) | 17.5% [10.1%, 24.0%] | 4.1% [1.2%, 8.2%] | 360 |
| D2 | +1h (`h=6`) | 1098 | conclusive | Twin (CQR upper) | 0.000 | 9.23% | 5114 (8.37%) | 89.3% [86.4%, 91.9%] | 19.2% [16.3%, 22.1%] | 60 |
| D2 | +1h (`h=6`) | 1098 | conclusive | Persistence rule | 1.353 | 9.23% | 5160 (8.45%) | 82.5% [78.0%, 86.4%] | 17.6% [14.5%, 20.8%] | 60 |
| D2 | +1h (`h=6`) | 1098 | conclusive | Trend rule | 1.253 | 9.23% | 5385 (8.82%) | 63.5% [58.1%, 68.8%] | 12.9% [10.1%, 15.7%] | 60 |
| D2 | +6h (`h=36`) | 1790 | conclusive | Twin (CQR upper) | 0.000 | 14.16% | 8191 (13.53%) | 85.1% [81.3%, 88.4%] | 18.6% [14.4%, 23.1%] | 360 |
| D2 | +6h (`h=36`) | 1790 | conclusive | Persistence rule | 2.087 | 14.16% | 8111 (13.39%) | 81.6% [76.2%, 86.1%] | 18.0% [13.6%, 22.4%] | 360 |
| D2 | +6h (`h=36`) | 1790 | conclusive | Trend rule | 0.337 | 14.16% | 8498 (14.03%) | 36.4% [33.7%, 39.0%] | 7.7% [4.9%, 10.5%] | 360 |

## Interpretation
- Because `Persistence` and `Trend` margins are fixed on the validation partition to match the Twin's validation alert rate, any shift in test-set temperature distribution or CQR upper-bound margin produces corresponding shifts in test alert counts.
- True-positive lead time equals the evaluated forecast horizon (`60` min for `+1h`, `360` min for `+6h`) because only hosts currently below `W` (`T(t) < W`) are eligible at issue time `t`.

Source artifact: `experiments/twin_hotspots_20261125/metrics.csv` (`scripts/eval_twin_hotspots.py`).
