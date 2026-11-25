# Planning-Interval Calibration Evaluation (NEXT-2 T5)

Evaluates two calibration strategies on the saved planning models (`models/twin/{d1,d2}_planning_0_*.txt`, unmodified) against a `90%` nominal target (`scripts/eval_planning_calibration.py`):
- **Baseline (`static_cqr`):** Global `q_hat` fit on `val_cal` (`81.09%` D1, `85.33%` D2).
- **(i) `rolling_7d`:** Causal 7-day sliding window over `[t - 7 days, t)` conformity scores strictly prior to `t` (fallback to `static_cqr` when `< 100` prior observations exist).
- **(ii) `regime_qhat`:** Separate CQR `q_hat` calibrated on `val_cal` for `idle` (`CPU_Load < 10%`), `active` (`10–50%`), and `heavy` (`> 50%`).

**Adoption rule:** Test coverage `>= 88.0%` AND mean width `<= 1.30x` static baseline on both clusters.

| Cluster | Method | Val Cov | Val Width | Test Cov | Test Width | Width Ratio | Idle Cov | Active Cov | Heavy Cov | Meets Rule |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| D1 | `static_cqr` | 90.00% | 16.11°C | **81.09%** | 15.87°C | 1.00x | 80.6% | 87.2% | 100.0% | NO |
| D1 | `rolling_7d` | 88.08% | 15.62°C | **89.36%** | 18.40°C | 1.16x | 89.1% | 92.7% | 100.0% | YES |
| D1 | `regime_qhat` | 90.00% | 16.12°C | **81.10%** | 15.90°C | 1.00x | 80.8% | 83.9% | 100.0% | NO |
| D2 | `static_cqr` | 90.00% | 19.63°C | **85.33%** | 17.87°C | 1.00x | 76.4% | 85.9% | 94.8% | NO |
| D2 | `rolling_7d` | 91.91% | 20.56°C | **89.09%** | 19.08°C | 1.07x | 80.6% | 89.4% | 98.6% | YES |
| D2 | `regime_qhat` | 90.01% | 20.47°C | **81.52%** | 18.36°C | 1.03x | 89.4% | 84.3% | 64.9% | NO |

## Regime Parameters (`val_cal`)
- **D1:** static `q_hat = 1.0542`, regime `q_hat = {'idle': 1.0984, 'active': 0.6029, 'heavy': 1.0542}`
- **D2:** static `q_hat = 0.0494`, regime `q_hat = {'idle': 3.2062, 'active': -0.1532, 'heavy': -2.1462}`

## Decision: ADOPTED (rolling_7d)
