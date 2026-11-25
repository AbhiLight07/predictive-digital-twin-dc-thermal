# Decisions Log

All project decisions not covered in the PROJECT_BRIEF.md are logged here with dates and rationale.

| Date | Decision | Rationale | Who |
|------|----------|-----------|-----|
| 2026-11-08 | Project initialised with structure from PROJECT_BRIEF.md | Following the brief's Phase 0 requirements | Agent |
| 2026-11-08 | Host column identified as `Series` | Auto-detected and verified in both datasets | Agent (audit) |
| 2026-11-08 | Time column identified as `Time` | Auto-detected and verified; format `YYYY-MM-DD HH:MM:SS` | Agent (audit) |
| 2026-11-08 | No disk/storage columns available | Neither dataset contains disk I/O or storage metrics; S2 feature set adjusted to exclude disk features | Agent (audit) |
| 2026-11-08 | Dataset 2 missingness relaxation to < 20% | D2 has 12–16% structural missingness in RAM and VMs across nodes; only 2 servers pass <10%. Relaxing to <20% qualifies 53 servers for cross-cluster evaluation while disqualifying servers with broken thermal sensors (`qh2-rcc263`, `rcc264`, `rcc226`, `rcc205`) | Agent |
| 2026-11-08 | 5 Dataset 1 servers lack inlet/fan data | `rcc196`–`rcc200` have 100% missing inlet temp and fan speeds but valid CPU temps; they participate in S0–S3 and planning regimes, but are excluded from S4 | Agent (audit) |
| 2026-11-08 | Gate verdict: PASS (76 qualifying servers from Dataset 1) | Exceeds the 40-server threshold; Dataset 1 is the primary dataset for all experiments | Agent (audit) |
| 2026-11-08 | 6-hour Embargo Buffer in Temporal Splits | Prevents rolling window overlap and autocorrelation leakage between train and test folds | Agent |
| 2026-11-08 | Correction: Current Temperature T(t) added to S1 and S4 | The forecast origin state T(t) was omitted in initial runs while persistence had access to T(t); using time-t state is valid and causal | Agent (review) |
| 2026-11-08 | Correction: Target re-formulated as Delta T = T(t+h) - T(t) | Absolute temperature targets penalise gradient boosting under distribution shifts; predicting incremental delta aligns model with thermal physics | Agent (review) |
| 2026-11-08 | Correction: GroupKFold (k=5) by server implemented | Fixes stratification bug where only lowest-load servers were tested; guarantees 100% of servers are evaluated across 5 folds | Agent (review) |
| 2026-11-08 | Correction: Split A re-aligned to 70/10/20 | Train 0–70%, Val 70–80%, Embargo 6h, Test 80%+6h to 100%, matching the protocol specification | Agent (review) |
| 2026-11-08 | Purging of unverified "Ahmed et al." citation | The dataset paper is Ilager et al. (UCC 2023); "Ahmed et al." was an unverified hallucination from prior draft | Agent (review) |
| 2026-11-08 | Dell PowerEdge hardware correction | As reported in the dataset paper (Ilager et al., UCC 2023): dual-CPU hosts; capacities as in the trace columns; other hardware specifications are unverified | Agent (review) |
| 2026-11-08 | Time-consistent cold-start pretraining | Pretrained models must strictly be trained on data prior to target server adaptation windows to prevent future leakage | Agent (review) |
| 2026-11-12 | Evaluation protocol frozen | Frozen protocol with 70/10/20 chronological Split A (36-step embargo), 5-fold GroupKFold Split B, and strict Split C; commit 6909f188 | Agent |
| 2026-11-24 | Regular 10-min grid alignment in twin serving | `TwinEngine._build_monitoring_features` now calls `fill_missing_timestamps(..., freq="10min")` before lag/rolling computation, and `forecast()` returns `unknown` when stale (>30 min) or when post-gap features are `NaN`, matching `scripts/train_twin_models.py` | Agent |
| 2026-11-25 | Pre-registered Hotspot-Event Evaluation Protocol | Locked before running evaluation (see details below) | Agent |

## Pre-Registered Hotspot-Event Protocol (2026-11-25)

- **Scope:** Saved models (`models/twin/`) and precomputed test forecast caches (`data/twin_cache/{d1,d2}_forecasts.parquet`) only; no retraining. Clusters: `D1` (`W = 73.93°C`) and `D2` (`W = 75.72°C`) from `configs/twin.yaml`. Horizons evaluated separately: `t + 1 h` (`h = 6`, `h_hours = 1.0`) and `t + 6 h` (`h = 36`, `h_hours = 6.0`).
- **Event Definition:** At evaluation timestamp `t` (30-min stride, aligned with forecast cache), a host currently strictly below the cluster warning threshold (`T_i(t) < W`) experiences a hotspot crossing event if its effective CPU temperature reaches or exceeds the warning threshold at `t + h` (`T_i(t + h) >= W`). Rows with missing `T_i(t)`, missing `T_i(t + h)`, or missing 1 h causal lookback `T_i(t - 6 steps)` are excluded uniformly across all detectors.
- **Detectors:**
  - **(A) Twin (CQR upper bound):** Alert iff `CQR_upper_h(t) >= W`.
  - **(B) Persistence rule:** Alert iff `T_i(t) >= W - m_persist` (equivalently `W - T_i(t) <= m_persist`).
  - **(C) Trend rule:** Alert iff `T_i(t) + slope_1h(t) * h_hours >= W - m_trend`, where `slope_1h(t) = (T_i(t) - T_i(t - 6 steps)) / 1.0 h` on the uniform 10-min grid.
- **Matched-Alert-Rate Calibration (Validation Partition Only):** On the validation partition (`val_sel_start <= t <= val_cal_end`, 30-min stride, strictly before `test_start`), compute the Twin's alert rate `r_val` among eligible (`T_i(t) < W`) rows. Choose `m_persist` and `m_trend` as the `(1 - r_val)` quantile of `(W - T_i(t))` and `(W - (T_i(t) + slope_1h(t) * h_hours))` respectively on validation so that (B) and (C) raise the same validation alert rate as (A). Evaluate **once** on `TEST`.
- **Metrics & Inference:** Event count (`n_events`), alert count (`n_alerts`), alert rate, recall, precision, and mean lead time (for each positive event at target `t + h`, lead time in minutes from the earliest alert on that host within `[t, t + h)` before `t + h`, which equals `h * 10` min for a single-horizon alert at `t` or earlier if alerted on an earlier stride inside `(t_cross - h*10m, t_cross)`), with 95% host-level bootstrap confidence intervals (`B = 2000` resamples over physical hosts, `seed = 42`), reported per cluster and horizon.
- **Conclusiveness Criterion:** If `n_events < 30` for any `(cluster, horizon)`, report the result as **`inconclusive`**.

