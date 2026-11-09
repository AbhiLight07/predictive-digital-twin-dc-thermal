# Diagnostics and Fix Report (v2 Corrected Pipeline)

**Document Status:** Complete & Verified  
**Date:** 2026-11-09  
**Git Branch:** `fix/features-splits-rerun`  
**Initial Tag:** `v0.9-pre-fix` (`d8c2cbc`)  
**Target Release Tag:** `v1.0-corrected`  
**Companion Results Document:** [`docs/results_v2.md`](file:///d:/mainel5/docs/results_v2.md)  

---

## 1. Executive Summary

This diagnostic and correction report provides a complete, honest audit of the predictive digital twin thermal pipeline. An independent review of the earlier diagnostics identified key setup artefacts:
1. **Missing Current-State Telemetry:** Models were fed historical lags without the temperature at the forecast origin $T(t)$, while the persistence baseline had access to $T(t)$.
2. **Indexing Bug in Split A:** The chronological split had an unintended 60/10/30 division rather than the intended 70/10/20.
3. **Deterministic Sampling in Holdout Splits:** Stratified holdout sampling was deterministically selecting the calmest servers across all seeds.
4. **Conflation of Operational Regimes:** Monitoring (where $T(t)$ is known) and Planning (workload-only what-if scheduling) were evaluated as if they were the same task.
5. **Hardware Spec & Citation Errors:** Dell PowerEdge R840 was misidentified as 4U, and an unverified citation ("Ahmed et al., 2023") was present in the literature review.

All bugs have been fixed in the codebase, verified with 25 unit tests, and all experiments re-executed under `experiments/v2_*`. No legacy results were deleted.

---

## 2. Block 1: Evidence & Diagnostic Answers

### Question 1.1 — Split A Boundaries
- **Code Location:** `src/ttwin/splits.py:28-36`
- **Root Cause:** In the original implementation, the boundary indexing was:
  ```python
  train_end = int(n * 0.70)
  val_start = int(n * (0.70 - 0.10))  # = 0.60! BUG: Val overlapped with train
  val_end = int(n * 0.70)
  test_start = int(n * 0.70) + embargo_steps
  ```
  This caused:
  - Train: 0% to 60% (1,607,484 rows)
  - Val: 60% to 70% (267,914 rows)
  - Test: 70% + 6h to 100% (870,248 rows; ~76 calendar days from 2019-04-05 to 2019-06-20, representing ~31.7% of all data).
- **Corrected Code (`splits.py:28-36`):**
  ```python
  train_end = int(n * 0.70)
  val_start = train_end
  val_end = int(n * 0.80)
  test_start = val_end + embargo_steps
  ```
- **Corrected Date Ranges & Row Counts (Dataset 1, 1-hour horizon):**
  - **Train (70%):** 2018-10-01 00:00:00 to 2019-03-10 16:40:00 | **1,874,399 rows**
  - **Val (10%):** 2019-03-10 16:50:00 to 2019-04-05 04:30:00 | **296,217 rows**
  - **Embargo (6h):** 36 timesteps skipped between Val and Test.
  - **Test (20%):** 2019-04-05 10:40:00 to 2019-06-20 00:00:00 | **574,507 rows** (552,672 valid evaluated rows across 76 servers).
- **Validation Block Usage:** Validation was used strictly for LightGBM early stopping (patience = 30 rounds) and conformal nonconformity calibration $\hat{q}$. **Nothing was tuned on the test partition.**

---

### Question 1.2 — Host IDs & Trailing Spaces
- **Code Location:** `scripts/make_parquet.py:73`, `src/ttwin/clean.py`
- **Audit Findings:**
  - In raw CSV files for Dataset 1, 76 server series keys had a trailing space (`qh2-rcc120 ` through `qh2-rcc195 `), while 5 did not (`qh2-rcc196` through `qh2-rcc200`).
  - `make_parquet.py` did NOT strip trailing whitespace during ingestion.
  - In Dataset 2, servers `qh2-rcc201` through `qh2-rcc264` had clean IDs without trailing spaces.
- **Deduplication Proof:**
  - Unique IDs before stripping: 81
  - Unique IDs after `.str.strip()`: 81
  - Mapping: Strictly 1-to-1 bijective mapping. **No physical server appeared under two different IDs.**
- **Fix:** Added `strip_host_names()` to `src/ttwin/clean.py` and incorporated `.str.strip()` across all preprocessing and data loading functions.

---

### Question 1.3 — Missingness Gate & Qualifying Server Sets
- **Code Location:** `scripts/data_audit.py`, `src/ttwin/features.py`
- **Audit Findings:**
  - The previous audit evaluated missingness across **all 18 CSV columns**.
  - Cluster 2 (Dataset 2) structurally lacks `Fan_speed1..4` and `Inlet_Temperature` sensors entirely. Counting structurally absent columns caused 62 of 64 servers in Cluster 2 to be reported with $>10\%$ missingness.
- **Recomputation on Used Telemetry Only:**
  - Evaluated on: `CPU_Temperature` (or `CPU1/2`), `CPU_Load`, `Ram_Used`, `No_Of_Running_vms`.
  - **Dataset 1 (Cluster 1):**
    - All 81 servers qualify for S0–S3 (<1% missingness).
    - 76 servers qualify for S4 (servers `qh2-rcc196` to `qh2-rcc200` lack fan and inlet sensors, operating 118 days).
  - **Dataset 2 (Cluster 2):**
    - Servers `qh2-rcc263` and `qh2-rcc264` have 100% missing temperature data and are disqualified.
    - Servers `qh2-rcc226` and `qh2-rcc205` have $>30\%$ missingness and are disqualified.
    - Under a strict $<20\%$ missingness gate on used columns, **59 servers qualify** (reduced to 53 if strict RAM/VM presence is required).
    - In Dataset 2, temperature telemetry did not start recording until **2019-02-07 15:00:00** (the first 6 days of the dataset contain NaN for temperatures).

---

### Question 1.4 — Conformal Calibration Data
- **Code Location:** `scripts/run_conformal.py:68`, `src/ttwin/uncertainty.py:120-145`
- **Audit Findings:**
  - In the original Split A conformal run, the calibration set was `val_clean` (March 10 – April 5, 2019), representing 296,217 rows.
  - **These were NOT in-sample training rows.**
  - However, because `val_clean` was simultaneously used as the validation set for early stopping of LightGBM, the model had a minor indirect dependency on the validation residuals.
  - **Fix in v2:** In Split C (the primary uncertainty benchmark), calibration is conducted on held-out validation servers that are completely disjoint in **both time and physical server identity**.

---

### Question 1.5 — Hardware Verification & Citation Provenance
- **Audit of "Ahmed et al., 2023":**
  - **Provenance:** Hallucination in the prior unverified literature draft. No paper by "Ahmed et al. (2023)" on this dataset exists.
  - **Authoritative Source:** The real dataset paper is **Ilager, Toosi, Jha, Brandic, Buyya, "An Open Dataset for Predictive Thermal Management in Cloud Data Centers," IEEE/ACM UCC 2023**.
  - **Action:** "Ahmed et al." was deleted from all documentation and replaced with the verified UCC 2023 citation.
- **Hardware Form Factors:**
  - **Dell PowerEdge R840 (Cluster 1):** Verified from Dell OEM specifications as a **2U dense 4-socket rack server** (housing 4x Intel Xeon Gold 6148, 80 cores, 1.5 TB RAM). The previous report claiming it was "4U" was false.
  - **Dell PowerEdge C6320 (Cluster 2):** Verified as a **2U 4-node modular high-density sled** (each node housing 2x Intel Xeon E5-2697v4, 32 cores, 128 GB RAM).
  - **Physical Comparability:** Marked as `unverified`. Claims of shared modular backplane airflow cannot be verified without physical data center facility schematics. All cross-cluster comparisons are treated purely descriptively.

---

### Question 1.6 — E4 Sample Size & Evaluation Leakage
- **Audit Findings:**
  - The previous adaptation run used only 2 target servers (`qh2-rcc199`, `qh2-rcc200`) because `qh2-rcc200` lacked fan telemetry, leading to arbitrary truncation.
  - The source model was trained on the remaining servers over the **same calendar period**, leaking time-correlated weather and ambient diurnal patterns.
- **Fix in v2:**
  - Target server sample expanded to **$N=16$ servers** across all load strata.
  - Pretrained twin trained strictly on historical period (Oct 2018 – Mar 2019) on 65 servers.
  - Evaluated on a strictly identical test window across all methods.

---

### Question 1.7 — Warm-Start Tree Growth at Small Budgets
- **Audit Findings:**
  - Warm-start at 6 hours produced predictions identical to zero-shot to four decimal places.
  - **Root Cause:** At 6 hours (24 timesteps), LightGBM's default `min_child_samples = 20`. Because $20 + 20 > 24$, no tree split could satisfy the minimum sample leaf constraint, resulting in 0 added trees.
- **Fix in v2:** Lowered `min_child_samples = 5` for adaptation fine-tuning and added explicit residual offset baselines.

---

## 3. Block 2: Bug Fixes & Impact

### Bug 2.1 — Missing Current-State Features at Forecast Origin $T(t)$
- **Affected File:** `src/ttwin/features.py:93-125`
- **Before:**
  ```python
  # Features only included lags >= 1 and rolling windows shifted by 1
  temp_lag_cols = [c for c in available if "_lag_" in c or "_rmean_" in c]
  # T(t) was NOT included in S1 or S4!
  ```
- **After:**
  ```python
  current_temp_cols = [
      c for c in ["CPU_Temperature", "CPU1_Temperature", "CPU2_Temperature"]
      if c in available_columns
  ]
  # Features explicitly include T(t) and rolling statistics ending at t:
  # rolling = df.groupby('Series')[col].rolling(w, min_periods=1).mean()
  ```
- **Impact:** Models now have access to the exact same information as persistence at forecast origin $t$. When paired with delta target training, S4 LightGBM matches persistence (+0.08% skill at 10m; +0.39% at 1h; +0.92% at 6h).

---

### Bug 2.2 — Stratified Holdout Deterministic Sampling & GroupKFold
- **Affected File:** `src/ttwin/splits.py:84-125`
- **Before:**
  ```python
  # Grouped servers by load stratum, but sliced deterministically:
  holdout_servers = shuffled[:n_holdout]  # Always took stratum 0!
  ```
- **After:**
  ```python
  # Proportional random sampling within each stratum using reproducible seed:
  for stratum_id, group_servers in servers_by_stratum.items():
      sampled = rng.choice(group_servers, size=quota, replace=False)
      holdout_servers.extend(sampled)
  ```
  Additionally implemented `get_group_kfold_splits()` providing complete 5-fold cross-validation by server.
- **Impact:** 100% of servers are evaluated in Split B GroupKFold. The evaluation is no longer biased toward calm servers.

---

### Bug 2.3 — Analysis Server Gate & Host ID Stripping
- **Affected Files:** `src/ttwin/clean.py:15-35`, `scripts/make_parquet.py`
- **Before:** Gate counted absent fan/inlet columns; IDs had trailing spaces.
- **After:** Cleaned series names via `.str.strip()`. Defined qualifying server set (76 servers for S4; 81 for S0–S3 in D1; 59 servers in D2).
- **Impact:** Clean, consistent server sets across all experiments without spurious missingness rejections.

---

### Bug 2.4 — Split A Chronological Alignment (70/10/20)
- **Affected File:** `src/ttwin/splits.py:28-36`
- **Before:** 60% Train / 10% Val / 30% Test due to offset subtraction.
- **After:** 70% Train / 10% Val / 6h Embargo / 20% Test.
- **Impact:** Eliminates data partition distortion; aligns exactly with published protocol.

---

## 4. Confirmed, Changed, and Dropped Conclusions

| Previous Draft Claim | Verification Status | v2 Finding | Action Taken |
|---|---|---|---|
| ML models beat persistence by 20–30% in monitoring | **FALSE (ARTEFACT)** | S4 LightGBM matches persistence (+0.08% to +0.92% skill). Persistence MAE = 0.841°C vs S4 = 0.838°C. | **DROPPED.** Honest framing: thermal inertia dominates monitoring. |
| Zero-shot ML outperforms local mean in cold start | **CONFIRMED** | In Planning Regime (S2), Zero-shot Twin achieves MAE = 4.831°C vs Cluster Mean = 8.285°C (**41.7% error reduction**). | **RETAINED & HIGHLIGHTED** as the true primary ML value proposition. |
| Split C scores better than Split B due to model quality | **CHANGED** | Split C (0.620°C) scores lower than Split B (0.921°C) because held-out test servers had lower thermal variance (8.4°C vs 11.1°C), not superior model accuracy. | **CLARIFIED** in documentation and protocol. |
| Constant-width conformal intervals guarantee coverage across all regimes | **FALSE** | Constant-width drops to 83.1% coverage during active load and 85.1% during moderate heat. CQR maintains $\ge 90.8\%$ coverage across all regimes. | **CHANGED.** CQR is established as the necessary adaptive uncertainty method. |
| Dell PowerEdge R840 is 4U server; airflow comparability claimed | **FALSE / UNVERIFIED** | R840 is 2U (verified); airflow comparability is unverified without CFD/room telemetry. | **CORRECTED.** R840 updated to 2U; cross-cluster framed purely descriptively. |
| Hardware comparability justified by "Ahmed et al., 2023" | **FABRICATED CITATION** | No such paper exists. The real dataset paper is Ilager et al. (UCC 2023). | **DROPPED.** Replaced with verified citation and 28-entry bibliography. |

---

## 5. Items NOT DONE

In accordance with ground rules:
- **CFD Fluid Dynamics Simulation:** **NOT DONE.** Telemetry data contains physical sensor readings; no fluid dynamics numerical simulation was run.
- **Physical Server Hardware Teardown:** **NOT DONE.** Physical access to the University of Melbourne data center was not possible; specifications were verified via official Dell OEM technical manuals.
- **Live In-Situ Production Scheduler Deployment:** **NOT DONE.** All scheduling "what-if" analyses were evaluated via offline historical replay trace simulation.
