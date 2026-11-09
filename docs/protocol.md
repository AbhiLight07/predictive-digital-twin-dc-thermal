# Evaluation Protocol (v2 Corrected)

**Status:** Frozen — Updated following comprehensive diagnostic audit (2026-11-08).

---

## 1. Experimental Splits

### Split A — Chronological (Temporal Holdout)
* **Partitions:** 70% Train, 10% Validation, 20% Test based on total dataset duration.
* **Embargo Buffer:** A 6-hour embargo buffer (36 sampling steps of 10 min) is strictly excluded between validation and test partitions to eliminate autocorrelation leakage across the rolling feature horizon.
* **Cutoffs:** Identical global temporal cutoffs applied to all servers within each cluster.

### Split B — GroupKFold by Server (Spatial Holdout)
* **Structure:** 5-fold cross-validation by server, ensuring 100% of servers are evaluated in exactly one test partition.
* **Stratification:** Servers are stratified by average CPU load across folds to ensure balanced load regimes in every fold.
* **Reporting:** Mean and standard deviation across the 5 folds, plus aggregated metrics on the combined out-of-fold predictions.

### Split C — Strict Spatiotemporal Holdout
* **Structure:** Evaluates unseen servers (20% held-out) at a strictly later time (the final 20% test duration).
* **Calibration:** Calibration is performed on seen servers during the validation window (70%–80% time), testing uncertainty validity under joint spatial and temporal distribution shift.

---

## 2. Model Targets & Feature Sets

### Target Formulation
* **Monitoring Regime (S1, S4):** Models predict the incremental delta:
  $$\Delta T = T(t+h) - T(t)$$
  The final temperature forecast is evaluated as $\hat{T}(t+h) = T(t) + \Delta\hat{T}$.
* **Planning Regime (S2, S3):** Because temperature at the forecast origin $T(t)$ is unknown, models predict absolute temperature $T(t+h)$ directly from workload and calendar telemetry.

### Feature Sets (Ablation Ladder)
* **S0 (Persistence Baseline):** $\hat{T}(t+h) = T(t)$
* **S1 (Temperature History + Current State):** Current temperatures at $t$ (`CPU_Temperature`, `CPU1_Temperature`, `CPU2_Temperature`), lagged temperatures ($t-1, t-2, t-3, t-6, t-12$), and rolling mean/std ending at $t$ ($w \in \{6, 12, 36\}$).
* **S2 (Workload-Only / Planning):** `CPU_Load`, `Ram`, `Ram_Used`, `CPU_cores`, `CPU_cores_used`, `Network_RX`, `Network_TX`, `No_Of_Running_vms`, and cyclic calendar features (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`).
* **S3 (Workload + Power):** S2 features plus `Power`.
* **S4 (Full Monitoring Telemetry):** S3 features plus current `Inlet_Temperature`, `Fan_speed1..4`, current temperatures at $t$, and all temperature lags and rolling statistics.

---

## 3. Evaluation Metrics on Identical Rows
All comparisons within a forecast horizon must be computed on strictly **identical rows**:
* **MAE** (°C): Mean Absolute Error
* **RMSE** (°C): Root Mean Squared Error
* **Skill Score vs Persistence:**
  $$\text{Skill} = 1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{persistence}}}$$
* **Per-Server Robustness:** Median per-server MAE and 90th percentile (worst-decile) per-server MAE.

---

## 4. Cold-Start Adaptation Protocol
* **Time-Consistent Pretraining:** Source models must be pretrained strictly on data timestamped prior to target server adaptation windows.
* **Hour-Scale Budgets:** $N \in \{0, 1, 6, 24, 72, 168\}$ hours.
* **Fixed Test Window:** All adaptation strategies evaluated on an identical test window occurring after $168\text{h} + 6\text{h}$ embargo.
* **Target Sample Size:** Evaluated on $\ge 15$ target servers with 95% bootstrap confidence intervals over servers.
