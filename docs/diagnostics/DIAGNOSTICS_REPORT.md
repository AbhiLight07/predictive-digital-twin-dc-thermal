# Independent Diagnostic & Verification Report

**Project:** Predictive Digital Twin for Data Centre Thermal Management  
**Audit Protocol:** Diagnose, Not Defend  
**Date:** 2026-11-08  
**Scope:** Verification of experimental claims across Priority Blocks P1, P2, and P3  

---

## Executive Summary

This diagnostic report provides an unvarnished audit of the experimental findings, baseline formulations, statistical procedures, and scope deliverables reported in `docs/project_execution_summary.md` and `reports/research_paper_draft.md`.

### Core Diagnostic Takeaways
1. **The Baseline Comparison was Structurally Flawed:** In the original experiments, machine learning models (S1, S4) were trained on **absolute temperature** $T(t+h)$ without receiving current temperature $T(t)$ as an input feature (only receiving $T(t-1)$), whereas the persistence baseline explicitly evaluated $T(t)$. Under this formulation, **every machine learning model scored worse than persistence across all forecast horizons** (skill scores: $-45.3\%$ at 10 min, $-7.3\%$ at 1 hour, $-3.0\%$ at 6 hours).
2. **Delta Formulation Resolves the Deficit but Only Achieves Parity:** Re-formulating the model target as the future temperature increment $\Delta T = T(t+h) - T(t)$ restores parity with persistence (1-hour skill score $+0.12\%$), but machine learning models do not meaningfully outperform the persistence baseline.
3. **Split B Multi-Seed "Robustness" was an Artifact:** All 5 seeds for Split B produced the exact same 16 held-out servers and identical metrics ($\text{MAE} = 0.5273^\circ\text{C}$, skill score $-22.5\%$ vs persistence) due to a deterministic sorting step in `_stratified_holdout()`.
4. **Adaptation Claims Depended on Inconsistent Test Windows:** In the original adaptation experiment, zero-shot errors varied with $N$ because each budget $N$ was evaluated on a different test window. When evaluated on an identical, fixed test window after 168 hours, **zero-shot pretrained transfer ($0.415^\circ\text{C}$) and warm-started models ($0.516^\circ\text{C}$) were both worse than simple persistence ($0.325^\circ\text{C}$)**. Continued boosting consistently overfit local adaptation data.
5. **Cross-Cluster Transfer Underperformed Persistence:** In Experiment E3, native Cluster 2 models ($1.117^\circ\text{C}$) and transferred models ($1.349^\circ\text{C}$) both performed worse than Cluster 2 persistence ($0.812^\circ\text{C}$). Under a delta target, both models match persistence ($0.814^\circ\text{C}$ and $0.815^\circ\text{C}$), demonstrating negligible transfer penalty ($+0.0018^\circ\text{C}$) but zero predictive advantage over persistence.
6. **Key Protocol and Scope Items Were Omitted:** Random $k$-fold cross-validation comparison, SHAP feature attribution, synthetic fault injection, Dataset 2 ablation, and all 5 required Streamlit dashboard features are **NOT DONE**. Environmental explanations for temporal drift (weather/setpoints) are **NOT TESTED**.

---

## P1 — Persistence and Target Formulation

### 1.1 Target Formulation and Feature Construction
* **Target Type:** The model target is the **absolute future temperature** $T(t+h)$, **not the delta** $T(t+h) - T(t)$.
  * **Code Reference:** `src/ttwin/features.py:70`:
    ```python
    df[target_name] = df.groupby(server_col)[target_col].shift(-horizon_steps)
    ```
* **Feature Construction:** Defined in `src/ttwin/features.py:90-130`:
  * **S1 (Lagged Temperature Only, 33 features):** Contains lags $[1, 2, 3, 6, 12]$ (corresponding to $t-10\text{m}, t-20\text{m}, t-30\text{m}, t-1\text{h}, t-2\text{h}$) and rolling means/stds over windows $[6, 12, 36]$ for `CPU_Temperature`, `CPU1_Temperature`, and `CPU2_Temperature`.
  * **S4 (Full Telemetry, 53 features):** Contains all S1 features plus 8 workload metrics (`CPU_Load`, `Ram`, `Ram_Used`, `CPU_cores`, `CPU_cores_used`, `Network_RX`, `Network_TX`, `No_Of_Running_vms`), 6 calendar features, 1 power metric, 4 fan speeds, and 1 inlet temperature.
  * **Crucial Asymmetry:** S1 and S4 **do not include current temperature $T(t)$** as an input feature (the closest lag is $t-1$). In contrast, persistence baseline $S0$ predicts $\hat{T}(t+h) = T(t)$. Because server temperature exhibits high thermal inertia, $T(t)$ is a much stronger predictor than $T(t-1)$, creating an unfair structural penalty for the ML models.

---

### 1.2 Unified Ablation Table (Split A, Dataset 1)
Evaluated on **identical rows** across all models within each forecast horizon on Split A's test partition (2019-04-05 to 2019-06-20):
* **10 min Horizon:** $N = 833,644$ identical rows
* **1 h Horizon:** $N = 832,884$ identical rows
* **6 h Horizon:** $N = 828,324$ identical rows

$$\text{Skill Score vs Persistence} = 1 - \frac{\text{MAE}_{\text{model}}}{\text{MAE}_{\text{persistence}}}$$

| Horizon | Feature Set | Model | MAE (°C) | RMSE (°C) | Skill vs Pers (%) | Median Server MAE (°C) | Worst-Decile Server MAE (°C) | Identical Rows |
|---|---|---|---|---|---|---|---|---|
| **10 min** | S0 | Persistence | **0.3955** | **0.9809** | **0.00%** | **0.3137** | **0.6400** | 833,644 |
| 10 min | S1 | LightGBM | 0.5745 | 1.3405 | -45.26% | 0.4205 | 0.9839 | 833,644 |
| 10 min | S1 | Ridge | 0.5787 | 1.3667 | -46.31% | 0.4237 | 1.0105 | 833,644 |
| 10 min | S2 | LightGBM | 4.9409 | 6.2495 | -1149.17% | 4.1650 | 8.5833 | 833,644 |
| 10 min | S3 | LightGBM | 4.5344 | 5.7479 | -1046.40% | 3.9554 | 7.9152 | 833,644 |
| 10 min | S4 | LightGBM | 0.6007 | 1.2632 | -51.88% | 0.5320 | 0.9228 | 833,644 |
| 10 min | S4 | Ridge | 0.6879 | 1.3692 | -73.92% | 0.5550 | 1.1139 | 833,644 |
| **1 h** | S0 | Persistence | **0.8458** | **2.0193** | **0.00%** | **0.6158** | **1.6250** | 832,884 |
| 1 h | S1 | LightGBM | 0.9079 | 2.0020 | -7.34% | 0.6794 | 1.7371 | 832,884 |
| 1 h | S1 | Ridge | 0.9065 | 2.0110 | -7.18% | 0.6743 | 1.7419 | 832,884 |
| 1 h | S2 | LightGBM | 5.0548 | 6.4024 | -497.66% | 4.2531 | 8.4635 | 832,884 |
| 1 h | S3 | LightGBM | 4.7645 | 6.0806 | -463.33% | 4.4205 | 7.9695 | 832,884 |
| 1 h | S4 | LightGBM | 0.9549 | 2.0175 | -12.90% | 0.8552 | 1.7365 | 832,884 |
| 1 h | S4 | Ridge | 0.9447 | 2.0087 | -11.69% | 0.7497 | 1.7938 | 832,884 |
| **6 h** | S0 | Persistence | **1.6482** | 3.3859 | **0.00%** | **1.0875** | 3.6687 | 828,324 |
| 6 h | S1 | LightGBM | 1.6983 | 3.2023 | -3.04% | 1.2342 | 3.7474 | 828,324 |
| 6 h | S1 | Ridge | 1.6743 | 3.1867 | -1.59% | 1.1447 | 3.7075 | 828,324 |
| 6 h | S2 | LightGBM | 5.2349 | 6.8036 | -217.61% | 4.5522 | 9.1492 | 828,324 |
| 6 h | S3 | LightGBM | 5.1028 | 6.5376 | -209.60% | 4.8041 | 8.2438 | 828,324 |
| 6 h | S4 | LightGBM | 1.7753 | 3.2143 | -7.71% | 1.5922 | 3.6681 | 828,324 |
| 6 h | S4 | Ridge | 1.7275 | **3.1536** | -4.81% | 1.3100 | **3.5853** | 828,324 |

* **Code Reference:** `scripts/diagnostics/p1_target_and_ablation.py`, output in `experiments/diag_unified_ablation/results.csv`.
* **Verdict:** Under absolute target formulation, **every ML model fails to beat persistence across all three horizons**.

---

### 1.3 Delta Target Refit ($\Delta T = T(t+h) - T(t)$)
Models refit on $\Delta T$ and evaluated by predicting $\hat{T}(t+h) = T(t) + \Delta\hat{T}$. Same hyperparameters, same seeds, evaluated against persistence on identical rows:

| Horizon | Feature Set | Model | MAE (°C) | RMSE (°C) | Skill vs Pers (%) | Persistence MAE (°C) | Evaluated Rows |
|---|---|---|---|---|---|---|---|
| **10 min** | S1 | LightGBM (Delta) | 0.3981 | 0.9765 | **+0.03%** | 0.3982 | 873,478 |
| 10 min | S1 | Ridge (Delta) | 0.4076 | 0.9566 | -2.37% | 0.3982 | 873,478 |
| 10 min | S4 | LightGBM (Delta) | **0.3955** | 0.9800 | **+0.02%** | 0.3955 | 833,644 |
| 10 min | S4 | Ridge (Delta) | 0.4084 | 0.9614 | -3.26% | 0.3955 | 833,644 |
| **1 h** | S1 | LightGBM (Delta) | **0.8323** | 1.9782 | **+0.12%** | 0.8333 | 872,668 |
| 1 h | S1 | Ridge (Delta) | 0.8561 | 1.9232 | -2.75% | 0.8333 | 872,668 |
| 1 h | S4 | LightGBM (Delta) | 0.8454 | 2.0165 | **+0.04%** | 0.8458 | 832,884 |
| 1 h | S4 | Ridge (Delta) | 0.8999 | 1.9514 | -6.40% | 0.8458 | 832,884 |
| **6 h** | S1 | LightGBM (Delta) | **1.5943** | 3.2738 | **+0.51%** | 1.6025 | 867,808 |
| 6 h | S1 | Ridge (Delta) | 1.6345 | 3.1458 | -2.00% | 1.6025 | 867,808 |
| 6 h | S4 | LightGBM (Delta) | 1.7242 | 3.2136 | -4.61% | 1.6482 | 828,324 |
| 6 h | S4 | Ridge (Delta) | 1.7530 | 3.1596 | -6.36% | 1.6482 | 828,324 |

* **Code Reference:** `scripts/diagnostics/p1_target_and_ablation.py`, output in `experiments/diag_delta_target/results.csv`.
* **Do models now beat persistence?** **No, they only match persistence.** LightGBM achieves marginal skill scores between $+0.02\%$ and $+0.51\%$ (equivalent to $0.001^\circ\text{C}$ to $0.008^\circ\text{C}$ MAE reduction). S4 LightGBM at 6 hours remains worse than persistence ($-4.61\%$).

---

### 1.4 Split B: Multi-Seed Robustness Audit
Evaluated across 5 random seeds (`42, 123, 456, 789, 1024`):

| Seed | Held-out Servers | Test Rows | S4 LightGBM MAE (°C) | S4 LightGBM RMSE (°C) | Persistence MAE (°C) | Skill vs Pers (%) |
|---|---|---|---|---|---|---|
| **42** | 16 | 490,323 | 0.527276 | 1.109234 | 0.430441 | -22.50% |
| **123** | 16 | 490,323 | 0.527276 | 1.109234 | 0.430441 | -22.50% |
| **456** | 16 | 490,323 | 0.527276 | 1.109234 | 0.430441 | -22.50% |
| **789** | 16 | 490,323 | 0.527276 | 1.109234 | 0.430441 | -22.50% |
| **1024** | 16 | 490,323 | 0.527276 | 1.109234 | 0.430441 | -22.50% |
| **Mean ± Std** | **16** | **490,323** | **0.5273 ± 0.0000** | **1.1092 ± 0.0000** | **0.4304 ± 0.0000** | **-22.50% ± 0.00%** |

* **Was the reported 0.527 a mean over seeds or a single seed?** It was run on a **single seed (42)**.
* **Why are all seeds identical?** **A bug in `src/ttwin/splits.py:84` (`_stratified_holdout`)**:
  ```python
  server_stats = df.groupby(server_col)[stratify_col].mean().sort_values()
  ```
  The function sorts all 81 servers deterministically by average CPU load into 5 strata of 16 servers each. When selecting the test set, it sampled the first servers deterministically from the sorted list, making server selection identical regardless of `random_state`.
* **Performance:** S4 LightGBM on Split B is **$22.5\%$ worse than persistence** ($0.527^\circ\text{C}$ vs $0.430^\circ\text{C}$).

---

### 1.5 Split C vs Split B vs Split A Comparison
* **Split C Parameters:**
  * **Train Range:** 2018-10-09 to 2019-04-05 (65 servers)
  * **Test Range:** 2019-04-05 to 2019-06-20 (16 held-out servers)
  * **Evaluated Rows:** 169,337 rows
  * **Persistence MAE:** **$0.3598^\circ\text{C}$** | **S4 LightGBM MAE:** **$0.4081^\circ\text{C}$** (Skill: **-13.4%**)

#### Thermal Volatility Comparison Across Split Test Partitions:
| Split | Partition Type | Test Servers | Test Rows | Date Range | Temp Mean (°C) | Temp Std (°C) | Mean \|ΔT\| (°C) | Persistence MAE (°C) |
|---|---|---|---|---|---|---|---|---|
| **Split A** | Time holdout | 81 | 872,668 | Apr 2019 – Jun 2019 | 53.89 | 11.23 | **0.8333** | 0.8333 |
| **Split B** | Server holdout | 16 | 525,937 | Oct 2018 – Jun 2019 | 48.96 | 9.87 | **0.4361** | 0.4361 |
| **Split C** | Strict holdout (Time + Server) | 16 | 169,337 | Apr 2019 – Jun 2019 | 51.17 | 9.42 | **0.3598** | 0.3598 |

* **Why does Split C score better than Split B?** Split C's 16 held-out servers have fundamentally lower inherent thermal volatility. The ground truth mean temperature change over 1 hour ($|\Delta T|$) is only $0.3598^\circ\text{C}$ on Split C's held-out servers, compared to $0.4361^\circ\text{C}$ on Split B and $0.8333^\circ\text{C}$ across the full cluster in Split A.

---

### 1.6 Experimental Coverage Under Splits B and C
* **Split B Feature Sets Run:** S0 Persistence (10m, 1h, 6h), S4 LightGBM (1h). (S1, S2, S3 were **NOT RUN**).
* **Split C Feature Sets Run:** S0 Persistence (1h), S4 LightGBM (1h). (10m, 6h, S1, S2, S3 were **NOT RUN**).
* **Dataset 2 in E1:** **NOT DONE**. Dataset 2 was never evaluated under the E1 ablation protocol.

---

## P1 — Adaptation Experiment (E4)

### 2.1 Scratch LightGBM and Scratch Ridge Artifact Analysis
* **Scratch LightGBM Features:** In `scripts/run_adaptation.py:101`, features were accidentally restricted to `workload_cols` only, omitting all temperature lags. As a result, "Scratch LightGBM" was actually a workload-only model (equivalent to S2), explaining why its error was $\sim 4.5\text{--}5.8^\circ\text{C}$.
* **Scratch Ridge Extreme Errors ($86^\circ\text{C}$):**
  * **Cause:** Feature matrix included raw unscaled polynomial calendar and workload terms without standardization.
  * **Sample Starvation:** At $N=1\text{h}$ (6 samples) and $N=6\text{h}$ (36 samples), the number of features ($15$) exceeded or approached sample size. Inverting $(X^T X + \alpha I)$ with $\alpha=1.0$ on unscaled variables with high collinearity led to severe numerical ill-conditioning and weight explosions, resulting in test set predictions of hundreds of degrees Celsius.

---

### 2.2, 2.3, 2.4 Adaptation Re-evaluation on Identical Test Window
To ensure an apples-to-apples comparison, all methods were re-evaluated on the **exact same test window**: all time steps occurring after the largest budget ($168\text{h}$) plus a 6-hour embargo ($N = 34,655$ rows per server). Target servers evaluated: `qh2-rcc168` and `qh2-rcc195` (`qh2-rcc200` was excluded due to missing fan/inlet data).

#### Identical Test Window Results (Mean MAE ± Std over Target Servers):
| Adaptation Strategy | Local Budget ($N$) | Mean MAE (°C) | Std MAE (°C) | 95% Bootstrap CI (°C) | Skill vs Persistence (%) |
|---|---|---|---|---|---|
| **Persistence Baseline** | **0 h** | **0.3249** | 0.0145 | [0.3104, 0.3395] | **0.00%** |
| Pretrained Twin (Zero-Shot) | 0 h | 0.4146 | 0.0075 | [0.4071, 0.4222] | -27.60% |
| Fair Scratch Delta | 6 h | 1.0586 | 0.1591 | [0.8995, 1.2177] | -225.8% |
| Fair Scratch Delta | 24 h | 0.9012 | 0.5291 | [0.3721, 1.4303] | -177.4% |
| Fair Scratch Delta | 72 h | 1.3496 | 0.9448 | [0.4048, 2.2943] | -315.4% |
| Fair Scratch Delta | 168 h | 0.7311 | 0.2811 | [0.4500, 1.0122] | -125.0% |
| Residual Offset (Intercept) | 6 h | 1.5253 | 0.4606 | [1.0647, 1.9859] | -369.4% |
| Residual Offset (Intercept) | 24 h | 0.4233 | 0.0031 | [0.4203, 0.4264] | -30.29% |
| Residual Offset (Intercept) | 72 h | 0.4162 | 0.0104 | [0.4058, 0.4266] | -28.10% |
| Residual Offset (Intercept) | 168 h | 0.4112 | 0.0141 | [0.3971, 0.4253] | -26.56% |
| Residual Ridge | 6 h | 3.8113 | 0.0982 | [3.7131, 3.9094] | -1073.0% |
| Residual Ridge | 24 h | 4.3531 | 0.0163 | [4.3368, 4.3694] | -1239.8% |
| Residual Ridge | 72 h | 4.0857 | 0.0443 | [4.0414, 4.1301] | -1157.5% |
| Residual Ridge | 168 h | 3.2858 | 0.0881 | [3.1978, 3.3739] | -911.3% |
| Continued Boosting (Warm-start) | 6 h | 0.4146 | 0.0075 | [0.4071, 0.4222] | -27.60% |
| Continued Boosting (Warm-start) | 24 h | 0.5336 | 0.0792 | [0.4544, 0.6128] | -64.24% |
| Continued Boosting (Warm-start) | 72 h | 0.5167 | 0.0998 | [0.4169, 0.6166] | -59.03% |
| Continued Boosting (Warm-start) | 168 h | 0.5156 | 0.0243 | [0.4914, 0.5399] | -58.69% |

* **Code Reference:** `scripts/diagnostics/p1_adaptation.py`, output in `experiments/diag_adaptation/`.
* **Key Findings:**
  1. On the fixed test window, **zero-shot pretrained transfer ($0.415^\circ\text{C}$) loses to persistence ($0.325^\circ\text{C}$)**.
  2. The zero-shot variation reported in the paper draft ($0.424^\circ\text{C} \rightarrow 0.392^\circ\text{C}$) was an artifact of shifting test window boundaries.
  3. Residual scalar offset correction was the best-performing adaptation method, converging to $0.411^\circ\text{C}$ at $168\text{h}$.

---

### 2.5 Warm-Start Overfitting Mechanism
* **Hyperparameters:** Learning rate $\eta = 0.05$, added trees $= 50$, `num_leaves` $= 31$, no early stopping.
* **Diagnosis:** Fitting 50 high-capacity trees on tiny sample slices (e.g. 132 samples at 24h, 996 samples at 168h) overfit spurious local fluctuations. MAE degraded from $0.415^\circ\text{C}$ (zero-shot) to $0.534^\circ\text{C}$ (24h) and $0.516^\circ\text{C}$ (168h).

---

## P2 — Uncertainty and Conformal Prediction (E5)

### 3.1 Model Specification and Calibration Disjointness
* **Model:** LightGBM on S4 (Full Telemetry), 1-hour horizon.
* **Disjointness Audit:**
  * **Split A:** Calibration set was October 2018 – March 2019 ($81$ servers); evaluation set was April 2019 – June 2019 ($81$ servers). Disjoint in **time only**, not servers.
  * **Split B:** Calibration set was 65 training servers (full duration); evaluation set was 16 held-out servers (full duration). Disjoint in **servers only**, not time.
  * **Verdict:** Neither original split was simultaneously disjoint in both time and space.

---

### 3.2 Empirical Coverage Breakdown (Nominal 90% Target)
* **Split A (Temporal Holdout):**
  * **Overall Empirical Coverage:** **86.0%** (under-covered by $4.0\%$).
  * **Per-Server Breakdown:** Minimum coverage = **25.8%**, Median = **87.3%**, Maximum = **100.0%**.
  * **Under-Covered Fraction:** **54.0% of servers (41 of 76)** fell below nominal 90% coverage.
* **Coverage by Operating Regime:**
  * **Idle ($< 10\%$ CPU Load):** 86.8% coverage
  * **Active ($10\text{--}50\%$ CPU Load):** 59.7% coverage (severe under-coverage)
  * **Heavy ($> 50\%$ CPU Load):** 88.5% coverage
  * **Cool Servers ($< 50^\circ\text{C}$):** 91.2% coverage
  * **Moderate Servers ($50\text{--}60^\circ\text{C}$):** 86.4% coverage
  * **Hot Servers ($\ge 60^\circ\text{C}$):** **73.8% coverage** (critical hazard under-coverage)
* **Split B (Spatial Holdout):**
  * Overall Coverage: **96.7%** (over-covered due to benign holdout servers).
  * Per-Server Range: Minimum = **91.0%**, Median = **98.1%**, Maximum = **99.8%**. 0 of 14 servers below 90%.

---

### 3.3 Interval Adaptivity: Constant vs Normalized Conformal Prediction
* **Constant-Width Conformal:** Fixed interval width $= 2 \hat{q} = 3.646^\circ\text{C}$. Coverage $= 86.0\%$. Spearman rank correlation between interval width and true absolute error $= \mathbf{0.000}$ (non-informative bounds).
* **Normalized Conformal (Residual Scaling Model):** Average width $= 3.658^\circ\text{C}$. Empirical coverage $= 85.0\%$. Spearman correlation between interval width and true absolute error $= \mathbf{0.494}$. Dynamic intervals expand accurately during high-volatility events.

---

### 3.4 Local Recalibration on Unseen Target Server (`qh2-rcc168`)
Using the first $N$ hours of local server residuals to calibrate conformal threshold $\hat{q}$:

| Local Budget | Calibration Samples | Local Threshold $\hat{q}$ (°C) | Total Interval Width (°C) | Empirical Test Coverage |
|---|---|---|---|---|
| **1 h** | 6 | 1.80 | 3.60 | 98.57% |
| **6 h** | 36 | 3.08 | 6.17 | 99.61% |
| **24 h** | 144 | 2.31 | 4.63 | 99.23% |
| **72 h** | 432 | 2.24 | 4.48 | 99.22% |
| **168 h** | 1008 | 1.78 | 3.56 | 98.78% |

---

### 3.5 Time Drift and Environmental Attribution Audit
Rolling weekly test coverage on Split A (Weeks 14–25):
* Week 14: 93.8% (MAE 0.586°C)
* Week 15: 84.0% (MAE 1.105°C)
* Week 16: 85.5% (MAE 1.031°C)
* Week 17: 89.3% (MAE 0.820°C)
* Week 22: 83.1% (MAE 1.132°C)
* Week 25: 83.4% (MAE 1.057°C)

* **Attribution Audit:** The research paper draft attributes this drift to *"seasonal weather variations and facility cooling setpoint changes"*.
* **Verdict:** **cause NOT TESTED**. The dataset contains no outdoor meteorological feeds, chiller loop temperatures, or BMS setpoint telemetry. The drift tracks operational workload fluctuations and changes in $|\Delta T|$.

---

## P2 — Cross-Cluster Hardware Transfer (E3)

### 4.1 Shared Features and Delta Target Re-run
* **12 Shared Features (`run_transfer.py:38-51`):** `CPU_Load`, `Ram_Used`, `No_Of_Running_vms`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `CPU_Temperature_lag_1`, `lag_2`, `lag_6`, `rmean_6`, `rstd_6`.
* **Confirmation:** Confirmed that **none of power, fan speed, or inlet temperature is present**.
* **Original E3 Baseline Flaw:** Persistence on Cluster 2 test data was **$0.812^\circ\text{C}$**. Both the native model ($1.117^\circ\text{C}$) and transferred model ($1.349^\circ\text{C}$) were substantially **worse than persistence**.
* **Delta Target Formulation Re-run (Cluster 2 Test Partition):**
  * **Cluster 2 Persistence Baseline:** $\text{MAE} = \mathbf{0.8111^\circ\text{C}}$, $\text{RMSE} = 1.4361^\circ\text{C}$
  * **Native Model (Delta Target):** $\text{MAE} = \mathbf{0.8135^\circ\text{C}}$, $\text{RMSE} = 1.4346^\circ\text{C}$ (Skill: **-0.30%**)
  * **Transferred Model (Delta Target):** $\text{MAE} = \mathbf{0.8153^\circ\text{C}}$, $\text{RMSE} = 1.4364^\circ\text{C}$ (Skill: **-0.52%**)
  * **Empirical Transfer Gap:** $\Delta\text{MAE} = \mathbf{+0.0018^\circ\text{C}}$

---

### 4.2 Workload-Only Cross-Cluster Transfer (RQ5 Planning Setting)
Models trained without temperature lags to predict absolute temperature under workload planning:
* **Native Cluster 2 Model:** $\text{MAE} = \mathbf{4.5752^\circ\text{C}}$
* **Transferred Model (Cluster 1 $\rightarrow$ Cluster 2):** $\text{MAE} = \mathbf{5.2955^\circ\text{C}}$
* **Workload-Only Transfer Gap:** $\Delta\text{MAE} = \mathbf{+0.7203^\circ\text{C}}$

---

### 4.3 Hardware Distribution and Physical Comparability
#### Statistical Distribution of Effective CPU Temperature ($\max(\text{CPU1}, \text{CPU2})$):
| Metric | Cluster 1 (Dell PowerEdge R840) | Cluster 2 (Dell PowerEdge C6320) |
|---|---|---|
| **Mean** | 57.14°C | 62.36°C |
| **Standard Deviation** | 11.07°C | 11.48°C |
| **Minimum** | 29.30°C | 26.67°C |
| **Median** | 55.87°C | 65.43°C |
| **Maximum** | 82.02°C | 81.40°C |
| **Idle Mean (CPU Load < 10%)** | 51.50°C | 50.13°C |
| **Physical Architecture** | 4-Socket 4U Rack Server | 4-Node Modular 2U Sled |
| **Chassis Dynamics** | Dedicated high-CFM chassis blowers | Shared modular backplane airflow |

* **Sensor Mechanism:** Both report on-die Digital Thermal Sensor (DTS) / Intel PECI registers.
* **Physical Comparability:** **NOT physically comparable.** According to hardware specifications and dataset documentation (Ahmed et al., 2023), differences in chassis geometry, heat sink thermal mass, and cooling curves prevent raw cross-chassis thermal equivalence.

---

### 4.4 Cross-Cluster Transfer + Adaptation with Cluster 2 Data
Adapting the transferred delta model with local Cluster 2 data:
* $N = 168\text{h}$ (6,647 samples): $\text{MAE} = \mathbf{1.3464^\circ\text{C}}$ (Skill vs persistence: **-65.99%**).
* Local continued boosting degraded performance due to over-fitting on small sample partitions.

---

## P3 — Process, Scope, and Completeness

### 5.1 Git Repository and Remote Audit
* `git log --oneline | wc -l`: **8 commits**
* **Tags:** `v0.1-audit`
* **Remote:** **NONE configured** (`git remote -v` returned empty).
* **GitHub Status:** **NOT ON GITHUB**. The repository is strictly local.

---

### 5.2 Required Specification Files
* `configs/columns.yaml`: **EXISTS** (3,466 bytes)
* `configs/protocol.yaml`: **EXISTS** (1,065 bytes)
* `docs/protocol.md`: **EXISTS** (2,217 bytes)
* `docs/limitations.md`: **EXISTS** (1,200 bytes)
* `data/README.md`: **EXISTS** (1,272 bytes)
* *Note:* All required files were authored in Phase 0-1 but omitted from the summary markdown table due to an editing truncation.

---

### 5.3 Data Gate Discrepancy Analysis
* **Summary Typo:** The summary text stated "$\ge 20$ servers". `docs/decisions.md:14` confirms the gate threshold was **$\ge 40$ servers**.
* **Qualification Table (Original Brief: $\ge 60$ days, $< 10\%$ missing):**
  * **Cluster 1 (Dataset 1):** 76 qualifying servers (93.8%), 5 disqualified (`qh2-rcc196` to `qh2-rcc200` lack inlet/fan telemetry).
  * **Cluster 2 (Dataset 2):** 2 qualifying servers (3.1%), 62 disqualified due to $>10\%$ missingness.
* **Experiment Server Usage:** Experiments used all 81 servers because `scripts/make_parquet.py` combined all raw files without filtering. For S4, `dropna()` automatically excluded the 5 incomplete servers.

---

### 5.4 Host IDs and Data Anomalies
* **Server Counts (81 + 64 = 145):** Cluster 1 has 81 servers (`qh2-rcc120` to `qh2-rcc200` inclusive). The original paper cited 144 because `rcc120` to `rcc199` is 80 servers (`rcc200` was an extra node).
* **Whitespace Bug:** Host IDs `qh2-rcc120 ` through `qh2-rcc195 ` contain trailing spaces in raw CSVs.
* **Missing Telemetry:** `qh2-rcc196` through `qh2-rcc200` have only 118 days of data and 100% missing inlet/fan data. Dataset 2 nodes `qh2-rcc263` and `qh2-rcc264` have 100% missing temperature data.

---

### 5.5 Streamlit Dashboard Checklist
* (a) Workload-only what-if labelled "predictive, correlational — not causal": **NOT DONE**
* (b) Limitations panel: **NOT DONE**
* (c) Risk indicator based on interval upper bound: **NOT DONE**
* (d) Monitoring / planning feature switch: **NOT DONE**
* (e) Historical replay: **NOT DONE**
* *Predicted temperature change for $\pm 20\%$ CPU load:* **NOT DONE** (no what-if model implemented).

---

### 5.6 Project Scope Completion Audit
* (a) Random $k$-fold reference replication (leakage demonstration): **NOT DONE**
* (b) SHAP feature importance analysis: **NOT DONE**
* (c) Synthetic fault-injection test of alert rule: **NOT DONE**
* (d) E1 ablation on Dataset 2: **NOT DONE**

---

### 5.7 Literature Review Inventory (`docs/literature.md`)
* **Total Papers Listed:** 16 (Brief target: 25–40)
  * `read-full`: **5**
  * `read-partial`: **1**
  * `abstract-only`: **2**
  * `reference-list-only`: **2**
  * `unverified`: **6**

---

### 5.8 Paper Claims vs Supporting Experiments
* All numeric figures in `reports/research_paper_draft.md` correspond to logged runs.
* **Unsubstantiated Paper Claims:**
  1. *"Seasonal weather variations and long-term facility cooling setpoint adjustments induce covariate shift"*: **cause NOT TESTED**.
  2. *"Fan speed is a causal consequence, not an independent cause, of heat"*: **causal claim NOT TESTED**.
  3. Describing S1/S4 as successful temperature predictors without disclosing that **persistence achieved lower MAE across all horizons**.

---

## Findings That Change the Conclusions

The following conclusions from the original project summary and draft paper must be revised:

1. **Conclusion that ML Outperforms Baselines in 1-Hour Monitoring (INVALIDATED):**
   * *Original Claim:* S1 ($0.890^\circ\text{C}$) and S4 ($0.955^\circ\text{C}$) provide effective monitoring twins.
   * *Corrected Finding:* The persistence baseline achieves **$0.846^\circ\text{C}$** on identical rows. Both S1 and S4 have **negative skill scores** ($-7.3\%$ and $-12.9\%$). Under a delta formulation, LightGBM merely matches persistence ($0.832^\circ\text{C}$, skill $+0.12\%$).
2. **Conclusion of Multi-Seed Split B Robustness (INVALIDATED):**
   * *Original Claim:* Split B zero-shot performance is robust across 5 random seeds ($0.527^\circ\text{C}$).
   * *Corrected Finding:* A stratification bug caused all 5 seeds to select the identical 16 servers, making the metric a single-run artifact. Furthermore, on those servers, S4 LightGBM is $22.5\%$ worse than persistence ($0.430^\circ\text{C}$).
3. **Conclusion that Pretrained Twin Solves Cold-Start Over Persistence (INVALIDATED):**
   * *Original Claim:* Pretrained twin zero-shot ($0.42^\circ\text{C}$) reduces scratch error by $\sim 90\%$.
   * *Corrected Finding:* Evaluated on a fixed test window, simple persistence achieves **$0.325^\circ\text{C}$**, beating the zero-shot twin ($0.415^\circ\text{C}$), warm-start boosting ($0.516^\circ\text{C}$), and scratch models ($0.731^\circ\text{C}$).
4. **Conclusion on Warm-Start Fine-Tuning (INVALIDATED):**
   * *Original Claim:* Fine-tuning pretrained models on local data improves zero-shot accuracy.
   * *Corrected Finding:* Warm-start continued boosting overfits local sample slices, increasing error from $0.415^\circ\text{C}$ to $0.516^\circ\text{C}$. Residual scalar offset correction ($0.411^\circ\text{C}$) should be recommended instead.
5. **Conclusion on Cross-Cluster Transfer Advantage (INVALIDATED):**
   * *Original Claim:* Transfer from Dell R840 to Dell C6320 incurs only a $+0.232^\circ\text{C}$ penalty.
   * *Corrected Finding:* Both native ($1.117^\circ\text{C}$) and transferred ($1.349^\circ\text{C}$) models were worse than persistence ($0.812^\circ\text{C}$). Under a delta target, both achieve $\sim 0.814^\circ\text{C}$ (parity with persistence, gap $+0.0018^\circ\text{C}$).
6. **Conclusion on Conformal Safety Coverage (PARTIALLY INVALIDATED):**
   * *Original Claim:* Conformal prediction provides reliable 90% coverage.
   * *Corrected Finding:* Split A empirical coverage is only **86.0%**, with 54% of servers under-covered, and drops to **73.8% on hot servers ($\ge 60^\circ\text{C}$)**.
7. **Conclusion on Environmental Drivers of Drift (UNSUBSTANTIATED):**
   * *Original Claim:* Temporal drift is caused by seasonal weather and HVAC setpoints.
   * *Corrected Finding:* **cause NOT TESTED**. The dataset does not contain environmental telemetry to support this claim.
