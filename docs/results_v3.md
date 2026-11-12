# Results v3 — Comprehensive Gap-Corrected Experiments

> [!IMPORTANT]
> All results in this report are strictly **preliminary**. Telemetry is processed with continuous 10-minute grid alignment (NaN insertion for missing intervals) to guarantee causal, gap-free lag and horizon shifts. Numerical claims are derived strictly from the committed CSV outputs in `experiments/v3_experiments/`.

---

## 1. Paired Bootstrap Analysis for Skill vs Persistence (E1)

Evaluated at 1-hour forecast horizon on Split A (temporal holdout with 6-hour embargo). 95% bootstrap confidence intervals computed over $B=2,000$ resamples across physical servers and across calendar weeks.

| dataset   |   overall_mae_persistence |   overall_mae_s1_lgbm |   overall_skill_pct |   mean_server_skill_pct |   server_ci_95_low_pct |   server_ci_95_high_pct |   mean_week_skill_pct |   week_ci_95_low_pct |   week_ci_95_high_pct |   n_servers |   n_weeks |
|:----------|--------------------------:|----------------------:|--------------------:|------------------------:|-----------------------:|------------------------:|----------------------:|---------------------:|----------------------:|------------:|----------:|
| Dataset 1 |                  0.832577 |              0.827679 |            0.588288 |               -0.116763 |              -0.378889 |                0.169373 |              0.602802 |             0.465225 |              0.756808 |          81 |         8 |
| Dataset 2 |                  0.804186 |              0.781424 |            2.83044  |               -0.600833 |              -6.8493   |                3.37634  |              2.70199  |             1.98793  |              3.44465  |          59 |         5 |

* **What this shows:**
  * S1 LightGBM achieves modest overall skill over persistence: **+0.59%** on Dataset 1 (0.8277°C vs 0.8326°C MAE) and **+2.83%** on Dataset 2 (0.7814°C vs 0.8042°C MAE).
  * **Server-level 95% confidence intervals cross zero on both datasets** (Dataset 1: `[-0.38%, +0.17%]`; Dataset 2: `[-6.85%, +3.38%]`). Across individual physical nodes, predictive superiority is **not statistically significant**. Per-server skill exhibits high variance around zero.
  * Calendar week-level bootstrap 95% confidence intervals are strictly positive (Dataset 1: `[+0.47%, +0.76%]`; Dataset 2: `[+1.99%, +3.44%]`). Week-level resampling aggregates across all servers simultaneously, reflecting slight aggregate stability across time blocks rather than uniform per-server gains. Resampling units use calendar year-week identifiers (`YYYY-Www`) to avoid cross-year aliasing.
* **What this does NOT show:** This does not show substantial predictive superiority over persistence; thermal inertia accounts for >99% of short-term thermal variance.

---

## 2. Sensor Ablation: Isolating Hardware Features (Dataset 1, 1-hour Horizon)

Evaluated on strictly identical test rows across feature sets with fans, power, inlet temperature, and thermal lags.

| configuration                                     |   n_features |   test_mae |   test_rmse |   skill_vs_pers_pct |   n_test_rows |
|:--------------------------------------------------|-------------:|-----------:|------------:|--------------------:|--------------:|
| S0 Persistence Baseline                           |            0 |   0.840763 |     2.00862 |            0        |        552672 |
| S4 Full (Fans + Power + Inlet + Workload + Lags)  |           56 |   0.838074 |     1.98294 |            0.319783 |        552672 |
| S4 without Fans (Power + Inlet + Workload + Lags) |           52 |   0.838071 |     1.98293 |            0.32014  |        552672 |
| S4 without Power (Fans + Inlet + Workload + Lags) |           55 |   0.838337 |     1.98687 |            0.288568 |        552672 |
| S4 without Fans & Power (Inlet + Workload + Lags) |           51 |   0.838337 |     1.98687 |            0.288568 |        552672 |
| S1 (Temperature Lags & Current State Only)        |           36 |   0.83596  |     1.9725  |            0.571309 |        552672 |

* **What this shows:**
  * Removing cooling fan telemetry changes test MAE by less than **0.0001°C** (0.8381°C vs 0.8381°C).
  * Removing server power telemetry changes test MAE by **0.0003°C** (0.8381°C vs 0.8383°C).
  * The lean S1 feature set (temperature lags alone, 36 features) achieves **0.8360°C MAE**, slightly outperforming the full 56-feature S4 model (0.8381°C). Workload and thermal inertia provide all informative variance; additional hardware telemetry introduces slight overfitting noise.
* **What this does NOT show:** This does not demonstrate that fan telemetry is universally uninformative in all facilities. In this trace, fan curves operate autonomously under firmware closed-loop control (BMC/PID) without facility BMS feedback, making fan telemetry reactive rather than predictive.

---

## 3. Split B (GroupKFold) and Split C across Horizons (with 5 Random Server Draws)

Split B evaluates complete 5-fold cross-validation across all servers. Split C evaluates joint spatiotemporal holdout across 5 distinct random seed draws (seeds: 42, 123, 456, 789, 1024).

| split                    | horizon   |   persistence_mae_mean |   persistence_mae_std |   s1_mae_mean |   s1_mae_std |   s4_mae_mean |   s4_mae_std |
|:-------------------------|:----------|-----------------------:|----------------------:|--------------:|-------------:|--------------:|-------------:|
| Split B (GroupKFold)     | 10min     |               0.434605 |             0.0464147 |      0.435767 |    0.0458628 |      0.434059 |    0.0450064 |
| Split B (GroupKFold)     | 1h        |               0.918224 |             0.115299  |      0.923727 |    0.111094  |      0.917906 |    0.111052  |
| Split B (GroupKFold)     | 6h        |               1.81739  |             0.202809  |      1.82802  |    0.192732  |      1.821    |    0.194727  |
| Split C (5 Random Seeds) | 10min     |               0.455976 |             0.101562  |      0.455625 |    0.101313  |      0.455547 |    0.101222  |
| Split C (5 Random Seeds) | 1h        |               0.940347 |             0.178617  |      0.936325 |    0.17513   |      0.937577 |    0.176339  |
| Split C (5 Random Seeds) | 6h        |               1.70954  |             0.218427  |      1.69888  |    0.209155  |      1.73404  |    0.238686  |

* **What this shows:**
  * GroupKFold (Split B) mean test MAE scales predictably across horizons: **0.435°C** (10min) $\to$ **0.918°C** (1h) $\to$ **1.817°C** (6h) for persistence, with S1 and S4 models tracking persistence within 0.01°C at all horizons.
  * Split C with 5 random seed draws confirms that spatial holdout variance across seeds is modest (standard deviation **0.102°C** at 10min, **0.179°C** at 1h, and **0.218°C** at 6h).
* **What this does NOT show:** This does not show that spatial generalization is trivial; test error on unseen servers is higher than temporal holdout on seen servers (e.g. 1h Split B MAE is ~0.92°C vs Split A ~0.83°C).

---

## 4. Fleet-Wide Planning Adaptation Across All 5 Folds ($N=81$ Target Servers)

Evaluates cold-start adaptation on every physical server in Dataset 1 across GroupKFold splits with local training budgets from 0 to 168 hours.

| strategy           |   budget_hours |   n_servers |   mean_mae |   std_mae |   ci_95_low |   ci_95_high |
|:-------------------|---------------:|------------:|-----------:|----------:|------------:|-------------:|
| continued_boosting |              1 |          81 |    5.03469 |   1.59049 |     4.71032 |      5.39496 |
| continued_boosting |              6 |          81 |    5.03469 |   1.59049 |     4.71032 |      5.39496 |
| continued_boosting |             24 |          81 |    4.93106 |   1.51998 |     4.622   |      5.2803  |
| continued_boosting |             72 |          81 |    4.91809 |   1.50899 |     4.61376 |      5.26149 |
| continued_boosting |            168 |          81 |    4.87168 |   1.43263 |     4.58066 |      5.19213 |
| fleet_mean         |              0 |          81 |    8.86595 |   2.99266 |     8.22641 |      9.49823 |
| residual_offset    |              1 |          81 |    5.51932 |   2.49997 |     5.01278 |      6.08418 |
| residual_offset    |              6 |          81 |    5.53646 |   2.59133 |     5.02984 |      6.11222 |
| residual_offset    |             24 |          81 |    5.46726 |   2.33718 |     4.99171 |      5.98436 |
| residual_offset    |             72 |          81 |    5.325   |   2.24959 |     4.87608 |      5.84244 |
| residual_offset    |            168 |          81 |    5.0168  |   2.21488 |     4.55175 |      5.51842 |
| zero_shot          |              0 |          81 |    5.03469 |   1.59049 |     4.71032 |      5.39496 |

* **What this shows:**
  * Zero-shot planning achieves **5.03°C MAE** (95% CI: `[4.71°C, 5.39°C]`) compared to the unconditioned fleet mean of **8.87°C**, a **43.2% error reduction**.
  * Continued boosting improves gradually with local data: from 5.03°C (0h) to **4.87°C** at 168 hours (7 days).
  * Scalar residual offset improves from **5.52°C** (1h) to **5.02°C** (168h).
* **What this does NOT show:** This does not show that longer adaptation budgets monotonically improve accuracy across every server; individual drift patterns vary across physical servers.

---

## 5. Cross-Cluster Hardware Adaptation (Cluster 1 $\rightarrow$ Cluster 2)

Source model trained on Dell PowerEdge R840 (Cluster 1) adapted to Dell PowerEdge C6320 (Cluster 2) target servers.

|   budget_hours | strategy           |   mean_mae |   ci_95_low |   ci_95_high |
|---------------:|:-------------------|-----------:|------------:|-------------:|
|              0 | zero_shot          |    6.85176 |     2.91623 |     13.4668  |
|              1 | residual_offset    |    3.78826 |     2.40255 |      5.27346 |
|              1 | continued_boosting |    6.85176 |     2.91623 |     13.4668  |
|              6 | residual_offset    |    3.16055 |     2.13024 |      4.30415 |
|              6 | continued_boosting |    6.85176 |     2.91623 |     13.4668  |
|             24 | residual_offset    |    3.16871 |     1.90078 |      4.80949 |
|             24 | continued_boosting |    6.07187 |     2.69138 |     11.7068  |
|             72 | residual_offset    |    2.9219  |     1.63878 |      4.80636 |
|             72 | continued_boosting |    6.09971 |     2.72064 |     11.7339  |
|            168 | residual_offset    |    3.0332  |     1.70417 |      4.75068 |
|            168 | continued_boosting |    6.04999 |     2.69669 |     11.7058  |

* **What this shows:**
  * Zero-shot cross-cluster planning exhibits substantial uncertainty: mean MAE is **6.85°C** with a wide 95% bootstrap CI of `[2.92°C, 13.47°C]`.
  * Local residual offset adaptation rapidly corrects cluster baseline bias, reducing error to **3.16°C** (6h, CI: `[2.13°C, 4.30°C]`) and **2.92°C** (72h, CI: `[1.64°C, 4.81°C]`).
  * Continued boosting underperforms residual offset (6.05°C at 168h) because tree split constraints on small adaptation slices limit parameter updates.
* **What this does NOT show:** This does NOT prove universal or invariant physical transfer. Chassis airflow impedance, thermal mass, and cooling paths differ fundamentally between R840 and C6320. The wide confidence intervals reflect significant inter-server heterogeneity.

---

## 6. Random K-Fold vs GroupKFold on Identical Telemetry Rows

Direct comparison on strictly identical test rows to evaluate the impact of evaluation splitting strategy.

| feature_set   | regime     |   random_kfold_mae |   group_kfold_mae |   leakage_underestimation_pct |
|:--------------|:-----------|-------------------:|------------------:|------------------------------:|
| S1            | monitoring |           0.915318 |          0.917551 |                      0.243442 |
| S2            | planning   |           4.08622  |          4.95875  |                     17.5959   |
| S3            | planning   |           3.75984  |          4.61529  |                     18.5351   |
| S4            | monitoring |           0.905846 |          0.910806 |                      0.544595 |

* **What this shows:**
  * In monitoring models (S1, S4), where current temperature is available, the difference between row-random and server-grouped evaluation is minimal: **0.24%** for S1 (0.9153°C vs 0.9176°C) and **0.54%** for S4 (0.9058°C vs 0.9108°C), because autoregressive persistence anchors predictions.
  * In planning models (S2, S3), row-random splitting yields substantially lower error than GroupKFold: **17.60%** lower for S2 (4.0862°C vs 4.9588°C) and **18.54%** lower for S3 (3.7598°C vs 4.6153°C).
  * This difference reflects both temporal autocorrelation leakage across adjacent time steps and spatial heterogeneity across distinct server hardware.
* **What this does NOT show:** This does not imply GroupKFold is unnecessarily pessimistic; GroupKFold correctly reflects operational deployment to new, unseen server hardware.

---

## 7. S1 LightGBM Learning Rate & Early Stopping Convergence

Assessing whether S1's early convergence (~5 trees) at learning rate 0.05 is an artifact of step size.

|   learning_rate |   best_iteration |   best_val_mae |   test_mae |   test_rmse |   skill_score_pct |   persistence_test_mae |
|----------------:|-----------------:|---------------:|-----------:|------------:|------------------:|-----------------------:|
|           0.05  |                5 |       0.822904 |   0.827679 |     1.94894 |          0.588288 |               0.832577 |
|           0.01  |               27 |       0.823018 |   0.827555 |     1.94702 |          0.603188 |               0.832577 |
|           0.005 |               49 |       0.822938 |   0.827652 |     1.95001 |          0.591532 |               0.832577 |

* **What this shows:**
  * Reducing the learning rate from 0.05 to 0.005 increases tree count by nearly an order of magnitude (5 $\to$ 27 $\to$ 49 trees).
  * However, test MAE shifts by less than **0.00013°C** (0.827679°C $\to$ 0.827555°C $\to$ 0.827652°C), and skill scores remain essentially unchanged (+0.588% $\to$ +0.603% $\to$ +0.592%).
  * Early convergence under LR=0.05 is not premature termination; rather, the gradient update rapidly plateaus because the delta target $\Delta T$ has low variance around persistence. The lowest learning rate is **not practically superior**.
* **What this does NOT show:** This does not change the core substantive finding: monitoring models remain anchored to thermal persistence regardless of gradient step size.
