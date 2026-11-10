# 06 Experiment Manifest

This document describes all 37 experiment output folders found in `experiments/`, listed in chronological sequence of execution. It accompanies the machine-readable catalog `docs/snapshot/06_experiment_manifest.csv`.

---

## 1. Chronological Overview of Experiment Generations

The repository contains three distinct generations of experimental outputs:
1. **Legacy Phase 2 Runs (`experiments/dataset1_*`, `experiments/adapt_*`, `experiments/conformal_*`, `experiments/transfer_*`):**
   - Executed on 2026-11-08 (commits `b4b9378` to `3f86008`).
   - Evaluated absolute temperature targets ($T(t+h)$) without providing current-state $T(t)$ to the model.
   - Used single fixed holdout seeds and legacy chronological partitioning (60/10/30).
   - Preserved untouched in accordance with ground rules.
2. **Diagnostic Audit Phase (`experiments/diag_*`):**
   - Executed during the P1–P3 diagnostic audit (commits `3f86008` to `d8c2cbc`).
   - Investigated target formulation, identical-window adaptation, and cross-cluster domain shifts.
3. **Corrected v2 Pipeline Runs (`experiments/v2_*`):**
   - Executed on 2026-11-09 (commit `aa0e974` and tag `v1.0-corrected`).
   - Standardised under strict protocol: current-state $T(t)$ provided, delta target predicted ($\Delta T = T(t+h) - T(t)$), 70/10/20 Split A with 6h embargo, complete 5-fold GroupKFold by server, and CQR uncertainty quantification.

---

## 2. Manifest Table of All 37 Experiment Folders

| # | Experiment Folder (`exp_id`) | Generation | Script | Split | Horizon | Model | Target Type | Headline Finding |
|---|---|---|---|---|---|---|---|---|
| 1 | `dataset1_A_S0_persistence_10min_20261108_145332` | Legacy | `run_experiment.py` | Split A | 10min | Persistence | Current $T$ | MAE = 0.4026°C |
| 2 | `dataset1_A_S0_persistence_1h_20261108_145458` | Legacy | `run_experiment.py` | Split A | 1h | Persistence | Current $T$ | MAE = 0.8408°C |
| 3 | `dataset1_A_S0_persistence_6h_20261108_145544` | Legacy | `run_experiment.py` | Split A | 6h | Persistence | Current $T$ | MAE = 1.6025°C |
| 4 | `dataset1_A_S1_lightgbm_10min_20261108_150840` | Legacy | `run_experiment.py` | Split A | 10min | LightGBM | Absolute $T$ (flawed) | MAE = 0.4102°C (negative skill) |
| 5 | `dataset1_A_S1_lightgbm_1h_20261108_150840` | Legacy | `run_experiment.py` | Split A | 1h | LightGBM | Absolute $T$ (flawed) | MAE = 0.8524°C (negative skill) |
| 6 | `dataset1_A_S1_lightgbm_6h_20261108_150842` | Legacy | `run_experiment.py` | Split A | 6h | LightGBM | Absolute $T$ (flawed) | MAE = 1.6321°C |
| 7 | `dataset1_A_S2_lightgbm_10min_20261108_145336` | Legacy | `run_experiment.py` | Split A | 10min | LightGBM | Absolute $T$ | MAE = 5.0994°C |
| 8 | `dataset1_A_S2_lightgbm_1h_20261108_145337` | Legacy | `run_experiment.py` | Split A | 1h | LightGBM | Absolute $T$ | MAE = 5.2417°C |
| 9 | `dataset1_A_S2_lightgbm_6h_20261108_145338` | Legacy | `run_experiment.py` | Split A | 6h | LightGBM | Absolute $T$ | MAE = 5.4116°C |
| 10 | `dataset1_A_S3_lightgbm_10min_20261108_150844` | Legacy | `run_experiment.py` | Split A | 10min | LightGBM | Absolute $T$ | MAE = 4.7309°C |
| 11 | `dataset1_A_S3_lightgbm_1h_20261108_150844` | Legacy | `run_experiment.py` | Split A | 1h | LightGBM | Absolute $T$ | MAE = 4.7412°C |
| 12 | `dataset1_A_S3_lightgbm_6h_20261108_150845` | Legacy | `run_experiment.py` | Split A | 6h | LightGBM | Absolute $T$ | MAE = 4.9198°C |
| 13 | `dataset1_A_S4_lightgbm_10min_20261108_145817` | Legacy | `run_experiment.py` | Split A | 10min | LightGBM | Absolute $T$ (flawed) | MAE = 0.4089°C |
| 14 | `dataset1_A_S4_lightgbm_1h_20261108_145819` | Legacy | `run_experiment.py` | Split A | 1h | LightGBM | Absolute $T$ (flawed) | MAE = 0.8510°C |
| 15 | `dataset1_A_S4_lightgbm_6h_20261108_145820` | Legacy | `run_experiment.py` | Split A | 6h | LightGBM | Absolute $T$ (flawed) | MAE = 1.6210°C |
| 16 | `dataset1_B_S0_persistence_10min_20261108_165305` | Legacy | `run_experiment.py` | Split B (single seed) | 10min | Persistence | Current $T$ | MAE = 0.3812°C |
| 17 | `dataset1_B_S0_persistence_1h_20261108_165222` | Legacy | `run_experiment.py` | Split B (single seed) | 1h | Persistence | Current $T$ | MAE = 0.7951°C |
| 18 | `dataset1_B_S0_persistence_6h_20261108_165332` | Legacy | `run_experiment.py` | Split B (single seed) | 6h | Persistence | Current $T$ | MAE = 1.5124°C |
| 19 | `dataset1_B_S4_lightgbm_1h_20261108_174012` | Legacy | `run_experiment.py` | Split B (single seed) | 1h | LightGBM | Absolute $T$ (flawed) | MAE = 0.8112°C |
| 20 | `dataset1_C_S0_persistence_1h_20261108_165358` | Legacy | `run_experiment.py` | Split C (single seed) | 1h | Persistence | Current $T$ | MAE = 0.6200°C |
| 21 | `dataset1_C_S4_lightgbm_1h_20261108_174147` | Legacy | `run_experiment.py` | Split C (single seed) | 1h | LightGBM | Absolute $T$ (flawed) | MAE = 0.6350°C |
| 22 | `adapt_dataset1_S4_1h_20261108_170601` | Legacy | `run_adaptation.py` | Holdout | 1h | Fine-tune LGBM | Absolute $T$ | Overfitting on small data |
| 23 | `conformal_dataset1_A_S4_1h_20261108_165926` | Legacy | `run_conformal.py` | Split A | 1h | Split Conformal | Absolute $T$ | 89.8% coverage |
| 24 | `conformal_dataset1_B_S4_1h_20261108_170141` | Legacy | `run_conformal.py` | Split B | 1h | Split Conformal | Absolute $T$ | 88.5% coverage |
| 25 | `transfer_c1_to_c2_1h_20261108_170327` | Legacy | `run_transfer.py` | Cross-Cluster | 1h | Direct Transfer | Absolute $T$ | Severe offset shift (+5.4°C) |
| 26 | `diag_adaptation` | Diagnostic | `p1_adaptation.py` | Holdout | 1h | Offset vs Boosting | Absolute $T$ | Proved identical test windows required |
| 27 | `diag_delta_target` | Diagnostic | `p1_target_and_ablation.py` | Split A | 1h | Delta LGBM | Delta $\Delta T$ | Proved delta formulation recovers skill |
| 28 | `diag_process` | Diagnostic | `p3_process.py` | Split A | 1h | Verification | — | Git and integrity audit |
| 29 | `diag_split_b_5seeds` | Diagnostic | `p1_adaptation.py` | Split B (5 seeds) | 1h | Multi-seed | Absolute $T$ | Showed seed variance across holdouts |
| 30 | `diag_transfer` | Diagnostic | `p2_transfer.py` | Cross-Cluster | 1h | Scalar Offset | Absolute $T$ | Showed offset recovers 90% of transfer error |
| 31 | `diag_uncertainty` | Diagnostic | `p2_uncertainty.py` | Split C | 1h | CQR vs Split | Absolute $T$ | Under-coverage in high temperature bins |
| 32 | `diag_unified_ablation` | Diagnostic | `p1_target_and_ablation.py` | Split A | 10m, 1h, 6h | S0–S4 Ladder | Delta $\Delta T$ | Precursor to v2 unified table |
| 33 | `v2_ablation` | Current | `run_v2_e1_ablation.py` | A, B, C, Random | 10m, 1h, 6h | LightGBM, Ridge, S0 | Delta / Absolute | Standard v2 benchmark table (Table 1–5) |
| 34 | `v2_adaptation` | Current | `run_v2_planning_adaptation.py` | GroupKFold (Fold 0) | 1h | Offset, Ridge, Boosting | Absolute (S2), Delta (S4) | Twin beats cluster mean +41.7%; loses to pers |
| 35 | `v2_fault_injection` | Current | `test_fault_injection.py` | Simulation | 10min | Conformal Alert Rule | Synthetic Perturbation | Detects sudden cooling loss in 20 min |
| 36 | `v2_transfer` | Current | `run_v2_cross_cluster.py` | C1 $\rightarrow$ C2 | 1h | Direct Transfer + Offset | Delta (S1), Absolute (S2) | Transfer monitoring penalty only +0.003°C |
| 37 | `v2_uncertainty` | Current | `run_v2_uncertainty.py` | Split C | 1h | Constant, Normalized, CQR| Delta + Quantile Loss | CQR achieves 91.19% cov, $r=0.470$ width vs err |

---

## 3. Folders with Missing Configs or Script Traceability

- **Folders with full traceable `config.yaml`:**
  All 21 legacy runs (`dataset1_*`, `adapt_*`, `conformal_*`, `transfer_*`) contain self-contained `config.yaml` and `results.json` files documenting exact parameters.
- **Folders without individual `config.yaml`:**
  - `experiments/diag_*` and `experiments/v2_*`:
    These folders were executed via standalone monolithic scripts (`run_v2_e1_ablation.py`, `run_v2_planning_adaptation.py`, `run_v2_cross_cluster.py`, `run_v2_uncertainty.py`, `test_fault_injection.py`) where configuration parameters were hardcoded in python script headers rather than written as a separate `config.yaml`.
  - All output CSVs in `v2_*` are fully reproducible from their named scripts.
