# 07 Claims Ledger

This ledger audits every numerical, comparative, and methodological claim across the repository's documentation, research paper draft, dashboard, and experiment outputs.

Each claim is mapped to its source file, exact experiment ID, underlying output file, re-read verification status, and whether it is **Supported**, **Superseded**, or **Unsupported**.

---

## 1. Claims Ledger Table

| File & Line | Claim Text | Experiment ID | Output File and Cell / Column | Verified by Re-reading? | Status | Audit Notes |
|---|---|---|---|---|---|---|
| `docs/results_v2.md:44` | S1 LightGBM 10-min MAE = 0.4024°C, Skill = +0.07% vs Persistence (0.4026°C) | `v2_ablation` | `results_split_a.csv`, row 3 | Yes | **Supported** | Matches file exactly; evaluated on 553,052 identical test rows. |
| `docs/results_v2.md:50` | S4 LightGBM 10-min MAE = 0.4023°C, Skill = +0.08% vs Persistence | `v2_ablation` | `results_split_a.csv`, row 9 | Yes | **Supported** | Marginal edge over persistence. |
| `docs/results_v2.md:53` | S1 LightGBM 1-hour MAE = 0.8362°C, Skill = +0.54% vs Persistence (0.8408°C) | `v2_ablation` | `results_split_a.csv`, row 12 | Yes | **Supported** | Re-verified independently in fresh repro check (0.8359°C vs 0.8408°C). |
| `docs/results_v2.md:59` | S4 LightGBM 1-hour MAE = 0.8375°C, Skill = +0.39% vs Persistence | `v2_ablation` | `results_split_a.csv`, row 18 | Yes | **Supported** | Adding fan/inlet features does not beat pure autoregression S1. |
| `docs/results_v2.md:62` | S1 LightGBM 6-hour MAE = 1.5917°C, Skill = +0.67% vs Persistence (1.6025°C) | `v2_ablation` | `results_split_a.csv`, row 21 | Yes | **Supported** | Long-horizon persistence remains highly competitive. |
| `docs/results_v2.md:68` | S4 LightGBM 6-hour MAE = 1.5877°C, Skill = +0.92% vs Persistence | `v2_ablation` | `results_split_a.csv`, row 27 | Yes | **Supported** | Matches file; full telemetry gives +0.92% skill at 6h. |
| `docs/results_v2.md:71` | Dataset 2 S1 LightGBM 10-min MAE = 0.4852°C, Skill = +3.19% vs Persistence (0.5012°C) | `v2_ablation` | `results_split_a.csv`, row 30 | Yes | **Supported** | Matches file on 216,205 identical rows. |
| `docs/results_v2.md:78` | Dataset 2 S1 LightGBM 1-hour MAE = 0.7809°C, Skill = +2.90% vs Persistence (0.8042°C) | `v2_ablation` | `results_split_a.csv`, row 37 | Yes | **Supported** | Ridge S1 achieved +4.14% (0.7709°C). |
| `docs/results_v2.md:85` | Dataset 2 S1 LightGBM 6-hour MAE = 1.2709°C, Skill = +1.14% vs Persistence (1.2856°C) | `v2_ablation` | `results_split_a.csv`, row 44 | Yes | **Supported** | Matches file on 213,951 identical rows. |
| `docs/results_v2.md:109` | Split B 5-Fold GroupKFold S1 MAE = 0.9324 ± 0.1213°C, Skill = -1.37% vs Pers (0.9207°C) | `v2_ablation` | `results_split_b_groupkfold.csv`, summary | Yes | **Supported** | LightGBM loses to persistence out-of-fold across 76 servers. |
| `docs/results_v2.md:109` | Split B 5-Fold GroupKFold S4 MAE = 0.9346 ± 0.1270°C, Skill = -1.55% vs Pers | `v2_ablation` | `results_split_b_groupkfold.csv`, summary | Yes | **Supported** | Negative skill confirmed across all 5 spatial folds. |
| `docs/results_v2.md:130` | Split C Dataset 1 S1 LightGBM MAE = 0.6205°C, Skill = -0.08% vs Pers (0.6200°C) | `v2_ablation` | `results_split_c.csv`, row 2 | Yes | **Supported** | Matches file; evaluated on 15 held-out hosts (101,808 rows). |
| `docs/results_v2.md:149` | Random 5-Fold CV reference MAE = 0.9225 ± 0.0225°C | `v2_ablation` | `results_random_kfold_reference.csv` | Yes | **Supported** | Run on 50,000 sampled rows. Did not show inflation over GroupKFold (0.9346°C) because test rows differed. |
| `docs/results_v2.md:163` | TreeSHAP top feature is `CPU_Temperature` (mean \|SHAP\| = 0.7466°C) | `v2_ablation` | `shap_summary.csv`, row 1 | Yes | **Supported** | State at $t$ dominates all workload features combined (`CPU_Load` is #17 at 0.0359°C). |
| `docs/results_v2.md:200` | Pretrained Planning Twin Zero-Shot MAE = 4.831°C (+41.7% over Cluster Mean 8.285°C) | `v2_adaptation` | `summary_table.csv`, row 6 | Yes | **Partially Supported (Inflated Claim)** | 4.831°C vs 8.285°C is in the file. However, against Ridge(CPU_Load) (5.98°C) gain drops to **+19.2%**, and against Load-Conditional Mean (6.11°C) drops to **+21.0%**. |
| `docs/results_v2.md:199` | Local Server Mean 168h MAE = 10.600°C (-27.9% vs Cluster Mean) | `v2_adaptation` | `summary_table.csv`, row 5 | Yes | **Supported** | Matches file. Caused by non-stationarity: mean drift between 168h adaptation window and test window is +3.99°C (range -10.6°C to +19.3°C). |
| `docs/results_v2.md:205` | Residual Offset (24h) Planning MAE = 5.474°C (worse than Zero-Shot 4.831°C) | `v2_adaptation` | `summary_table.csv`, row 12 | Yes | **Supported** | Offset hurts because Pearson correlation between 168h adaptation offset and test offset is $r = -0.026$ ($p=0.92$). |
| `docs/results_v2.md:210` | Pretrained Monitoring Twin Zero-Shot MAE = 1.182°C (-7.8% vs Persistence 1.096°C) | `v2_adaptation` | `summary_table.csv`, row 18 | Yes | **Supported** | Shrinkage with $\lambda = 0.10$ recovers MAE to 1.0877°C (+0.43% skill). |
| `docs/results_v2.md:235` | Conformal CQR Empirical Coverage = 91.19% at 90% target, Width = 3.679°C | `v2_uncertainty` | `conformal_methods_comparison.csv`, row 3 | Yes | **Supported** | Evaluated on Split C test set (109,170 samples). |
| `docs/results_v2.md:245` | CQR keeps coverage $\ge 90.8\%$ in all regimes (text in FIX_REPORT) | `v2_uncertainty` | `coverage_by_regime_and_bin.csv` | Yes | **Contradicted by Table 8** | Table 8 shows Moderate Temperature Bin (50–60°C) CQR coverage is **89.09%**, which is below 90.8%. |
| `docs/results_v2.md:251` | 40.0% of individual servers have empirical coverage below 90% for both Constant-Width and CQR | `v2_uncertainty` | `weekly_coverage_drift.csv` / script logs | Yes | **Supported** | Marginal coverage guarantee $\ge 90\%$ does NOT guarantee conditional per-server coverage. |
| `docs/results_v2.md:288` | Cross-cluster monitoring transfer penalty = +0.0030°C vs native Cluster 2 model | `v2_transfer` | `cross_cluster_monitoring_results.csv` | Yes | **Supported** | Transferred MAE = 0.8074°C vs Native = 0.8044°C vs Pers = 0.8055°C. |
| `docs/results_v2.md:291` | Cross-cluster planning transfer penalty = +0.6599°C vs native Cluster 2 planning | `v2_transfer` | `cross_cluster_planning_results.csv` | Yes | **Supported** | Transferred MAE = 4.8835°C vs Native = 4.2236°C vs Cluster 2 Mean = 10.8845°C. |
| `docs/results_v2.md:313` | Fault injection: Cooling loss detected in 20 min (2 steps), TPR = 93.3%, FPR = 0.0% | `v2_fault_injection` | `fault_injection_results.json` | Yes | **Supported (with caveats)** | Matches JSON. Caveat: evaluated on synthetic noise predictions; model input features were not perturbed. |
| `reports/research_paper_draft.md:15` | "Omitting thermal state history increases 1-hour forecasting error from 0.89°C to 5.28°C" | Legacy Phase 2 | Legacy `dataset1_A_*` | Yes | **Superseded** | Old legacy numbers. v2 numbers are 0.8362°C (S1) vs 5.2417°C (S2). |
| `reports/research_paper_draft.md:16` | "Pooled model generalises zero-shot to entirely unseen servers with an MAE of 0.527°C" | Legacy Phase 2 | `dataset1_B_S4_lightgbm_1h_20261108_174012` | Yes | **Superseded** | Flawed single-seed holdout on calm servers. v2 5-fold GroupKFold is **0.9346°C**. |
| `reports/research_paper_draft.md:17` | "Pretraining provides ~90% error reduction over scratch on cold-start (0.42°C vs 4.00°C–5.81°C)" | Legacy Phase 2 | `adapt_dataset1_S4_1h_20261108_170601` | Yes | **Superseded** | Flawed legacy comparison with non-identical test windows and scratch models without $T(t)$. |
| `reports/research_paper_draft.md:18` | "Conformal prediction achieves 96.7% empirical coverage at 90% nominal target" | Legacy Phase 2 | `conformal_dataset1_B_S4_1h_20261108_170141` | Yes | **Superseded** | Legacy split conformal on Split B. v2 CQR on Split C achieves **91.19%**. |
| `reports/research_paper_draft.md:19` | "Cross-cluster hardware transfer incurs only +0.232°C error degradation" | Legacy Phase 2 | `transfer_c1_to_c2_1h_20261108_170327` | Yes | **Superseded** | Legacy direct transfer. v2 monitoring transfer gap is **+0.0030°C**; planning gap is **+0.6599°C**. |
| `app/streamlit_app.py:147` | "0.0% to +0.2% Skill vs Persistence" | `app/streamlit_app.py` | UI text widget | Yes | **Supported** | Reflects empirical findings of Split A (+0.07% at 10m, +0.39% to +0.54% at 1h). |
| `app/streamlit_app.py:183` | `pred_base_temp = 45.0 + 0.28 * cpu + 0.015 * ram + 0.1 * vms` | `app/streamlit_app.py` | Code lines 183–184 | Yes | **Unsupported by Model Binary** | Hardcoded heuristic formula; does not load a trained model binary. |
| `app/streamlit_app.py:227` | `q_hat = 1.80` | `app/streamlit_app.py` | Code line 227 | Yes | **Unsupported by Dynamic Calibration** | Hardcoded scalar quantile; does not compute empirical quantile dynamically from telemetry. |

---

## 2. Summary of Claims Status

- **Total Claims Audited:** 32
- **Fully Supported by v2 Experiments:** 22
- **Partially Supported (Inflated baseline comparison):** 1 (Planning 41.7% gain vs cluster mean)
- **Directly Contradicted by Stored Data:** 1 (FIX_REPORT claiming CQR $\ge 90.8\%$ in all regimes vs Table 8 showing 89.09%)
- **Superseded Legacy Claims in Paper Draft:** 5 (All legacy draft abstract claims)
- **Unsupported Hardcoded Heuristics in Dashboard:** 2 (Heuristic what-if formula and hardcoded $q = 1.80^\circ\text{C}$)
- **Claims Without Experiment Behind Them:** 1 (Claim that chassis fans operate via proportional-integral control loops — qualitative domain inference, unverified from telemetry).
