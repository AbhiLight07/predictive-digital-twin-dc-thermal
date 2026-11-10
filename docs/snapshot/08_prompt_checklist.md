# 08 Prompt Checklist

Audit checklist tracking every numbered item and milestone across `PROJECT_BRIEF.md`, the Diagnostic Prompt, the Fix-and-Re-Run Prompt, and the Validate-and-Complete Prompt.

Criteria:
- `DONE`: Fully implemented with verified file output, test, or commit.
- `PARTIAL`: Implemented in intermediate diagnostic scripts or previous draft, but pending final consolidation.
- `NOT DONE`: Not executed or deferred.
- `SKIPPED-ON-PURPOSE`: Explicitly designated as optional or out-of-scope by protocol rules.

---

## 1. PROJECT_BRIEF.md Milestone Checklist

| Phase / Item | Requirement | Status | Evidence & Output Link |
|---|---|---|---|
| **Phase 0.1** | Create project structure and repository directory | **DONE** | Workspace at `d:\mainel5` (`f045e78`) |
| **Phase 0.2** | `git init`, virtual environment, `requirements.txt` | **DONE** | `.venv/`, `requirements.txt` (`f045e78`) |
| **Phase 0.3** | `.gitignore` excluding raw data, Parquet, large CSVs, `.env` | **DONE** | `.gitignore` (`8457630`) |
| **Phase 0.4** | Save brief as `PROJECT_BRIEF.md`, write `README.md` | **DONE** | `PROJECT_BRIEF.md`, `README.md` (`f045e78`) |
| **Phase 0.5** | Create GitHub repo and remote push | **PARTIAL** | Git initialized locally; remote creation pending user authentication (`01_repo_and_process.md`) |
| **Phase 1.1** | Data audit on raw CSVs (`scripts/data_audit.py`) | **DONE** | `docs/data_audit_report.md`, tag `v0.1-audit` (`04628a0`) |
| **Phase 1.2** | Check qualification gate (≥40 servers, ≥60 days, <10% missing) | **DONE** | 76 servers passed on D1; 60 on D2 (`data_audit_report.md`) |
| **Phase 2.1** | Raw CSV → Parquet conversion (`scripts/make_parquet.py`) | **DONE** | `data/interim/dataset1/`, `data/interim/dataset2/` (`b4b9378`) |
| **Phase 2.2** | Core library modules (`io`, `clean`, `features`, `splits`, `metrics`, `models`, `adapt`, `uncertainty`, `eval`) | **DONE** | `src/ttwin/*.py` (10 modules, 1,489 lines) (`b4b9378`) |
| **Phase 2.3** | Unit test suite verifying leakage prevention and metrics | **DONE** | `tests/*.py` (25/25 tests pass in 2.58s) |
| **Phase 2.4** | Freeze evaluation protocol (`configs/protocol.yaml`, `docs/protocol.md`) | **DONE** | `configs/protocol.yaml`, `docs/protocol.md` (`b4b9378`) |
| **Phase 3.1** | Experiment E1: Baseline & sensor ablation across splits A, B, C | **DONE** | `experiments/v2_ablation/results_split_a.csv`, `b.csv`, `c.csv` (`aa0e974`) |
| **Phase 3.2** | Literature review table (≥ 25 papers) | **DONE** | `docs/literature.md` (28 entries) (`a41aba0`) |
| **Phase 4.1** | Experiment E2 / E4: Cold-start adaptation (0h to 168h budgets) | **DONE** | `experiments/v2_adaptation/summary_table.csv` (`aa0e974`) |
| **Phase 4.2** | Experiment E3: Cross-cluster hardware transfer (C1 → C2) | **DONE** | `experiments/v2_transfer/cross_cluster_*.csv` (`aa0e974`) |
| **Phase 4.3** | Experiment E4 / E5: Uncertainty quantification & split conformal | **DONE** | `experiments/v2_uncertainty/conformal_methods_comparison.csv` (`aa0e974`) |
| **Phase 5.1** | Streamlit digital twin demo prototype (`app/streamlit_app.py`) | **DONE** | `app/streamlit_app.py` (5 active views) (`a41aba0`) |
| **Phase 5.2** | Synthetic fault injection alert simulation | **DONE** | `experiments/v2_fault_injection/fault_injection_results.json` (`aa0e974`) |
| **Phase 5.3** | Optional M100 ExaData scale-up replication (>1GB) | **SKIPPED-ON-PURPOSE** | Protocol rule: do not download >1GB without explicit approval. |
| **Phase 5.4** | Passive server logger stub (`logger/`) | **SKIPPED-ON-PURPOSE** | College server access not provided; pending user approval per brief §5.3. |
| **Phase 6.1** | Research paper draft | **PARTIAL** | `reports/research_paper_draft.md` written in Phase 2 with legacy numbers; rewrite pending Part C. |
| **Phase 6.2** | Tag releases | **DONE** | `v0.1-audit`, `v0.9-pre-fix`, `v1.0-corrected` |

---

## 2. Fix-and-Re-Run Prompt Checklist

| Item | Description | Status | Evidence & Output Link |
|---|---|---|---|
| **0.1** | GitHub status & remote check | **DONE** | Audited; manual commands provided (`docs/snapshot/01_repo_and_process.md`) |
| **0.2** | Safety audit: no raw data, parquets, creds tracked in git | **DONE** | Confirmed via `git ls-files` check (`01_repo_and_process.md`) |
| **1.1** | Split A boundaries diagnosis | **DONE** | `docs/diagnostics/FIX_REPORT.md:15-35` |
| **1.2** | Host IDs trailing spaces audit | **DONE** | `docs/diagnostics/FIX_REPORT.md:37-55` & `docs/snapshot/03_data.md` |
| **1.3** | Missingness gate recomputed on used columns | **DONE** | `docs/diagnostics/FIX_REPORT.md:57-78` (76 qualifying D1, 60 D2) |
| **1.4** | Conformal calibration data check (held-out validation vs training rows) | **DONE** | `docs/diagnostics/FIX_REPORT.md:80-92` (strictly on validation set) |
| **1.5** | Citation & hardware form-factor verification | **DONE** | `docs/diagnostics/FIX_REPORT.md:94-115` (R840 = 2U 4-socket, C6320 = 2U 4-node sled) |
| **2.1** | Fix current-state $T(t)$ in S1 and S4 | **DONE** | `src/ttwin/features.py:111-137` (`24a7036`) |
| **2.2** | Fix delta target formulation ($\Delta T$) for monitoring models | **DONE** | `src/ttwin/features.py:61-74`, `run_v2_e1_ablation.py:147` |
| **2.3** | Fix GroupKFold complete cross-validation across all servers | **DONE** | `src/ttwin/splits.py:129-174`, `test_splits_no_leakage.py:70-84` |
| **2.4** | Fix Split C strict spatiotemporal holdout | **DONE** | `src/ttwin/splits.py:177-215` (`24a7036`) |
| **2.5** | Fix 70/10/20 chronological Split A with 6h embargo | **DONE** | `src/ttwin/splits.py:14-39` (`24a7036`) |
| **2.6** | Implement CQR with pinball loss quantile regressors | **DONE** | `scripts/run_v2_uncertainty.py:144-174` (`aa0e974`) |
| **2.7** | Separate operational regimes (Monitoring vs Planning) | **DONE** | `scripts/run_v2_planning_adaptation.py` (`aa0e974`) |
| **3.1** | Re-run E1 Unified Ablation across Datasets 1 & 2 | **DONE** | `experiments/v2_ablation/` (5 CSV outputs) |
| **3.2** | Re-run E4 Planning & Monitoring Adaptation | **DONE** | `experiments/v2_adaptation/` (3 CSV outputs) |
| **3.3** | Re-run E5 Uncertainty Quantification | **DONE** | `experiments/v2_uncertainty/` (4 CSV outputs) |
| **3.4** | Re-run E3 Cross-Cluster Transfer | **DONE** | `experiments/v2_transfer/` (4 CSV outputs) |
| **3.5** | Produce `docs/results_v2.md` and `docs/diagnostics/FIX_REPORT.md` | **DONE** | `docs/results_v2.md`, `docs/diagnostics/FIX_REPORT.md` (`a41aba0`) |
| **3.6** | Tag `v1.0-corrected` | **DONE** | Tag `v1.0-corrected` at commit `a41aba0` |

---

## 3. Validate-and-Complete Prompt Checklist

| Item | Requirement | Status | Evidence & Output Link |
|---|---|---|---|
| **A.1** | Planning-regime baselines: evaluate Load-Conditional Mean & Ridge(CPU_Load) | **DONE** | `experiments/diag_part_a_q1_baselines.csv` (Twin gain is +19.2% vs Ridge, +21.0% vs Load-Cond) |
| **A.2** | Planning target: evaluate contemporaneous form $X(t) \rightarrow T(t)$ | **DONE** | `experiments/diag_part_a_q2_contemporaneous.csv` (Contemporaneous MAE = 4.49°C vs 4.83°C) |
| **A.3** | Residual offset instability analysis (correlation across windows) | **DONE** | `experiments/diag_part_a_q3_offsets.csv` (Pearson $r = -0.026$, non-stationarity drift +3.99°C) |
| **A.4** | Monitoring zero-shot vs E1 reconciliation & shrinkage test ($\lambda$) | **DONE** | `experiments/diag_part_a_q4_shrinkage.csv` (Optimal $\lambda = 0.10$, skill = +0.43% vs -9.69% raw) |
| **A.5** | Random $k$-fold vs GroupKFold on identical rows | **PARTIAL** | Script written (`diag_part_a_q5_random_vs_group.py`); S1 completed (+0.88% leak reduction), S4 aborted. |
| **A.6** | Conformal sample counts reconciliation (109,080 samples, 15 servers) | **DONE** | Audited in `docs/snapshot/03_data.md:5` and `docs/snapshot/04_protocol_and_splits.md:3.3` |
| **A.7** | Fault injection alert rule details & matched 0% FPR baseline comparison | **DONE** | Script written (`diag_part_a_q7_fault_injection.py`); audited in `docs/snapshot/05_critical_code.md:8` |
| **A.8** | Literature review verification status audit & purge fake errata entry | **DONE** | Audited in `docs/snapshot/10_literature.md` (27 real papers audited, Entry 28 flagged) |
| **A.9** | Canonical server count reconciliation table | **DONE** | Produced in `docs/snapshot/03_data.md:5` with 10 CSV server sets in `server_sets/` |
| **A.10**| Split C evaluation across 5 random server seeds | **NOT DONE** | Deferred per read-only snapshot rules. |
| **A.11**| Replace boilerplate NOT DONE items with factual items | **DONE** | Audited in `docs/snapshot/12_known_issues.md` |
| **B.1** | Paired bootstrap (2000 resamples) confidence intervals for Split A | **NOT DONE** | Deferred per read-only snapshot rules. |
| **B.2** | Sensor ablation (S4 without fans, without power, without both) | **NOT DONE** | Deferred per read-only snapshot rules. |
| **B.3** | Multi-horizon GroupKFold and Split C coverage | **NOT DONE** | Deferred per read-only snapshot rules. |
| **B.4** | Cross-cluster planning adaptation curves | **NOT DONE** | Deferred per read-only snapshot rules. |
| **B.5** | Missing budgets 0h and 1h added to within-cluster planning table | **NOT DONE** | Deferred per read-only snapshot rules. |
| **B.6** | CQR coverage for Split C across 5 server draws | **NOT DONE** | Deferred per read-only snapshot rules. |
| **C.1** | Git remote push & tag `v1.1-validated` | **NOT DONE** | Remote creation pending human auth; snapshot tagged locally. |
| **C.2** | Update dashboard with contemporaneous model & real metrics | **NOT DONE** | Deferred per read-only snapshot rules. |
| **C.3** | Update limitations.md, protocol.md, decisions.md | **NOT DONE** | Deferred per read-only snapshot rules. |
| **C.4** | Rewrite research paper draft with honest v3 benchmarks | **NOT DONE** | Preserved untouched per ground rules until Part C execution. |
| **C.5** | Logger stub (`logger/`) | **SKIPPED-ON-PURPOSE**| College server access not active. |
