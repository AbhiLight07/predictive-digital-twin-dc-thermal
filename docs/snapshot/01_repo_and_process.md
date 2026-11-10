# 01 Repo and Process

## 1. Git Repository State Commands (Raw Output)

### 1.1 `git status -sb`
```text
## main
?? docs/project_execution_summary.md
?? docs/results_summary.md
?? docs/snapshot/
?? experiments/adapt_dataset1_S4_1h_20261108_170601/
?? experiments/conformal_dataset1_A_S4_1h_20261108_165926/
?? experiments/conformal_dataset1_B_S4_1h_20261108_170141/
?? experiments/dataset1_A_S0_persistence_10min_20261108_145332/
?? experiments/dataset1_A_S0_persistence_1h_20261108_145458/
?? experiments/dataset1_A_S0_persistence_6h_20261108_145544/
?? experiments/dataset1_A_S1_lightgbm_10min_20261108_150840/
?? experiments/dataset1_A_S1_lightgbm_1h_20261108_150840/
?? experiments/dataset1_A_S1_lightgbm_6h_20261108_150842/
?? experiments/dataset1_A_S2_lightgbm_10min_20261108_145336/
?? experiments/dataset1_A_S2_lightgbm_1h_20261108_145337/
?? experiments/dataset1_A_S2_lightgbm_6h_20261108_145338/
?? experiments/dataset1_A_S3_lightgbm_10min_20261108_150844/
?? experiments/dataset1_A_S3_lightgbm_1h_20261108_150844/
?? experiments/dataset1_A_S3_lightgbm_6h_20261108_150845/
?? experiments/dataset1_A_S4_lightgbm_10min_20261108_145817/
?? experiments/dataset1_A_S4_lightgbm_1h_20261108_145819/
?? experiments/dataset1_A_S4_lightgbm_6h_20261108_145820/
?? experiments/dataset1_B_S0_persistence_10min_20261108_165305/
?? experiments/dataset1_B_S0_persistence_1h_20261108_165222/
?? experiments/dataset1_B_S0_persistence_6h_20261108_165332/
?? experiments/dataset1_B_S4_lightgbm_1h_20261108_174012/
?? experiments/dataset1_C_S0_persistence_1h_20261108_165358/
?? experiments/dataset1_C_S4_lightgbm_1h_20261108_174147/
?? experiments/diag_part_a_q1_baselines.csv
?? experiments/diag_part_a_q2_contemporaneous.csv
?? experiments/diag_part_a_q3_offsets.csv
?? experiments/diag_part_a_q4_shrinkage.csv
?? experiments/transfer_c1_to_c2_1h_20261108_170327/
?? pyproject.toml
?? reports/
?? scripts/diagnostics/diag_part_a_q1_to_q4.py
?? scripts/diagnostics/diag_part_a_q5_random_vs_group.py
?? scripts/diagnostics/diag_part_a_q7_fault_injection.py
?? scripts/diagnostics/generate_server_sets.py
?? scripts/diagnostics/inspect_data.py
?? scripts/generate_report_figures.py
?? scripts/run_adaptation.py
?? scripts/run_conformal.py
?? scripts/run_transfer.py
```

### 1.2 `git branch -a`
```text
  fix/features-splits-rerun
* main
```

### 1.3 `git log --date=iso --pretty="%h %ad %an %s"` (Full)
```text
a41aba0 2026-11-09 20:03:11 +0530 RVCE Thermal Twin Team docs: v2 results, fix report, literature expansion, and protocol alignments
aa0e974 2026-11-09 19:59:59 +0530 RVCE Thermal Twin Team exp: v2 re-runs — ablation, planning adaptation, uncertainty, cross-cluster transfer
24a7036 2026-11-08 22:24:52 +0530 RVCE Thermal Twin Team fix: current-state features at t, GroupKFold splits, stratified holdout sampling, and 70/10/20 Split A
d8c2cbc 2026-11-08 20:52:31 +0530 RVCE Thermal Twin Team docs: complete comprehensive diagnostics report (P1-P3 audit)
98c175d 2026-11-08 20:48:58 +0530 RVCE Thermal Twin Team exp: diagnostics P3: process, scope, completeness
f622e97 2026-11-08 20:48:34 +0530 RVCE Thermal Twin Team exp: diagnostics P2: uncertainty, cross-cluster transfer
3f86008 2026-11-08 18:43:05 +0530 RVCE Thermal Twin Team exp: diagnostics P1: target formulation, unified ablation, adaptation
b4b9378 2026-11-08 14:41:28 +0530 RVCE Thermal Twin Team feat: Phase 2 core modules (io, clean, features, splits, metrics, models), 23/23 tests pass, Parquet conversion done
04628a0 2026-11-08 14:29:30 +0530 RVCE Thermal Twin Team docs: Phase 1 data audit complete -- gate PASS (76 servers), column mappings verified, audit report written
e3b2154 2026-11-08 14:16:57 +0530 RVCE Thermal Twin Team fix: replace Unicode chars with ASCII for Windows cp1252 compat, add 'series' to host column detection
8457630 2026-11-08 14:00:43 +0530 RVCE Thermal Twin Team fix: update .gitignore to track data/README.md for download instructions and attribution
f045e78 2026-11-08 13:58:17 +0530 RVCE Thermal Twin Team chore: initial project structure — repo scaffold, docs, configs, module stubs, tests, and data audit script
```

### 1.4 `git tag -n`
```text
v0.1-audit      docs: Phase 1 data audit complete -- gate PASS (76 servers), column mappings verified, audit report written
v0.9-pre-fix    docs: complete comprehensive diagnostics report (P1-P3 audit)
v1.0-corrected  docs: v2 results, fix report, literature expansion, and protocol alignments
```

### 1.5 `git remote -v`
```text
(empty - no remote configured)
```

### 1.6 `git ls-files | wc -l`
```text
70
```

### 1.7 Check for Large or Forbidden Tracked Files
Commands:
`git ls-files | ForEach-Object { $f = $_; $info = Get-Item $f -ErrorAction SilentlyContinue; if ($info -and $info.Length -gt 5MB) { "$f $($info.Length)" } }`
Output: (empty — no files > 5 MB are tracked).

Command:
`git ls-files | Where-Object { $_ -match '(^data/|\.parquet$|\.csv$|\.env$)' }`
Output:
```text
data/README.md
experiments/diag_adaptation/adaptation_identical_window_results.csv
experiments/diag_adaptation/adaptation_summary_table.csv
experiments/diag_delta_target/results.csv
experiments/diag_split_b_5seeds/results.csv
experiments/diag_transfer/transfer_adaptation_results.csv
experiments/diag_uncertainty/local_recalibration_results.csv
experiments/diag_uncertainty/weekly_coverage_drift.csv
experiments/diag_unified_ablation/results.csv
experiments/v2_ablation/results_random_kfold_reference.csv
experiments/v2_ablation/results_split_a.csv
experiments/v2_ablation/results_split_b_groupkfold.csv
experiments/v2_ablation/results_split_c.csv
experiments/v2_ablation/shap_summary.csv
experiments/v2_adaptation/monitoring_adaptation_results.csv
experiments/v2_adaptation/planning_adaptation_results.csv
experiments/v2_adaptation/summary_table.csv
experiments/v2_transfer/cross_cluster_monitoring_results.csv
experiments/v2_transfer/cross_cluster_planning_results.csv
experiments/v2_transfer/hardware_distributions.csv
experiments/v2_transfer/transfer_adaptation_results.csv
experiments/v2_uncertainty/conformal_methods_comparison.csv
experiments/v2_uncertainty/coverage_by_regime_and_bin.csv
experiments/v2_uncertainty/local_recalibration_results.csv
experiments/v2_uncertainty/weekly_coverage_drift.csv
```
*Note on tracked CSVs:* No raw telemetry CSVs or Parquet binaries are tracked. The only CSV files tracked under `experiments/` are aggregate tabular metric outputs (each < 30 KB). No `.env` credentials files exist or are tracked.

---

## 2. Process and State Questions

| Question | Value / Answer | Evidence / Details |
|---|---|---|
| **Is anything pushed?** | **NO** | `git remote -v` returns empty string; no remote was added. |
| **Time of the last push?** | **NEVER** | No remote ever contacted from this repository. |
| **Commit of tag `v0.1-audit`** | `04628a0` | Commit: `docs: Phase 1 data audit complete -- gate PASS (76 servers)...` |
| **Commit of tag `v0.9-pre-fix`** | `d8c2cbc` | Commit: `docs: complete comprehensive diagnostics report (P1-P3 audit)` |
| **Commit of tag `v1.0-corrected`** | `a41aba0` | Commit: `docs: v2 results, fix report, literature expansion, and protocol alignments` |
| **Is the working tree clean?** | **NO** | 35 untracked directories/files present (untracked legacy runs in `experiments/`, intermediate diagnostic scripts, `reports/`, and new `docs/snapshot/`). |
