# 02 Files and Tests

## 1. Repository Source Tree (Excluding `data/`, `.venv/`, `.git/`, `__pycache__/`)

Sizes and line counts for every tracked and workspace file:

```text
.gitignore  (0.5 KB, 26 lines)
PROJECT_BRIEF.md  (24.8 KB, 152 lines)
README.md  (6.0 KB, 103 lines)
pyproject.toml  (0.6 KB, 25 lines)
requirements.txt  (0.3 KB, 21 lines)

app/
  streamlit_app.py  (17.1 KB, 336 lines)

configs/
  columns.yaml  (3.4 KB, 138 lines)
  protocol.yaml  (1.0 KB, 29 lines)
  experiments/
    .gitkeep  (0.0 KB, 0 lines)

docs/
  data_audit_report.md  (8.1 KB, 139 lines)
  decisions.md  (3.0 KB, 19 lines)
  limitations.md  (4.0 KB, 22 lines)
  literature.md  (9.2 KB, 35 lines)
  project_execution_summary.md  (19.1 KB, 219 lines)
  protocol.md  (3.5 KB, 42 lines)
  results_summary.md  (7.3 KB, 80 lines)
  results_v2.md  (22.2 KB, 318 lines)
  diagnostics/
    DIAGNOSTICS_REPORT.md  (28.4 KB, 321 lines)
    FIX_REPORT.md  (13.0 KB, 170 lines)
  paper/

reports/
  research_paper_draft.md  (19.5 KB, 165 lines)
  figures/
    fig1_ablation_horizons.png  (91.9 KB, binary)
    fig2_adaptation_curves.png  (114.8 KB, binary)
    fig3_conformal_coverage.png  (146.3 KB, binary)
    fig4_cross_cluster_transfer.png  (75.6 KB, binary)

scripts/
  data_audit.py  (18.5 KB, 431 lines)
  generate_report_figures.py  (15.0 KB, 243 lines)
  make_parquet.py  (4.0 KB, 134 lines)
  run_adaptation.py  (5.8 KB, 149 lines)
  run_conformal.py  (5.4 KB, 129 lines)
  run_experiment.py  (14.1 KB, 334 lines)
  run_transfer.py  (7.1 KB, 161 lines)
  run_v2_cross_cluster.py  (13.7 KB, 219 lines)
  run_v2_e1_ablation.py  (20.4 KB, 456 lines)
  run_v2_planning_adaptation.py  (12.0 KB, 266 lines)
  run_v2_uncertainty.py  (12.8 KB, 268 lines)
  test_fault_injection.py  (6.0 KB, 148 lines)
  diagnostics/
    diag_part_a_q1_to_q4.py  (16.2 KB, 257 lines)
    diag_part_a_q5_random_vs_group.py  (6.0 KB, 119 lines)
    diag_part_a_q7_fault_injection.py  (5.8 KB, 125 lines)
    generate_server_sets.py  (1.9 KB, 50 lines)
    inspect_data.py  (1.2 KB, 30 lines)
    p1_adaptation.py  (14.8 KB, 302 lines)
    p1_target_and_ablation.py  (18.7 KB, 395 lines)
    p2_transfer.py  (13.0 KB, 242 lines)
    p2_uncertainty.py  (13.8 KB, 241 lines)
    p3_process.py  (10.3 KB, 181 lines)

src/ttwin/
  __init__.py  (0.1 KB, 2 lines)
  adapt.py  (7.8 KB, 220 lines)
  clean.py  (3.3 KB, 100 lines)
  eval.py  (7.0 KB, 146 lines)
  features.py  (6.3 KB, 175 lines)
  io.py  (1.9 KB, 40 lines)
  metrics.py  (2.7 KB, 82 lines)
  models.py  (6.9 KB, 208 lines)
  splits.py  (8.9 KB, 236 lines)
  uncertainty.py  (6.2 KB, 186 lines)

tests/
  __init__.py  (0.0 KB, 0 lines)
  test_features_no_future.py  (5.1 KB, 125 lines)
  test_metrics.py  (1.7 KB, 63 lines)
  test_splits_no_leakage.py  (3.2 KB, 87 lines)
```

---

## 2. Test Execution & Environment Details

### 2.1 Python and Package Versions
Command:
```powershell
python --version
python -c "import pandas,numpy,sklearn,lightgbm,xgboost,shap,streamlit,pyarrow; print(f'pandas={pandas.__version__}'); print(f'numpy={numpy.__version__}'); print(f'sklearn={sklearn.__version__}'); print(f'lightgbm={lightgbm.__version__}'); print(f'xgboost={xgboost.__version__}'); print(f'shap={shap.__version__}'); print(f'streamlit={streamlit.__version__}'); print(f'pyarrow={pyarrow.__version__}')"
```
Raw Output:
```text
Python 3.11.9
pandas=3.1.1
numpy=2.4.6
sklearn=1.9.1
lightgbm=4.7.0
xgboost=3.2.0
shap=0.51.0
streamlit=1.67.0
pyarrow=26.0.1
```

### 2.2 Full Pytest Output (`pytest -q`)
Command:
`pytest -q`
Raw Output:
```text
.........................                                                [100%]
25 passed in 2.58s
```

Detailed test names (from `pytest -v --tb=no`):
```text
tests/test_features_no_future.py::test_lag_features_no_future PASSED     [  4%]
tests/test_features_no_future.py::test_rolling_features_end_at_t PASSED  [  8%]
tests/test_features_no_future.py::test_perturbing_future_leaves_features_at_t_unchanged PASSED [ 12%]
tests/test_features_no_future.py::test_target_creation_correct_horizon_and_delta PASSED [ 16%]
tests/test_features_no_future.py::test_feature_set_s1_and_s4_include_current_temp PASSED [ 20%]
tests/test_features_no_future.py::test_effective_cpu_temp_is_max PASSED  [ 24%]
tests/test_features_no_future.py::test_calendar_features_correct PASSED  [ 28%]
tests/test_features_no_future.py::test_feature_set_s0_is_empty PASSED    [ 32%]
tests/test_features_no_future.py::test_feature_set_s2_excludes_power_and_fan PASSED [ 36%]
tests/test_metrics.py::test_mae_computation PASSED                       [ 40%]
tests/test_metrics.py::test_mae_zero PASSED                              [ 44%]
tests/test_metrics.py::test_rmse_computation PASSED                      [ 48%]
tests/test_metrics.py::test_rmse_zero PASSED                             [ 52%]
tests/test_metrics.py::test_skill_score_perfect PASSED                   [ 56%]
tests/test_metrics.py::test_skill_score_equal PASSED                     [ 60%]
tests/test_metrics.py::test_skill_score_worse PASSED                     [ 64%]
tests/test_metrics.py::test_per_server_metrics PASSED                    [ 68%]
tests/test_metrics.py::test_summarize_per_server PASSED                  [ 72%]
tests/test_splits_no_leakage.py::test_no_future_timestamps_in_train PASSED [ 76%]
tests/test_splits_no_leakage.py::test_no_server_overlap_in_leave_out PASSED [ 80%]
tests/test_splits_no_leakage.py::test_strict_split_no_overlap PASSED     [ 84%]
tests/test_splits_no_leakage.py::test_leave_out_reproducibility PASSED   [ 88%]
tests/test_splits_no_leakage.py::test_leave_out_different_seeds_give_different_sets PASSED [ 92%]
tests/test_splits_no_leakage.py::test_group_kfold_covers_all_servers_disjointly PASSED [ 96%]
tests/test_splits_no_leakage.py::test_split_b_has_5_seeds PASSED         [100%]
```

---

## 3. Test File Invariants and Assert Quotations

### 3.1 `tests/test_features_no_future.py`
Enforces temporal causal invariants on feature engineering: lagged and rolling features must end at observation time $t$ and never access future timestamps $> t$.

Quoted Assert Lines for Feature Causal Invariance:
```python
# Line 37: Lag 1 uses strictly t-1
assert df.iloc[i]["CPU_Temperature_lag_1"] == df.iloc[i - 1]["CPU_Temperature"]

# Lines 48-50: Rolling window strictly ends at t
np.testing.assert_almost_equal(df.iloc[i]["CPU_Temperature_rmean_3"], expected, decimal=5)

# Lines 71-72: Future perturbation does not alter features at t
pd.testing.assert_series_equal(features_at_t_orig, features_at_t_pert)

# Lines 83-84: Target horizon step is strictly shifted by +h
assert df.iloc[i][target_col] == sample_server_df.iloc[i + 3]["CPU_Temperature"]
assert df.iloc[i][delta_col] == sample_server_df.iloc[i + 3]["CPU_Temperature"] - sample_server_df.iloc[i]["CPU_Temperature"]

# Lines 96-99: Current state T(t) is explicitly included in S1 and S4
assert "CPU_Temperature" in s1_cols
assert "CPU_Temperature" in s4_cols
```

### 3.2 `tests/test_splits_no_leakage.py`
Enforces partition disjointness and temporal ordering: no future timestamps in training partitions, no physical server overlap between train and test partitions in spatial splits, and GroupKFold partitions covering 100% of servers disjointly.

Quoted Assert Lines for Partition Leakage Tests:
```python
# Lines 37-39: Split A temporal arrow and embargo
assert verify_no_future_leak(train, test)
assert train["Time"].max() < test["Time"].min()
assert val["Time"].max() < test["Time"].min()

# Lines 44-45: Split B server disjointness
assert verify_no_server_overlap(train, test)
assert len(set(train_servers) & set(test_servers)) == 0

# Lines 52-54: Split C spatiotemporal disjointness
assert len(train_servers & test_servers) == 0
assert train["Time"].max() < test["Time"].min()

# Lines 78-83: GroupKFold complete disjoint partition coverage
assert len(set(train_s) & set(test_s)) == 0
assert len(all_test_servers) == len(servers)
assert set(all_test_servers) == servers
```

### 3.3 `tests/test_metrics.py`
Enforces numerical definitions and symmetry of evaluation metrics: MAE, RMSE, and Skill Score relative to the persistence baseline.

Quoted Assert Lines:
```python
# Line 15: MAE computation
assert mae(y_true, y_pred) == pytest.approx(0.5)

# Line 26: RMSE computation
assert rmse(y_true, y_pred) == pytest.approx(np.sqrt(1 / 3))

# Lines 35-43: Skill score bounds
assert skill_score_vs_persistence(0.0, 1.0) == pytest.approx(1.0)
assert skill_score_vs_persistence(1.0, 1.0) == pytest.approx(0.0)
assert skill_score_vs_persistence(2.0, 1.0) == pytest.approx(-1.0)
```
