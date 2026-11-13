# Research Studio — Repository Data & Artifact Inventory
**Generated:** 2026-11-13  
**Repository:** `predictive-digital-twin-dc-thermal`  
**Purpose:** Comprehensive audit of all raw datasets, interim/processed datasets, feature engineering pipelines, trained model checkpoints, experiment results, split protocols, and test suites prior to the implementation of `app/research_studio.py`.

---

## 1. Artifact Verification Summary Table

| Artifact | File path | Type | Purpose | Verified? | Status / Notes |
|:---|:---|:---|:---|:---|:---|
| **Raw Dataset 1** | `data/raw/dataset1_combined.csv` | CSV | Original telemetry observations (Cluster 1: Dell PowerEdge R840) | **Yes** | 613,212,066 bytes (~584.8 MB), 2,748,525 rows, 18 columns, 81 servers (`qh2-rcc120` to `qh2-rcc200`). |
| **Raw Dataset 2** | `data/raw/dataset2_combined.csv` | CSV | Original telemetry observations (Cluster 2: Dell PowerEdge C6320) | **Yes** | 184,178,675 bytes (~175.6 MB), 1,281,024 rows, 13 columns, 64 servers (`qh2-rcc201` to `qh2-rcc264`). |
| **Interim Dataset 1 Combined** | `data/interim/dataset1/_combined.parquet` | Parquet | Cleaned, deduplicated, temperature-clipped observations | **Yes** | 93,004,513 bytes (~88.7 MB), 2,748,039 rows, 18 columns, 81 servers. Deduplicated & clipped [-10°C, 110°C]. |
| **Interim Dataset 1 Per-Server** | `data/interim/dataset1/{server}.parquet` | Parquet | Cleaned per-server telemetry slices | **Yes** | 81 individual files (`qh2-rcc120.parquet` to `qh2-rcc200.parquet`). |
| **Interim Dataset 2 Combined** | `data/interim/dataset2/_combined.parquet` | Parquet | Cleaned, deduplicated, temperature-clipped observations | **Yes** | 32,018,709 bytes (~30.5 MB), 1,280,640 rows, 13 columns, 64 servers. |
| **Interim Dataset 2 Per-Server** | `data/interim/dataset2/{server}.parquet` | Parquet | Cleaned per-server telemetry slices | **Yes** | 64 individual files (`qh2-rcc201.parquet` to `qh2-rcc264.parquet`). |
| **Feature Dataset (Persisted)** | `data/processed/` | Directory | Persisted pre-computed feature matrix ready for modelling | **No** | Directory exists but is empty (`[]`). Feature engineering is computed in-memory via `src/ttwin/features.py:build_features()`. |
| **Trained Model Checkpoint** | `models/` or `checkpoints/` | Model file (`.joblib`, `.pkl`, `.pt`, `.onnx`, booster) | Serialized trained predictor for offline inference | **No** | **MISSING**. No saved model files exist in the repository. Previous experiments trained models in-memory. |
| **Experiment Outputs (v3)** | `experiments/v3_experiments/*.csv`, `*.json` | CSV & JSON | Gap-corrected Tier B benchmark results & execution manifest | **Yes** | 7 CSVs + 1 JSON execution manifest: `e1_bootstrap_summary.csv`, `sensor_ablation.csv`, `split_b_and_c_horizons.csv`, `planning_adaptation_all_folds.csv`, `cross_cluster_adaptation.csv`, `random_vs_group_identical_rows.csv`, `s1_learning_rate_study.csv`, `execution_manifest.json`. |
| **Experiment Outputs (v2)** | `experiments/v2_*/*.csv`, `*.json` | CSV & JSON | v2 ablation, adaptation, uncertainty, transfer, fault injection | **Yes** | All files verified and loadable: `v2_ablation/results_split_a.csv`, `results_split_b_groupkfold.csv`, `results_split_c.csv`, `v2_adaptation/summary_table.csv`, `v2_uncertainty/conformal_methods_comparison.csv`, `coverage_by_regime_and_bin.csv`, `v2_transfer/*.csv`, `v2_fault_injection/*.json`. |
| **Experiment Outputs (Legacy v1)** | `experiments/dataset1_*`, `conformal_*`, `adapt_*`, `transfer_*` | CSV & JSON | Early preliminary experiment runs | **Yes** | Multiple legacy directories containing `metrics.csv` and `results.json`. |

---

## 2. Detailed Technical Audit (Items 1 – 12)

### 1. Exact Location of Every Raw Dataset
- `data/raw/dataset1_combined.csv`
  - Size: 613,212,066 bytes (584.80 MB)
  - Raw row count: 2,748,525 rows
  - Raw column count: 18 columns
  - Hardware: Dell PowerEdge R840 (Cluster 1), 81 servers (`qh2-rcc120` to `qh2-rcc200`)
  - Date range: 2018-10-09 11:20:00 to 2019-06-20 10:00:00 (253.9 days)
  - Columns: `Series`, `Time`, `CPU_Load`, `Power`, `Network_RX`, `Network_TX`, `Inlet_Temperature`, `CPU1_Temperature`, `CPU2_Temperature`, `Fan_speed1`, `Fan_speed2`, `Fan_speed3`, `Fan_speed4`, `Ram`, `Ram_Used`, `No_Of_Running_vms`, `CPU_cores`, `CPU_cores_used`
- `data/raw/dataset2_combined.csv`
  - Size: 184,178,675 bytes (175.65 MB)
  - Raw row count: 1,281,024 rows
  - Raw column count: 13 columns
  - Hardware: Dell PowerEdge C6320 (Cluster 2), 64 servers (`qh2-rcc201` to `qh2-rcc264`)
  - Date range: 2019-02-01 11:10:00 to 2019-06-20 10:00:00 (139.0 days)
  - Columns: `Series`, `Time`, `CPU_Load`, `CPU1_Temperature`, `CPU2_Temperature`, `Ram`, `Ram_Used`, `CPU_cores`, `CPU_cores_used`, `Power`, `Network_RX`, `Network_TX`, `No_Of_Running_vms`
  - Notice: No fan speeds (`Fan_speed1..4`) and no `Inlet_Temperature` are recorded in Dataset 2.

### 2. Exact Location of Every Processed / Interim Dataset
- `data/interim/dataset1/_combined.parquet` (93,004,513 bytes, 2,748,039 rows, 18 columns, 81 servers)
- `data/interim/dataset1/{server}.parquet` (81 files: `qh2-rcc120.parquet` to `qh2-rcc200.parquet`)
- `data/interim/dataset2/_combined.parquet` (32,018,709 bytes, 1,280,640 rows, 13 columns, 64 servers)
- `data/interim/dataset2/{server}.parquet` (64 files: `qh2-rcc201.parquet` to `qh2-rcc264.parquet`)
- `data/processed/`: Empty directory (`[]`). Feature matrices are generated dynamically during execution.

### 3. Preprocessing Scripts and Execution Order
1. **Primary Script:** `scripts/make_parquet.py`
   - Invocation: `python scripts/make_parquet.py [--chunksize 100000] [--dataset both]`
   - Execution pipeline:
     1. Reads CSV in chunks (100k rows)
     2. Timestamp parsing: `pd.to_datetime(chunk['Time'], errors='coerce')`
     3. Group by server (`Series`)
     4. Timestamp sorting: `df.sort_values('Time').reset_index(drop=True)`
     5. Duplicate removal: `remove_duplicate_timestamps()` (drops duplicate `(Series, Time)` pairs, keeping first)
     6. Temperature clipping: `clip_temperature()` (sets values $< -10.0^\circ\text{C}$ or $> 110.0^\circ\text{C}$ to NaN across temperature/inlet columns)
     7. Numeric downcasting: `downcast_dtypes()` (`float64` $\to$ `float32`, `int64` $\to$ `int32`/`int16`)
     8. Writes individual `{server}.parquet` files and concatenated `_combined.parquet`.
2. **Library functions in `src/ttwin/clean.py`:**
   - `strip_host_names(df, host_col="Series")`
   - `parse_timestamps(df, time_col="Time")`
   - `remove_duplicate_timestamps(df, host_col="Series", time_col="Time")`
   - `fill_missing_timestamps(df, host_col="Series", time_col="Time", freq="10min")`
   - `clip_temperature(df, temp_cols=None, min_temp=-10.0, max_temp=110.0)`
   - `drop_constant_columns(df, exclude=None)`
   - `clean_dataset(df)`

### 4. Feature-Engineering Implementation
Located in `src/ttwin/features.py`:
- `fill_missing_timestamps()`: Reindexes each server to a strict 10-minute frequency grid ($t_{\min}$ to $t_{\max}$), inserting NaN for missing observations to guarantee that lag/rolling shifts represent true wall-clock elapsed time.
- `compute_effective_cpu_temp()`: `CPU_Temperature = max(CPU1_Temperature, CPU2_Temperature)`
- `add_calendar_features()`: `hour`, `dayofweek`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`
- `add_lag_features()`: Temperature lags for steps $l \in [1, 2, 3, 6, 12]$ (corresponding to 10m, 20m, 30m, 1h, 2h).
- `add_rolling_features()`: Causal rolling statistics ending at time $t$ for windows $w \in [6, 12, 36]$ (1h, 2h, 6h): `rmean_{w}` and `rstd_{w}`.
- Feature Sets defined via `get_feature_set(feature_set, available_columns)`:
  - **S0**: Empty feature set (Persistence baseline: $\hat{T}(t+h) = T(t)$)
  - **S1 (Monitoring, Temperature Only)**: Current $T(t)$ (`CPU_Temperature`, `CPU1_Temperature`, `CPU2_Temperature`) + lag and rolling temperature features (36 features).
  - **S2 (Planning, Workload Only)**: Workload (`CPU_Load`, `Ram`, `Ram_Used`, `CPU_cores`, `CPU_cores_used`, `Network_RX`, `Network_TX`, `No_Of_Running_vms`) + calendar cyclic encodings (14 features).
  - **S3 (Planning with Power)**: S2 + `Power` (15 features).
  - **S4 (Full Monitoring Telemetry)**: S3 + Fan speeds (`Fan_speed1..4`) + `Inlet_Temperature` + current temperatures + temperature lags & rolling statistics (56 features on Dataset 1).

### 5. Target-Generation Logic
Implemented in `src/ttwin/features.py:create_target()`:
- **Absolute Target:** `df[f"{target_col}_target_{h}"] = df.groupby(server_col)[target_col].shift(-h)`
- **Delta Target:** `df[f"{target_col}_delta_{h}"] = future_val - df[target_col]`
- **Operational Alignment:**
  - **Monitoring Regime (S1, S4):** Models are trained on the incremental delta target $\Delta T = T(t+h) - T(t)$. At inference, the forecast is reconstructed via $\hat{T}(t+h) = T(t) + \Delta \hat{T}$.
  - **Planning Regime (S2, S3):** Models are trained directly on the absolute target $T(t+h)$ because the current thermal state $T(t)$ is assumed unmeasured or hypothetical.

### 6. Trained Model Checkpoints
- **Status:** **NOT COMMITTED / MISSING**.
- The directories `models/` and `checkpoints/` do not exist in the repository.
- No serialized model objects (`.joblib`, `.pkl`, `.pickle`, `.pt`, `.onnx`, or LightGBM model booster files) exist.
- Historical scripts trained models in-memory, evaluated metrics, and saved summary tables.
- **Rule for Research Studio:** Per instructions, the dashboard must NOT fake inference. If no checkpoint exists, the app will show: *"Trained checkpoint unavailable. Run the verified training pipeline to generate a compatible model."*
- To support genuine inference without fabrication, a verified offline training script can be executed (or provided) that trains a standard LightGBM S1 model on Split A seen servers and saves the booster to `models/dataset1_split_a_s1_lightgbm_1h.joblib`. We will ask the user for approval regarding this.

### 7. Model Feature Order and Expected Input Schema
For S1 (36 features):
1. `CPU_Temperature`, `CPU1_Temperature`, `CPU2_Temperature`
2. Lags $[1, 2, 3, 6, 12]$ for `CPU_Temperature` (5 features)
3. Rolling mean & std $[6, 12, 36]$ for `CPU_Temperature` (6 features)
4. Lags $[1, 2, 3, 6, 12]$ for `CPU1_Temperature` (5 features)
5. Rolling mean & std $[6, 12, 36]$ for `CPU1_Temperature` (6 features)
6. Lags $[1, 2, 3, 6, 12]$ for `CPU2_Temperature` (5 features)
7. Rolling mean & std $[6, 12, 36]$ for `CPU2_Temperature` (6 features)
- Expected dtypes: all `float32` / `float64`. Missing values: handled natively by LightGBM or dropped for evaluation.

### 8. Experiment Result Files
All verified and committed in `experiments/`:
1. `experiments/v3_experiments/`:
   - `e1_bootstrap_summary.csv` (2 rows x 12 cols): 95% bootstrap CIs across servers and weeks for D1 and D2.
   - `sensor_ablation.csv` (6 rows x 6 cols): S0, S4 Full, S4 without Fans, S4 without Power, S4 without Fans & Power, S1.
   - `split_b_and_c_horizons.csv` (6 rows x 8 cols): Split B and Split C at 10m, 1h, 6h across 5 seeds.
   - `planning_adaptation_all_folds.csv` (12 rows x 7 cols): Fleet-wide planning adaptation across all 5 folds, budgets 0..168h.
   - `cross_cluster_adaptation.csv` (11 rows x 5 cols): C1 $\to$ C2 adaptation, budgets 0..168h.
   - `random_vs_group_identical_rows.csv` (4 rows x 5 cols): Leakage evaluation across S1, S2, S3, S4.
   - `s1_learning_rate_study.csv` (3 rows x 7 cols): Learning rates 0.05, 0.01, 0.005.
   - `execution_manifest.json`: Verification manifest with commit hash, seeds, and execution status.
2. `experiments/v2_ablation/`:
   - `results_split_a.csv`, `results_split_b_groupkfold.csv`, `results_split_c.csv`, `results_random_kfold_reference.csv`, `shap_summary.csv`
3. `experiments/v2_adaptation/`:
   - `planning_adaptation_results.csv`, `monitoring_adaptation_results.csv`, `summary_table.csv`
4. `experiments/v2_uncertainty/`:
   - `conformal_methods_comparison.csv`, `coverage_by_regime_and_bin.csv`, `local_recalibration_results.csv`, `weekly_coverage_drift.csv`
5. `experiments/v2_transfer/`:
   - `cross_cluster_monitoring_results.csv`, `cross_cluster_planning_results.csv`, `transfer_adaptation_results.csv`, `hardware_distributions.csv`
6. `experiments/v2_fault_injection/`:
   - `fault_injection_results.json`

### 9. Split Definitions and Evaluation Protocols
Frozen in `configs/protocol.yaml`:
- **Split A (Chronological):** Global time cutoff per cluster; 70% Train, 10% Validation, 20% Test with a 6-hour embargo (36 steps of 10 min) before test.
- **Split B (GroupKFold):** 5-fold cross-validation by physical server ID (`Series`), stratified by mean CPU load. All servers are evaluated in held-out test folds.
- **Split C (Strict Spatiotemporal Holdout):** 20% held-out servers evaluated exclusively on later time (final 20% duration with 6-hour embargo).
- **Conformal Calibration Disjointness (4-Way Partitioning):**
  - Train: 70% (model fitting)
  - Val-Select: 5% (early stopping & scale estimator)
  - Val-Cal: 5% (conformal nonconformity scores & quantile $\hat{q}$)
  - Embargo: 6 hours (strictly excluded)
  - Test: 20% - 6h embargo (final test evaluation)
  - Guarantees zero timestamp overlap between model selection and calibration.

### 10. Existing Dashboard Calculations & Illustrative Demonstrations
In `app/streamlit_app.py`:
- **Real Calculations:**
  - Page 1 (Overview & Ablation): Reads real CSV `experiments/v2_ablation/results_split_a.csv`.
  - Page 4 (Conformal Coverage): Reads real CSVs `experiments/v2_uncertainty/conformal_methods_comparison.csv` and `coverage_by_regime_and_bin.csv`.
- **Illustrative Demonstrations (Not Trained Models):**
  - Page 2 (What-If Simulator): Uses a hardcoded linear correlation heuristic:
    $$\hat{T} = 45.0 + 0.28 \cdot \text{CPU} + 0.015 \cdot \text{RAM} + 0.1 \cdot \text{VMs}$$
    with an illustrative static $\pm 4.8^\circ\text{C}$ uncertainty band.
  - Page 3 (Historical Replay & Risk): Reads actual telemetry from `data/interim/dataset1/*.parquet`, but computes predictions using an illustrative sinusoidal offset:
    $$\hat{T}(t+1\text{h}) = T(t) + 0.3 \cdot \sin(\text{step} / 10)$$
    and a fixed static conformal threshold $\hat{q} = 1.80^\circ\text{C}$.

### 11. Existing Test Coverage
- Total: **63 tests, 100% passing** in ~13.5 seconds (`pytest tests/`).
- Test suite breakdown:
  - `tests/test_calibration_independence.py` (9 tests)
  - `tests/test_clean.py` (3 tests)
  - `tests/test_features_no_future.py` (10 tests)
  - `tests/test_metrics.py` (9 tests)
  - `tests/test_run_v3_reproducibility.py` (7 tests)
  - `tests/test_splits_no_leakage.py` (7 tests)
  - `tests/test_streamlit_app.py` (3 tests)
  - `tests/test_uncertainty.py` (8 tests)
  - `tests/test_v3_report_validation.py` (7 tests)

### 12. Missing Files or Dependencies
- **Missing Files:**
  - Serialized model checkpoint file (`.joblib` / `.pkl`) for offline inference.
  - Pre-computed feature datasets in `data/processed/` (features are currently computed on-the-fly in python scripts).
  - `app/research_studio.py` (to be created in subsequent stages).
  - `src/ttwin/research_studio/` modules (to be created in Stage 1).
- **Environment & Dependencies:**
  - Python 3.11.9 in `.venv` with `streamlit` (1.67.0), `lightgbm` (4.7.0), `scikit-learn` (1.9.1), `pandas` (3.1.1), `numpy` (2.4.6), `matplotlib` (3.11.2), `seaborn` (0.13.2), `pytest` (9.2.0), `shap` (0.47.0), `pyyaml` (6.0.3), `pyarrow` (20.0.0).
  - All essential libraries are installed and functional.
