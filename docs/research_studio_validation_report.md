# Research Studio Validation Report
**Platform:** Thermal Twin Research Demonstration Studio  
**Application Entry Point:** `app/research_studio.py`  
**Legacy Dashboard Status:** `app/streamlit_app.py` strictly untouched (`git diff` = clean)  
**Execution Timestamp:** 2026-11-13  
**Test Suite Status:** 81 passed / 81 tests (100% passing)  

---

## 1. Executive Summary

A dedicated research demonstration platform (`app/research_studio.py`) has been constructed inside the `predictive-digital-twin-dc-thermal` repository. The application operates strictly as an empirical research demonstrator: all displayed data tables, sensor trajectories, before/after preprocessing comparisons, model predictions, ablation metrics, generalisation analyses, adaptation curves, and uncertainty calibrations are derived directly from actual repository data and genuine offline model execution. No mock formulas or synthetic numbers are utilized.

---

## 2. Implemented Features & Architecture

### Stage 1: Reusable Data Access Layer (`src/ttwin/research_studio/`)
- `data_registry.py`: Programmatic catalog of raw datasets, interim parquets, experiment CSVs, and model checkpoints.
- `dataset_loader.py`: High-performance, schema-validated dataset loading with pagination, sampling, and memory-safe chunking.
- `preprocessing_view.py`: Tracks verified data cleaning steps, row dropped counts (486 duplicate collision rows in D1, 384 in D2), clipping boundaries, and missingness distributions.
- `feature_view.py`: Explains causal feature engineering (lags, rolling stats), distinguishes monitoring delta from planning absolute targets, and generates dynamic feature correlation heatmaps.
- `model_loader.py`: Safe discovery and deserialization of verified offline checkpoints from `models/`.
- `inference.py`: Executes genuine model predictions on actual held-out telemetry observations, computing model error, persistence baseline, and skill.
- `experiment_loader.py`: Unified access to committed v3 and v2 experimental outputs with source file annotation.
- `validation.py`: Automated schema validation, data integrity checks, and data leakage assertions.

### Stage 2: Dataset Explorer (6 Comprehensive Tabs)
- **Tab A (Overview):** Verified hardware descriptions, provenance, row counts, variables, date ranges, and sampling cadences.
- **Tab B (Raw Telemetry):** Unmodified original observations directly from CSV, with column filters, pagination, first 10, last 10, random 10-row samples, and download buttons.
- **Tab C (Processed Telemetry):** Cleaned interim Parquet telemetry, per-server slices, observation distribution charts, and download options.
- **Tab D (Raw vs Processed):** Side-by-side comparison tables, tracking exact row counts before and after, dropped duplicate records (486 rows), clipping boundaries, and memory downcasting.
- **Tab E (Data Quality):** Sensor missingness heatmaps/bars, temperature measurement distributions, and sampling interval continuity verification.
- **Tab F (Server Explorer):** Deep-dive profiler for individual physical server telemetry with dual-axis temperature and CPU load trajectories.

### Stage 3: Preprocessing & Feature Engineering Lab
- Step-by-step walkthrough of the 7-stage lineage from raw CSV to model-ready tensors.
- Clear distinction between **Monitoring** ($\Delta T = T(t+h) - T(t)$) and **Planning** ($\hat{T}(t+h) = \text{absolute temperature}$).
- Live feature vector extraction for selected servers.
- Dynamic Pearson correlation heatmaps on actual feature values.

### Stage 4: Real Model Inference Lab
- Genuine offline inference on selected observations using a verified trained LightGBM model.
- Transparent reporting of evaluation partition membership (Train vs Validation vs Embargo vs Test).
- Side-by-side error comparison against the naive persistence baseline.
- Interactive actual-versus-predicted trajectory plot.
- Downloadable verified prediction JSON record.

### Stage 5: Model Benchmark & Ablation Lab
- Interactive filtering of committed benchmark CSVs by dataset, split, horizon, feature set, and model.
- Dedicated hardware sensor ablation tab (isolating the predictive utility of cooling fans and power).
- Scientific explanations of why persistence is an essential baseline and why small percentage gains must not be overstated.

### Stage 6: Spatial Generalisation Lab
- Detailed breakdown of Split A (Chronological), Split B (GroupKFold), and Split C (Strict Holdout).
- Empirical demonstration of data leakage: naive random row splitting underestimates planning error by **17.6% (S2)** and **18.5% (S3)**.
- Robustness evaluation across 5 random seed draws for Split C.

### Stage 7: Adaptation Lab
- Interactive budget scrubber across [0h, 1h, 6h, 24h, 72h, 168h] local calibration data.
- Fleet-wide planning adaptation across all 81 servers in Dataset 1 (43.2% error reduction for zero-shot over fleet mean).
- Cross-cluster hardware transfer from Dell PowerEdge R840 to Dell PowerEdge C6320.

### Stage 8: Conformal Uncertainty & Calibration Lab
- Comparison of Constant-width, Normalized, and Conformalized Quantile Regression (CQR).
- Verification of 4-way disjoint calibration partitions (Train, Val-Select, Val-Cal, 6h Embargo, Test).
- Conditional coverage breakdowns showing constant-width under-coverage (83.1%) on active loads.
- Weekly temporal coverage drift over 8 test weeks.
- Exact mathematical formulation of finite-sample conformal quantiles and exchangeability limitations.

### Stage 9: Experiment Reproduction Centre
- Sandboxed subprocess execution of `scripts/run_v3_experiments.py`, `scripts/run_v2_uncertainty.py`, and `scripts/train_baseline_model.py`.
- Support for `--dry-run`, `--reuse-existing`, and `--force` modes.
- Real-time stdout/stderr capture, execution duration tracking, and output artifact verification.

### Stage 10: Research Conclusions & Disclosures
- Comprehensive synthesis of research findings.
- Mandatory disclosures: offline prototype status, no live telemetry feeds, zero physical cooling actuators, no outdoor meteorological sensors.

---

## 3. Files Created or Modified

| Path | Action | Description |
|:---|:---|:---|
| `app/research_studio.py` | **Created** | Dedicated Streamlit research demonstration application. |
| `app/streamlit_app.py` | **Unchanged** | Legacy prototype preserved in exact original state. |
| `src/ttwin/research_studio/__init__.py` | **Created** | Research studio package exports. |
| `src/ttwin/research_studio/data_registry.py` | **Created** | Authoritative data, experiment, and model artifact catalog. |
| `src/ttwin/research_studio/dataset_loader.py` | **Created** | High-performance, memory-safe data access layer. |
| `src/ttwin/research_studio/preprocessing_view.py` | **Created** | Before/after comparison and data quality analysis. |
| `src/ttwin/research_studio/feature_view.py` | **Created** | Pipeline lineage, feature engineering, and correlation analysis. |
| `src/ttwin/research_studio/model_loader.py` | **Created** | Model checkpoint discovery and deserialization. |
| `src/ttwin/research_studio/inference.py` | **Created** | Genuine offline model inference engine. |
| `src/ttwin/research_studio/experiment_loader.py` | **Created** | Loader for committed benchmark results. |
| `src/ttwin/research_studio/validation.py` | **Created** | Schema validation and zero-leakage assertions. |
| `src/ttwin/research_studio/ui/*.py` | **Created** | 10 modular Streamlit UI components and dark theme styles. |
| `scripts/train_baseline_model.py` | **Created** | Verified offline training script for Split A S1 LightGBM checkpoint. |
| `models/dataset1_split_a_s1_lightgbm_60min.*` | **Created** | Trained booster checkpoint (`.joblib`, `.txt`) and metadata (`.json`). |
| `tests/test_research_studio_data_access.py` | **Created** | 12 automated unit tests for data access modules. |
| `tests/test_research_studio_app.py` | **Created** | 6 automated tests for research studio app configuration and integrity. |
| `docs/research_studio_data_inventory.md` | **Created** | Comprehensive repository audit document. |
| `docs/research_studio_validation_report.md` | **Created** | Final validation report. |

---

## 4. Verification & Automated Test Results

The complete test suite was executed using pytest:
```bash
pytest
```
- **Total tests:** 81
- **Passed:** 81 (100%)
- **Failed:** 0
- **Duration:** 26.77 seconds

### Test Suites Verified:
1. `tests/test_calibration_independence.py` (9 passed): 4-way disjoint partition, zero timestamp overlap, 6h embargo enforcement.
2. `tests/test_clean.py` (3 passed): Timestamp parsing, deduplication, temperature clipping.
3. `tests/test_features_no_future.py` (10 passed): Causal lag and rolling feature calculations without future leakage.
4. `tests/test_metrics.py` (9 passed): MAE, RMSE, skill scores, persistence predictions.
5. `tests/test_research_studio_app.py` (6 passed): Page configuration, no hardcoded drive paths, no physical actuation claims, untouched legacy app.
6. `tests/test_research_studio_data_access.py` (12 passed): Dataset loading, pagination, preprocessing views, genuine inference, schema validation.
7. `tests/test_run_v3_reproducibility.py` (7 passed): Manifest generation, CLI arguments (`--reuse-existing`, `--force`, `--dry-run`).
8. `tests/test_splits_no_leakage.py` (7 passed): Chronological, GroupKFold, and strict holdout integrity.
9. `tests/test_streamlit_app.py` (3 passed): Legacy application integrity.
10. `tests/test_uncertainty.py` (8 passed): Conformal quantile mathematics, finite-sample coverage guarantees.
11. `tests/test_v3_report_validation.py` (7 passed): Cross-verification of report text against committed CSV outputs.

---

## 5. Verified Dataset & Checkpoint Paths

- **Raw Dataset 1:** `data/raw/dataset1_combined.csv` (613,212,066 bytes, 2,748,525 rows)
- **Raw Dataset 2:** `data/raw/dataset2_combined.csv` (184,178,675 bytes, 1,281,024 rows)
- **Interim Dataset 1:** `data/interim/dataset1/_combined.parquet` (93,004,513 bytes, 2,748,039 rows)
- **Interim Dataset 2:** `data/interim/dataset2/_combined.parquet` (32,018,709 bytes, 1,280,640 rows)
- **Trained Checkpoint:** `models/dataset1_split_a_s1_lightgbm_60min.joblib` (LightGBM Booster)
- **Checkpoint Metadata:** `models/dataset1_split_a_s1_lightgbm_60min.json` (Test MAE: 0.8277°C, Persistence MAE: 0.8326°C, Skill: +0.59%)

---

## 6. Known Limitations & Scope Boundaries

1. **Offline Telemetry Only:** Telemetry originates from historical 2018–2019 cloud traces. There are no live sensor connections.
2. **Zero Closed-Loop Actuation:** The system cannot adjust physical CRAC units, chillers, or fan speeds.
3. **No Meteorological Sensors:** The dataset contains no outdoor weather telemetry. Seasonal temperature shifts cannot be attributed to weather.
4. **Physical Dissimilarity:** Dell PowerEdge R840 (dense 4-socket 2U) and C6320 (modular 4-node 2U sled) have distinct aerodynamic and thermal profiles; cross-cluster transfer is treated purely descriptively.
