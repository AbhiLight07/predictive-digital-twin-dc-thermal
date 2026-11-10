# 09 Dashboard Audit

This document audits the user-facing Streamlit dashboard application located at `app/streamlit_app.py`.

---

## 1. Run Command and Execution Environment

```powershell
streamlit run app/streamlit_app.py
```
- **Port:** Default `http://localhost:8501`
- **Framework:** Streamlit 1.67.0
- **Primary Data Sources:** `data/interim/dataset1/*.parquet` and `experiments/v2_*/*.csv`

---

## 2. Feature-by-Feature Verification Matrix

| Required Feature | Exists? | Works? | Exact Code Location (`app/streamlit_app.py`) | Model File / Backing Logic | Mock / Hard-Coded Details |
|---|---|---|---|---|---|
| **(a) Workload-Only What-If Simulator** | **YES** | **YES** | Lines 158–200 (`🔮 What-If Simulator (Planning)`) | **None (Hard-Coded Heuristic)** | Uses hardcoded formula: `pred = 45.0 + 0.28*cpu + 0.015*ram + 0.1*vms`. Does not load a saved LightGBM model binary. |
| **Labelled "predictive, correlational — not causal"** | **YES** | **YES** | Lines 162–168 | Orange markdown disclaimer banner explicitly warning operators that simulations reflect correlational historical data, not causal cooling responses. | None (Label is active and visible). |
| **(b) Limitations Panel** | **YES** | **YES** | Lines 295–320 (`⚠️ Limitations Panel`) | Markdown text panel | Displays 5 core limitations: offline prototype, single datacenter facility, correlational nature, conformal exchangeability drift, open-loop shadow. |
| **(c) Risk Indicator from Conformal Upper Bound** | **YES** | **YES** | Lines 232–248 (`⏱️ Historical Replay & Risk`) | Evaluates simulated interval upper bound against static thresholds: **Critical: $\ge 75.0^\circ\text{C}$**, **Elevated: $\ge 70.0^\circ\text{C}$**, **Normal: $< 70.0^\circ\text{C}$**. | Upper bound is calculated from simulated noise: `pred = cur_temp + sin(step/10)*0.3` and hardcoded quantile `q_hat = 1.80`. |
| **(d) Monitoring / Planning Operational Switch** | **YES** | **YES** | Lines 111–115 (Sidebar radio toggle) | Radio selector toggling between `Monitoring Mode (Telemetry + Temp Lags)` and `Planning Mode (Workload Only)`. | Toggles sidebar mode description; does not dynamically re-wire backend model pipelines. |
| **(e) Historical Replay Time-Scrubber** | **YES** | **YES** | Lines 202–230 (`⏱️ Historical Replay & Risk`) | Loads real server telemetry from `data/interim/dataset1/<server>.parquet` and exposes an interactive timeline slider (`step_idx`) across 500 timesteps. | Real telemetry replay; predictions superimposed on top are mock. |

---

## 3. Backing Model Files and Technical Architecture

1. **Overview & Ablation View (Page 1):**
   - Directly loads and renders `experiments/v2_ablation/results_split_a.csv`.
   - KPI cards dynamically display 1-hour MAE values: Persistence ($0.841^\circ\text{C}$), S1 LightGBM ($0.836^\circ\text{C}$), S4 LightGBM ($0.838^\circ\text{C}$).
2. **What-If Simulator (Page 2):**
   - Inputs: Baseline CPU (0–100%), CPU Delta ($\pm 20\%$), RAM Used (10–256 GB), Running VMs (0–50).
   - Calculation: Hardcoded linear thermal sensitivity ($+0.28^\circ\text{C}$ per $1\%$ CPU utilization).
3. **Historical Replay & Risk Monitor (Page 3):**
   - Inputs: Server dropdown (populates dynamically from `data/interim/dataset1/*.parquet`), timeline scrubber slider.
   - Plots: Historical true temperature series alongside simulated conformal upper/lower bounds.
4. **Conformal Coverage View (Page 4):**
   - Loads and renders `experiments/v2_uncertainty/conformal_methods_comparison.csv` and `coverage_by_regime_and_bin.csv`.

---

## 4. Headless Environment Note on Screenshots

Because the current execution environment is a headless Windows terminal without a GUI desktop display or automated browser screenshot driver installed, UI elements were verified directly through code inspection and AST analysis. Mockups of the dashboard pages are preserved in the text specifications of `app/streamlit_app.py`.
