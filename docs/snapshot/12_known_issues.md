# 12 Known Issues, Vulnerabilities, and Confidence Assessment

This document provides a strictly adversarial self-assessment of the codebase, experimental results, and methodological limitations. It is written for an external reviewer who does not trust previous narrative summaries.

---

## 1. Bugs Known or Suspected (Fixed or Unfixed)

| Bug Description | Location in Code | Status | Technical Details & Impact |
|---|---|---|---|
| **Duplicate Column Slicing in Feature Selection** | `src/ttwin/features.py:133-137` | **Workaround in scripts; unfixed in core module** | `get_feature_set("S4", ...)` appends current temperature columns and lag columns to base features without deduplication. If `CPU_Temperature` appears in both base and temp lists, slicing `df[feats]` produces a 57-column matrix instead of 56, causing LightGBM dimension mismatch crashes. Script workaround `list(dict.fromkeys(...))` was applied, but `features.py` itself lacks internal deduplication. |
| **Omission of Host Name Stripping in `clean_dataset()`** | `src/ttwin/clean.py:90-100` | **Workaround in scripts; unfixed in core module** | `strip_host_names()` was written in `clean.py:44`, but `clean_dataset()` only calls `parse_timestamps()`, `remove_duplicate_timestamps()`, and `clip_temperature()`. As a result, interim Parquet files for Dataset 1 were saved with trailing whitespace (`"qh2-rcc120 .parquet"`). Handled downstream via `.str.strip()` in splitters. |
| **Legacy Split A Boundary Overlap** | `src/ttwin/splits.py` (legacy `d8c2cbc`) | **FIXED in v2 (`24a7036`)** | The legacy splitter used `val_start = int(n * (0.70 - 0.10))`, causing the validation set to overlap with the training partition and inflating the test set to 31.7% of calendar time (~76 days). Fixed in v2 to exact 70% Train / 10% Val / 6h Embargo / 20% Test duration. |
| **Hard-Coded Mock Logic in User Dashboard** | `app/streamlit_app.py:183, 227` | **UNFIXED (Design Limitation)** | The Streamlit dashboard's "What-If Simulator" does not query a saved LightGBM model; it uses a hardcoded formula (`45.0 + 0.28*cpu + 0.015*ram + 0.1*vms`). Similarly, the Historical Replay view generates intervals using sinusoidal mock noise (`sin(step/10)*0.3`) and a fixed scalar `q_hat = 1.80`. |
| **Synthetic Baseline in Fault Injection Script** | `scripts/test_fault_injection.py:58-62` | **UNFIXED (Evaluation Shortcut)** | The fault injection simulation evaluates alert rules against synthetic Gaussian noise (`pred = y_true + N(0, 0.4)`) rather than querying a model trained on perturbed telemetry features. |

---

## 2. Experimental Results We Would NOT Defend

1. **The Claimed "+41.7% Planning Improvement Over Baselines":**
   - *Why:* In `docs/results_v2.md:221`, the zero-shot planning model (MAE = $4.831^\circ\text{C}$) is claimed to provide a "41.7% error reduction" over the cluster baseline ($8.285^\circ\text{C}$). The cluster mean is an artificially weak strawman. When evaluated against stronger, trivial baselines:
     - `Ridge(CPU_Load)` achieves MAE = $5.98^\circ\text{C}$ (Twin gain drops to **+19.2%**).
     - `Load-Conditional Mean` (10 load decile bins) achieves MAE = $6.11^\circ\text{C}$ (Twin gain drops to **+21.0%**).
   - *Verdict:* We would defend an operational advantage of ~19%–21% over load-aware baselines, but **not** the headline claim of 41.7%.

2. **Local Server Mean (168h) Performing Worse (10.60°C) than Cluster Mean (8.29°C):**
   - *Why:* It seems absurd that 7 days of a server's own data produces a worse temperature forecast than the global fleet average. This occurs because the datacenter experienced significant non-stationary seasonal warming: mean temperature drift between the 168h adaptation period and the post-embargo test window averaged **$+3.99^\circ\text{C}$** across servers (ranging from $-10.64^\circ\text{C}$ to $+19.25^\circ\text{C}$). Because the baseline was evaluated as a static mean rather than an exponentially weighted moving average, it suffered severe bias.
   - *Verdict:* We would not defend this as evidence that "local data is useless"; it is an artifact of static baseline evaluation under temporal non-stationarity.

3. **Residual Offset Adaptation in Planning Regime:**
   - *Why:* Adding a scalar residual offset fitted on 6h to 24h of target telemetry increased error from $4.83^\circ\text{C}$ (zero-shot) to $5.47^\circ\text{C}$–$5.69^\circ\text{C}$. This occurs because the Pearson correlation between local residuals in the adaptation period and local residuals in the test window is **$r = -0.026$ ($p=0.92$)**—essentially zero. Server offsets in this trace are unstable over monthly horizons.

4. **Synthetic Fault Injection Detection Metrics (20 min latency, 93.3% TPR, 0.0% FPR):**
   - *Why:* The test was conducted by manually adding a linear ramp to a single 1,000-step telemetry slice and evaluating a mathematical inequality ($|y - \hat{y}| > 3\hat{q}$). Real physical fan failures cause complex thermodynamic interactions, airflow turbulence, chassis throttling, and dynamic setpoint compensation that cannot be validated from offline synthetic scalar addition.

5. **Legacy Abstract Metrics in `reports/research_paper_draft.md`:**
   - *Why:* The paper draft currently cites legacy metrics (e.g., Split B MAE = 0.527°C, ~90% error reduction on cold start, 96.7% conformal coverage). These were computed under optimistic, flawed legacy partitioning and must be completely replaced with v2 numbers.

---

## 3. Assumptions Made Without Empirical Evidence

1. **Hardware Server Form Factors:** Dataset 1 was documented as Dell PowerEdge R840 (originally claimed as 4U, corrected to 2U 4-socket); Dataset 2 was documented as Dell PowerEdge C6320 (2U 4-node modular sled). These specifications originate from Zenodo dataset descriptions and vendor marketing sheets, not physical machine inspection.
2. **Fan Controller PI Feedback Dynamics:** We argued that including fan speed does not improve forecasting skill because chassis fans operate via closed-loop feedback controllers. While standard in server firmware, the dataset contains no control setpoint logs or PID loop variables to empirically prove this causality.
3. **Exchangeability in Conformal Prediction:** Split conformal prediction guarantees marginal coverage only when calibration and test samples are exchangeable. Over a 9-month timeframe subject to ambient seasonal shifts and hardware workload migrations, exchangeability is routinely violated.

---

## 4. Discrepancies Where Reports Claimed What Code Does Not Do

1. **CQR Coverage Bound in `FIX_REPORT.md`:** The text in `FIX_REPORT.md:120` claims that CQR "keeps coverage $\ge 90.8\%$ across all regimes". However, Table 8 in `docs/results_v2.md` explicitly shows that in the Moderate Temperature Bin (50–60°C), CQR empirical coverage is **89.09%**, which falls below 90.8%.
2. **Host Name Stripping in `clean.py`:** `docs/data_audit_report.md` implied that host identifiers were stripped during interim cleaning, but `clean_dataset()` omitted calling `strip_host_names()`.
3. **Literature Verification in `literature.md`:** `docs/literature.md` marks all 28 entries as `read-full`. In reality, zero sources were fetched or opened in the current session, over half are abstract-only or reference-list citations, and Entry 28 was an errata note.

---

## 5. Top Five Checks an External Reviewer Should Run First

If an external auditor has only 15 minutes to inspect this repository, they should run these five checks in order:

1. **Verify Partition Leakage Prevention:**
   ```powershell
   pytest tests/test_splits_no_leakage.py tests/test_features_no_future.py -v
   ```
   *What it proves:* Confirms that feature engineering strictly ends at $t$ and that train/val/test splits maintain zero server overlap and strict temporal ordering.

2. **Execute the End-to-End Reproducibility Check from Raw Data:**
   ```powershell
   python reproduce_d1_1h_split_a.py
   ```
   *What it proves:* Recomputes Split A 1-hour persistence ($0.8408^\circ\text{C}$) and S1 LightGBM ($0.8359^\circ\text{C}$, skill $+0.57\%$) end to end from the 613 MB raw CSV in under 60 seconds.

3. **Inspect the Real Planning Advantage Against Strong Baselines:**
   Examine `experiments/diag_part_a_q1_baselines.csv`.
   *What it proves:* Shows that the twin's advantage over `Ridge(CPU_Load)` is **+19.2%** and over `Load-Conditional Mean` is **+21.0%**, deflating the claimed +41.7% gain.

4. **Inspect the Conformal Coverage Breakdown:**
   Examine `experiments/v2_uncertainty/coverage_by_regime_and_bin.csv` and `weekly_coverage_drift.csv`.
   *What it proves:* Proves that marginal 91.19% coverage masks conditional under-coverage (40% of individual servers fall below the nominal 90% target).

5. **Inspect the Streamlit Dashboard Source Code:**
   Open `app/streamlit_app.py` and inspect lines 182–185 and 227–230.
   *What it proves:* Confirms that what-if simulations and historical replay intervals are generated via heuristic equations and sinusoidal mock logic rather than dynamically queried model binaries.
