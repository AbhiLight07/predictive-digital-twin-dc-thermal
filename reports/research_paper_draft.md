# Predictive Digital Twin for Data Centre Thermal Management
## Generalisation, Limited-Data Adaptation, and Uncertainty-Aware Server Temperature Prediction

**Authors:** Student Research Team (CS, AI, ISE), Fifth Semester Experiential Learning  
**Target Venue:** Regional/National Peer-Reviewed Engineering Journal / Conference  
**Alignment:** UN Sustainable Development Goal 9 (*Industry, Innovation, and Infrastructure*)  
**Primary Dataset:** Melbourne Research-Cloud Datacenter Trace (Zenodo Record 10069402, CC BY 4.0)  

---

## Abstract
Modern cloud data centres consume substantial electrical power, with cooling infrastructure forming a major operational cost to prevent server thermal emergencies. While machine learning (ML) models have been proposed to forecast server temperatures from telemetry, prior work often trains bespoke models per physical node, evaluates with optimistic random cross-validation that leaks temporal state, overlooks cold-start onboarding regimes, and reports point estimates without uncertainty intervals.

This study investigates how reliably server-temperature prediction models generalise across unseen servers and distinct hardware clusters, how much local data is required to adapt a pretrained model to a new server, and whether conformal prediction intervals remain calibrated under temporal and server-level distribution shift. Using ~4.03 million rows of 10-minute production telemetry across two clusters (81 hosts in Cluster 1; 64 hosts in Cluster 2) from the University of Melbourne research-cloud trace (Ilager et al., UCC 2023), we show that:
1. Short-term forecasting is governed by thermal inertia: omitting temperature history in the workload-only planning regime increases 1-hour GroupKFold MAE from `0.9176°C` (S1) to `4.9588°C` (S2), while random $k$-fold splits underestimate S2 error by `17.60%` (`4.0862°C` vs `4.9588°C`).
2. Across Dataset 1 (`81` servers) and Dataset 2 (`59` servers), 1-hour Split A S1 LightGBM yields modest overall skill over persistence (`0.8277°C` vs `0.8326°C`, `+0.59%` on D1; `0.7814°C` vs `0.8042°C`, `+2.83%` on D2), with per-server 95% bootstrap confidence intervals crossing zero (`[-0.38%, +0.17%]` on D1; `[-6.85%, +3.38%]` on D2).
3. On unseen servers across all 5 GroupKFold folds ($N=81$), zero-shot planning reduces error by `43.2%` compared to the unconditioned fleet mean (`5.03°C` vs `8.87°C`), and 7 days of continued boosting reaches `4.87°C`.
4. Under strict spatiotemporal holdout (Split C), Conformalized Quantile Regression (CQR) achieves `91.19%` empirical coverage at a `90%` nominal target with a mean width of `3.679°C`, though conditional coverage dips to `89.09%` in moderate temperatures (`50–60°C`) and `40.0%` of held-out servers fall below `90%`.
5. Zero-shot cross-cluster planning transfer from Cluster 1 to Cluster 2 incurs high spread (`6.85°C` MAE, 95% CI `[2.92°C, 13.47°C]`), which 6 hours of local scalar residual-offset adaptation reduces to `3.16°C`.

These models are integrated into an offline, data-driven **predictive digital twin prototype** combining a FastAPI backend, calibrated CQR forecast cache, and an offline Three.js 3D schematic visualisation.

---

## 1. Introduction

Data centres consume substantial global electricity, with cooling systems needed to prevent server thermal emergencies where CPU temperatures exceed operational limits and trigger throttling or hardware degradation. Predictive thermal monitoring aims to anticipate temperature rises before thresholds are crossed. However, prior data-driven studies exhibit key methodological gaps:
- **Evaluation Leakage:** Standard random $k$-fold cross-validation leaks adjacent time steps and node identities across train/test splits, inflating accuracy estimates especially when temperature history is absent.
- **Node-Bespoke Isolation:** Training separate models per physical server fails when commissioning new nodes that lack local history.
- **Unquantified Uncertainty:** Operators need calibrated prediction intervals rather than point predictions alone when conditions shift over time or across servers.
- **Non-Causal Telemetry Features:** Fan speeds are typically firmware feedback responses to temperature rather than independent causes; mixing fan telemetry into what-if interventions introduces reverse-causality confusion.

### Research Questions
This work evaluates five research questions:
* **RQ1 (Honest Baselines & Sensor Ablation):** How does prediction performance change from chronological testing (seen servers, future time) to unseen-server testing, and how much does accuracy depend on power, fan speed, and temperature history?
* **RQ2 (Zero-Shot Generalisation):** Can a model trained on existing servers predict an unseen server's temperature with zero local data?
* **RQ3 (Limited-Data Adaptation):** How much local data from a new server is needed to adapt a pretrained model, and does adaptation beat training from scratch on the same local data?
* **RQ4 (Uncertainty Quantification):** Are prediction intervals reliable (coverage) on unseen servers and shifting conditions, and can local calibration restore coverage?
* **RQ5 (Cross-Hardware Transfer):** Does a model transfer across the two hardware clusters when restricted to shared features?

---

## 2. Telemetry Datasets & Quality Verification

This study uses the public University of Melbourne research-cloud trace (Ilager et al., UCC 2023; Zenodo `10069402`, CC BY 4.0) spanning two clusters sampled at 10-minute cadence:
- **Cluster 1 (Dataset 1):** `81` hosts (`qh2-rcc120` to `qh2-rcc200`), `2,748,525` raw rows (~253.9 days, `2018-10-09` to `2019-06-20`). Instrumented with dual CPU DTS temperatures, CPU load, RAM, cores, running VMs, network RX/TX, power, inlet temperature, and four fan speeds (five nodes `qh2-rcc196`–`rcc200` lack fan and inlet telemetry and span ~118 days).
- **Cluster 2 (Dataset 2):** `64` hosts (`qh2-rcc201` to `qh2-rcc264`), `1,281,024` raw rows (~139.0 days, `2019-02-01` to `2019-06-20`). Lacks fan-speed and inlet-temperature sensors entirely, and exhibits higher structural missingness (`12%–16%` in RAM/VM metrics, plus broken temperature registers on `qh2-rcc263` and `rcc264`).
- **Hardware Specification Policy:** As reported in the dataset paper (Ilager et al., UCC 2023): dual-CPU hosts; capacities as in the trace columns; other hardware specifications are unverified. Effective CPU temperature is defined as $T_t = \max(\text{CPU1}_t, \text{CPU2}_t)$.

### Quality Gate Audit
Before modeling (`scripts/data_audit.py` and `docs/data_audit_report.md`):
- **Regular Grid Alignment:** Because raw gaps cause row shifts (`shift(-h)`) to jump across irregular intervals, telemetry is reindexed onto a uniform 10-minute grid (`fill_missing_timestamps`, inserting `NaN` for missing timestamps) prior to causal lag and target construction.
- **Qualification Gate:** Using the phase-1 audit gate (`>= 60` days continuous coverage, `< 10%` missing temperature/workload), `76` of `81` Cluster 1 servers qualify against the `40`-server minimum gate. For cross-cluster experiments on Cluster 2, relaxing missingness to `< 20%` yields `53`–`59` qualifying servers depending on strict column completeness.

---

## 3. Methodology

```
┌────────────────────────────────────────────────────────────────────────┐
│            Telemetry (10-min uniform grid, NaN on gaps)                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      Causal Feature Engineering            Leakage-Free Split Protocols
      ├─ Cyclical Calendar (sin/cos)        ├─ Split A: 70/10/20 + 6h Embargo
      ├─ Rolling Stats (6, 12, 36 steps)    ├─ Split B: 5-Fold GroupKFold
      └─ Temp Lags (1, 2, 3, 6, 12)         └─ Split C: Unseen Hosts + Later Time
                  │                                   │
                  └─────────────────┬─────────────────┘
                                    ▼
                 Models: Persistence (S0), LightGBM, Ridge
                                    │
                  ┌─────────────────┴─────────────────┐
                  ▼                                   ▼
      E2 Cold-Start Adaptation               E4 Conformal Uncertainty
      ├─ Zero-Shot Transfer                  ├─ Split Conformal / Normalized
      ├─ Residual Offset & Boosted Tune      ├─ Conformalized Quantile Reg (CQR)
      └─ Scratch Local Baselines             └─ Conditional & Drift Coverage
```

### 3.1 Leakage-Free Experimental Protocols (`configs/protocol.yaml`)
1. **Split A (Chronological Shift):** Global `70%` train / `10%` validation / `20%` test cutoff per cluster with a **6-hour embargo buffer** (`36` timesteps) separating partitions to prevent lag and rolling-window leakage.
2. **Split B (Unseen Server Generalisation):** 5-fold `GroupKFold` grouped by host ID (`Series`) so every physical server is evaluated while held out.
3. **Split C (Strict Spatiotemporal Holdout):** Evaluated across held-out servers and future time (`5` random server seeds: `42, 123, 456, 789, 1024`).

### 3.2 Feature Ablation Ladder
- **S0 (Persistence Baseline):** $\hat{T}_{t+h} = T_t$.
- **S1 (Autoregressive Temperature State):** Current `CPU_Temperature`, `CPU1_Temperature`, `CPU2_Temperature`, lags (`1, 2, 3, 6, 12`), and rolling mean/std (`6, 12, 36`).
- **S2 (Workload + Calendar — Planning Setting):** CPU load, RAM, cores, running VMs, network RX/TX, and cyclic calendar terms (`hour_sin/cos`, `dow_sin/cos`, plus twin planning rolling means). No temperature history, power, or fan speeds.
- **S3 (Workload + Power):** S2 plus server `Power`.
- **S4 (Full Monitoring):** S3 plus fan speeds (`Fan_speed1–4`), `Inlet_Temperature`, and S1 temperature lags (D1 only).

### 3.3 Uncertainty Quantification (E4)
We compare constant-width split conformal intervals, normalised residual conformal intervals, and Conformalized Quantile Regression (CQR, targeting `90%` nominal coverage via symmetric pinball-loss quantile estimators calibrated on validation residuals).

---

## 4. Empirical Results & Discussion

*(Gap-corrected v3 results from `docs/results_v3.md` and `experiments/v3_experiments/`; conformal comparison from `docs/results_v2.md` §8.)*

### 4.1 RQ1: Feature Ablation, Persistence Parity & Evaluation Leakage (Split A)

Across `552,672` identical Dataset 1 test rows at the 1-hour horizon (`docs/results_v3.md` §2):

| Configuration | Features | Test MAE (°C) | Test RMSE (°C) | Skill vs Persistence |
|---|---:|---:|---:|---:|
| **S0 Persistence Baseline** | 0 | 0.8408 | 2.0086 | 0.00% |
| **S4 Full** (Fans + Power + Inlet + Workload + Lags) | 56 | 0.8381 | 1.9829 | +0.32% |
| **S4 without Fans** (Power + Inlet + Workload + Lags) | 52 | 0.8381 | 1.9829 | +0.32% |
| **S4 without Power** (Fans + Inlet + Workload + Lags) | 55 | 0.8383 | 1.9869 | +0.29% |
| **S4 without Fans & Power** (Inlet + Workload + Lags) | 51 | 0.8383 | 1.9869 | +0.29% |
| **S1 Temperature Lags & Current State Only** | 36 | 0.8360 | 1.9725 | +0.57% |

In the paired bootstrap analysis ($B=2,000$ resamples, `docs/results_v3.md` §1) on Split A (1-hour horizon):
- **Dataset 1 (`81` servers, `8` weeks):** S1 LightGBM `0.8277°C` vs persistence `0.8326°C` (`+0.59%` aggregate skill), mean per-server skill `-0.12%` (95% CI `[-0.38%, +0.17%]`, crossing zero).
- **Dataset 2 (`59` servers, `5` weeks):** S1 LightGBM `0.7814°C` vs persistence `0.8042°C` (`+2.83%` aggregate skill), mean per-server skill `-0.60%` (95% CI `[-6.85%, +3.38%]`, crossing zero).

**Key Findings:**
1. **Persistence Parity:** Thermal inertia dominates 10-min to 1-hour dynamics; individual per-server skill confidence intervals span zero. Reducing S1 learning rate from `0.05` (`5` trees) to `0.01` (`27` trees) and `0.005` (`49` trees) shifts test MAE by less than `0.00013°C` (`docs/results_v3.md` §7), confirming convergence reflects low $\Delta T$ variance around persistence rather than premature stopping.
2. **Hardware Telemetry Marginality:** Dropping fan speeds changes 1-hour test MAE by `<0.0001°C`; fan speed reflects closed-loop chassis BMC/PID reaction rather than causal cooling leverage.
3. **Random K-Fold Leakage (`docs/results_v3.md` §6):** Evaluating identical telemetry rows with random $k$-fold vs server-grouped `GroupKFold` underestimates planning error by `17.60%` for S2 (`4.0862°C` vs `4.9588°C`) and `18.54%` for S3 (`3.7598°C` vs `4.6153°C`), versus `0.24%` (`S1`) and `0.54%` (`S4`) in monitoring settings.

### 4.2 RQ2: Zero-Shot Unseen Server Generalisation (Split B & Split C)

Across 5-fold `GroupKFold` (`N=81` Dataset 1 hosts, `docs/results_v3.md` §3):

| Protocol | Horizon | Persistence Mean ± Std (°C) | S1 Mean ± Std (°C) | S4 Mean ± Std (°C) |
|---|---|---:|---:|---:|
| **Split B (`GroupKFold`)** | 10 min | 0.435 ± 0.046 | 0.436 ± 0.046 | 0.434 ± 0.045 |
| **Split B (`GroupKFold`)** | 1 h | 0.918 ± 0.115 | 0.924 ± 0.111 | 0.918 ± 0.111 |
| **Split B (`GroupKFold`)** | 6 h | 1.817 ± 0.203 | 1.828 ± 0.193 | 1.821 ± 0.195 |
| **Split C (`5` Seeds)** | 10 min | 0.456 ± 0.102 | 0.456 ± 0.101 | 0.456 ± 0.101 |
| **Split C (`5` Seeds)** | 1 h | 0.940 ± 0.179 | 0.936 ± 0.175 | 0.938 ± 0.176 |
| **Split C (`5` Seeds)** | 6 h | 1.710 ± 0.218 | 1.699 ± 0.209 | 1.734 ± 0.239 |

Unseen-server 1-hour error (`~0.92°C` Split B) exceeds seen-server chronological error (`~0.83°C` Split A), demonstrating real inter-server heterogeneity rather than universal zero-cost transfer.

### 4.3 RQ3: Fleet-Wide Planning Adaptation (`N=81`)

Evaluating planning adaptation across all 5 GroupKFold folds ($N=81$ Dataset 1 servers, `docs/results_v3.md` §4; `docs/results_v4_addendum.md` §1):

| Strategy | Local Budget | Mean MAE (°C) | Std (°C) | 95% CI (°C) |
|---|---|---:|---:|---:|
| **Fleet Mean** | 0 h | 8.87 | 2.99 | [8.23, 9.50] |
| **Ridge (`CPU_Load` only)** | 0 h | 5.70 | 2.15 | [5.24, 6.17] |
| **Load-Bin Mean (10% bins)** | 0 h | 5.67 | 2.14 | [5.21, 6.13] |
| **Zero-Shot Pretrained (`S2`)** | 0 h | 5.03 | 1.59 | [4.71, 5.39] |
| **Continued Boosting** | 24 h | 4.93 | 1.52 | [4.62, 5.28] |
| **Continued Boosting** | 168 h (7 d) | 4.87 | 1.43 | [4.58, 5.19] |
| **Residual Offset** | 1 h | 5.52 | 2.50 | [5.01, 6.08] |
| **Residual Offset** | 168 h (7 d) | 5.02 | 2.21 | [4.55, 5.52] |

Zero-shot `S2` LightGBM reduces error by `43.21%` (95% CI `[38.57%, 47.30%]`) relative to the unconditioned fleet mean (`5.03°C` vs `8.87°C`). However, simple load-aware baselines already capture most of that gain (`Ridge`: `5.70°C`; `load_bin_mean`: `5.67°C`), leaving `S2` with a more modest overall reduction of `+11.67%` (95% CI `[+5.37%, +17.56%]`; mean per-server `+5.80%`, CI `[-0.71%, +11.67%]`) over Ridge on `CPU_Load` and `+11.14%` (CI `[+4.24%, +17.02%]`; mean per-server `+5.01%`, CI `[-1.45%, +10.77%]`) over the 10%-binned load mean. Small local continued-boosting improvements accumulate gradually (`5.03°C → 4.87°C` at `168 h`), whereas prior lab testbed studies (Brännvall et al., Energies 2023) established earlier that fine-tuning beats scratch training.

### 4.4 RQ4: Conformal Uncertainty & Shift Coverage (`docs/results_v2.md` §8)

At the `90.0%` nominal target under Split C (`experiments/v2_uncertainty/`):

| Method | Empirical Coverage | Mean Width (°C) | Spearman $r$ (Width vs Error) |
|---|---:|---:|---:|
| **Constant-Width Conformal** | 90.33% | 4.204 | 0.000 |
| **Normalized Conformal** | 86.44% | 3.502 | 0.419 |
| **Conformalized Quantile Regression (CQR)** | **91.19%** | **3.679** | **0.470** |

CQR adapts interval widths positively with true error (`r=0.470`) and covers load regimes (`Idle 90.85%`, `Active 93.33%`, `Heavy 98.53%`), `Cool <50°C` (`92.24%`), and `Hot >=60°C` (`92.67%`), while under-covering slightly in `Moderate 50–60°C` (`89.09%`). Per-server coverage across `N=15` held-out hosts ranges from `84.7%` to `96.9%` (median `91.6%`, `40.0%` of servers `<90%`), and weekly temporal coverage drifts from `88.26%` (Week 18) to `92.21%` (Week 23) due to **operational drift** (ambient weather and HVAC cooling units are not recorded; cause untested).

### 4.5 RQ5: Cross-Cluster Hardware Adaptation (Cluster 1 → Cluster 2)

Transferring a Cluster 1 planning model to `N=12` Split C held-out target servers in Cluster 2 (`docs/results_v3.md` §5; `docs/results_v4_addendum.md` §2):
- **Zero-Shot Transfer:** `6.85°C` MAE (wide 95% bootstrap CI `[2.92°C, 13.47°C]`). Under the earlier v2 chronological Split A protocol across all `59` qualifying C2 servers (pooled row-weighted MAE, early-stopped C1 model), zero-shot MAE was lower (`4.88°C` with v2's `lags=[1,2,6]` set; `4.97°C` under v3 features), whereas evaluating v2's model on the same `12` Split C servers after the 174-hour buffer yields `6.67°C`. The difference reflects unseen-server selection and unweighted per-server aggregation rather than a modeling regression.
- **Residual Offset Adaptation:** Drops bias rapidly to `3.79°C` (`1 h`, CI `[2.40, 5.27]`), `3.16°C` (`6 h`, CI `[2.13, 4.30]`), `2.92°C` (`72 h`, CI `[1.64, 4.81]`), and `3.03°C` (`168 h`).
- **Continued Boosting:** Remains higher (`6.07°C` at `24 h`; `6.05°C` at `168 h`) because tree split constraints resist global baseline shifts across heterogeneous clusters.

---

## 5. Predictive Digital Twin & 3D Prototype (`twin3d`)

Beyond offline experiment scripts, the trained models are deployed in an **offline, data-driven predictive digital twin prototype** (`v2.1-twin3d` → `v2.4-evidence`) backed by `TwinEngine` (`src/ttwin/twin/engine.py`), saved LightGBM point + CQR quantile models (`models/twin/`), and precomputed 30-minute stride test caches (`550,848` D1 rows; `205,671` D2 rows).

### 5.1 Deployed Twin Model Performance & Planning Calibration (`models/twin/`, `docs/planning_calibration_eval.md`)

| Cluster | Regime | Horizon | Model MAE (°C) | Persistence MAE (°C) | Static CQR Cov | Mean Width (°C) |
|---|---|---|---:|---:|---:|---:|
| **D1** (76 hosts) | Monitoring (S1) | 10 min (`h=1`) | 0.4023 | 0.4025 | 89.96% | 1.6578 |
| **D1** (76 hosts) | Monitoring (S1) | 1 h (`h=6`) | 0.8371 | 0.8394 | 90.00% | 3.3510 |
| **D1** (76 hosts) | Monitoring (S1) | 6 h (`h=36`) | 1.6069 | 1.5991 | 88.14% | 6.7149 |
| **D1** (76 hosts) | Planning (Workload) | Contemporaneous | 5.4844 | — | 81.09% (89.36% roll-7d) | 15.8726 (18.402 roll-7d) |
| **D2** (60 hosts) | Monitoring (S1) | 10 min (`h=1`) | 0.4845 | 0.5014 | 89.79% | 2.0820 |
| **D2** (60 hosts) | Monitoring (S1) | 1 h (`h=6`) | 0.7807 | 0.8018 | 89.31% | 3.5339 |
| **D2** (60 hosts) | Monitoring (S1) | 6 h (`h=36`) | 1.2696 | 1.2817 | 86.15% | 7.2528 |
| **D2** (60 hosts) | Planning (Workload) | Contemporaneous | 3.8926 | — | 85.33% (89.09% roll-7d) | 17.8727 (19.079 roll-7d) |

Because static planning CQR under-covers the `90%` target (`81.09%` D1, `85.33%` D2) under temporal drift and regime-wise `q_hat` (`idle`/`active`/`heavy`) fails to restore coverage (`81.10%` D1, `81.52%` D2), causal **7-day rolling recalibration (`rolling_7d`)** raises test coverage to `89.36%` on D1 (`18.402°C`, `1.16x` width) and `89.09%` on D2 (`19.079°C`, `1.07x` width), and is integrated as an opt-in `TwinEngine` calibration mode (`models/twin/planning_calibration.json`). *(Honest note: D1 6-hour monitoring MAE `1.6069°C` trails persistence slightly at `1.5991°C`.)*

### 5.2 Architecture, Frame Rate & Alert Evaluations (`docs/twin3d_validation.md`, `docs/twin3d_fps_gpu.md`)
1. **Local FastAPI Service (`src/ttwin/twin3d/server.py`):** Exposes `/api/meta`, `/api/snapshot` (p50 `35.74 ms`, p95 `48.03 ms` on D1), `/api/compare` (strict causal lookup matching target time `t` against forecast issued at `t - h`, reporting `"missing"` rather than `"ok"` when observations are absent), `/api/series`, `/api/alerts`, `POST /api/whatif`, and `/api/health`. Serving feature construction enforces 10-minute grid alignment (`fill_missing_timestamps`) to match training.
2. **Offline Three.js Visualisation (`twin3d/web/`, vendored `three@0.164.1`):** Schematic 42U rack rows (`C1-R1..R4`, `C2-R1..R3`) rendered with instanced meshes (`13` WebGL draw calls in compare mode, `0` external CDN dependencies, `0` console errors offline), CQR uncertainty halos, dual-scissor compare mode (across 3 TEST-period days at `h=6`: MAE `0.50–0.64°C` / `90.79–97.37%` inside on D1; `0.51–0.89°C` / `87.27–90.91%` on D2), live planning what-if ghost predictions (`"predictive, correlational - not causal"`), and a Server Anatomy view (`docs/twin3d_screens/`). In 30-second compare-mode benchmarks (`docs/twin3d_fps_gpu.md`), headed GPU Chromium (`ANGLE Intel UHD Graphics D3D11`) sustains **60.0 median FPS** (`60.0–61.0` FPS) vs **6.0 median FPS** under headless software rendering (`SwiftShader`). Hosts without fan/inlet sensors (Cluster 2) display static grey parts explicitly labelled `"no fan sensor"` and `"no inlet sensor"`. (The earlier illustrative `app/streamlit_app.py` mock has been retired to a deprecation stub.)
3. **Pre-Registered Hotspot Early-Warning Evaluation (`docs/twin_hotspot_evaluation.md`):** On real TEST crossings from below `W` (`73.93°C` D1, `75.72°C` D2) to `>= W` (all 4 cluster-horizon pairs conclusive with `n_events = 758`–`1,790`), matching validation alert rates via `val_sel + val_cal` thresholds:
   - **D1 (`+1h`, `758` events):** Twin achieves `63.5%` recall / `17.3%` precision (`1.54%` test alert rate) vs Persistence `47.1%` / `9.2%` (`2.14%`) and Trend `34.8%` / `7.0%` (`2.08%`).
   - **D1 (`+6h`, `1,328` events):** Twin achieves `50.4%` recall / `11.6%` precision (`3.20%`) vs Persistence `45.4%` / `9.8%` (`3.40%`) and Trend `17.5%` / `4.1%` (`3.11%`).
   - **D2 (`+1h`, `1,098` events):** Twin achieves `89.3%` recall / `19.2%` precision (`8.37%`) vs Persistence `82.5%` / `17.6%` (`8.45%`) and Trend `63.5%` / `12.9%` (`8.82%`).
   - **D2 (`+6h`, `1,790` events):** Twin achieves `85.1%` recall / `18.6%` precision (`13.53%`) vs Persistence `81.6%` / `18.0%` (`13.39%`) and Trend `36.4%` / `7.7%` (`14.03%`).
4. **Synthetic Early-Warning Threshold-Crossing Evaluation (`docs/twin_alert_evaluation_v2.md`):** Framed strictly as **early-warning threshold crossing, not a fault detector** (**SYNTHETIC FAULTS — not real-world detection accuracy**), across `100` clean TEST windows per cluster with baselines calibrated to the Twin's exact `clean_control` false-alarm rate (`15.0%` D1; `0.0%` D2), the Twin detects `100.0%` of threshold-crossing windows across all 6 perturbation profiles, whereas matched-FPR `Rolling z-score` and `One-step change` baselines achieve `0.0%`–`61.5%` on gradual ramps (`ramp_6C_2h`, `ramp_10C_6h`) and `0.0%`–`80.0%` on slow drift (`drift_0.5C_per_h`).

---

## 6. Threats to Validity & Limitations

1. **Telemetry & Infrastructure Blind Spots:** The trace lacks rack coordinates, CRAC/CRAH cooling telemetry, indoor/outdoor weather sensors, and per-VM placement details. The 3D hall layout is schematic, and airflow streaks/fan caps (`12 rad/s`) are visual encodings rather than CFD physics.
2. **Open-Loop Digital Shadow:** The system performs offline historical replay without live ingestion or physical actuation (`no cooling control`). Planning what-if responses are strictly correlational.
3. **Short-Horizon Persistence & Modest Load-Aware Planning Gains:** Monitoring skill gains over persistence are small or cross zero at the server level; planning MAE remains `3.89–5.48°C` without thermal history, and zero-shot `S2` improves over simple `CPU_Load` Ridge/bin-mean baselines by `+11.1%` to `+11.7%` overall (per-server CIs cross zero).
4. **Conformal Shift Sensitivity & Early-Warning Scope:** Static conformal validity erodes under operational drift (`81.09%` D1, `85.33%` D2 planning coverage), requiring rolling 7-day causal recalibration (`89.36%` D1, `89.09%` D2) while still lacking unrecorded ambient/HVAC covariates. Cross-cluster Split C transfer covers `N=12` held-out Cluster 2 hosts with wide variance (`[2.92°C, 13.47°C]`), and synthetic perturbation benchmarks evaluate early-warning threshold crossing rather than real-world physical fault diagnosis.

---

## 7. Conclusion

Across ~4.03 million telemetry records evaluated without temporal or server-identity leakage, short-horizon server thermal prediction tracks persistence closely (`~0.40–0.48°C` at 10 min; `~0.78–0.84°C` at 1 h), whereas workload-only planning suffers `3.89–5.48°C` error and exhibits `17.6%` evaluation inflation under naive random $k$-fold splits. Zero-shot pooled models cut planning error by `43.2%` compared to unconditioned fleet averages, while CQR provides adaptive uncertainty intervals (`91.19%` under Split C) integrated end-to-end into an offline, honest 3D predictive digital twin prototype.

---

## References
1. Ilager, S., Ramamohanarao, K., & Buyya, R. (2021). Thermal prediction for efficient energy management of clouds using machine learning. *IEEE Transactions on Parallel and Distributed Systems*, 32(5), 1044–1056.
2. Ilager, S., Toosi, A. N., Jha, M., Brandic, I., & Buyya, R. (2023). A data-driven analysis of a cloud data center: Statistical characterization of workload, energy and temperature. In *IEEE/ACM International Conference on Utility and Cloud Computing (UCC)*.
3. Veigas, C., Chinnici, M., De Chiara, M., & Chinnici, A. (2025). Towards energy efficiency of HPC data centers: A data-driven analytical visualization dashboard prototype approach. *Electronics*, 14(16), 3170.
4. Borghesi, A., et al. (2023). M100 ExaData: A data collection campaign on the CINECA’s Marconi100 Tier-0 supercomputer. *Scientific Data*, 10(1), 288.
5. Seyedkazemi Ardebili, M., et al. (2024). HazardNet: A thermal hazard prediction framework for datacenters. *Future Generation Computer Systems*, 155, 340–353.
6. Brännvall, R., Gustafsson, J., & Sandin, F. (2023). Modular and transferable machine learning for heat management and reuse in edge data centers. *Energies*, 16(5), 2255.
7. Angelopoulos, A. N., & Bates, S. (2021). A gentle introduction to conformal prediction and distribution-free uncertainty quantification. *arXiv:2107.07511*.
8. Romano, Y., Patterson, E., & Candès, E. (2019). Conformalized quantile regression. *NeurIPS*.
