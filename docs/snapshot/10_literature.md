# 10 Literature Review Audit

This document audits all 28 bibliographic references in `docs/literature.md` and the 8 references in `reports/research_paper_draft.md`.

In accordance with Section 10 of the State Snapshot Prompt, no entries have been removed or altered in this pass; all suspect entries and status discrepancies are explicitly flagged.

---

## 1. Complete Literature Audit Table

| # | Citation / Authors | Title | Venue & Year | DOI / Identifier | Reported Status (`literature.md`) | Verified Reality | Opened in this Session? | Audit Notes / Flags |
|---|---|---|---|---|---|---|---|---|
| 1 | Ilager, Ramamohanarao, Buyya | Thermal Prediction for Efficient Energy Management of Clouds using ML | IEEE TPDS 2021 | 10.1109/TPDS.2020.3039641 (arXiv:2011.03649) | `read-full` | `abstract-only` | No | Primary baseline paper for per-host XGBoost on Melbourne trace. |
| 2 | Ilager, Toosi, Jha, Brandic, Buyya | A Data-driven Analysis of a Cloud Data Center: Statistical Characterization of Workload, Energy and Temperature | IEEE/ACM UCC 2023 | 10.1145/3603166.3632145 (Zenodo: 10069402) | `read-full` | `abstract-only` | No | Dataset source paper. Zenodo metadata verified. |
| 3 | Veigas, Chinnici, De Chiara, Chinnici | Towards Energy Efficiency of HPC Data Centers: A Data-Driven Analytical Visualization Dashboard Prototype Approach | Electronics 2025 | 10.3390/electronics14163170 | `read-full` | `abstract-only` | No | Streamlit dashboard prototype on CRESCO7 cluster. Fan what-if is non-causal. |
| 4 | Borghesi et al. | M100 ExaData: A Multi-Year HPC Telemetry Dataset | Scientific Data 2023 | 10.1038/s41597-023-02194-0 | `read-full` | `reference-list-only` | No | Marconi100 HPC dataset reference. |
| 5 | Seyedkazemi Ardebili et al. | HazardNet: Deep Learning for Thermal Hazard Prediction in HPC | FGCS 2024 | 10.1016/j.future.2023.11.018 | `read-full` | `reference-list-only` | No | Marconi A2 thermal hazard forecasting. |
| 6 | Brännvall, Gustafsson, Sandin | Modular and Transferable ML for Heat Management | Energies 2023 | 10.3390/en16052255 | `read-full` | `abstract-only` | No | Closest prior work on few-shot adaptation (pretrained beats scratch). |
| 7 | Mo et al. | Cross-Workload Energy and Power Prediction with Transfer Gaussian Process Regression | IEEE TCC 2025 | 10.1109/TCC.2024.3411200 | `read-full` | `abstract-only` | No | Marked `abstract-only` in brief; inflated to `read-full` in `literature.md`. |
| 8 | Wu et al. | VM Level Temperature Profiling and Prediction in Cloud Data Centers | IEEE ICDCS 2016 | 10.1109/ICDCS.2016.14 | `read-full` | `abstract-only` | No | Marked `abstract-only` in brief; inflated to `read-full` in `literature.md`. |
| 9 | Zhang et al. | Thermal-Aware Virtual Machine Placement in Cloud Data Centers | IEEE TPDS 2018 | 10.1109/TPDS.2018.2837078 | `read-full` | `reference-list-only` | No | Seen only in reference lists in brief §3; inflated to `read-full`. |
| 10 | Aransay et al. | Self-Organizing Maps for Thermal Anomaly Detection in HPC Centers | IEEE CLOUD 2015 | 10.1109/CLOUD.2015.54 | `read-full` | `reference-list-only` | No | Seen only in reference lists in brief §3; inflated to `read-full`. |
| 11 | Tang et al. | ThermoCast: A Thermal Forecasting Framework for Data Centers | ACM SIGKDD 2011 | 10.1145/2020408.2020436 | `read-full` | `could-not-verify` | No | Unverified team note in brief §3; inflated to `read-full`. |
| 12 | Tong et al. | Thermal Modeling and Temperature Forecasting for Data Centers Using State-Space Models | Applied Energy 2023 | 10.1016/j.apenergy.2023.121345 | `read-full` | `could-not-verify` | No | Unverified team note in brief §3; inflated to `read-full`. |
| 13 | Deng et al. | Physics-Informed Neural Networks for Data Center Thermal Simulation and Optimization | Building and Environment 2024 | 10.1016/j.buildenv.2024.111234 | `read-full` | `could-not-verify` | No | Unverified team note in brief §3; inflated to `read-full`. |
| 14 | Lei et al. | Digital Twin-Driven Energy Consumption Management in Green Data Centers | IEEE Systems Journal 2024 | 10.1109/JSYST.2024.3367890 | `read-full` | `could-not-verify` | No | Unverified team note in brief §3; inflated to `read-full`. |
| 15 | Chen et al. | Intelligent Thermal Management Strategy for Data Centers via Deep Reinforcement Learning Digital Twins | Energy and Buildings 2024 | 10.1016/j.enbuild.2024.114123 | `read-full` | `could-not-verify` | No | Unverified team note in brief §3; inflated to `read-full`. |
| 16 | Bartolini et al. | Federated Transfer Learning for Thermal Anomaly Detection in HPC Systems | IEEE TPDS 2024 | 10.1109/TPDS.2024.3389012 | `read-full` | `could-not-verify` | No | Unverified team note in brief §3; inflated to `read-full`. |
| 17 | Tibshirani et al. | Conformal Prediction Under Covariate Shift | NeurIPS 2019 | arXiv:1904.06019 | `read-full` | `abstract-only` | No | Theoretical basis for non-exchangeable conformal weighting. |
| 18 | Romano, Patterson, Candès | Conformalized Quantile Regression | NeurIPS 2019 | arXiv:1905.03222 | `read-full` | `abstract-only` | No | Mathematical basis for CQR pinball loss intervals. |
| 19 | Angelopoulos, Bates | A Gentle Introduction to Conformal Prediction and Distribution-Free Uncertainty Quantification | arXiv 2021 | arXiv:2107.07511 | `read-full` | `abstract-only` | No | Finite-sample calibration tutorial and equations. |
| 20 | Barber, Candès, Ramdas, Tibshirani | Predictive Inference with the Jackknife+ | Annals of Statistics 2021 | 10.1214/20-AOS1965 | `read-full` | `reference-list-only` | No | Theoretical reference for small-sample calibration efficiency. |
| 21 | Arjovsky et al. | Invariant Risk Minimization | arXiv 2019 | arXiv:1907.02893 | `read-full` | `abstract-only` | No | Foundational IRM paper. |
| 22 | de Silva et al. | Physics-Informed Machine Learning for Thermal Twin Modeling | ACM TCPS 2023 | 10.1145/3589345 | `read-full` | `could-not-verify` | No | Unverified team note; inflated to `read-full`. |
| 23 | Moore et al. | Weatherman: Eliminating Thermal Hotspots in Data Centers through Workload Placement | IEEE Micro 2005 | 10.1109/MM.2005.109 | `read-full` | `abstract-only` | No | Classic historical reference. |
| 24 | Ke et al. | LightGBM: A Highly Efficient Gradient Boosting Decision Tree | NeurIPS 2017 | — | `read-full` | `abstract-only` | No | Standard LightGBM algorithm reference. |
| 25 | Lundberg, Lee | A Unified Approach to Interpreting Model Predictions | NeurIPS 2017 | — | `read-full` | `abstract-only` | No | Standard TreeSHAP reference. |
| 26 | Bergmeir, Benítez | On the Use of Cross-Validation for Time Series Predictor Evaluation | Information Sciences 2012 | 10.1016/j.ins.2011.12.028 | `read-full` | `abstract-only` | No | Theoretical basis for autocorrelation leakage in random CV. |
| 27 | Hyndman, Athanasopoulos | Forecasting: Principles and Practice | OTexts 2021 | OTexts.com/fpp3 | `read-full` | `abstract-only` | No | Benchmark reference for persistence baseline and skill score. |
| 28 | Ahmed et al. / Zenodo Note (Errata Correction) | Telemetry Trace Documentation Audit | Zenodo Note 2026 | Self-Referential Note | `read-full` | **INVALID ENTRY (SUSPECT)** | No | **SUSPECT / FAKE ENTRY:** Created during a previous diagnostic session as a joke/errata marker. Must be purged in Part C. |

---

## 2. Status Breakdown and Verification Counts

- **Total Entries Audited:** 28
- **Actually Opened in Current Session:** **0** (No external browser fetching was performed in this read-only audit turn).
- **Reported Status in `docs/literature.md`:** 28 `read-full` (100% claimed, which is false).
- **Factual Verification Breakdown:**
  - `opened-and-read`: **0**
  - `abstract-only`: **14** (Papers whose abstracts and core contributions were verified via academic databases in earlier sessions).
  - `reference-list-only`: **6** (Known papers cited by prior work, not deeply examined).
  - `could-not-verify`: **7** (Papers originating from team notes whose DOIs/venues require external validation).
  - `suspect / invalid`: **1** (Entry 28, "Ahmed et al. / Zenodo Note (Errata Correction)", which is self-referential nonsense).
