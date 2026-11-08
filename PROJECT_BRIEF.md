# PROJECT BRIEF — Predictive Digital Twin for Data Centre Thermal Management

**Subtitle:** Generalisation, Limited-Data Adaptation and Uncertainty-Aware Server Temperature Prediction

> You are the engineering agent for a 4-student university research project. Read this whole file first, then start Phase 0 immediately. Do not ask questions unless a **STOP-AND-ASK checkpoint** (section 12) is reached. This file is the single source of truth: copy it into the repo as `PROJECT_BRIEF.md` and keep `docs/decisions.md` for anything you decide that is not written here.

---

## 0. Start now (first 60 minutes, no questions)

1. Create a project folder `predictive-digital-twin-dc-thermal/` (use the current workspace if it is empty).
2. `git init`, create the repo structure (section 8), a Python 3.10+ virtual environment, and `requirements.txt`.
3. Add `.gitignore` (must exclude `data/`, `*.parquet`, `*.csv` larger than 5 MB, `.venv/`, `__pycache__/`, `.env`, model binaries, notebooks checkpoints).
4. Save this brief as `PROJECT_BRIEF.md`, write a short `README.md`, and make the first commit.
5. Create the GitHub repository (section 9) and push.
6. Move to Phase 1 (data audit). If the dataset files are not in `data/raw/`, STOP and tell the user the exact download links (section 5).

---

## 1. Context

- **Who:** a team of four RVCE (Bengaluru) undergraduates (1 CS, 1 AI, 2 ISE), fifth-semester Experiential Learning project, ~8 weeks, alongside other coursework.
- **Constraints:** must map to an SDG (this project: **SDG 9 – Industry, Innovation and Infrastructure**); software-oriented; must build on existing papers with a valid research gap; final goal is a **paper submission** (realistic target: national/regional journal or conference — do not claim top-venue ambitions); must be impressive yet easily explainable.
- **Mentor requirement:** the work must be ground-level and use **real measured data**. **Never promise or pre-announce results.** Frame everything as research questions; report null or negative findings honestly.
- **SDG 9 link:** energy-efficient, resilient digital infrastructure. Servers that run hot raise cooling cost and failure risk; reliable thermal prediction supports efficient and resilient data centres. (A reported motivating statistic from the dataset paper — most machines under 20% CPU yet drawing a large share of peak power — must be re-verified before it is quoted.)
- **Ideas already rejected — do not revisit:** carbon-aware / data-centre job scheduling (already taken by others), renewable/off-grid data-centre forecasting dataset (provenance could not be verified), cooling optimisation and automatic control, federated dropout prediction, supply-chain alert prioritisation, industrial predictive maintenance.

---

## 2. Research problem

Machine-learning models can predict server temperature from operational telemetry, but published work mostly reports accuracy under evaluation setups that may be optimistic, trains one model per server, and gives no reliability estimate. In practice, new servers have little history, hardware differs across clusters, and operators need to know when a prediction can be trusted.

**Problem statement (use in all documents):**
> This project investigates how reliably server-temperature prediction models generalise to unseen servers and unfamiliar hardware, how much local data is needed to adapt a pretrained model to a new server, and whether prediction intervals can flag unreliable predictions. The models are integrated into a lightweight, offline, data-driven predictive digital twin for server-level thermal monitoring.

**Title (keep):** *Predictive Digital Twin for Data Centre Thermal Management.* Terminology rule: call the demo a **"predictive digital twin prototype (offline, data-driven)"** or "digital shadow with predictive capability". Never claim a production, real-time, bidirectional twin.

### Research questions
- **RQ1.** How does prediction performance change from chronological testing (seen servers, future time) to unseen-server testing, and how much does it depend on which sensors (power, fan speed, temperature history) are used as inputs?
- **RQ2.** Can a model trained on existing servers predict a new server's temperature with zero local data?
- **RQ3.** How much local data from a new server is needed to adapt a pretrained model, and does adaptation beat training from scratch on the same local data?
- **RQ4.** Are prediction intervals reliable (coverage) on unseen servers and unfamiliar conditions, and can a few hours of local data recalibrate them?
- **RQ5 (conditional).** Does a model transfer across the two hardware clusters when only shared, physically comparable features are used?

### Honest novelty position
Temperature forecasting, thermal-hazard prediction, data-centre digital twins and transfer learning for thermal models all exist already. **Never write "first" claims.** The contribution is the systematic combination: leakage-aware evaluation, measured data-amount curves for cold-start adaptation, and coverage of uncertainty under server-level shift, on production telemetry — and even that must be re-checked against the literature before claiming it.

---

## 3. Literature (status matters — never invent citations)

Maintain `docs/literature.md` as a table: citation | what it did | relevance | **verification status** (`read-full`, `abstract-only`, `reference-list-only`, `unverified`). Target 25–40 relevant papers by Week 4. Before citing anything, verify it exists via web search; mark anything you could not open as `unverified` and never quote numbers from it.

**Verified (checked directly by the human's assistant):**
1. Ilager, Ramamohanarao, Buyya — *Thermal Prediction for Efficient Energy Management of Clouds using Machine Learning*, IEEE TPDS 2021 (arXiv 2011.03649). `read-full`. Per-host XGBoost; 10-fold cross-validation with no mention of time-ordered splits; inputs include host power and fan speeds (fan speed is a controller response to heat — possible circularity, to be **tested** not assumed); models intended only for the data centre they were trained on; near-idle hosts produced extreme predictions, patched with value bounds and averaging other hosts' models. Also contains a thermal-aware scheduling part (out of our scope).
2. Ilager, Toosi, Jha, Brandic, Buyya — *A Data-driven Analysis of a Cloud Data Center: Statistical Characterization of Workload, Energy and Temperature*, UCC 2023. Dataset paper for our primary data. Reportedly random 70/30 splits, per-cluster models, next-interval CPU load and power prediction. `read-partial`; re-read the evaluation section.
3. Veigas, Chinnici, De Chiara, Chinnici — *Towards Energy Efficiency of HPC Data Centers: A Data-Driven Analytical Visualization Dashboard Prototype Approach*, Electronics 2025, 14(16), 3170. `read-full`. CRESCO7: 144 nodes, ~4 months, LightGBM, Streamlit dashboard, forecasts of cluster-wide averages, per-node models retrained on the fly (80/20), what-if sliders on fan speed and supply air, health score. Advises retraining on local data when hardware differs; **does not measure how much data is needed; uncertainty only as future work; data not public.** Has internal date/number inconsistencies — treat as a secondary source. Its fan-speed what-if is causally unsound (fan speed correlates *positively* with temperature because it responds to heat) — do **not** copy this.
4. Borghesi et al. — *M100 ExaData: a data collection campaign on the CINECA's Marconi100 Tier-0 supercomputer*, Scientific Data 10:288 (2023). 980+ nodes, 2.5 years, 49.9 TB uncompressed, monthly Zenodo entries in partitioned Parquet; companion code at gitlab.com/ecs-lab/exadata.
5. Seyedkazemi Ardebili et al. — *HazardNet: a thermal hazard prediction framework for datacenters*, FGCS 155 (2024) 340–353. Room/rack-level hazard prediction on Marconi A2 data; dataset and code public.
6. Brännvall, Gustafsson, Sandin — *Modular and Transferable Machine Learning for Heat Management and Reuse in Edge Data Centers*, Energies 2023, 16(5), 2255. `read-full`. Fine-tuning a pretrained physics-informed RNN on ~5 h of data beat training from scratch (lab testbed, six servers → one server in a wind tunnel). **Closest prior work on "pretrained beats scratch" — we must not claim that finding as new.**
7. Mo et al. — cross-workload power prediction with transfer Gaussian process regression, IEEE Trans. Cloud Computing 13(3), 2025. `abstract-only`.
8. Wu et al. — *Virtual Machine Level Temperature Profiling and Prediction in Cloud Datacenters*, ICDCS 2016. `abstract-only`.
9. Seen only in reference lists (find and verify before citing): Zhang et al., IEEE TPDS 2018 (ML temperature prediction for runtime thermal management); Aransay et al., IEEE CLOUD 2015 (self-organising maps for thermal anomalies).

**Unverified (came from the team's own notes — verify before use):** ThermoCast (2011); Tong et al. (2023, time-varying state-space rack temperature prediction); Deng et al. (2026, CFD + ML for a 180-rack server room); *Digital Twin-Driven Energy Consumption Management of Integrated Heat Pipe Cooling System for a Data Center* (2024); *An Intelligent Thermal Management Strategy for a Data Center Prototype Based on Digital Twin Technology* (2025); *Federated Transfer Learning for Anomaly Detection in HPC Systems* (2025/26).

**Search topics to cover:** server/rack temperature forecasting; thermal anomaly and hazard prediction; cross-server / cross-cluster generalisation; domain adaptation and transfer learning for regression; few-shot / cold-start time-series; conformal prediction and quantile regression under distribution shift; digital-twin thermal management; leakage and evaluation pitfalls in time-series ML.

---

## 4. Concrete research design

### 4.1 Prediction target
Primary: **CPU temperature** (per server; if two CPU sensors exist, use the maximum as the effective CPU temperature, and also keep each sensor). Secondary if available: inlet-to-CPU temperature difference. Horizons: **10 min, 1 h, 6 h** ahead (adjust to the real sampling interval — expected 10 min).

### 4.2 Feature sets (ablation ladder — run everything on each applicable rung)
- **S0 Persistence:** next value = last value (mandatory baseline; at 10 minutes it will be very strong — always report skill against it).
- **S1 History-only:** lagged temperature only.
- **S2 Workload-only ("planning setting"):** CPU load, memory, network, disk/data usage, VM count, calendar features (hour, weekday). No power, no fan, no temperature history. This is the only setting valid for what-if scenarios and for a brand-new server.
- **S3 Workload + power.**
- **S4 Full ("monitoring setting"):** S3 + fan speeds + inlet temperature + temperature lags (only where the columns exist; the second cluster reportedly lacks fan and inlet data — verify in the audit).

Rule: any feature that is a consequence of temperature (fan speed, possibly power) must be reported separately, never silently mixed with workload features.

### 4.3 Evaluation protocol (freeze in `configs/protocol.yaml` and `docs/protocol.md` BEFORE running any model; changing it afterwards requires an entry in `docs/decisions.md` with a reason)
- **Split A — chronological:** one **global** time cutoff per cluster (train earlier ~70%, test later ~30%), same cutoff for all servers, plus an **embargo gap** at least as long as the longest lag/horizon so no information leaks across the boundary. A validation block is carved from the end of training for tuning.
- **Split B — leave-servers-out:** hold out ~20% of servers entirely (stratified by mean load if possible); **5 repeats** with fixed different seeds; report mean and spread across repeats.
- **Split C — strict:** held-out servers **and** later time.
- **Test-set discipline:** all tuning on validation only; each final configuration is evaluated on test **once**; save the config hash and git commit for every test run. Never tune on test.
- **Metrics:** MAE and RMSE in °C; **skill score vs persistence** (1 − MAE_model / MAE_persistence); report **per-server distributions** (median, IQR, worst decile), not only global means. Handle idle servers explicitly (the earlier literature reported extreme predictions on near-idle hosts).
- **Seeds, logging, reproducibility:** fixed seeds; every run writes `experiments/<exp_id>/{config.yaml, metrics.csv, notes.md}` with the git commit hash.
- **Leakage tests (pytest, must pass in CI/commit):** no future timestamps in train for any test row; no server overlap in leave-servers-out; lag/rolling features computed only from the past; scalers/encoders fit on train only.

### 4.4 Models
Persistence, ridge/linear, LightGBM (primary), XGBoost, random forest (optional), small MLP or LSTM (optional, only if time allows). Pooled models use cluster/hardware indicators and static capacity features (cores, RAM) **but not server ID** for unseen-server tests. Keep it simple and explainable (SHAP on the best tree model for driver analysis).

### 4.5 Experiments (core = E1, E2, E4, E5; E3 conditional)
- **E1 — Honest baselines + sensor ablation (RQ1).** Run S0–S4 under splits A, B, C at all horizons. Replicate the "random k-fold, per-server model" protocol of the earlier literature as a *reference only*, to quantify how much random folds inflate accuracy.
- **E2 — Cold start (RQ2, RQ3).** Treat a held-out server as "new". Budgets of local data: **0 h (zero-shot), 1 h, 6 h, 24 h, 72 h** (and 7 days if available), taken from the start of the new server's test period; evaluate on the remaining later data. Compare: (a) scratch on local data only, (b) pooled zero-shot, (c) pooled + adaptation. Adaptation methods to try (tree models do not fine-tune like neural nets): continued boosting from the pooled model, a residual-correction model on top of the pooled prediction, and (optionally) neural fine-tuning. Repeat over many target servers and several start offsets; report data-amount curves with confidence bands.
- **E3 — Cross-cluster transfer (RQ5, conditional).** Train on one cluster and adapt/test on the other **only using shared features** (S2-type), and only if the temperature quantities are physically comparable (same sensor meaning, units, sampling). If not comparable, skip and document why.
- **E4 — Uncertainty (RQ4).** Quantile regression (LightGBM quantile objective) and split conformal prediction (e.g., conformalised quantile regression). Report empirical coverage at the nominal 90% level and mean interval width **per held-out server**, under chronological vs unseen-server testing, and test recalibration using the first few hours of the new server's data. State plainly that conformal guarantees assume exchangeability, which shift can break.
- **E5 — Twin demo.** A Streamlit app (section 7). Hotspot alerts use the interval's **upper bound** crossing a threshold. Because the data has no labelled anomalies, validate alerts only with **clearly labelled synthetic faults** (e.g., injected temperature offsets/drift) and qualitative case studies — never present synthetic detection scores as real-world accuracy.

---

## 5. Data

### 5.1 Primary dataset — University of Melbourne research-cloud trace (public)
- Zenodo record **10.5281/zenodo.10069402** — "A Data-driven Analysis of a Cloud Data Center: Statistical Characterization of Workload, Energy and Temperature" (Ilager, TU Wien). **Licence CC BY 4.0 — attribute and cite the UCC 2023 paper.**
- Files: `dataset1_combined.csv` (~613 MB) and `dataset2_combined.csv` (~184 MB). 10-minute samples, collection starting 2018-10-01. Two clusters on different hardware (reported as Dell R840 and Dell C6320); CPU load, memory, network, power, CPU temperatures; the first cluster reportedly also has fan speeds and inlet temperature, the second does not. **None of these details have been verified on the actual files — the audit must establish them.**
- **The human downloads these manually** (browser) into `data/raw/`; do not try to fetch them if the network blocks it; never commit them.

### 5.2 Optional scale-up — M100 ExaData (HPC, public)
Use at most **one month** (e.g., a single monthly Zenodo entry, ~33 GB) for an unseen-node replication across hundreds of nodes. **Do not download anything over 1 GB without explicit user approval.** First check licence, node-ID anonymisation structure, and metric names via the companion repo. M100 is an HPC supercomputer, not a commercial cloud — state this limitation.

### 5.3 The college server (Phase 2, not available yet)
The team will later get read-only access to a college server (possibly an HPE server with iLO). Currently available: historical CPU utilisation, data usage, and about one week of power data; **temperature and other hardware sensors only in real time**. Rules:
- It plays the role of the **"new server"** in the cold-start story and a deployment demo. It is a **case study, not a basis for generalisation claims** (one server, short history).
- **Do not run anything on that server until the user says so.** When asked, prepare `logger/` with a **read-only, passive** poller (iLO/Redfish or IPMI/`sensors`, whichever the user confirms) logging every 1–5 minutes to CSV: UTC timestamp, CPU utilisation, memory, network/disk traffic, power, inlet and CPU temperatures, fan speeds. Anonymise hostnames; no configuration changes; get the administrator's written permission and ask about publication of anonymised results.
- Verify that its temperature sensors are comparable to the public data's before attempting transfer.

---

## 6. Phases and acceptance criteria

| Phase | When | Work | Done when |
|---|---|---|---|
| 0 | Hour 0–1 | Repo, env, structure, first commit, GitHub push | Repo on GitHub with README and brief |
| 1 | Week 1 | Data audit (`scripts/data_audit.py`) on both files; write `docs/data_audit_report.md` | Report states servers, coverage, missingness, column meanings, gate verdict |
| 2 | Weeks 1–2 | CSV → Parquet (chunked), cleaning, EDA, `configs/columns.yaml`, features, splits + leakage tests, baselines S0–S2; freeze protocol | Tests pass; protocol committed |
| 3 | Weeks 3–4 | E1 complete (S0–S4, splits A/B/C, all horizons); literature table ≥ 25 papers | Tables + figures in `experiments/`, written interpretation |
| 4 | Weeks 4–6 | E2, E3 (if valid), E4 | Data-amount curves; coverage tables |
| 5 | Week 7 | E5 Streamlit twin; optional M100 replication; logger (only if user says) | Demo runs from README instructions |
| 6 | Week 8 | Paper draft (IEEE-style structure), limitations, reproducibility check, tag `v1.0` | Fresh clone reproduces key tables |

**Phase 1 gate (suggested thresholds):** at least **40 servers** with **≥ 60 days** of continuous coverage and **< 10% missing** in temperature and workload columns, and a licence that permits publication. If the gate fails, STOP and report; fallback options to evaluate (unverified): NREL ESIF HPC data-centre PUE data, IEEE DataPort DC-HVAC-DT23 cooling telemetry.

**Known pitfalls to handle:** very strong persistence baseline at 10 minutes; autocorrelation (use blocked splits, not random); duplicate or irregular timestamps; sensor dropouts and constant columns; idle servers and extreme predictions; units/time zones; unseen categories; small number of servers (report spread, not just means); fan/power circularity.

---

## 7. Predictive digital twin (demo layer — not the novelty)

Streamlit app in `app/`:
- Server selector; historical replay of load, power and temperature; predicted temperature with **prediction interval**; risk indicator (interval upper bound vs threshold); model/feature-set switch (monitoring vs planning).
- **What-if:** only **workload variables** (e.g., scale CPU load ±X%), labelled **"predictive, correlational — not causal"**. **No fan-speed or cooling-setpoint sliders. No automatic control of anything.**
- A visible "limitations" panel (offline data-driven prototype; trained on a specific data centre; intervals not guaranteed under shift).
- Keep it light: precomputed predictions are fine; target responsiveness under a few seconds.

---

## 8. Repository structure
```
predictive-digital-twin-dc-thermal/
  README.md            PROJECT_BRIEF.md       requirements.txt      .gitignore
  configs/             columns.yaml  protocol.yaml  experiments/*.yaml
  data/                (git-ignored) raw/ interim/ processed/  + data/README.md (how to fetch, checksums, licence/attribution)
  src/ttwin/           io.py clean.py features.py splits.py metrics.py models.py adapt.py uncertainty.py eval.py
  scripts/             data_audit.py  make_parquet.py  run_experiment.py
  experiments/         <exp_id>/ config.yaml metrics.csv notes.md  (small files only)
  app/                 streamlit_app.py
  docs/                literature.md  protocol.md  decisions.md  limitations.md  data_audit_report.md  paper/
  tests/               test_splits_no_leakage.py  test_features_no_future.py  test_metrics.py
  logger/              (Phase 5, only when the user approves)
  notebooks/           EDA only; results must come from scripts
```
Stack: Python 3.10+, pandas, numpy, pyarrow, scikit-learn, lightgbm, xgboost, shap, matplotlib, streamlit, pytest, pyyaml (add a conformal library only if it genuinely helps; custom CQR is acceptable). Read large CSVs in chunks, convert to Parquet, downcast dtypes.

`scripts/data_audit.py` already exists (the human has it): chunked per-server audit printing servers, coverage days, missingness and temperature-like columns; it guesses host/time columns and accepts `--host` and `--time`. Copy it into `scripts/` and fix it if it fails on the real files.

---

## 9. Git and GitHub rules
- Create the remote with the GitHub CLI if installed and authenticated: `gh repo create predictive-digital-twin-dc-thermal --private --source=. --push`. If `gh` is missing or not logged in, do `git init` and commit locally, then tell the user exactly what to run to authenticate or create the remote; do not stall other work.
- **Commit early and often** — after every logical unit (a module + its test, an audit report, a finished experiment, a doc update). Small commits; never leave work uncommitted at the end of a session. **Push after every commit or at least at the end of each work block.**
- Conventional messages: `feat: …`, `fix: …`, `test: …`, `docs: …`, `exp: …` (experiment result), `chore: …`. Example: `exp: E1 chronological baselines S0-S2 at 10min/1h/6h`.
- Branching: `main` stays runnable; use short-lived `feat/<topic>` branches for risky changes and merge when tests pass. Tag milestones: `v0.1-audit`, `v0.2-protocol`, `v0.3-e1`, `v0.4-e2-e4`, `v0.5-twin`, `v1.0`.
- **Never commit:** raw data, large Parquet/CSV (> ~5 MB), credentials, tokens, `.env`, hostnames or IPs of the college server. If something sensitive is committed by mistake, stop and tell the user.
- Add a LICENSE only after the user chooses one; keep the dataset attribution (CC BY 4.0) in `data/README.md`.

---

## 10. Working rules for the agent
1. **Be efficient:** start immediately; batch related work; don't ask permission for routine steps already covered here. Use approval prompts as the human's safety net, not as a reason to pause.
2. **Honesty:** no fabricated citations, numbers or results; label every uncertain statement; never write "first", "novel", or "state of the art" without evidence; report negative results.
3. **No tuning to the test set**; no changing the protocol after seeing results without logging why.
4. **Verify before trusting:** inspect actual column names and values — do not guess silently; record mappings in `configs/columns.yaml` and sanity-check units and ranges.
5. **Explain as you go:** the team is learning. Each experiment folder gets a `notes.md` in plain language: what was run, what the numbers say, what it does **not** show.
6. **Status reports:** at the end of every phase (and every long session) give a short report: done, key findings, problems, decisions needed, next step.
7. **Scope discipline:** no cooling control, no scheduling, no job placement, no deep-learning rabbit holes before the baselines are solid.

---

## 11. Deliverables
Public/private GitHub repo with reproducible code and tests; frozen protocol and results tables/figures; literature table (25–40 verified papers); Streamlit digital-twin prototype; limitations document; paper draft (Introduction, Related Work, Data, Method, Experiments, Results, Limitations, Conclusion); documents for the course (problem statement, literature review, methodology) when the user asks.

---

## 12. STOP-AND-ASK checkpoints (only these)
1. Dataset files missing from `data/raw/` → give the download link (https://zenodo.org/records/10069402) and wait.
2. After the Phase 1 audit → present the gate verdict and column findings; wait for the user's go-ahead before freezing the protocol.
3. Any download > 1 GB (including M100) or any use of the college server → ask first.
4. GitHub authentication needed → tell the user the exact command; continue local work meanwhile.
5. A required decision not covered here (licence, venue choice, scope change) → ask one clear question with a recommended default.
