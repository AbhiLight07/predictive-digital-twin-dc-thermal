# State Snapshot Evidence Pack Index

This directory (`docs/snapshot/`) contains an immutable, read-only evidence pack auditing every aspect of the Predictive Digital Twin thermal forecasting repository as of commit `a41aba0` and tag `v1.0-corrected`.

---

## Index of Evidence Pack Documents

- `00_README.md`: Master index and overview of all snapshot audit documents.
- `01_repo_and_process.md`: Raw Git status, full commit log, tags, branches, remote status, tracked file counts, and forbidden file safety checks.
- `02_files_and_tests.md`: Repository file tree with sizes and line counts, full pytest output (25/25 passed), Python/package versions, and test invariant assert quotes.
- `03_data.md`: Raw file SHA256 checksums, dataset dimensions, host ID whitespace audit, verbatim `columns.yaml`, missingness breakdown, and canonical analysis server sets.
- `04_protocol_and_splits.md`: Verbatim protocol documents, exact code-generated partition bounds for Splits A/B/C across both datasets, held-out server IDs, and proof of zero test leakage.
- `05_critical_code.md`: Verbatim code listings for target formulation, feature sets S0–S4, splitters, persistence and planning baselines, adaptation functions, CQR conformal inference, and fault injection alert rules.
- `06_experiment_manifest.csv`: Machine-readable catalog of all 37 experiment folders across legacy, diagnostic, and v2 runs with hyper-parameters, row counts, and headline metrics.
- `06_experiment_manifest.md`: Chronological narrative accompanying the experiment manifest, classifying legacy vs v2 runs and detailing config traceability.
- `07_claims_ledger.md`: Item-by-item audit of 32 numerical and comparative claims across results docs, paper draft, and dashboard, classified as supported, superseded, or unsupported.
- `08_prompt_checklist.md`: Comprehensive milestone audit tracking all requirements from `PROJECT_BRIEF.md`, Diagnostic Prompt, Fix-and-Re-Run Prompt, and Validate-and-Complete Prompt.
- `09_dashboard.md`: Audit of the Streamlit application (`app/streamlit_app.py`) verifying operational views, run commands, disclaimers, and identifying hard-coded heuristic logic.
- `10_literature.md`: Complete audit of all 28 bibliographic references in `docs/literature.md` and paper draft, reporting factual verification statuses and flagging suspect entries.
- `11_repro_check.md`: End-to-end verification reproducing Split A 1-hour persistence ($0.8408^\circ\text{C}$) and S1 LightGBM ($0.8359^\circ\text{C}$, skill $+0.57\%$) from raw CSV in 57.5 seconds in a clean virtual environment.
- `12_known_issues.md`: Adversarial self-assessment detailing known and suspected bugs, unsupportable claims (deflating planning gain from 41.7% to 19.2%), unverified assumptions, and top five reviewer checks.
- `server_sets/`: Directory containing canonical CSV server rosters for all 10 experimental configurations across Datasets 1 and 2.
