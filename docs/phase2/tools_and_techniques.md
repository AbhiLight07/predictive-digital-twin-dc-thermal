# Tools and Techniques — Phase 2

> [!IMPORTANT]
> All results are **preliminary**. Numbers sourced only from actual experiment CSVs and codebase; none invented.

| Item | Why Chosen | Where Applied (file:function) | Evidence (experiment id) |
|------|------------|-------------------------------|--------------------------|
| pandas / pyarrow | Fast tabular IO; Parquet columnar storage reduces I/O overhead | `src/ttwin/io.py:load_parquet`, `io.py:load_all_servers` | v2_ablation (all CSVs) |
| scikit-learn (Ridge) | Linear baseline — no hyperparams to tune, fully interpretable coefficients | `src/ttwin/models.py:RidgeModel`, `models.py:get_model` | v2_ablation/results_split_a.csv (S1/S4 ridge rows) |
| LightGBM | Gradient-boosted trees; handles missing values natively; fast training on tabular data | `src/ttwin/models.py:LightGBMModel`, `models.py:get_model` | v2_ablation, v2_adaptation, v2_transfer, v2_uncertainty |
| XGBoost | Gradient boosting alternative | `src/ttwin/models.py:XGBoostModel` | defined, never run |
| SHAP (limited) | Post-hoc feature attribution; 1 000-sample budget keeps runtime tractable | `scripts/run_v2_e1_ablation.py` | v2_ablation/shap_summary.csv |
| matplotlib | Static, reproducible figure generation (200 dpi, 16:9) | `scripts/phase2/make_pack.py` | Phase 2 figures (fig01–fig10) |
| Streamlit | Offline prototype dashboard; rapid iteration without server deployment | `app/streamlit_app.py` | docs/snapshot/09_dashboard.md |
| pytest (25 tests) | Regression tests for splits, leakage, and metric correctness | `tests/test_*.py` | 25/25 pass, commit b4b9378 |
| Git / GitHub | Version control; 3 milestone tags anchor reproducibility | `.git/` | tags v0.1-audit, v0.9-pre-fix, v1.0-corrected |
| Google Antigravity | AI coding agent — all code produced with AI assistance and human review | all scripts | all experiment commits |
| Persistence baseline & Skill score | Lower bound reference; Skill = 1 − MAE_model / MAE_persist | `src/ttwin/metrics.py:skill_score_vs_persistence`, `metrics.py:persistence_predictions` | v2_ablation/results_split_a.csv |
| Delta target (ΔT) | Reduces autocorrelation in monitoring models; raw T(t+h) is highly autocorrelated | `src/ttwin/features.py:create_target` | v2_ablation (S1/S4 feature_set) |
| S0–S4 feature ladder | Controlled ablation of sensor/workload information; isolates contribution of each feature group | `src/ttwin/features.py:get_feature_set`, `features.py:build_features` | v2_ablation/results_split_a.csv |
| Chronological / GroupKFold / Strict splits + embargo | Prevent temporal and spatial leakage; 6 h embargo enforced at all boundaries | `src/ttwin/splits.py:split_chronological`, `splits.py:get_group_kfold_splits`, `splits.py:split_strict` | v2_ablation (splits A/B/C) |
| Monitoring vs Planning regimes | Operationally distinct: Monitoring — T(t) known; Planning — T(t) unknown | `scripts/run_v2_planning_adaptation.py` | v2_adaptation/summary_table.csv |
| Cold-start adaptation methods | Zero-shot, continued boosting, residual offset — cover realistic deployment scenarios (0–168 h of local data) | `src/ttwin/adapt.py:direct_transfer`, `adapt.py:adapt_ridge`, `adapt.py:adapt_lightgbm_finetune` | v2_adaptation/summary_table.csv |
| Constant / Normalised / CQR conformal intervals | Coverage validity under distribution shift; CQR achieves 91.2 % coverage vs 90 % target | `src/ttwin/uncertainty.py:conformalize`, `uncertainty.py:coverage`, `uncertainty.py:average_width` | v2_uncertainty/conformal_methods_comparison.csv |
| Leakage tests | Verify no test-set contamination at validation/test boundaries | `tests/test_splits_no_leakage.py:test_no_server_overlap_group_kfold`, `test_split_a_no_leakage` | 25/25 tests pass |
