# Slide Deck Text — Thermal Digital Twin for HPC Datacentres

> [!IMPORTANT]
> All results are **preliminary**. All numbers match experiment CSVs; none invented.

---

## Slide 1 — Title
**Rubric criterion:** *(title slide — no criterion)*

**Title:** Thermal Digital Twin for HPC Datacentres

**Bullets:**
- SDG 9: Industry, Innovation, Infrastructure
- Problem: predict server CPU temperature 10 min to 6 h ahead
- Goal: enable proactive energy management and fault prevention
- Datasets: Melbourne (D1 — 76 servers) and CRESCO (D2 — 60 servers)

**Speaker notes:**
> Introduce yourself and the project. Note that this is a Phase 2 interim submission; all results are labelled PRELIMINARY throughout. The two datasets come from two independent HPC sites, giving geographic diversity.

---

## Slide 2 — Topic Evolution
**Rubric criterion:** Execution — Methodology

**Bullets:**
- Original idea: ML-driven HPC job scheduler
- Literature search found that domain is already well-covered
- Pivot: thermal prediction digital twin — a complementary gap
- Kept from original plan: ML pipeline structure, HPC focus, Melbourne/CRESCO datasets
- Pivot approved; documented in `docs/snapshot/08_prompt_checklist.md`

**Speaker notes:**
> See `docs/phase2/plan_vs_actual.md` rows 0.1–1.2 for the transition. The pivot is a methodology strength, not a weakness — it shows literature awareness and scope management.

---

## Slide 3 — Methodology
**Rubric criterion:** Execution — Methodology

**Bullets:**
- CRISP-DM aligned: Business Understanding → Data → Modelling → Evaluation → Deployment
- 3-split protocol: A chronological (70/10/20%), B GroupKFold (5-fold, 76 servers), C strict spatio-temporal (61 seen / 15 unseen servers)
- 2 operational regimes: Monitoring (T(t) known) vs Planning (T(t) unknown)
- Cold-start adaptation: 0, 6, 24, 72, 168 h of local labelled data
- Uncertainty quantification: Constant, Normalised, and CQR conformal intervals

**Speaker notes:**
> Refer to `docs/snapshot/04_protocol_and_splits.md` for split details and `src/ttwin/splits.py` for implementation. The regime distinction was added after the diagnostic audit revealed a critical methodological conflation — this is recorded as a planned deviation in `docs/phase2/plan_vs_actual.md`.

---

## Slide 4 — Tools & Techniques
**Rubric criterion:** Tools — Selection / Application

**Bullets:**
- Models: LightGBM (gradient-boosted trees) + Ridge regression (interpretable linear baseline)
- Explainability: SHAP feature attribution (1 000-sample budget)
- Uncertainty: CQR + Normalised conformal intervals (`src/ttwin/uncertainty.py`)
- Testing: pytest — 25/25 tests pass in 2.58 s (commit b4b9378)
- Version control: Git with 3 milestone tags (v0.1-audit, v0.9-pre-fix, v1.0-corrected)
- AI assistance: Google Antigravity coding agent — human review applied to all outputs

**Speaker notes:**
> See `docs/phase2/tools_and_techniques.md` for the full table with justifications and evidence experiment IDs. Each tool choice is justified by a specific technical requirement, not convenience.

---

## Slide 5 — Pipeline & Integration
**Rubric criterion:** Tools — Integration

**Bullets:**
- 8-stage pipeline: Raw CSV → Clean & Convert → Feature Engineering → Splits A/B/C → Models → Evaluation → Uncertainty → Dashboard/Twin
- Each stage maps to a specific tool (see `docs/phase2/fig01_pipeline.png`)
- Full data-to-result traceability: every figure links to an experiment CSV
- Dashboard: Streamlit prototype (`app/streamlit_app.py`) — offline, known mock limitation

**Speaker notes:**
> Show `docs/phase2/fig01_pipeline.png`. Each box in the diagram names the tool responsible. `src/ttwin/*.py` (10 modules, 1 489 lines) implements stages 2–7; the Streamlit app implements stage 8.

---

## Slide 6 — Execution Timeline
**Rubric criterion:** Execution — Timeliness

**Bullets:**
- Repo scaffold → v0.1-audit tag → modules + 25 tests → diagnostic audit → v2 re-runs → v1.0-corrected → Phase 2 snapshot
- Key milestones: v0.1-audit (commit 04628a0), v0.9-pre-fix (commit d8c2cbc, 2026-11-08), v1.0-corrected (commit aa0e974, 2026-11-09), Snapshot (commit a6e99ee, 2026-11-12)
- Plan vs actual: 20/23 items DONE · 2 PARTIAL · 1 SKIPPED (server access not provided)
- Both skipped/partial items are external blockers, not implementation gaps

**Speaker notes:**
> Refer to `docs/phase2/fig02_timeline.png` for the visual timeline and `docs/phase2/plan_vs_actual.md` for the full status table. The two deviations (planning regime, budget units) both improved methodology.

---

## Slide 7 — Quality
**Rubric criterion:** Execution — Quality

**Bullets:**
- 4 bugs identified by diagnostic audit and fixed: Split A boundary overlap, feature deduplication (S4 dim: 57→56 cols), host trailing spaces, conformal calibration set contamination
- 25/25 leakage tests pass (see `tests/test_splits_no_leakage.py`)
- 6 h embargo enforced at all temporal split boundaries
- All result figures and tables labelled **PRELIMINARY**

**Speaker notes:**
> See `docs/phase2/fig04_bugs_before_after.png` for before/after metrics on each bug, and `docs/phase2/bugs_table.md` for the summary table. `docs/snapshot/12_known_issues.md` and `docs/diagnostics/FIX_REPORT.md` are the primary audit records. See also `docs/phase2/fig03_splits.png` for the corrected split boundaries.

---

## Slide 8 — Results: Monitoring & Planning
**Rubric criterion:** Partial Results

**Bullets:**
- **Monitoring (fig05):** S1/S4 LightGBM MAE 0.84–1.59 °C across horizons (D1, Split A); skill ~+0.5 % above persistence — close, feature engineering incomplete
- **Planning baselines (fig06):** Zero-shot LightGBM 4.83 °C vs cluster mean 8.29 °C vs Ridge(CPU) 5.98 °C (diagnostic subset, 16 servers, 1 h horizon)
- Gain over load-aware baselines: ~19–21 %
- No confidence intervals computed yet — further cross-validation pending

**Speaker notes:**
> Refer to `v2_ablation/results_split_a.csv` (monitoring) and `diag_part_a_q1_baselines.csv` (planning). The close monitoring skill is an honest finding — it motivates the Phase 3 feature engineering goal. All numbers are from pre-computed CSVs.

---

## Slide 9 — Results: Adaptation, Uncertainty & Transfer
**Rubric criterion:** Partial Results — Analysis

**Bullets:**
- **Adaptation (fig07):** Continued boosting reaches 4.80 °C MAE at 6 h of local data; stable thereafter (`v2_adaptation/summary_table.csv`)
- **Uncertainty (fig08):** CQR achieves 91.2 % empirical coverage at 3.68 °C interval width vs 90 % target (`v2_uncertainty/conformal_methods_comparison.csv`)
- **Cross-cluster transfer (fig09):** C1→C2 transferred model adds only +0.66 °C gap vs native training (4.88 vs 4.22 °C MAE) (`v2_transfer/cross_cluster_planning_results.csv`)
- CQR outperforms constant and normalised conformal on both coverage and width

**Speaker notes:**
> These three figures (`fig07`, `fig08`, `fig09`) together address research questions E2–E5. The small transfer gap (+0.66 °C) is encouraging for generalisation to new clusters without retraining from scratch.

---

## Slide 10 — Limitations & Next Steps
**Rubric criterion:** Partial Results — Clarity

**Bullets:**
- **Gap issue:** Time gaps up to ~4 days in telemetry may corrupt row-shifted horizon targets; gap audit (`docs/phase2/gap_audit.md`) documents the scope
- **Monitoring skill:** Close to persistence — feature engineering incomplete (Phase 3 priority)
- **Dashboard:** Prototype contains hard-coded mock data, not model-backed predictions (known issue)
- **Fault injection:** Synthetic only — real fault logs not available
- **Paper draft:** Uses legacy numbers → full rewrite scheduled for Phase 3

**Speaker notes:**
> Transparency about limitations is a methodological strength. The gap fix is the highest-priority Phase 3 action item because it affects all horizon targets. All limitations are documented in `docs/snapshot/12_known_issues.md`. The snapshot commit (a6e99ee, 2026-11-12) is the reference baseline for Phase 3.
