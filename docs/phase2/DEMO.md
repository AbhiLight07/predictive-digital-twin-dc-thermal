# DEMO — Live Demo Script

> [!CAUTION]
> **Do NOT re-run experiments.** Figures come from pre-computed CSVs. Only run the commands below.

---

## Step 1 — Run the test suite (wall-clock: 8.7 s)

```powershell
python -m pytest tests/ -q
```

**Actual output:**
```
.........................                                                [100%]
25 passed in 4.94s
```

Demonstrates: 25/25 regression tests pass, covering splits, leakage guards, and metric correctness.

---

## Step 2 — Regenerate all figures (wall-clock: 16.2 s)

```powershell
python scripts/phase2/make_pack.py
```

**Actual output:**
```
Generating figures → docs/phase2/

  ✓  docs/phase2\fig01_pipeline.png
  ✓  docs/phase2\fig02_timeline.png
  ✓  docs/phase2\fig03_splits.png
  ✓  docs/phase2\fig04_bugs_before_after.png
  ✓  docs/phase2\fig05_skill_vs_persistence.png
  ✓  docs/phase2\fig06_planning_baselines.png
  ✓  docs/phase2\fig07_adaptation_curve.png
  ✓  docs/phase2\fig08_coverage_by_regime.png
  ✓  docs/phase2\fig09_cross_cluster.png
  ✓  docs/phase2\fig10_shap.png

── Summary ──
Succeeded (10/10): fig01_pipeline, fig02_timeline, fig03_splits, fig04_bugs_before_after, fig05_skill_vs_persistence, fig06_planning_baselines, fig07_adaptation_curve, fig08_coverage_by_regime, fig09_cross_cluster, fig10_shap
All 10 figures generated successfully.
```

Demonstrates: full pipeline from pre-computed CSVs → publication-ready PNGs + caption .txt files, without re-running any experiments.

---

## Step 3 — Run the gap audit (wall-clock: 2.8 s)

```powershell
python scripts/phase2/gap_audit.py
```

**Actual output:**
```
D1: expected=2776274, actual=2661621, gap_frac=0.041
D2: expected=1198195, actual=1198440, gap_frac=0.000
  D1 10min: corrupted target est. = 4.1%
  D1 1h: corrupted target est. = 12.4%
  D1 6h: corrupted target est. = 41.3%
  D2 10min: corrupted target est. = 0.0%
  D2 1h: corrupted target est. = 0.0%
  D2 6h: corrupted target est. = 0.0%

Persistence MAE (S0, Split A):
 dataset horizon      mae
dataset1   10min 0.402645
dataset1      1h 0.840763
dataset1      6h 1.602473
dataset2   10min 0.501216
dataset2      1h 0.804186
dataset2      6h 1.285580

Note: gap-excluded MAE is not computable without Parquet timestamp access.
All gap fractions are UPPER BOUNDS — some servers may have shorter legitimate spans.
```

Demonstrates: quantitative gap audit assessing horizon target corruption from missing timestamps.

---

## Step 4 — Browse the output

Open `docs/phase2/` in Windows File Explorer:

- **10 PNG figures**: `fig01_pipeline.png` through `fig10_shap.png`
- **10 caption .txt files**: 2-line plain English description and experiment id / source file alongside each figure
- **Markdown documentation**: `tools_and_techniques.md`, `plan_vs_actual.md`, `rubric_map.md`, `slides_text.md`, `DEMO.md`, `gap_audit.md`, `bugs_table.md`

Point out figure titles, real physical units on all axes (°C, %, hours, server count), and the `PRELIMINARY` stamp on results figures (fig04–fig10; omitted on schematic fig01–fig03).

---

> [!NOTE]
> Commit b4b9378 is the snapshot baseline for the 25-test suite. Git tags: v0.1-audit, v0.9-pre-fix, v1.0-corrected.
