# DEMO — 3-Minute Live Demo Script

> [!CAUTION]
> **Do NOT re-run experiments.** Figures come from pre-computed CSVs. Only run the three commands below.

---

## Step 1 — Run the test suite (≈ 30 s)

```powershell
cd d:\mainel5 && python -m pytest tests/ -q
```

**Expected output:**
```
25 passed in ~3s
```

Demonstrates: 25/25 regression tests pass, covering splits, leakage guards, and metric correctness.

---

## Step 2 — Regenerate all figures (≈ 60 s)

```powershell
cd d:\mainel5 && python scripts/phase2/make_pack.py
```

**Expected output:**
```
Generated 10 figures in docs/phase2/
```

Demonstrates: full pipeline from pre-computed CSVs → publication-ready PNGs + caption .txt files, without re-running any experiments.

---

## Step 3 — Browse the output (≈ 60 s)

Open `docs/phase2/` in Windows File Explorer:

- **10 PNG figures**: fig01_pipeline.png through fig10_shap.png
- **10 caption .txt files**: one alongside each figure naming the experiment id
- **5 markdown files**: tools_and_techniques.md, plan_vs_actual.md, rubric_map.md, slides_text.md, DEMO.md

Point out figure titles, axis labels (all with units), and the `PRELIMINARY` label printed on results figures.

---

> [!NOTE]
> Commit b4b9378 is the snapshot baseline for the 25-test suite. Git tags: v0.1-audit, v0.9-pre-fix, v1.0-corrected.
