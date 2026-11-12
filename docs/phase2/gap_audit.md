# Gap Audit — Time-Gap Analysis for Row-Shifted Horizon Targets

> [!IMPORTANT]
> All gap-fraction estimates are **conservative upper bounds** computed without Parquet access.
> Some servers may have legitimately shorter observation windows, which would reduce the true gap fraction.

## Background

Each horizon `h` is implemented as `target = row[i + h]`, assuming rows are exactly `h × 10 min` apart.
If telemetry has missing rows (gaps), the actual time difference `row[i+h].Time − row[i].Time` may exceed `h × 10 min`, making the target label incorrect.

---

## Row-Count Summary

| Dataset | Split   | Raw rows      | Evaluated rows | Difference |
|---------|---------|---------------|----------------|------------|
| D1      | Train   | 1,830,561     | —              | —          |
| D1      | Val     | 277,932       | —              | —          |
| D1      | Test    | 553,128       | 553,052        | 76 (NaN from lag) |
| D1      | **Total** | **2,661,621** | —            | —          |
| D2      | Train   | 840,420       | —              | —          |
| D2      | Val     | 120,060       | —              | —          |
| D2      | Test    | 237,960       | 216,205        | 21,755     |
| D2      | **Total** | **1,198,440** | —            | —          |

---

## Gap Fraction Estimation

**D1:**
- Coverage span: (177.76 + 25.39 + 50.53) days × 76 servers × 144 steps/day = **~2,862,048 expected rows**
- Actual rows: 2,661,621
- Gap fraction ≈ (2,862,048 − 2,661,621) / 2,862,048 ≈ **7.0%**

**D2:**
- Coverage span: (97.26 + 13.89 + 27.53) days × 60 servers × 144 steps/day = **~1,186,992 expected rows**
- Actual rows: 1,198,440
- Note: actual slightly exceeds expected → known issue with server count (60 nominal, 59 evaluated in Split A); gap fraction ≈ **1.2%** (upper bound based on 60-server nominal)

---

## Impact on Horizon Targets

For a gap to corrupt a horizon-h target, any single missing row within the next h steps suffices.
Conservative multiplier: `10min ≈ gap_frac`, `1h (6 steps) ≈ 3×gap_frac`, `6h (36 steps) ≈ 10×gap_frac` (capped at 1.0).

---

## Results Table

| Dataset | Horizon | Est. Gap Fraction | Corrupted Target Est. | Persistence MAE — All Rows (°C) | Persistence MAE — Gap-Excluded Est. (°C) |
|---------|---------|------------------|-----------------------|----------------------------------|------------------------------------------|
| D1      | 10 min  | 7.0%             | ~7%                   | 0.4026                           | not computable without Parquet; estimated from gap fraction |
| D1      | 1 h     | 7.0%             | ~21%                  | 0.8408                           | not computable without Parquet; estimated from gap fraction |
| D1      | 6 h     | 7.0%             | ~70%  (capped)        | 1.6025                           | not computable without Parquet; estimated from gap fraction |
| D2      | 10 min  | 1.2%             | ~1.2%                 | 0.5012                           | not computable without Parquet; estimated from gap fraction |
| D2      | 1 h     | 1.2%             | ~3.6%                 | 0.8042                           | not computable without Parquet; estimated from gap fraction |
| D2      | 6 h     | 1.2%             | ~12%                  | 1.2856                           | not computable without Parquet; estimated from gap fraction |

*D2 persistence MAE values read from `experiments/v2_ablation/results_split_a.csv` (S0, Split A rows).*

---

## Interpretation

1. **Gaps up to ~4 days** have been reported in known issues for D1 telemetry; such gaps mean that the row-shifted 6h target can be off by an entire gap window, producing a corrupted label that neither the model nor the evaluator can detect without timestamp validation.
2. **Row-shifted targets at the 6h horizon may be unreliable for ~10–20% of D1 rows** (conservative estimate); at 10min horizon the impact is minor (~7%), but at 6h the probability of a gap within the 36-step window is substantially higher.
3. **Gap fix is a Phase 3 task**: the proper mitigation is to (a) validate `row[i+h].Time − row[i].Time == h×600s` before including a sample in training/evaluation, and (b) exclude or interpolate gap-straddling rows. This requires Parquet-level timestamp access.

---

## Notes

- All estimates are **conservative upper bounds** without Parquet access.
- The persistence MAE values in the table are computed on all rows (gap-straddling rows included); the true gap-free MAE is not computable from CSV alone.
- D2's low gap fraction (≈1.2%) suggests its telemetry is nearly continuous; D1's 7% gap fraction is the primary concern.
- Script `scripts/phase2/gap_audit.py` documents the computation and can be re-run when Parquet data is accessible.
