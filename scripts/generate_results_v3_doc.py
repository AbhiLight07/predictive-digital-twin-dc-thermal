import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "experiments" / "v3_experiments"

t1 = pd.read_csv(OUT_DIR / "e1_bootstrap_summary.csv")
t2 = pd.read_csv(OUT_DIR / "sensor_ablation.csv")
t3 = pd.read_csv(OUT_DIR / "split_b_and_c_horizons.csv")
t4 = pd.read_csv(OUT_DIR / "planning_adaptation_all_folds.csv")
t5 = pd.read_csv(OUT_DIR / "cross_cluster_adaptation.csv")
t6 = pd.read_csv(OUT_DIR / "random_vs_group_identical_rows.csv")
t7 = pd.read_csv(OUT_DIR / "s1_learning_rate_study.csv")

md_content = f"""# Results v3 — Comprehensive Gap-Corrected Experiments

> [!IMPORTANT]
> All results in this report are strictly **preliminary**. Telemetry is processed with continuous 10-minute grid alignment (NaN insertion for missing intervals) to guarantee causal, gap-free lag and horizon shifts.

---

## 1. Paired Bootstrap Analysis for Skill vs Persistence (E1)

Evaluated at 1-hour forecast horizon on Split A (temporal holdout with 6-hour embargo). 95% bootstrap confidence intervals computed over $B=2,000$ resamples across physical servers and across calendar weeks.

{t1.to_markdown(index=False)}

* **What this shows:** On both clusters, S1 LightGBM achieves statistically confirmed, but tiny positive skill over persistence (+0.59% on D1, +0.33% on D2). Server-level 95% CIs are bounded within [0.03%, 1.15%], proving predictive superiority is real but practically marginal.
* **What this does NOT show:** This does not show double-digit error reductions; thermal inertia accounts for >99% of short-term thermal variance.

---

## 2. Sensor Ablation: Isolating Hardware Features (Dataset 1, 1-hour Horizon)

Evaluated on strictly identical test rows across feature sets with fans, power, inlet temperature, and thermal lags.

{t2.to_markdown(index=False)}

* **What this shows:** Removing cooling fans and server power features from full S4 telemetry changes test MAE by less than 0.003°C. Workload and thermal inertia provide all informative variance.
* **What this does NOT show:** This does not show fan telemetry is useless in real facilities; in this trace, fan curves operate autonomously under firmware PID without facility feedback.

---

## 3. Split B (GroupKFold) and Split C across Horizons (with 5 Random Server Draws)

Split B evaluates complete 5-fold cross-validation across all servers. Split C evaluates joint spatiotemporal holdout across 5 distinct random seed draws (seeds: 42, 123, 456, 789, 1024).

{t3.to_markdown(index=False)}

* **What this shows:** GroupKFold MAE scales predictably across horizons (10min: 0.43°C, 1h: 0.92°C, 6h: 1.82°C). Split C with 5 random seed draws confirms that spatial holdout variance across seeds is modest (std ~0.10°C to 0.22°C).
* **What this does NOT show:** This does not show that spatial generalization is trivial; test error increases when predicting entirely unseen physical servers.

---

## 4. Fleet-Wide Planning Adaptation Across All 5 Folds ($N=81$ Target Servers)

Evaluates cold-start adaptation on every physical server in Dataset 1 across GroupKFold splits with local training budgets from 0 to 168 hours.

{t4.to_markdown(index=False)}

* **What this shows:** Zero-shot planning achieves 5.03°C MAE (95% CI: [4.71°C, 5.39°C]) vs Fleet Mean (8.87°C), a 43.2% error reduction. Continued boosting improves performance to 4.87°C at 168 hours. Residual offset improves from 5.52°C (1h) to 5.02°C (168h).
* **What this does NOT show:** This does not show that longer adaptation budgets monotonically improve performance on every server; individual drift patterns vary across nodes.

---

## 5. Cross-Cluster Hardware Adaptation (Cluster 1 $\\rightarrow$ Cluster 2)

Source model trained on Dell PowerEdge R840 (Cluster 1) adapted to Dell PowerEdge C6320 (Cluster 2) target servers.

{t5.to_markdown(index=False)}

* **What this shows:** Zero-shot transfer from C1 to C2 achieves 6.85°C MAE. Local residual offset adaptation rapidly cuts error down to 3.16°C at 6 hours and 2.92°C at 72 hours, demonstrating that global workload physics transfer well once local thermal baseline offsets are calibrated.
* **What this does NOT show:** This does not show thermodynamic equivalence; chassis impedance differences require local offset alignment.

---

## 6. Random K-Fold vs GroupKFold on Identical Telemetry Rows

Direct comparison on strictly identical test rows to measure data leakage inflation from random temporal splitting.

{t6.to_markdown(index=False)}

* **What this shows:** In planning models (S2, S3), Random K-Fold artificially deflates error by 17.6% to 18.5% compared to GroupKFold by leaking temporal autocorrelation across adjacent 10-minute records. In monitoring models (S1, S4), persistence anchoring restricts the leakage gap to <0.6%.
* **What this does NOT show:** This does not imply GroupKFold is pessimistic; GroupKFold correctly reflects production deployment to new, unseen server hardware.

---

## 7. S1 LightGBM Learning Rate & Early Stopping Convergence

Assessing whether S1's early convergence (~5 trees) at learning rate 0.05 is an artifact of step size.

{t7.to_markdown(index=False)}

* **What this shows:** Reducing learning rate from 0.05 to 0.005 extends tree growth from 5 trees to 49 trees, with validation loss stabilizing at 0.823°C. Test MAE shifts minimally from 0.8277°C to 0.8276°C (skill moves from +0.59% to +0.60%).
* **What this does NOT show:** This does not change the core substantive finding: monitoring models remain anchored to thermal persistence regardless of gradient step size.
"""

with open(ROOT / "docs" / "results_v3.md", "w", encoding="utf-8") as f:
    f.write(md_content)

print("Generated docs/results_v3.md successfully.")
