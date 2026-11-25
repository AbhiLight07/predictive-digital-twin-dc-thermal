# SUPERSEDED — Do Not Cite

`experiments/v2_fault_injection/` (`scripts/test_fault_injection.py`) used **pseudo-predictions** rather than saved LightGBM model inferences over perturbed input features.

- **Do not cite** these results (including 20 min latency / 93.3% TPR / 0.0% FPR).
- **Replaced by:** `experiments/twin_alerts_synthetic_20261124` (`scripts/eval_twin_alerts.py`, `docs/twin_alert_evaluation.md`) and `docs/twin_alert_evaluation_v2.md`.
