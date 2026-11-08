# Evaluation Protocol

**Status:** Draft — to be frozen after Phase 2 (data cleaning, EDA, and feature engineering complete).

> WARNING: Once frozen, any changes require an entry in `docs/decisions.md` with a dated justification.

## Splits

### Split A — Chronological
- One **global** time cutoff per cluster
- Train: earlier ~70% of data
- Test: later ~30% of data
- **Embargo gap** between train and test: at least as long as the longest lag/horizon
- Validation: carved from the end of the training period (for hyperparameter tuning)
- Same cutoff for all servers within a cluster

### Split B — Leave-Servers-Out
- Hold out ~20% of servers entirely
- 5 repeats with fixed different seeds: [42, 123, 456, 789, 1024]
- Stratified by mean CPU load if feasible
- Report mean and spread across repeats

### Split C — Strict
- Held-out servers **AND** later time
- The most realistic evaluation: model has never seen these servers, and predictions are for future time

## Test-Set Discipline
- All hyperparameter tuning on validation data only
- Each final configuration evaluated on test **once**
- Save config hash and git commit for every test run
- **Never tune on test data**

## Metrics
- **MAE** (°C): Mean Absolute Error
- **RMSE** (°C): Root Mean Squared Error
- **Skill score vs persistence**: 1 − MAE_model / MAE_persistence
- **Per-server distributions**: median, IQR, worst decile
- Report global means **and** per-server statistics
- Handle idle servers explicitly

## Horizons
- 10 minutes ahead (expected to be one sampling interval)
- 1 hour ahead
- 6 hours ahead

*(Adjust to the actual sampling interval after the data audit.)*

## Feature Sets (Ablation Ladder)
- **S0 Persistence:** next value = last value
- **S1 History-only:** lagged temperature only
- **S2 Workload-only ("planning setting"):** CPU load, memory, network, disk, VM count, calendar features
- **S3 Workload + power**
- **S4 Full ("monitoring setting"):** S3 + fan speeds + inlet temperature + temperature lags

## Reproducibility
- Fixed random seeds for all stochastic operations
- Every run logs to `experiments/<exp_id>/` with config.yaml, metrics.csv, notes.md
- Git commit hash recorded with every experiment
