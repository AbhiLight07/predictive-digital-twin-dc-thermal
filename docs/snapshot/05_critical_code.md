# 05 Critical Code (Verbatim with Line Numbers)

## 1. Target Construction and Forecast Evaluation

File: `src/ttwin/features.py`, lines 61–74:
```python
def create_target(
    df: pd.DataFrame,
    target_col: str,
    horizon_steps: int,
    server_col: str = "Series",
) -> pd.DataFrame:
    """Create the prediction targets: absolute value and delta at horizon_steps."""
    df = df.copy()
    target_name = f"{target_col}_target_{horizon_steps}"
    delta_name = f"{target_col}_delta_{horizon_steps}"
    future_val = df.groupby(server_col)[target_col].shift(-horizon_steps)
    df[target_name] = future_val
    df[delta_name] = future_val - df[target_col]
    return df
```

How $T(t)$ is added back at inference for delta models:
File: `scripts/run_v2_e1_ablation.py`, lines 183–184:
```python
pred_lgb_target = booster.predict(X_test)
pred_lgb_abs = (y_test_curr + pred_lgb_target) if is_monitoring else pred_lgb_target
```

---

## 2. Feature Builders for S0–S4

File: `src/ttwin/features.py`, lines 93–140:
```python
def get_feature_set(feature_set: str, available_columns: List[str]) -> List[str]:
    """Return feature columns for a given feature set name.

    S0: empty (persistence baseline)
    S1: current temperature state at t (CPU, CPU1, CPU2) + lagged/rolling temperature
    S2: workload + calendar (planning setting, no temperature history)
    S3: S2 + power (planning setting with power)
    S4: S3 + fan + inlet + current temperatures + temperature lags/rolling (full monitoring)
    """
    workload_cols = [
        "CPU_Load", "Ram", "Ram_Used", "CPU_cores", "CPU_cores_used",
        "Network_RX", "Network_TX", "No_Of_Running_vms",
    ]
    calendar_cols = ["hour", "dayofweek", "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    power_cols = ["Power"]
    fan_cols = ["Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4"]
    inlet_cols = ["Inlet_Temperature"]

    current_temp_cols = [
        c for c in ["CPU_Temperature", "CPU1_Temperature", "CPU2_Temperature"]
        if c in available_columns
    ]

    temp_lag_cols = [
        c for c in available_columns
        if ("Temperature" in c or "CPU_Temperature" in c)
        and ("_lag_" in c or "_rmean_" in c or "_rstd_" in c)
    ]

    if feature_set == "S0":
        return []
    elif feature_set == "S1":
        return current_temp_cols + [c for c in temp_lag_cols if c in available_columns]
    elif feature_set == "S2":
        base = workload_cols + calendar_cols
        return [c for c in base if c in available_columns]
    elif feature_set == "S3":
        base = workload_cols + calendar_cols + power_cols
        return [c for c in base if c in available_columns]
    elif feature_set == "S4":
        base = workload_cols + calendar_cols + power_cols + fan_cols + inlet_cols
        features = [c for c in base if c in available_columns]
        features += current_temp_cols
        features += [c for c in temp_lag_cols if c in available_columns]
        return features
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")
```

### Exact Column Lists Produced per Feature Set

#### Dataset 1 (Dell PowerEdge R840)
- **S0 (0 features):** `[]`
- **S1 (36 features):**
  `['CPU_Temperature', 'CPU1_Temperature', 'CPU2_Temperature', 'CPU_Temperature_lag_1', 'CPU_Temperature_lag_2', 'CPU_Temperature_lag_3', 'CPU_Temperature_lag_6', 'CPU_Temperature_lag_12', 'CPU1_Temperature_lag_1', 'CPU1_Temperature_lag_2', 'CPU1_Temperature_lag_3', 'CPU1_Temperature_lag_6', 'CPU1_Temperature_lag_12', 'CPU2_Temperature_lag_1', 'CPU2_Temperature_lag_2', 'CPU2_Temperature_lag_3', 'CPU2_Temperature_lag_6', 'CPU2_Temperature_lag_12', 'CPU_Temperature_rmean_6', 'CPU_Temperature_rstd_6', 'CPU_Temperature_rmean_12', 'CPU_Temperature_rstd_12', 'CPU_Temperature_rmean_36', 'CPU_Temperature_rstd_36', 'CPU1_Temperature_rmean_6', 'CPU1_Temperature_rstd_6', 'CPU1_Temperature_rmean_12', 'CPU1_Temperature_rstd_12', 'CPU1_Temperature_rmean_36', 'CPU1_Temperature_rstd_36', 'CPU2_Temperature_rmean_6', 'CPU2_Temperature_rstd_6', 'CPU2_Temperature_rmean_12', 'CPU2_Temperature_rstd_12', 'CPU2_Temperature_rmean_36', 'CPU2_Temperature_rstd_36']`
- **S2 (14 features):**
  `['CPU_Load', 'Ram', 'Ram_Used', 'CPU_cores', 'CPU_cores_used', 'Network_RX', 'Network_TX', 'No_Of_Running_vms', 'hour', 'dayofweek', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos']`
- **S3 (15 features):**
  `S2` + `['Power']`
- **S4 (56 features):**
  `S3` + `['Fan_speed1', 'Fan_speed2', 'Fan_speed3', 'Fan_speed4', 'Inlet_Temperature']` + 36 temperature features of `S1`.

#### Dataset 2 (Dell PowerEdge C6320)
- **S0 (0 features):** `[]`
- **S1 (36 features):** Same 36 temperature features as D1.
- **S2 (14 features):** Same 14 workload and calendar features as D1.
- **S3 (15 features):** `S2` + `['Power']`.
- **S4:** Not available on Dataset 2 (no fan speed tachometers or inlet temperature sensors exist in the trace).

---

## 3. Splitters and Stratified Holdout Functions

File: `src/ttwin/splits.py`, lines 14–39:
```python
def split_chronological(
    df: pd.DataFrame,
    time_col: str = "Time",
    train_frac: float = 0.70,
    val_frac: float = 0.10,
    embargo_steps: int = 36,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split A: chronological split with embargo gap.

    Follows 70% train / 10% validation / 20% test with 6h embargo before test.
    Returns (train, val, test). The embargo gap is excluded entirely.
    """
    times = df[time_col].sort_values()
    min_time, max_time = times.min(), times.max()
    total_duration = max_time - min_time

    train_end = min_time + total_duration * train_frac
    val_end = train_end + total_duration * val_frac
    embargo_duration = pd.Timedelta(minutes=10 * embargo_steps)
    test_start = val_end + embargo_duration

    train = df[df[time_col] < train_end].copy()
    val = df[(df[time_col] >= train_end) & (df[time_col] < val_end)].copy()
    test = df[df[time_col] >= test_start].copy()

    return train, val, test
```

File: `src/ttwin/splits.py`, lines 47–99:
```python
def _stratified_holdout(
    df: pd.DataFrame,
    n_holdout: int,
    rng: np.random.RandomState,
    server_col: str = "Series",
    stratify_col: Optional[str] = "CPU_Load",
) -> Tuple[List[str], List[str]]:
    """Select holdout servers via stratified random sampling across load strata.

    Divides servers into strata based on stratify_col and samples randomly within
    each stratum using rng so that different seeds produce different held-out sets
    representing low, medium, and high load regimes.
    """
    servers = get_server_list(df, server_col)

    if stratify_col and stratify_col in df.columns:
        server_means = (
            df.assign(**{server_col: df[server_col].astype(str).str.strip()})
            .groupby(server_col)[stratify_col]
            .mean()
            .reindex(servers)
        )
        sorted_servers = server_means.sort_values().index.tolist()
        n_groups = min(5, len(sorted_servers))
        group_size = len(sorted_servers) // n_groups

        strata = []
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else len(sorted_servers)
            strata.append(sorted_servers[start:end])

        base_quota = n_holdout // n_groups
        remainder = n_holdout % n_groups

        # Randomize which strata receive the remainder quota using rng
        strata_indices = list(range(n_groups))
        rng.shuffle(strata_indices)
        quotas = {idx: base_quota for idx in range(n_groups)}
        for idx in strata_indices[:remainder]:
            quotas[idx] += 1

        test_servers = []
        for i, group in enumerate(strata):
            q = quotas[i]
            if q > 0 and len(group) > 0:
                selected = rng.choice(group, size=min(q, len(group)), replace=False).tolist()
                test_servers.extend(selected)
    else:
        test_servers = rng.choice(servers, size=n_holdout, replace=False).tolist()

    train_servers = [s for s in servers if s not in test_servers]
    return train_servers, test_servers
```

File: `src/ttwin/splits.py`, lines 129–174:
```python
def get_group_kfold_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    server_col: str = "Series",
    stratify_col: Optional[str] = "CPU_Load",
    seed: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """Generate 5-fold cross-validation by server so every server is tested exactly once.

    Servers are stratified by CPU load across folds to balance regimes.
    Returns list of 5 tuples: (train_servers, test_servers).
    """
    df_clean = df.copy()
    df_clean[server_col] = df_clean[server_col].astype(str).str.strip()
    servers = get_server_list(df_clean, server_col)
    rng = np.random.RandomState(seed)

    if stratify_col and stratify_col in df_clean.columns:
        server_means = (
            df_clean.groupby(server_col)[stratify_col].mean().reindex(servers)
        )
        sorted_servers = server_means.sort_values().index.tolist()

        folds = [[] for _ in range(n_splits)]
        n_groups = min(n_splits, len(sorted_servers))
        group_size = len(sorted_servers) // n_groups
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else len(sorted_servers)
            group = sorted_servers[start:end].copy()
            rng.shuffle(group)
            for j, srv in enumerate(group):
                folds[j % n_splits].append(srv)
    else:
        shuffled = servers.copy()
        rng.shuffle(shuffled)
        folds = [[] for _ in range(n_splits)]
        for i, srv in enumerate(shuffled):
            folds[i % n_splits].append(srv)

    splits = []
    for f_idx in range(n_splits):
        test_srvs = sorted(folds[f_idx])
        train_srvs = sorted([s for s in servers if s not in test_srvs])
        splits.append((train_srvs, test_srvs))
    return splits
```

File: `src/ttwin/splits.py`, lines 177–215:
```python
def split_strict(
    df: pd.DataFrame,
    time_col: str = "Time",
    server_col: str = "Series",
    train_frac: float = 0.7,
    holdout_frac: float = 0.2,
    embargo_steps: int = 36,
    seed: int = 42,
    stratify_col: Optional[str] = "CPU_Load",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split C: held-out servers AND later time.

    Train: seen servers, earlier time (0 to 70%)
    Val: seen servers, validation time (70% to 80%)
    Test: unseen servers, later time (80% + 6h to 100%)
    """
    df_clean = df.copy()
    df_clean[server_col] = df_clean[server_col].astype(str).str.strip()
    servers = get_server_list(df_clean, server_col)
    n_holdout = max(1, int(len(servers) * holdout_frac))
    rng = np.random.RandomState(seed)

    train_servers, test_servers = _stratified_holdout(
        df_clean, n_holdout, rng, server_col, stratify_col
    )

    times = df_clean[time_col].sort_values()
    min_time, max_time = times.min(), times.max()
    total_duration = max_time - min_time
    train_end = min_time + total_duration * train_frac
    val_end = train_end + total_duration * 0.10
    embargo_duration = pd.Timedelta(minutes=10 * embargo_steps)
    test_start = val_end + embargo_duration

    train = df_clean[(df_clean[server_col].isin(train_servers)) & (df_clean[time_col] < train_end)].copy()
    val = df_clean[(df_clean[server_col].isin(train_servers)) & (df_clean[time_col] >= train_end) & (df_clean[time_col] < val_end)].copy()
    test = df_clean[(df_clean[server_col].isin(test_servers)) & (df_clean[time_col] >= test_start)].copy()

    return train, val, test
```

---

## 4. Baselines Implementation

### 4.1 Persistence Baseline
File: `src/ttwin/metrics.py`, lines 27–29:
```python
def persistence_predictions(y_series: pd.Series, horizon_steps: int) -> pd.Series:
    """Generate persistence baseline: prediction = value `horizon_steps` ago."""
    return y_series.shift(horizon_steps)
```
In ablation scripts, evaluated on current temperature:
```python
mae_pers = mae(y_test_abs, y_test_curr)
```

### 4.2 Fleet Cluster Mean Baseline
File: `scripts/run_v2_planning_adaptation.py`, lines 102 and 142:
```python
cluster_mean_temp = float(clean_s2_src[target_abs].mean())
...
mae_cluster_mean = mae(y_test_abs, np.full_like(y_test_abs, cluster_mean_temp))
```

### 4.3 Local Server Mean Baseline (First $N$ Hours)
File: `scripts/run_v2_planning_adaptation.py`, lines 181–185:
```python
server_mean_local = float(np.mean(y_ad_abs))
planning_records.append({
    "server": srv, "budget_hours": bh, "strategy": "server_mean_local",
    "regime": "planning", "mae": mae(y_test_abs, np.full_like(y_test_abs, server_mean_local)),
})
```

### 4.4 Stronger Planning Baselines: Load-Conditional Mean and Ridge on CPU Load
File: `scripts/diagnostics/diag_part_a_q1_to_q4.py`, lines 75–93:
```python
# Ridge on CPU Load only
ridge_cpu = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100.0))])
ridge_cpu.fit(clean_s2_src[["CPU_Load"]].values, clean_s2_src[target_abs].values)
pred_ridge_cpu = ridge_cpu.predict(clean_test[["CPU_Load"]].values)
mae_ridge_cpu = mae(y_test_abs, pred_ridge_cpu)

# Load-Conditional Mean (10 load decile bins fitted on source training data)
load_bins = np.linspace(0, 100, 11)
src_load_bin = pd.cut(clean_s2_src["CPU_Load"], bins=load_bins, labels=range(10))
bin_means = clean_s2_src.groupby(src_load_bin, observed=False)[target_abs].mean().fillna(cluster_mean_temp).to_dict()

test_load_bin = pd.cut(clean_test["CPU_Load"], bins=load_bins, labels=range(10))
pred_load_cond = test_load_bin.map(bin_means).fillna(cluster_mean_temp).values
mae_load_cond = mae(y_test_abs, pred_load_cond)
```

---

## 5. Planning Regime Definition

File: `scripts/run_v2_planning_adaptation.py`, lines 69–72, 95–101:
```python
horizon_step = 6  # 1 hour
target_abs = f"CPU_Temperature_target_{horizon_step}"

# Pretrain Planning Model (S2, workload-only, predicting absolute temperature)
clean_s2_src = df_source_train[s2_feats + [target_abs]].dropna()
dtrain_s2 = lgb.Dataset(clean_s2_src[s2_feats].values, label=clean_s2_src[target_abs].values, feature_name=s2_feats)
lgb_planning = lgb.train(
    {"objective": "regression", "metric": "mae", "learning_rate": 0.05, "num_leaves": 63, "verbosity": -1, "seed": 42},
    dtrain_s2, num_boost_round=200,
)
```

Target used in v2: $T(t+h)$ where $h=6$ (1 hour ahead).
Features: Workload (`CPU_Load`, `Ram`, `Ram_Used`, `CPU_cores`, `CPU_cores_used`, `Network_RX`, `Network_TX`, `No_Of_Running_vms`) and cyclic calendar encodings.

---

## 6. Cold-Start Adaptation Functions

File: `scripts/run_v2_planning_adaptation.py`, lines 187–237:

### 6.1 Residual Offset
```python
ad_pred_plan = lgb_planning.predict(X_ad_s2)
offset_plan = float(np.mean(y_ad_abs - ad_pred_plan))
pred_plan_offset = pred_zs_plan + offset_plan
```

### 6.2 Residual Ridge
```python
res_plan = y_ad_abs - ad_pred_plan
ridge_pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=50.0))])
ridge_pipe.fit(X_ad_s2, res_plan)
pred_plan_ridge = pred_zs_plan + ridge_pipe.predict(X_test_s2)
```

### 6.3 Continued Boosting
```python
dtrain_ad_plan = lgb.Dataset(X_ad_s2, label=y_ad_abs, feature_name=s2_feats)
ws_plan = lgb.train(
    {"objective": "regression", "metric": "mae", "learning_rate": 0.01, "num_leaves": 15, "min_child_samples": 3, "verbosity": -1},
    dtrain_ad_plan, num_boost_round=min(30, max(5, n_adapt // 10)), init_model=lgb_planning,
)
```
- Minimum sample threshold: `if n_adapt < 6: continue` (at least 1 hour of telemetry).
- Early stopping in continued boosting: None (fixed conservative round budget $R = \min(30, \max(5, n_{\text{adapt}} // 10))$ with low learning rate 0.01).

---

## 7. Conformal Prediction Code

File: `scripts/run_v2_uncertainty.py`, lines 114–174:

```python
alpha = 0.10  # 90% nominal coverage
n_cal = len(abs_err_val)
q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal

# 1. Constant-Width Split Conformal
q_hat_const = float(np.quantile(abs_err_val, min(1.0, q_level)))
lower_const = pred_te_abs - q_hat_const
upper_const = pred_te_abs + q_hat_const

# 2. Normalized Conformal (Residual Scaling)
ridge_scale = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100.0))])
ridge_scale.fit(X_val, abs_err_val)
sigma_val = np.clip(ridge_scale.predict(X_val), 0.1, 15.0)
sigma_te = np.clip(ridge_scale.predict(X_te), 0.1, 15.0)
norm_scores_val = abs_err_val / sigma_val
q_hat_norm = float(np.quantile(norm_scores_val, min(1.0, q_level)))
lower_norm = pred_te_abs - q_hat_norm * sigma_te
upper_norm = pred_te_abs + q_hat_norm * sigma_te

# 3. Conformalized Quantile Regression (CQR)
m_low = lgb.train({"objective": "quantile", "alpha": 0.05, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1}, dtrain, num_boost_round=150)
m_high = lgb.train({"objective": "quantile", "alpha": 0.95, "learning_rate": 0.05, "num_leaves": 31, "verbosity": -1}, dtrain, num_boost_round=150)
q_low_val = y_val_curr + m_low.predict(X_val)
q_high_val = y_val_curr + m_high.predict(X_val)
q_low_te = y_te_curr + m_low.predict(X_te)
q_high_te = y_te_curr + m_high.predict(X_te)
cqr_scores_val = np.maximum(q_low_val - y_val_abs, y_val_abs - q_high_val)
q_hat_cqr = float(np.quantile(cqr_scores_val, min(1.0, q_level)))
lower_cqr = q_low_te - q_hat_cqr
upper_cqr = q_high_te + q_hat_cqr
```
- Calibration rows: `X_val`, `y_val_abs` (from `val_df`, held-out seen servers, 223,077 rows).
- Evaluation rows: `X_te`, `y_te_abs` (from `test_df`, held-out unseen servers in test time window, 109,170 rows).

---

## 8. Fault Injection Code

File: `scripts/test_fault_injection.py`, lines 58–99:
```python
# Simulated conformal band: q_hat = 1.8°C (90% interval = +/- 1.8°C)
q_hat = 1.80
pred = y_true + np.random.normal(0, 0.4, size=n_steps)  # baseline predictor
upper = pred + q_hat

# 1. Nominal False Alarm Rate
nominal_crit_alerts = (upper >= 75.0) | (np.abs(y_true - pred) > 3 * q_hat)
nominal_far = float(np.mean(nominal_crit_alerts))

# Scenario A: Simulated Sudden Fan Failure (+15°C ramp starting at step 300)
y_fan_fail = y_true.copy()
fault_start = 300
fault_duration = 30
ramp = np.linspace(0, 15.0, 5)
y_fan_fail[fault_start:fault_start+5] += ramp
y_fan_fail[fault_start+5:fault_start+fault_duration] += 15.0

upper_fan = pred + q_hat
alerts_fan = (upper_fan >= 75.0) | (np.abs(y_fan_fail - pred) > 3 * q_hat)
detection_step = np.where(alerts_fan[fault_start:])[0]
latency_steps = int(detection_step[0]) if len(detection_step) > 0 else None
tpr_fan = float(np.mean(alerts_fan[fault_start:fault_start+fault_duration]))
```
- **Alert Rule:** `(upper >= 75.0) | (np.abs(y_measured - pred) > 3 * q_hat)`.
- **Consecutive exceedance rule:** None implemented in code (a single instantaneous exceedance triggers an alert).
- **Does the injected fault modify input features as well as measured value?** **NO.** The perturbation is added purely to the synthetic target array `y_fan_fail`. The baseline predictions `pred` are generated by adding random Gaussian noise to nominal temperatures; no feature matrix $X$ is fed through a trained model during this fault-injection script.

---

## 9. Hyper-Parameters Actually Used

### 9.1 LightGBM Point Regressor (Ablation E1)
File: `scripts/run_v2_e1_ablation.py`, lines 173–178:
```python
lgb_params = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbosity": -1,
    "seed": 42,
}
# num_boost_round = 250, early_stopping = 30 rounds
```

### 9.2 Ridge Regressors
- Ablation E1: `Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100.0))])`
- Residual Ridge Adaptation E4: `Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=50.0))])`
- Conformal Residual Scaling E5: `Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=100.0))])`
