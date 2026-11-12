"""Feature engineering module for the thermal twin project."""

import pandas as pd
import numpy as np
from typing import List
from ttwin.clean import fill_missing_timestamps


def add_calendar_features(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """Add hour-of-day and day-of-week features (raw + cyclic encodings)."""
    df = df.copy()
    dt = df[time_col]
    if not pd.api.types.is_datetime64_any_dtype(dt):
        dt = pd.to_datetime(dt)
    df["hour"] = dt.dt.hour
    df["dayofweek"] = dt.dt.day_of_week
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_lag_features(
    df: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    server_col: str = "Series",
) -> pd.DataFrame:
    """Add lagged features per server. Lags are in timesteps (10 min each).

    Must be called BEFORE train/test split. The NaN rows at the start of each
    server's data are natural and do not leak future information.
    """
    df = df.copy()
    for col in columns:
        for lag in lags:
            df[f"{col}_lag_{lag}"] = df.groupby(server_col)[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    server_col: str = "Series",
) -> pd.DataFrame:
    """Add rolling mean and std features per server.

    The rolling window ends at t (includes observation at time t and preceding w-1 timesteps).
    This represents the current state up to time t and introduces no future information.
    """
    df = df.copy()
    for col in columns:
        for w in windows:
            rolling = df.groupby(server_col)[col].rolling(window=w, min_periods=1)
            df[f"{col}_rmean_{w}"] = rolling.mean().reset_index(level=0, drop=True)
            df[f"{col}_rstd_{w}"] = rolling.std().reset_index(level=0, drop=True).fillna(0)
    return df


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


def compute_effective_cpu_temp(
    df: pd.DataFrame,
    cpu1_col: str = "CPU1_Temperature",
    cpu2_col: str = "CPU2_Temperature",
) -> pd.DataFrame:
    """Compute effective CPU temperature as max(CPU1, CPU2)."""
    df = df.copy()
    if cpu1_col in df.columns and cpu2_col in df.columns:
        df["CPU_Temperature"] = df[[cpu1_col, cpu2_col]].max(axis=1)
    elif cpu1_col in df.columns:
        df["CPU_Temperature"] = df[cpu1_col]
    elif cpu2_col in df.columns:
        df["CPU_Temperature"] = df[cpu2_col]
    return df


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
        return list(dict.fromkeys(features))
    else:
        raise ValueError(f"Unknown feature set: {feature_set}")


def build_features(
    df: pd.DataFrame,
    server_col: str = "Series",
    time_col: str = "Time",
    target_col: str = "CPU_Temperature",
    horizons: List[int] | None = None,
    lags: List[int] | None = None,
    rolling_windows: List[int] | None = None,
) -> pd.DataFrame:
    """Full feature engineering pipeline.

    Args:
        horizons: prediction horizons in timesteps (1=10min, 6=1h, 36=6h)
        lags: lag steps for temperature features
        rolling_windows: rolling window sizes for temperature features
    """
    if horizons is None:
        horizons = [1, 6, 36]
    if lags is None:
        lags = [1, 2, 3, 6, 12]
    if rolling_windows is None:
        rolling_windows = [6, 12, 36]

    df = fill_missing_timestamps(df, host_col=server_col, time_col=time_col, freq="10min")
    df = compute_effective_cpu_temp(df)
    df = add_calendar_features(df, time_col)

    temp_cols = [c for c in ["CPU_Temperature", "CPU1_Temperature", "CPU2_Temperature"] if c in df.columns]
    df = add_lag_features(df, temp_cols, lags, server_col)
    df = add_rolling_features(df, temp_cols, rolling_windows, server_col)

    for h in horizons:
        df = create_target(df, target_col, h, server_col)

    return df
