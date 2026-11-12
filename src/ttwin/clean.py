"""Data cleaning module for the thermal twin project."""

import pandas as pd
import numpy as np


def parse_timestamps(df: pd.DataFrame, time_col: str = "Time") -> pd.DataFrame:
    """Parse time column to datetime and sort."""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    df = df.sort_values(time_col).reset_index(drop=True)
    return df


def remove_duplicate_timestamps(
    df: pd.DataFrame, host_col: str = "Series", time_col: str = "Time"
) -> pd.DataFrame:
    """Remove rows with duplicate timestamps within each server, keeping first."""
    return df.drop_duplicates(subset=[host_col, time_col], keep="first").reset_index(drop=True)


def fill_missing_timestamps(
    df: pd.DataFrame,
    host_col: str = "Series",
    time_col: str = "Time",
    freq: str = "10min",
) -> pd.DataFrame:
    """Ensure regular time grid per server by reindexing to expected frequency.

    Missing rows are inserted with NaN values.
    """
    result = []
    for server, group in df.groupby(host_col):
        group = group.set_index(time_col)
        full_idx = pd.date_range(group.index.min(), group.index.max(), freq=freq)
        group = group.reindex(full_idx)
        group[host_col] = server
        group.index.name = time_col
        group = group.reset_index()
        result.append(group)
    return pd.concat(result, ignore_index=True)


def strip_host_names(df: pd.DataFrame, host_col: str = "Series") -> pd.DataFrame:
    """Strip leading and trailing whitespace from host/server identifiers."""
    df = df.copy()
    if host_col in df.columns:
        df[host_col] = df[host_col].astype(str).str.strip()
    return df


def clip_temperature(
    df: pd.DataFrame,
    temp_cols: list[str] | None = None,
    min_temp: float = -10.0,
    max_temp: float = 110.0,
) -> pd.DataFrame:
    """Clip temperature values to physically plausible range.

    Values outside [min_temp, max_temp] are set to NaN.
    """
    df = df.copy()
    if temp_cols is None:
        temp_cols = [c for c in df.columns if "temp" in c.lower() or "inlet" in c.lower()]
    for col in temp_cols:
        if col in df.columns:
            mask = (df[col] < min_temp) | (df[col] > max_temp)
            if mask.any():
                print(f"  [clean] {col}: {mask.sum()} values outside [{min_temp}, {max_temp}] set to NaN")
                df.loc[mask, col] = np.nan
    return df


def drop_constant_columns(
    df: pd.DataFrame, exclude: list[str] | None = None
) -> tuple[pd.DataFrame, list[str]]:
    """Drop columns with zero variance. Returns (df, dropped_columns)."""
    exclude = exclude or []
    dropped = []
    for col in df.columns:
        if col in exclude:
            continue
        if df[col].nunique(dropna=True) <= 1:
            dropped.append(col)
    if dropped:
        df = df.drop(columns=dropped)
    return df, dropped


def clean_dataset(
    df: pd.DataFrame,
    host_col: str = "Series",
    time_col: str = "Time",
    freq: str = "10min",
) -> pd.DataFrame:
    """Full cleaning pipeline: strip host names, parse time, dedup, clip temps."""
    df = strip_host_names(df, host_col)
    df = parse_timestamps(df, time_col)
    df = remove_duplicate_timestamps(df, host_col, time_col)
    df = clip_temperature(df)
    return df
