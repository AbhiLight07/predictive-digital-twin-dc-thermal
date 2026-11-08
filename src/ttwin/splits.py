"""Train/val/test split implementations for the thermal twin project.

Split A: chronological (time cutoff with embargo gap)
Split B: leave-servers-out (hold out ~20% of servers, 5 seeds)
Split C: strict (held-out servers AND later time)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


def split_chronological(
    df: pd.DataFrame,
    time_col: str = "Time",
    train_frac: float = 0.7,
    val_frac: float = 0.1,
    embargo_steps: int = 36,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split A: chronological split with embargo gap.

    Returns (train, val, test). The embargo gap is excluded entirely.
    """
    times = df[time_col].sort_values()
    min_time, max_time = times.min(), times.max()
    total_duration = max_time - min_time

    train_end = min_time + total_duration * train_frac
    embargo_duration = pd.Timedelta(minutes=10 * embargo_steps)
    test_start = train_end + embargo_duration

    val_start = min_time + total_duration * (train_frac - val_frac)

    train = df[df[time_col] < val_start].copy()
    val = df[(df[time_col] >= val_start) & (df[time_col] < train_end)].copy()
    test = df[df[time_col] >= test_start].copy()

    return train, val, test


def get_server_list(df: pd.DataFrame, server_col: str = "Series") -> List[str]:
    """Get sorted list of unique servers."""
    return sorted(df[server_col].unique().tolist())


def _stratified_holdout(
    df: pd.DataFrame,
    n_holdout: int,
    rng: np.random.RandomState,
    server_col: str = "Series",
    stratify_col: Optional[str] = "CPU_Load",
) -> Tuple[List[str], List[str]]:
    """Select holdout servers with optional stratification."""
    servers = get_server_list(df, server_col)

    if stratify_col and stratify_col in df.columns:
        server_means = df.groupby(server_col)[stratify_col].mean().reindex(servers)
        sorted_servers = server_means.sort_values().index.tolist()
        n_groups = min(5, len(sorted_servers))
        group_size = len(sorted_servers) // n_groups
        shuffled = []
        for i in range(n_groups):
            start = i * group_size
            end = start + group_size if i < n_groups - 1 else len(sorted_servers)
            group = sorted_servers[start:end]
            rng.shuffle(group)
            shuffled.extend(group)
        test_servers = shuffled[:n_holdout]
    else:
        test_servers = rng.choice(servers, size=n_holdout, replace=False).tolist()

    train_servers = [s for s in servers if s not in test_servers]
    return train_servers, test_servers


def split_leave_servers_out(
    df: pd.DataFrame,
    holdout_frac: float = 0.2,
    seed: int = 42,
    server_col: str = "Series",
    stratify_col: Optional[str] = "CPU_Load",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Split B: hold out servers entirely.

    Returns (train_df, test_df, train_servers, test_servers).
    """
    servers = get_server_list(df, server_col)
    n_holdout = max(1, int(len(servers) * holdout_frac))
    rng = np.random.RandomState(seed)

    train_servers, test_servers = _stratified_holdout(
        df, n_holdout, rng, server_col, stratify_col
    )

    train_df = df[df[server_col].isin(train_servers)].copy()
    test_df = df[df[server_col].isin(test_servers)].copy()

    return train_df, test_df, train_servers, test_servers


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

    Train: seen servers, earlier time
    Val: seen servers, later time
    Test: unseen servers, later time
    """
    servers = get_server_list(df, server_col)
    n_holdout = max(1, int(len(servers) * holdout_frac))
    rng = np.random.RandomState(seed)

    train_servers, test_servers = _stratified_holdout(
        df, n_holdout, rng, server_col, stratify_col
    )

    times = df[time_col].sort_values()
    min_time, max_time = times.min(), times.max()
    total_duration = max_time - min_time
    train_end = min_time + total_duration * train_frac
    embargo_duration = pd.Timedelta(minutes=10 * embargo_steps)
    test_start = train_end + embargo_duration

    train = df[(df[server_col].isin(train_servers)) & (df[time_col] < train_end)].copy()
    val = df[(df[server_col].isin(train_servers)) & (df[time_col] >= test_start)].copy()
    test = df[(df[server_col].isin(test_servers)) & (df[time_col] >= test_start)].copy()

    return train, val, test


def get_split_b_seeds() -> List[int]:
    """Return the 5 fixed seeds for Split B repeats."""
    return [42, 123, 456, 789, 1024]


def verify_no_server_overlap(
    train_df: pd.DataFrame, test_df: pd.DataFrame, server_col: str = "Series"
) -> bool:
    """Check that no servers appear in both train and test."""
    train_servers = set(train_df[server_col].unique())
    test_servers = set(test_df[server_col].unique())
    return len(train_servers & test_servers) == 0


def verify_no_future_leak(
    train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str = "Time"
) -> bool:
    """Check that max train time < min test time."""
    return train_df[time_col].max() < test_df[time_col].min()
