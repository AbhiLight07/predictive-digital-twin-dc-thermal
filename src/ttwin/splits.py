"""Train/val/test split implementations for the thermal twin project.

Split A: chronological (70% train / 10% val / 20% test with 6h embargo gap)
Split B: leave-servers-out (hold out ~20% of servers, stratified by load across 5 seeds)
Split B (GroupKFold): 5-fold cross-validation by server (all servers tested)
Split C: strict holdout (held-out servers AND later time)
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Optional


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


def get_server_list(df: pd.DataFrame, server_col: str = "Series") -> List[str]:
    """Get sorted list of unique servers with whitespace stripped."""
    return sorted(list(set(df[server_col].astype(str).str.strip().unique())))


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


def split_leave_servers_out(
    df: pd.DataFrame,
    holdout_frac: float = 0.2,
    seed: int = 42,
    server_col: str = "Series",
    stratify_col: Optional[str] = "CPU_Load",
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """Split B: hold out servers entirely (stratified random across load strata).

    Returns (train_df, test_df, train_servers, test_servers).
    """
    df_clean = df.copy()
    df_clean[server_col] = df_clean[server_col].astype(str).str.strip()
    servers = get_server_list(df_clean, server_col)
    n_holdout = max(1, int(len(servers) * holdout_frac))
    rng = np.random.RandomState(seed)

    train_servers, test_servers = _stratified_holdout(
        df_clean, n_holdout, rng, server_col, stratify_col
    )

    train_df = df_clean[df_clean[server_col].isin(train_servers)].copy()
    test_df = df_clean[df_clean[server_col].isin(test_servers)].copy()

    return train_df, test_df, train_servers, test_servers


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


def get_split_b_seeds() -> List[int]:
    """Return the 5 fixed seeds for Split B repeats."""
    return [42, 123, 456, 789, 1024]


def verify_no_server_overlap(
    train_df: pd.DataFrame, test_df: pd.DataFrame, server_col: str = "Series"
) -> bool:
    """Check that no servers appear in both train and test."""
    train_servers = set(train_df[server_col].astype(str).str.strip().unique())
    test_servers = set(test_df[server_col].astype(str).str.strip().unique())
    return len(train_servers & test_servers) == 0


def verify_no_future_leak(
    train_df: pd.DataFrame, test_df: pd.DataFrame, time_col: str = "Time"
) -> bool:
    """Check that max train time < min test time."""
    return train_df[time_col].max() < test_df[time_col].min()


def partition_validation_conformal(
    val_df: pd.DataFrame,
    select_ratio: float = 0.5,
    time_col: str = "Time",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Chronologically partition a validation block into disjoint model-selection and calibration sets.

    Ensures model selection (e.g. early stopping, hyperparameter tuning) and conformal residual
    calibration use strictly disjoint observations.

    Args:
        val_df: Validation DataFrame spanning [val_start, val_end).
        select_ratio: Fraction of validation duration allocated to model selection (default 0.5).
        time_col: Column containing timestamps.

    Returns:
        (val_select_df, val_cal_df): Disjoint DataFrames ordered in time.
    """
    times = val_df[time_col].sort_values()
    if len(times) == 0:
        return val_df.copy(), val_df.copy()
    min_time, max_time = times.min(), times.max()
    duration = max_time - min_time
    split_time = min_time + duration * select_ratio

    val_select = val_df[val_df[time_col] < split_time].copy()
    val_cal = val_df[val_df[time_col] >= split_time].copy()
    return val_select, val_cal


def split_strict_4way(
    df: pd.DataFrame,
    time_col: str = "Time",
    server_col: str = "Series",
    train_frac: float = 0.70,
    val_select_frac: float = 0.05,
    val_cal_frac: float = 0.05,
    holdout_frac: float = 0.20,
    embargo_steps: int = 36,
    seed: int = 42,
    stratify_col: Optional[str] = "CPU_Load",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split C with 4-way disjoint chronological partition for conformal calibration independence.

    Partitions:
        1. Train: Seen servers, earlier time [min_time, train_end).
        2. Val-Select: Seen servers, model-selection window [train_end, select_end).
           Used strictly for early stopping, model selection, and variance scaling fit.
        3. Val-Cal: Seen servers, calibration window [select_end, val_end).
           Used strictly for computing conformal nonconformity scores and quantiles.
        4. Test: Unseen servers, final time window [val_end + embargo_duration, max_time].
           Used strictly for final evaluation.

    Guarantees:
        - Strict chronological order: train < val_select < val_cal < test.
        - Preserves the 6-hour embargo gap (embargo_steps * 10 min) before test.
        - Zero timestamp overlap across all partitions.
        - Calibration data never overlaps with model fitting or early stopping.
        - Protocol fractions preserved: train_frac=0.70, total val=(val_select_frac + val_cal_frac)=0.10.
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
    select_end = train_end + total_duration * val_select_frac
    val_end = select_end + total_duration * val_cal_frac
    embargo_duration = pd.Timedelta(minutes=10 * embargo_steps)
    test_start = val_end + embargo_duration

    train = df_clean[(df_clean[server_col].isin(train_servers)) & (df_clean[time_col] < train_end)].copy()
    val_select = df_clean[(df_clean[server_col].isin(train_servers)) & (df_clean[time_col] >= train_end) & (df_clean[time_col] < select_end)].copy()
    val_cal = df_clean[(df_clean[server_col].isin(train_servers)) & (df_clean[time_col] >= select_end) & (df_clean[time_col] < val_end)].copy()
    test = df_clean[(df_clean[server_col].isin(test_servers)) & (df_clean[time_col] >= test_start)].copy()

    return train, val_select, val_cal, test


def split_chronological_4way(
    df: pd.DataFrame,
    time_col: str = "Time",
    train_frac: float = 0.70,
    val_select_frac: float = 0.05,
    val_cal_frac: float = 0.05,
    embargo_steps: int = 36,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split A with 4-way disjoint chronological partition for conformal calibration independence."""
    times = df[time_col].sort_values()
    min_time, max_time = times.min(), times.max()
    total_duration = max_time - min_time

    train_end = min_time + total_duration * train_frac
    select_end = train_end + total_duration * val_select_frac
    val_end = select_end + total_duration * val_cal_frac
    embargo_duration = pd.Timedelta(minutes=10 * embargo_steps)
    test_start = val_end + embargo_duration

    train = df[df[time_col] < train_end].copy()
    val_select = df[(df[time_col] >= train_end) & (df[time_col] < select_end)].copy()
    val_cal = df[(df[time_col] >= select_end) & (df[time_col] < val_end)].copy()
    test = df[df[time_col] >= test_start].copy()

    return train, val_select, val_cal, test


def verify_no_timestamp_overlap(
    df1: pd.DataFrame, df2: pd.DataFrame, time_col: str = "Time"
) -> bool:
    """Check that two DataFrames share zero common timestamps."""
    times1 = set(df1[time_col].unique())
    times2 = set(df2[time_col].unique())
    return len(times1 & times2) == 0


def verify_embargo_enforced(
    df_early: pd.DataFrame,
    df_late: pd.DataFrame,
    expected_embargo_steps: int = 36,
    step_minutes: int = 10,
    time_col: str = "Time",
) -> bool:
    """Check that the gap between max(df_early) and min(df_late) is >= expected embargo duration."""
    gap = df_late[time_col].min() - df_early[time_col].max()
    expected_duration = pd.Timedelta(minutes=expected_embargo_steps * step_minutes)
    return gap >= expected_duration
