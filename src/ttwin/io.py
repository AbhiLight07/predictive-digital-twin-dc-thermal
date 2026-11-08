"""Data I/O utilities for the thermal twin project."""

import pandas as pd
from pathlib import Path
from typing import Iterator


def read_csv_chunked(csv_path: str | Path, chunksize: int = 100_000) -> Iterator[pd.DataFrame]:
    """Yield chunks from a large CSV file."""
    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        yield chunk


def read_raw_csv(csv_path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    """Read a raw CSV file (or first nrows) into a DataFrame."""
    return pd.read_csv(csv_path, nrows=nrows, low_memory=False)


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast numeric columns to save memory.

    float64 -> float32, int64 -> int32/int16 where possible.
    """
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def save_parquet(df: pd.DataFrame, path: str | Path, compression: str = "snappy") -> None:
    """Save DataFrame as Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, compression=compression, index=False)


def load_parquet(path: str | Path) -> pd.DataFrame:
    """Load a Parquet file."""
    return pd.read_parquet(path)


def load_all_servers(parquet_dir: str | Path) -> pd.DataFrame:
    """Load all per-server Parquet files from a directory and concatenate."""
    parquet_dir = Path(parquet_dir)
    dfs = []
    for f in sorted(parquet_dir.glob("*.parquet")):
        if f.name.startswith("_"):
            continue  # Skip combined files
        df = pd.read_parquet(f)
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No Parquet files found in {parquet_dir}")
    return pd.concat(dfs, ignore_index=True)
