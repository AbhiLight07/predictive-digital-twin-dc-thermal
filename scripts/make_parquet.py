#!/usr/bin/env python3
"""Convert raw CSVs to per-server Parquet files with downcast dtypes.

Usage:
    python scripts/make_parquet.py [--chunksize N]

Reads dataset1_combined.csv and dataset2_combined.csv from data/raw/,
cleans and converts to per-server Parquet files in data/interim/.
"""

import argparse
import sys
from pathlib import Path

# Add project src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import numpy as np
from ttwin.io import downcast_dtypes, save_parquet
from ttwin.clean import parse_timestamps, remove_duplicate_timestamps, clip_temperature


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
INTERIM_DIR = Path(__file__).resolve().parent.parent / "data" / "interim"

DATASETS = {
    "dataset1": {
        "file": "dataset1_combined.csv",
        "host_col": "Series",
        "time_col": "Time",
    },
    "dataset2": {
        "file": "dataset2_combined.csv",
        "host_col": "Series",
        "time_col": "Time",
    },
}


def convert_dataset(dataset_name: str, chunksize: int = 100_000) -> None:
    """Convert a raw CSV to per-server Parquet files."""
    config = DATASETS[dataset_name]
    csv_path = RAW_DIR / config["file"]
    out_dir = INTERIM_DIR / dataset_name

    if not csv_path.exists():
        print(f"[SKIP] {csv_path} not found")
        return

    print(f"\n{'='*60}")
    print(f"Converting {dataset_name}: {csv_path.name}")
    print(f"Output: {out_dir}")
    print(f"{'='*60}")

    host_col = config["host_col"]
    time_col = config["time_col"]

    # Accumulate per server
    server_chunks: dict[str, list[pd.DataFrame]] = {}
    total_rows = 0
    chunk_count = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk_count += 1
        total_rows += len(chunk)

        # Parse timestamps
        chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")

        # Group by server and accumulate
        for server, group in chunk.groupby(host_col):
            server = str(server)
            if server not in server_chunks:
                server_chunks[server] = []
            server_chunks[server].append(group)

        if chunk_count % 10 == 0:
            print(f"  ... processed {total_rows:,} rows ({chunk_count} chunks)")

    print(f"\nTotal rows read: {total_rows:,}")
    print(f"Servers found: {len(server_chunks)}")

    # Write per-server Parquet files
    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0

    for server in sorted(server_chunks.keys()):
        df = pd.concat(server_chunks[server], ignore_index=True)

        # Clean
        df = df.sort_values(time_col).reset_index(drop=True)
        df = remove_duplicate_timestamps(df, host_col, time_col)
        df = clip_temperature(df)

        # Downcast
        df = downcast_dtypes(df)

        # Save
        out_path = out_dir / f"{server}.parquet"
        save_parquet(df, out_path)
        n_written += 1

    print(f"\nWrote {n_written} Parquet files to {out_dir}")

    # Also write a combined file for convenience (if not too large)
    print(f"Writing combined Parquet...")
    all_dfs = []
    for f in sorted(out_dir.glob("*.parquet")):
        if f.name == "_combined.parquet":
            continue
        all_dfs.append(pd.read_parquet(f))
    combined = pd.concat(all_dfs, ignore_index=True)
    save_parquet(combined, out_dir / "_combined.parquet")
    print(f"Combined file: {out_dir / '_combined.parquet'} ({len(combined):,} rows)")


def main():
    parser = argparse.ArgumentParser(description="Convert raw CSVs to Parquet.")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk")
    parser.add_argument("--dataset", choices=["dataset1", "dataset2", "both"],
                        default="both", help="Which dataset to convert")
    args = parser.parse_args()

    datasets = ["dataset1", "dataset2"] if args.dataset == "both" else [args.dataset]

    for ds in datasets:
        convert_dataset(ds, chunksize=args.chunksize)

    print("\nDone!")


if __name__ == "__main__":
    main()
