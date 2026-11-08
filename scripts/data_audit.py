#!/usr/bin/env python3
"""Data audit script for the Melbourne cloud data-centre trace.

Reads large CSVs in chunks and produces a per-server summary:
  - Number of rows and coverage (date range, days)
  - Missingness per column (%)
  - Temperature-like columns detected
  - Basic statistics for key columns

Usage:
    python scripts/data_audit.py <csv_file> [--host HOST_COL] [--time TIME_COL] [--chunksize N]

Examples:
    python scripts/data_audit.py data/raw/dataset1_combined.csv
    python scripts/data_audit.py data/raw/dataset2_combined.csv --host hostname --time timestamp
    python scripts/data_audit.py data/raw/dataset1_combined.csv --chunksize 50000
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np


def guess_host_column(columns: list[str]) -> str | None:
    """Guess which column identifies the host/server."""
    candidates = [
        "hostname", "host", "server", "node", "machine",
        "host_name", "server_name", "node_name", "machine_name",
        "hostid", "host_id", "server_id", "node_id", "machine_id",
        "series",
    ]
    cols_lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    # Fallback: look for columns containing 'host' or 'server'
    for col_lower, col_orig in cols_lower.items():
        if "host" in col_lower or "server" in col_lower:
            return col_orig
    return None


def guess_time_column(columns: list[str]) -> str | None:
    """Guess which column is the timestamp."""
    candidates = [
        "timestamp", "time", "datetime", "date", "ts",
        "time_stamp", "date_time",
    ]
    cols_lower = {c.lower().strip(): c for c in columns}
    for cand in candidates:
        if cand in cols_lower:
            return cols_lower[cand]
    for col_lower, col_orig in cols_lower.items():
        if "time" in col_lower or "date" in col_lower:
            return col_orig
    return None


def detect_temperature_columns(columns: list[str]) -> list[str]:
    """Find columns that look like temperature measurements."""
    temp_keywords = ["temp", "thermal", "cpu_temp", "inlet", "outlet", "exhaust"]
    result = []
    for col in columns:
        col_lower = col.lower()
        for kw in temp_keywords:
            if kw in col_lower:
                result.append(col)
                break
    return result


def detect_column_categories(columns: list[str]) -> dict[str, list[str]]:
    """Categorise columns by likely meaning."""
    categories = {
        "temperature": [],
        "power": [],
        "fan": [],
        "cpu_load": [],
        "memory": [],
        "network": [],
        "disk": [],
        "vm": [],
        "other": [],
    }
    for col in columns:
        cl = col.lower()
        if any(kw in cl for kw in ["temp", "thermal", "inlet", "outlet", "exhaust"]):
            categories["temperature"].append(col)
        elif any(kw in cl for kw in ["power", "watt", "energy"]):
            categories["power"].append(col)
        elif any(kw in cl for kw in ["fan", "rpm"]):
            categories["fan"].append(col)
        elif any(kw in cl for kw in ["cpu", "load", "util"]):
            categories["cpu_load"].append(col)
        elif any(kw in cl for kw in ["mem", "ram"]):
            categories["memory"].append(col)
        elif any(kw in cl for kw in ["net", "byte", "packet", "bandwidth"]):
            categories["network"].append(col)
        elif any(kw in cl for kw in ["disk", "io", "storage", "data"]):
            categories["disk"].append(col)
        elif any(kw in cl for kw in ["vm", "virtual", "instance", "container"]):
            categories["vm"].append(col)
        else:
            categories["other"].append(col)
    return categories


def audit_file(
    csv_path: str,
    host_col: str | None = None,
    time_col: str | None = None,
    chunksize: int = 100_000,
) -> dict:
    """Run a chunked audit on a CSV file.

    Returns a dict with:
      - file_info: basic file metadata
      - columns: list of column names
      - column_categories: categorised columns
      - host_col / time_col: resolved column names
      - per_server: dict of per-server statistics
      - global_stats: overall statistics
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        print(f"ERROR: File not found: {csv_path}")
        sys.exit(1)

    file_size_mb = csv_path.stat().st_size / (1024 * 1024)
    print(f"\n{'='*80}")
    print(f"DATA AUDIT: {csv_path.name}")
    print(f"File size: {file_size_mb:.1f} MB")
    print(f"{'='*80}\n")

    # Read a small sample to detect columns
    sample = pd.read_csv(csv_path, nrows=5)
    all_columns = list(sample.columns)
    print(f"Columns ({len(all_columns)}):")
    for i, col in enumerate(all_columns):
        print(f"  [{i}] {col} (dtype: {sample[col].dtype})")

    # Resolve host and time columns
    if host_col is None:
        host_col = guess_host_column(all_columns)
        if host_col:
            print(f"\n[AUTO] Host column guessed: '{host_col}'")
        else:
            print("\n[WARN] Could not guess host column. Use --host to specify.")
            print("       Will treat entire file as a single server.")

    if time_col is None:
        time_col = guess_time_column(all_columns)
        if time_col:
            print(f"[AUTO] Time column guessed: '{time_col}'")
        else:
            print("[WARN] Could not guess time column. Use --time to specify.")

    # Categorise columns
    categories = detect_column_categories(all_columns)
    print(f"\nColumn categories:")
    for cat, cols in categories.items():
        if cols:
            print(f"  {cat}: {cols}")

    temp_cols = categories["temperature"]
    if temp_cols:
        print(f"\nTemperature columns detected: {temp_cols}")
    else:
        print("\n[WARN] No temperature columns detected!")

    # Chunked processing
    print(f"\nProcessing in chunks of {chunksize:,} rows...")

    # Accumulators
    server_stats: dict[str, dict] = {}
    total_rows = 0
    col_missing_counts = {col: 0 for col in all_columns}
    col_dtypes = {}
    chunk_count = 0

    for chunk in pd.read_csv(csv_path, chunksize=chunksize, low_memory=False):
        chunk_count += 1
        n = len(chunk)
        total_rows += n

        if chunk_count == 1:
            col_dtypes = {col: str(chunk[col].dtype) for col in all_columns}

        # Parse time column if available
        if time_col and time_col in chunk.columns:
            chunk[time_col] = pd.to_datetime(chunk[time_col], errors="coerce")

        # Count missing values per column
        for col in all_columns:
            if col in chunk.columns:
                col_missing_counts[col] += chunk[col].isna().sum()

        # Per-server stats
        if host_col and host_col in chunk.columns:
            for server, group in chunk.groupby(host_col):
                server = str(server)
                if server not in server_stats:
                    server_stats[server] = {
                        "n_rows": 0,
                        "time_min": None,
                        "time_max": None,
                        "col_missing": {col: 0 for col in all_columns},
                    }
                    # Add accumulators for numeric columns
                    for col in all_columns:
                        if col not in [host_col, time_col]:
                            server_stats[server][f"_sum_{col}"] = 0.0
                            server_stats[server][f"_sum2_{col}"] = 0.0
                            server_stats[server][f"_min_{col}"] = np.inf
                            server_stats[server][f"_max_{col}"] = -np.inf
                            server_stats[server][f"_count_{col}"] = 0

                ss = server_stats[server]
                ss["n_rows"] += len(group)

                # Time range
                if time_col and time_col in group.columns:
                    valid_times = group[time_col].dropna()
                    if len(valid_times) > 0:
                        chunk_min = valid_times.min()
                        chunk_max = valid_times.max()
                        if ss["time_min"] is None or chunk_min < ss["time_min"]:
                            ss["time_min"] = chunk_min
                        if ss["time_max"] is None or chunk_max > ss["time_max"]:
                            ss["time_max"] = chunk_max

                # Missing counts
                for col in all_columns:
                    if col in group.columns:
                        ss["col_missing"][col] += group[col].isna().sum()

                # Running stats for numeric columns
                for col in all_columns:
                    if col not in [host_col, time_col] and col in group.columns:
                        numeric_vals = pd.to_numeric(group[col], errors="coerce").dropna()
                        if len(numeric_vals) > 0:
                            ss[f"_sum_{col}"] += numeric_vals.sum()
                            ss[f"_sum2_{col}"] += (numeric_vals ** 2).sum()
                            ss[f"_min_{col}"] = min(ss[f"_min_{col}"], numeric_vals.min())
                            ss[f"_max_{col}"] = max(ss[f"_max_{col}"], numeric_vals.max())
                            ss[f"_count_{col}"] += len(numeric_vals)
        else:
            # No host column — treat as single server
            server = "__ALL__"
            if server not in server_stats:
                server_stats[server] = {
                    "n_rows": 0,
                    "time_min": None,
                    "time_max": None,
                    "col_missing": {col: 0 for col in all_columns},
                }
            ss = server_stats[server]
            ss["n_rows"] += n
            if time_col and time_col in chunk.columns:
                valid_times = chunk[time_col].dropna()
                if len(valid_times) > 0:
                    chunk_min = valid_times.min()
                    chunk_max = valid_times.max()
                    if ss["time_min"] is None or chunk_min < ss["time_min"]:
                        ss["time_min"] = chunk_min
                    if ss["time_max"] is None or chunk_max > ss["time_max"]:
                        ss["time_max"] = chunk_max
            for col in all_columns:
                if col in chunk.columns:
                    ss["col_missing"][col] += chunk[col].isna().sum()

        if chunk_count % 10 == 0:
            print(f"  ... processed {total_rows:,} rows ({chunk_count} chunks)")

    print(f"\nTotal rows: {total_rows:,}")
    print(f"Total chunks: {chunk_count}")

    # ── Global missingness ──
    print(f"\n{'-'*60}")
    print("GLOBAL COLUMN MISSINGNESS")
    print(f"{'-'*60}")
    print(f"{'Column':<40} {'Missing':>10} {'%':>8}")
    print(f"{'-'*40} {'-'*10} {'-'*8}")
    for col in all_columns:
        missing = col_missing_counts[col]
        pct = 100.0 * missing / total_rows if total_rows > 0 else 0
        flag = " [!]" if pct > 10 else ""
        print(f"{col:<40} {missing:>10,} {pct:>7.2f}%{flag}")

    # ── Per-server summary ──
    n_servers = len(server_stats)
    print(f"\n{'-'*60}")
    print(f"PER-SERVER SUMMARY ({n_servers} servers)")
    print(f"{'-'*60}")

    # Build a summary table
    summary_rows = []
    for server in sorted(server_stats.keys()):
        ss = server_stats[server]
        days = None
        if ss["time_min"] is not None and ss["time_max"] is not None:
            delta = ss["time_max"] - ss["time_min"]
            days = delta.total_seconds() / 86400

        # Temperature missingness
        temp_missing_pct = {}
        for tc in temp_cols:
            if tc in ss["col_missing"]:
                pct = 100.0 * ss["col_missing"][tc] / ss["n_rows"] if ss["n_rows"] > 0 else 0
                temp_missing_pct[tc] = pct

        # Mean temperature
        temp_means = {}
        for tc in temp_cols:
            count_key = f"_count_{tc}"
            sum_key = f"_sum_{tc}"
            if count_key in ss and ss[count_key] > 0:
                temp_means[tc] = ss[sum_key] / ss[count_key]

        # Temperature range
        temp_ranges = {}
        for tc in temp_cols:
            min_key = f"_min_{tc}"
            max_key = f"_max_{tc}"
            if min_key in ss and ss[min_key] != np.inf:
                temp_ranges[tc] = (ss[min_key], ss[max_key])

        summary_rows.append({
            "server": server,
            "n_rows": ss["n_rows"],
            "days": days,
            "time_min": ss["time_min"],
            "time_max": ss["time_max"],
            "temp_missing_pct": temp_missing_pct,
            "temp_means": temp_means,
            "temp_ranges": temp_ranges,
        })

    # Print summary table
    print(f"\n{'Server':<25} {'Rows':>8} {'Days':>8} {'Start':<22} {'End':<22}")
    print(f"{'-'*25} {'-'*8} {'-'*8} {'-'*22} {'-'*22}")
    for row in summary_rows:
        days_str = f"{row['days']:.1f}" if row['days'] is not None else "N/A"
        start_str = str(row['time_min'])[:19] if row['time_min'] is not None else "N/A"
        end_str = str(row['time_max'])[:19] if row['time_max'] is not None else "N/A"
        print(f"{row['server']:<25} {row['n_rows']:>8,} {days_str:>8} {start_str:<22} {end_str:<22}")

    # Temperature column details per server
    if temp_cols:
        print(f"\n{'-'*60}")
        print("TEMPERATURE COLUMNS -- PER-SERVER STATISTICS")
        print(f"{'-'*60}")
        for tc in temp_cols:
            print(f"\n  Column: {tc}")
            print(f"  {'Server':<25} {'Mean':>8} {'Min':>8} {'Max':>8} {'Missing%':>10}")
            print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
            for row in summary_rows:
                mean_str = f"{row['temp_means'].get(tc, float('nan')):.1f}" if tc in row['temp_means'] else "N/A"
                if tc in row['temp_ranges']:
                    min_str = f"{row['temp_ranges'][tc][0]:.1f}"
                    max_str = f"{row['temp_ranges'][tc][1]:.1f}"
                else:
                    min_str = "N/A"
                    max_str = "N/A"
                miss_str = f"{row['temp_missing_pct'].get(tc, 0):.2f}%"
                print(f"  {row['server']:<25} {mean_str:>8} {min_str:>8} {max_str:>8} {miss_str:>10}")

    # ── Gate check ──
    print(f"\n{'='*60}")
    print("PHASE 1 GATE CHECK")
    print(f"{'='*60}")

    # Count servers with >= 60 days coverage and < 10% missing in temp + workload
    qualifying_servers = []
    workload_cols = categories["cpu_load"] + categories["memory"] + categories["network"] + categories["disk"]
    gate_cols = temp_cols + workload_cols

    for row in summary_rows:
        if row["days"] is None or row["days"] < 60:
            continue
        # Check missingness in gate columns
        ss = server_stats[row["server"]]
        passes = True
        for gc in gate_cols:
            if gc in ss["col_missing"]:
                pct = 100.0 * ss["col_missing"][gc] / ss["n_rows"] if ss["n_rows"] > 0 else 100
                if pct >= 10:
                    passes = False
                    break
        if passes:
            qualifying_servers.append(row["server"])

    n_qualifying = len(qualifying_servers)
    gate_pass = n_qualifying >= 40

    print(f"\nServers with >=60 days coverage & <10% missing (temp + workload): {n_qualifying}")
    print(f"Gate threshold: 40 servers")
    print(f"Gate verdict: {'PASS' if gate_pass else 'FAIL'}")

    if not gate_pass:
        print(f"\n[WARN] Only {n_qualifying} servers qualify. Gate requires 40.")
        if n_qualifying > 0:
            print(f"Qualifying servers: {qualifying_servers[:20]}{'...' if len(qualifying_servers) > 20 else ''}")

    # ── Sampling interval ──
    print(f"\n{'-'*60}")
    print("SAMPLING INTERVAL ESTIMATE")
    print(f"{'-'*60}")
    # Read a small sample to estimate
    if time_col:
        try:
            sample_df = pd.read_csv(csv_path, nrows=10000)
            if host_col and host_col in sample_df.columns:
                first_server = sample_df[host_col].iloc[0]
                server_sample = sample_df[sample_df[host_col] == first_server].copy()
            else:
                server_sample = sample_df.copy()
            server_sample[time_col] = pd.to_datetime(server_sample[time_col], errors="coerce")
            server_sample = server_sample.dropna(subset=[time_col]).sort_values(time_col)
            diffs = server_sample[time_col].diff().dropna()
            if len(diffs) > 0:
                median_diff = diffs.median()
                print(f"Median sampling interval (first server sample): {median_diff}")
                print(f"Min: {diffs.min()}, Max: {diffs.max()}")
            else:
                print("Could not compute sampling interval.")
        except Exception as e:
            print(f"Could not estimate sampling interval: {e}")
    else:
        print("No time column available for interval estimation.")

    # ── Column dtype summary ──
    print(f"\n{'-'*60}")
    print("COLUMN DTYPES (from first chunk)")
    print(f"{'-'*60}")
    for col, dtype in col_dtypes.items():
        print(f"  {col:<40} {dtype}")

    # Return results dict
    return {
        "file": str(csv_path),
        "file_size_mb": file_size_mb,
        "total_rows": total_rows,
        "n_servers": n_servers,
        "columns": all_columns,
        "column_categories": {k: v for k, v in categories.items() if v},
        "col_dtypes": col_dtypes,
        "host_col": host_col,
        "time_col": time_col,
        "temp_cols": temp_cols,
        "n_qualifying_servers": n_qualifying,
        "gate_pass": gate_pass,
        "summary_rows": summary_rows,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Audit a data-centre CSV dataset (chunked processing).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("csv_file", help="Path to the CSV file to audit")
    parser.add_argument("--host", default=None, help="Column name for host/server identifier")
    parser.add_argument("--time", default=None, help="Column name for timestamp")
    parser.add_argument("--chunksize", type=int, default=100_000, help="Rows per chunk (default: 100000)")
    args = parser.parse_args()

    results = audit_file(
        csv_path=args.csv_file,
        host_col=args.host,
        time_col=args.time,
        chunksize=args.chunksize,
    )

    print(f"\n{'='*60}")
    print("AUDIT COMPLETE")
    print(f"{'='*60}")
    print(f"File: {results['file']}")
    print(f"Size: {results['file_size_mb']:.1f} MB")
    print(f"Rows: {results['total_rows']:,}")
    print(f"Servers: {results['n_servers']}")
    print(f"Gate: {'PASS' if results['gate_pass'] else 'FAIL'} ({results['n_qualifying_servers']} qualifying servers)")


if __name__ == "__main__":
    main()
