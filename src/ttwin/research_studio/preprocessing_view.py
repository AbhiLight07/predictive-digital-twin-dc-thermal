"""Preprocessing View & Audit Module.

Provides exact side-by-side comparisons of raw observations versus
cleaned interim telemetry, tracking transformations, row dropped counts,
temperature clipping thresholds, and data quality metrics.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader


class PreprocessingView:
    """Computes verifiable transformations, data quality profiles, and before/after comparisons."""

    def __init__(self, loader: Optional[DatasetLoader] = None):
        self.loader = loader or DatasetLoader()
        self.registry = self.loader.registry

    def get_transformation_pipeline_spec(self, dataset_name: str) -> List[Dict[str, Any]]:
        """Return the exact sequence of preprocessing steps executed in the repository."""
        d1_diff = 2_748_525 - 2_748_039  # 486 duplicate rows dropped in D1
        d2_diff = 1_281_024 - 1_280_640  # 384 duplicate rows dropped in D2
        dropped_dups = d1_diff if dataset_name == "dataset1" else d2_diff

        return [
            {
                "step": 1,
                "name": "Strip Hostname Whitespace",
                "function": "ttwin.clean.strip_host_names",
                "source_file": "src/ttwin/clean.py",
                "description": "Strips leading/trailing whitespace from server identifier strings.",
                "input_columns": ["Series"],
                "output_columns": ["Series"],
                "effect": "Prevents key mismatch in groupby operations and spatial splitting.",
                "affected_rows": "All rows with whitespace padding",
            },
            {
                "step": 2,
                "name": "Parse Timestamps",
                "function": "ttwin.clean.parse_timestamps",
                "source_file": "src/ttwin/clean.py",
                "description": "Converts string timestamp column to ISO datetime64[ns] and sorts chronologically.",
                "input_columns": ["Time"],
                "output_columns": ["Time (datetime64[ns])"],
                "effect": "Establishes strict chronological ordering.",
                "affected_rows": "All rows",
            },
            {
                "step": 3,
                "name": "Deduplicate Timestamps",
                "function": "ttwin.clean.remove_duplicate_timestamps",
                "source_file": "src/ttwin/clean.py",
                "description": "Drops rows sharing identical (Series, Time), keeping the first occurrence.",
                "input_columns": ["Series", "Time"],
                "output_columns": ["Series", "Time"],
                "effect": f"Removes exactly {dropped_dups:,} duplicate telemetry collision records.",
                "affected_rows": f"{dropped_dups:,} rows dropped",
            },
            {
                "step": 4,
                "name": "Physical Temperature Clipping",
                "function": "ttwin.clean.clip_temperature",
                "source_file": "src/ttwin/clean.py",
                "description": "Detects temperature readings outside physically plausible range [-10.0°C, 110.0°C] and converts them to NaN.",
                "input_columns": ["CPU1_Temperature", "CPU2_Temperature", "Inlet_Temperature (if present)"],
                "output_columns": ["Clipped temperature columns"],
                "effect": "Eliminates unphysical thermal sensor anomalies without distorting plausible high-heat spikes.",
                "affected_rows": "Anomalous readings converted to NaN",
            },
            {
                "step": 5,
                "name": "Memory Downcasting",
                "function": "ttwin.io.downcast_dtypes",
                "source_file": "src/ttwin/io.py",
                "description": "Downcasts float64 to float32 and int64 to int32/int16.",
                "input_columns": "All numeric columns",
                "output_columns": "Downcasted numeric columns",
                "effect": "Reduces in-memory DataFrame footprint by >50%.",
                "affected_rows": "All rows",
            },
            {
                "step": 6,
                "name": "Chunked Parquet Serialization",
                "function": "scripts.make_parquet.convert_dataset",
                "source_file": "scripts/make_parquet.py",
                "description": "Writes per-server Parquet files and concatenated _combined.parquet using Snappy compression.",
                "input_columns": "All columns",
                "output_columns": "Snappy Parquet files",
                "effect": "Reduces disk storage from ~585 MB (CSV) to ~89 MB (Parquet) on Dataset 1.",
                "affected_rows": "All rows",
            },
            {
                "step": 7,
                "name": "10-Min Continuous Grid Alignment",
                "function": "ttwin.clean.fill_missing_timestamps",
                "source_file": "src/ttwin/clean.py",
                "description": "Reindexes per server to a regular 10-minute frequency grid, inserting NaNs for unrecorded steps.",
                "input_columns": ["Series", "Time"],
                "output_columns": ["Series", "Time", "Aligned grid"],
                "effect": "Guarantees causal, gap-free lag and horizon time shifts without data leakage.",
                "affected_rows": "Missing 10-min intervals inserted as NaN rows",
            },
        ]

    def get_comparison_summary(self, dataset_name: str) -> Dict[str, Any]:
        """Return before vs after transformation statistics using verified repository artifacts."""
        raw_meta = self.loader.get_raw_metadata(dataset_name)
        interim_meta = self.loader.get_interim_metadata(dataset_name)

        raw_rows = raw_meta.get("total_rows", 0)
        interim_rows = interim_meta.get("total_rows", 0)
        dropped_rows = raw_rows - interim_rows

        return {
            "dataset_name": dataset_name,
            "raw_file": raw_meta.get("file_path"),
            "raw_size_mb": raw_meta.get("file_size_mb"),
            "raw_rows": raw_rows,
            "raw_cols": raw_meta.get("total_columns"),
            "raw_servers": raw_meta.get("server_count"),
            "interim_file": interim_meta.get("file_path"),
            "interim_size_mb": interim_meta.get("file_size_mb"),
            "interim_rows": interim_rows,
            "interim_cols": interim_meta.get("total_columns"),
            "interim_servers": interim_meta.get("server_count"),
            "rows_dropped": dropped_rows,
            "rows_dropped_reason": "Duplicate (Series, Time) timestamps removed (keeping first)",
            "columns_added_or_removed": "Zero columns added or removed in interim step (dtypes downcasted)",
            "temperature_clipping_bounds": "[-10.0°C, 110.0°C]",
            "null_counts_interim": interim_meta.get("null_counts", {}),
        }

    def compute_data_quality_report(
        self,
        dataset_name: str,
        server_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute missing value percentages, distributions, and sampling intervals on actual data."""
        if server_name:
            df = self.loader.load_interim_server(dataset_name, server_name)
        else:
            # Sample 50,000 rows across combined dataset for responsive quality assessment
            df = self.loader.load_interim_combined(dataset_name, max_rows=50_000)

        n_rows = len(df)
        null_counts = df.isnull().sum().to_dict()
        null_pct = {k: round((v / n_rows) * 100, 2) for k, v in null_counts.items()}

        # Temperature statistics
        temp_stats = {}
        for col in ["CPU1_Temperature", "CPU2_Temperature", "Inlet_Temperature"]:
            if col in df.columns:
                valid = df[col].dropna()
                if len(valid) > 0:
                    temp_stats[col] = {
                        "count": int(valid.count()),
                        "mean": float(valid.mean()),
                        "std": float(valid.std()),
                        "min": float(valid.min()),
                        "q25": float(valid.quantile(0.25)),
                        "median": float(valid.median()),
                        "q75": float(valid.quantile(0.75)),
                        "max": float(valid.max()),
                    }

        # Sampling interval check
        time_col = "Time"
        time_gaps_summary = {}
        if time_col in df.columns and len(df) > 1:
            df_sorted = df.sort_values(time_col)
            time_diffs = (df_sorted[time_col].diff().dt.total_seconds() / 60.0).dropna()
            time_gaps_summary = {
                "median_gap_min": float(time_diffs.median()),
                "p95_gap_min": float(time_diffs.quantile(0.95)),
                "max_gap_min": float(time_diffs.max()),
                "pct_10min_exact": float((time_diffs == 10.0).mean() * 100),
            }

        return {
            "n_rows_evaluated": n_rows,
            "null_counts": null_counts,
            "null_pct": null_pct,
            "temp_stats": temp_stats,
            "time_gaps_summary": time_gaps_summary,
        }
