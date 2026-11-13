"""Schema and Data Integrity Validation Module.

Provides automated checks for schema compliance, column presence,
causal ordering, embargo enforcement, and zero-leakage assertions.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np


class SchemaValidator:
    """Validates telemetry DataFrames and experiment outputs against frozen protocols."""

    @staticmethod
    def validate_dataset_schema(
        df: pd.DataFrame,
        expected_columns: List[str],
        time_col: str = "Time",
        server_col: str = "Series",
    ) -> Dict[str, Any]:
        """Verify that a loaded telemetry dataset conforms to the expected columns and types."""
        errors: List[str] = []
        warnings: List[str] = []

        # 1. Column Presence
        actual_cols = set(df.columns)
        missing_cols = set(expected_columns) - actual_cols
        if missing_cols:
            errors.append(f"Missing required columns: {sorted(list(missing_cols))}")

        # 2. Key column checks
        if time_col not in df.columns:
            errors.append(f"Missing required timestamp column '{time_col}'")
        if server_col not in df.columns:
            errors.append(f"Missing required server column '{server_col}'")

        # 3. Numeric checks
        numeric_cols = [c for c in expected_columns if c not in [time_col, server_col]]
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    warnings.append(f"Column '{col}' is not numeric dtype: {df[col].dtype}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "column_count": len(df.columns),
            "row_count": len(df),
        }

    @staticmethod
    def validate_split_integrity(
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        val_df: Optional[pd.DataFrame] = None,
        time_col: str = "Time",
        server_col: str = "Series",
        embargo_minutes: int = 360,  # 6 hours
        check_server_disjoint: bool = False,
    ) -> Dict[str, Any]:
        """Verify chronological separation, embargo gap, and spatial server isolation."""
        errors: List[str] = []

        # Ensure datetime dtypes
        t_train_max = pd.to_datetime(train_df[time_col]).max()
        t_test_min = pd.to_datetime(test_df[time_col]).min()

        # Check chronological direction
        if t_train_max >= t_test_min:
            errors.append(f"Temporal leakage detected: max train ({t_train_max}) >= min test ({t_test_min})")

        # Check embargo gap
        gap_duration = (t_test_min - t_train_max).total_seconds() / 60.0
        if gap_duration < embargo_minutes:
            errors.append(
                f"Embargo buffer violated: gap is {gap_duration:.1f} min, expected >= {embargo_minutes} min"
            )

        # Check server disjointness if requested
        if check_server_disjoint:
            train_servers = set(train_df[server_col].astype(str).str.strip().unique())
            test_servers = set(test_df[server_col].astype(str).str.strip().unique())
            overlap = train_servers.intersection(test_servers)
            if overlap:
                errors.append(f"Spatial server leakage detected: {len(overlap)} servers in both train and test: {overlap}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "gap_minutes": gap_duration if "gap_duration" in locals() else None,
        }

    @staticmethod
    def validate_experiment_results(
        df: pd.DataFrame,
        required_cols: List[str],
        metric_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Verify that an experimental result CSV is well-formed with valid positive error metrics."""
        errors: List[str] = []
        warnings: List[str] = []

        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            errors.append(f"Missing required metric columns: {missing}")

        metrics = metric_cols or [c for c in ["mae", "rmse", "test_mae", "test_rmse"] if c in df.columns]
        for m in metrics:
            if m in df.columns:
                vals = pd.to_numeric(df[m], errors="coerce").dropna()
                if (vals < 0).any():
                    errors.append(f"Negative error metrics detected in column '{m}'")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "row_count": len(df),
        }
