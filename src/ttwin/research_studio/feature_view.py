"""Feature View & Engineering Lab Module.

Explains and generates the feature sets (S0, S1, S2, S3, S4),
distinguishes Monitoring vs Planning regimes, computes feature correlations,
and validates causal alignment without data leakage.
"""

from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np

from ttwin.features import build_features, get_feature_set, compute_effective_cpu_temp
from ttwin.research_studio.dataset_loader import DatasetLoader


class FeatureView:
    """Manages feature definitions, feature matrix generation for inspection, and correlation analysis."""

    def __init__(self, loader: Optional[DatasetLoader] = None):
        self.loader = loader or DatasetLoader()

    def get_pipeline_stages(self) -> List[Dict[str, Any]]:
        """Return the complete 7-stage lineage from raw telemetry to model-ready tensors."""
        return [
            {
                "stage": 1,
                "name": "Raw Telemetry Ingestion",
                "function": "ttwin.io.read_csv_chunked",
                "source_file": "src/ttwin/io.py",
                "inputs": "Original CSV (Series, Time, raw sensor metrics)",
                "outputs": "In-memory DataFrame chunks",
                "description": "Stream-reads multi-hundred megabyte raw CSV traces safely.",
            },
            {
                "stage": 2,
                "name": "Cleaning & Plausibility Gate",
                "function": "ttwin.clean.clean_dataset",
                "source_file": "src/ttwin/clean.py",
                "inputs": "Raw strings, timestamps, noisy sensor numbers",
                "outputs": "Cleaned rows with deduplicated timestamps and clipped temps [-10°C, 110°C]",
                "description": "Standardizes hostnames, eliminates collision duplicates, converts invalid spikes to NaN.",
            },
            {
                "stage": 3,
                "name": "Continuous 10-Minute Grid Alignment",
                "function": "ttwin.clean.fill_missing_timestamps",
                "source_file": "src/ttwin/clean.py",
                "inputs": "Cleaned telemetry with irregular gaps",
                "outputs": "Strictly continuous 10-min grid with NaNs inserted for unrecorded intervals",
                "description": "Enforces uniform time delta so shifting by k rows strictly equals k * 10 minutes.",
            },
            {
                "stage": 4,
                "name": "Effective CPU Temperature Synthesis",
                "function": "ttwin.features.compute_effective_cpu_temp",
                "source_file": "src/ttwin/features.py",
                "inputs": "CPU1_Temperature, CPU2_Temperature",
                "outputs": "CPU_Temperature = max(CPU1, CPU2)",
                "description": "Derives the dominant thermal hotspot for dual-socket rack chassis.",
            },
            {
                "stage": 5,
                "name": "Causal Feature Engineering",
                "function": "ttwin.features.add_lag_features, add_rolling_features, add_calendar_features",
                "source_file": "src/ttwin/features.py",
                "inputs": "Continuous time series up to time t",
                "outputs": "Calendar encodings, temperature lags [1..12], causal rolling mean/std [6..36]",
                "description": "Windows end strictly at t; zero future leakage into the predictor.",
            },
            {
                "stage": 6,
                "name": "Target Formulation (Regime-Specific)",
                "function": "ttwin.features.create_target",
                "source_file": "src/ttwin/features.py",
                "inputs": "Target column at time t + h",
                "outputs": "Monitoring Delta: T(t+h) - T(t) | Planning Absolute: T(t+h)",
                "description": "Separates monitoring delta targets from planning absolute targets.",
            },
            {
                "stage": 7,
                "name": "Disjoint Spatiotemporal Partitioning",
                "function": "ttwin.splits.split_chronological, get_group_kfold_splits, split_strict_4way",
                "source_file": "src/ttwin/splits.py",
                "inputs": "Model-ready DataFrame",
                "outputs": "Strictly disjoint Train, Validation-Select, Calibration, and Test partitions",
                "description": "Enforces 6-hour embargo buffers and complete physical server isolation.",
            },
        ]

    def get_feature_set_descriptions(self) -> Dict[str, Dict[str, Any]]:
        """Return definitions and scientific rationale for each feature set."""
        return {
            "S0": {
                "name": "S0 (Persistence Baseline)",
                "regime": "Monitoring Baseline",
                "feature_count": 0,
                "description": "Predicts future temperature as current observed temperature: T_hat(t+h) = T(t). Zero ML features.",
                "rationale": "Thermal inertia dominates short horizons; this serves as the foundational bar all models must beat.",
            },
            "S1": {
                "name": "S1 (Temperature History & Current State)",
                "regime": "Monitoring (Telemetry Active)",
                "feature_count": 36,
                "description": "Current temperatures (CPU, CPU1, CPU2) + past temperature lags [1, 2, 3, 6, 12] + rolling stats [6, 12, 36].",
                "rationale": "Lean autoregressive model. Evaluates whether pure thermal memory is sufficient for monitoring.",
            },
            "S2": {
                "name": "S2 (Workload & Calendar Only)",
                "regime": "Planning (Workload Scheduling)",
                "feature_count": 14,
                "description": "Workload (CPU load, RAM used, cores used, network I/O, VM count) + cyclic calendar encodings (hour/dow sin/cos).",
                "rationale": "Simulates hypothetical workload placement where temperature sensors do not yet exist.",
            },
            "S3": {
                "name": "S3 (Workload + Calendar + Power)",
                "regime": "Planning (With Server Power)",
                "feature_count": 15,
                "description": "S2 features + real-time server Power (Watts).",
                "rationale": "Measures the incremental value of server power draw for planning predictions.",
            },
            "S4": {
                "name": "S4 (Full Multimodal Telemetry)",
                "regime": "Monitoring (Comprehensive)",
                "feature_count": 56,
                "description": "S3 + chassis fan speeds (1-4) + Inlet temperature + current temperatures + temperature lags & rolling stats.",
                "rationale": "The comprehensive telemetry twin incorporating all available hardware sensors.",
            },
        }

    def generate_server_feature_sample(
        self,
        dataset_name: str,
        server_name: str,
        n_rows: int = 50,
        horizons: Optional[List[int]] = None,
    ) -> pd.DataFrame:
        """Load a single server slice and compute verified features and targets on-the-fly."""
        df_srv = self.loader.load_interim_server(dataset_name, server_name)
        # Take a bounded window for responsive calculation
        if len(df_srv) > 500:
            df_srv = df_srv.iloc[1000 : 1000 + n_rows + 50].copy()

        df_feat = build_features(
            df_srv,
            horizons=horizons or [1, 6, 36],
            lags=[1, 2, 3, 6, 12],
            rolling_windows=[6, 12, 36],
        )
        return df_feat.iloc[:n_rows].reset_index(drop=True)

    def compute_feature_correlations(
        self,
        df_features: pd.DataFrame,
        feature_set: str = "S1",
        target_col: str = "CPU_Temperature_delta_6",
    ) -> pd.DataFrame:
        """Compute Pearson correlation matrix between features and target on actual observations."""
        cols = get_feature_set(feature_set, df_features.columns.tolist())
        target = target_col if target_col in df_features.columns else None

        active_cols = [c for c in cols if c in df_features.columns]
        if target and target not in active_cols:
            active_cols.append(target)

        sub_df = df_features[active_cols].dropna()
        if sub_df.empty or len(active_cols) < 2:
            return pd.DataFrame()

        # If too many features, select the most informative subset
        if len(active_cols) > 20:
            top_cols = [c for c in active_cols if not c.startswith("CPU1_") and not c.startswith("CPU2_")]
            if target and target not in top_cols:
                top_cols.append(target)
            sub_df = sub_df[top_cols]

        return sub_df.corr().round(3)
