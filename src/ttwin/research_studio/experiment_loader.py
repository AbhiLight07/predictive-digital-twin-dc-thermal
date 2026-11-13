"""Experiment Loader for the Research Studio.

Provides unified, validated access to committed experimental outputs (v3, v2, v1),
enabling filtering by dataset, split, horizon, feature set, and model.
Always displays the source CSV file path alongside every result.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import pandas as pd
import numpy as np

from ttwin.research_studio.data_registry import DataRegistry, ExperimentFileDescriptor


class ExperimentLoader:
    """Loads and standardizes experimental results from committed CSV and JSON artifacts."""

    def __init__(self, registry: Optional[DataRegistry] = None):
        self.registry = registry or DataRegistry()
        self._cache: Dict[str, pd.DataFrame] = {}

    def load_experiment_df(self, key: str) -> pd.DataFrame:
        """Load an experiment CSV by its registry key with caching."""
        if key in self._cache:
            return self._cache[key].copy()

        desc = self.registry.get_experiment(key)
        if not desc.path.exists():
            raise FileNotFoundError(f"Experiment artifact not found: {desc.path}")

        if desc.path.suffix.lower() == ".csv":
            df = pd.read_csv(desc.path)
            # Annotate metadata
            df.attrs["source_file"] = desc.relative_path
            df.attrs["description"] = desc.description
            self._cache[key] = df
            return df.copy()
        elif desc.path.suffix.lower() == ".json":
            with open(desc.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.json_normalize(data)
            df.attrs["source_file"] = desc.relative_path
            df.attrs["description"] = desc.description
            self._cache[key] = df
            return df.copy()
        else:
            raise ValueError(f"Unsupported experiment file type: {desc.path.suffix}")

    def load_unified_ablation_benchmarks(
        self,
        dataset: Optional[str] = None,
        split: Optional[str] = None,
        horizon: Optional[str] = None,
        feature_set: Optional[str] = None,
        model: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, str]:
        """Load the primary v2 Split A benchmark table with flexible filtering.

        Returns (filtered_df, source_file_path).
        """
        df = self.load_experiment_df("v2_split_a")
        source_path = "experiments/v2_ablation/results_split_a.csv"

        if dataset and dataset != "all":
            clean_d = dataset.lower().replace(" ", "").replace("_", "")
            df = df[df["dataset"].astype(str).str.lower().str.replace(" ", "").str.replace("_", "") == clean_d]

        if horizon and horizon != "all":
            df = df[df["horizon"].str.lower() == horizon.lower()]

        if feature_set and feature_set != "all":
            df = df[df["feature_set"].str.startswith(feature_set)]

        if model and model != "all":
            df = df[df["model"].str.lower() == model.lower()]

        return df.reset_index(drop=True), source_path

    def load_sensor_ablation(self) -> Tuple[pd.DataFrame, str]:
        """Load the v3 hardware sensor ablation study (fans, power, S1)."""
        df = self.load_experiment_df("v3_sensor_ablation")
        return df, "experiments/v3_experiments/sensor_ablation.csv"

    def load_split_generalisation(self) -> Tuple[pd.DataFrame, str]:
        """Load GroupKFold vs Strict Holdout cross-horizon evaluation."""
        df = self.load_experiment_df("v3_splits_bc")
        return df, "experiments/v3_experiments/split_b_and_c_horizons.csv"

    def load_leakage_comparison(self) -> Tuple[pd.DataFrame, str]:
        """Load Random K-Fold vs GroupKFold on identical telemetry rows."""
        df = self.load_experiment_df("v3_leakage_study")
        return df, "experiments/v3_experiments/random_vs_group_identical_rows.csv"

    def load_adaptation_fleet(self) -> Tuple[pd.DataFrame, str]:
        """Load fleet-wide planning adaptation across all 5 folds."""
        df = self.load_experiment_df("v3_planning_adaptation")
        return df, "experiments/v3_experiments/planning_adaptation_all_folds.csv"

    def load_cross_cluster_adaptation(self) -> Tuple[pd.DataFrame, str]:
        """Load cross-cluster hardware adaptation (Cluster 1 -> Cluster 2)."""
        df = self.load_experiment_df("v3_cross_cluster")
        return df, "experiments/v3_experiments/cross_cluster_adaptation.csv"

    def load_uncertainty_conformal(self) -> Tuple[pd.DataFrame, str]:
        """Load conformal prediction methods comparison (Constant vs Normalized vs CQR)."""
        df = self.load_experiment_df("v2_conformal_comparison")
        return df, "experiments/v2_uncertainty/conformal_methods_comparison.csv"

    def load_uncertainty_regimes(self) -> Tuple[pd.DataFrame, str]:
        """Load conditional conformal coverage across load regimes and temperature bins."""
        df = self.load_experiment_df("v2_conformal_regimes")
        return df, "experiments/v2_uncertainty/coverage_by_regime_and_bin.csv"

    def load_weekly_drift(self) -> Tuple[pd.DataFrame, str]:
        """Load temporal uncertainty coverage tracking across 8 test weeks."""
        df = self.load_experiment_df("v2_weekly_drift")
        return df, "experiments/v2_uncertainty/weekly_coverage_drift.csv"

    def load_execution_manifest(self) -> Dict[str, Any]:
        """Load verified execution manifest from experiments/v3_experiments/execution_manifest.json."""
        manifest_desc = self.registry.get_experiment("v3_manifest")
        if not manifest_desc.path.exists():
            return {}
        with open(manifest_desc.path, "r", encoding="utf-8") as f:
            return json.load(f)
