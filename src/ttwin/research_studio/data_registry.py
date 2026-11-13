"""Data Registry for the Thermal Twin Research Studio.

Provides an authoritative, programmatic catalog of all raw datasets,
interim parquets, feature engineering artifacts, trained model checkpoints,
and experiment result files.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any
import json
import yaml


@dataclass
class DatasetDescriptor:
    name: str
    display_name: str
    hardware_description: str
    citation: str
    license: str
    raw_path: Path
    interim_dir: Path
    combined_parquet: Path
    expected_servers: int
    expected_columns: List[str]
    time_column: str = "Time"
    server_column: str = "Series"
    sampling_interval_min: int = 10
    has_fan_telemetry: bool = True
    has_inlet_telemetry: bool = True


@dataclass
class ExperimentFileDescriptor:
    key: str
    name: str
    category: str
    relative_path: str
    path: Path
    description: str
    required_columns: List[str] = field(default_factory=list)


@dataclass
class ModelCheckpointDescriptor:
    name: str
    path: Path
    format: str
    feature_set: str
    target_type: str
    horizon_steps: int
    horizon_label: str
    dataset: str
    feature_names: List[str]
    model_class: str
    created_at: Optional[str] = None
    git_commit: Optional[str] = None


class DataRegistry:
    """Central registry identifying actual available research artifacts."""

    def __init__(self, root_dir: Optional[Path] = None):
        if root_dir is None:
            # Resolves from src/ttwin/research_studio -> src/ttwin -> src -> repo_root
            self.root_dir = Path(__file__).resolve().parents[3]
        else:
            self.root_dir = Path(root_dir).resolve()

        self.data_dir = self.root_dir / "data"
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.experiments_dir = self.root_dir / "experiments"
        self.models_dir = self.root_dir / "models"
        self.checkpoints_dir = self.root_dir / "checkpoints"
        self.configs_dir = self.root_dir / "configs"

        self._datasets: Dict[str, DatasetDescriptor] = self._init_datasets()
        self._experiments: Dict[str, ExperimentFileDescriptor] = self._init_experiments()

    def _init_datasets(self) -> Dict[str, DatasetDescriptor]:
        return {
            "dataset1": DatasetDescriptor(
                name="dataset1",
                display_name="Cluster 1 (Dell PowerEdge R840)",
                hardware_description="2U dense 4-socket rack server (4x Intel Xeon Gold 6148, 80 cores, 1.5 TB RAM)",
                citation="Ilager et al., UCC 2023 (Zenodo 10.5281/zenodo.10069402)",
                license="CC BY 4.0",
                raw_path=self.raw_dir / "dataset1_combined.csv",
                interim_dir=self.interim_dir / "dataset1",
                combined_parquet=self.interim_dir / "dataset1" / "_combined.parquet",
                expected_servers=81,
                has_fan_telemetry=True,
                has_inlet_telemetry=True,
                expected_columns=[
                    "Series", "Time", "CPU_Load", "Power", "Network_RX", "Network_TX",
                    "Inlet_Temperature", "CPU1_Temperature", "CPU2_Temperature",
                    "Fan_speed1", "Fan_speed2", "Fan_speed3", "Fan_speed4",
                    "Ram", "Ram_Used", "No_Of_Running_vms", "CPU_cores", "CPU_cores_used",
                ],
            ),
            "dataset2": DatasetDescriptor(
                name="dataset2",
                display_name="Cluster 2 (Dell PowerEdge C6320)",
                hardware_description="2U 4-node modular sled (each node 2x Intel Xeon E5-2697v4, 32 cores, 128 GB RAM)",
                citation="Ilager et al., UCC 2023 (Zenodo 10.5281/zenodo.10069402)",
                license="CC BY 4.0",
                raw_path=self.raw_dir / "dataset2_combined.csv",
                interim_dir=self.interim_dir / "dataset2",
                combined_parquet=self.interim_dir / "dataset2" / "_combined.parquet",
                expected_servers=64,
                has_fan_telemetry=False,
                has_inlet_telemetry=False,
                expected_columns=[
                    "Series", "Time", "CPU_Load", "CPU1_Temperature", "CPU2_Temperature",
                    "Ram", "Ram_Used", "CPU_cores", "CPU_cores_used", "Power",
                    "Network_RX", "Network_TX", "No_Of_Running_vms",
                ],
            ),
        }

    def _init_experiments(self) -> Dict[str, ExperimentFileDescriptor]:
        exps = [
            # v3 Comprehensive Experiments
            ExperimentFileDescriptor(
                key="v3_e1_bootstrap",
                name="Paired Bootstrap Skill vs Persistence",
                category="v3_experiments",
                relative_path="experiments/v3_experiments/e1_bootstrap_summary.csv",
                path=self.experiments_dir / "v3_experiments" / "e1_bootstrap_summary.csv",
                description="Paired bootstrap resampling over physical servers and calendar weeks for 1h horizon.",
                required_columns=["dataset", "overall_mae_persistence", "overall_mae_s1_lgbm", "overall_skill_pct"],
            ),
            ExperimentFileDescriptor(
                key="v3_sensor_ablation",
                name="Sensor & Telemetry Ablation Study",
                category="v3_experiments",
                relative_path="experiments/v3_experiments/sensor_ablation.csv",
                path=self.experiments_dir / "v3_experiments" / "sensor_ablation.csv",
                description="Isolating the predictive utility of cooling fans, power, and inlet temperature.",
                required_columns=["configuration", "n_features", "test_mae", "test_rmse", "skill_vs_pers_pct"],
            ),
            ExperimentFileDescriptor(
                key="v3_splits_bc",
                name="GroupKFold and Strict Holdout Across Horizons",
                category="v3_experiments",
                relative_path="experiments/v3_experiments/split_b_and_c_horizons.csv",
                path=self.experiments_dir / "v3_experiments" / "split_b_and_c_horizons.csv",
                description="Split B 5-fold cross-validation and Split C across 10min, 1h, 6h horizons with 5 random seed draws.",
                required_columns=["split", "horizon", "persistence_mae_mean", "s1_mae_mean", "s4_mae_mean"],
            ),
            ExperimentFileDescriptor(
                key="v3_planning_adaptation",
                name="Fleet-Wide Planning Adaptation (All 5 Folds)",
                category="v3_experiments",
                relative_path="experiments/v3_experiments/planning_adaptation_all_folds.csv",
                path=self.experiments_dir / "v3_experiments" / "planning_adaptation_all_folds.csv",
                description="Cold-start adaptation on 81 servers across budgets 0h to 168h in planning regime.",
                required_columns=["strategy", "budget_hours", "n_servers", "mean_mae", "std_mae"],
            ),
            ExperimentFileDescriptor(
                key="v3_cross_cluster",
                name="Cross-Cluster Hardware Adaptation (C1 -> C2)",
                category="v3_experiments",
                relative_path="experiments/v3_experiments/cross_cluster_adaptation.csv",
                path=self.experiments_dir / "v3_experiments" / "cross_cluster_adaptation.csv",
                description="Transfer from Dell R840 to Dell C6320 target nodes across adaptation budgets.",
                required_columns=["budget_hours", "strategy", "mean_mae", "ci_95_low", "ci_95_high"],
            ),
            ExperimentFileDescriptor(
                key="v3_leakage_study",
                name="Random KFold vs GroupKFold (Leakage Demonstration)",
                category="v3_experiments",
                relative_path="experiments/v3_experiments/random_vs_group_identical_rows.csv",
                path=self.experiments_dir / "v3_experiments" / "random_vs_group_identical_rows.csv",
                description="Quantifies optimism bias and data leakage of random row splitting on identical telemetry rows.",
                required_columns=["feature_set", "regime", "random_kfold_mae", "group_kfold_mae", "leakage_underestimation_pct"],
            ),
            ExperimentFileDescriptor(
                key="v3_learning_rates",
                name="S1 LightGBM Learning Rate Convergence",
                category="v3_experiments",
                relative_path="experiments/v3_experiments/s1_learning_rate_study.csv",
                path=self.experiments_dir / "v3_experiments" / "s1_learning_rate_study.csv",
                description="Verifies whether early stopping convergence of S1 is an artifact of learning rate.",
                required_columns=["learning_rate", "best_iteration", "test_mae", "skill_score_pct"],
            ),
            ExperimentFileDescriptor(
                key="v3_manifest",
                name="v3 Execution Metadata Manifest",
                category="v3_experiments",
                relative_path="experiments/v3_experiments/execution_manifest.json",
                path=self.experiments_dir / "v3_experiments" / "execution_manifest.json",
                description="Full execution audit trail including commit hash, random seeds, and per-experiment status.",
            ),

            # v2 Benchmark Experiments
            ExperimentFileDescriptor(
                key="v2_split_a",
                name="v2 Chronological Ablation (Split A)",
                category="v2_ablation",
                relative_path="experiments/v2_ablation/results_split_a.csv",
                path=self.experiments_dir / "v2_ablation" / "results_split_a.csv",
                description="Full model & feature set ablation under 70/10/20 chronological split with 6h embargo.",
                required_columns=["dataset", "horizon", "feature_set", "model", "mae", "rmse"],
            ),
            ExperimentFileDescriptor(
                key="v2_split_b_groupkfold",
                name="v2 GroupKFold Cross-Validation (Split B)",
                category="v2_ablation",
                relative_path="experiments/v2_ablation/results_split_b_groupkfold.csv",
                path=self.experiments_dir / "v2_ablation" / "results_split_b_groupkfold.csv",
                description="5-fold GroupKFold by server across all physical nodes.",
                required_columns=["dataset", "horizon", "feature_set", "model", "mae"],
            ),
            ExperimentFileDescriptor(
                key="v2_split_c",
                name="v2 Strict Spatiotemporal Holdout (Split C)",
                category="v2_ablation",
                relative_path="experiments/v2_ablation/results_split_c.csv",
                path=self.experiments_dir / "v2_ablation" / "results_split_c.csv",
                description="Unseen physical servers evaluated on future time window.",
                required_columns=["dataset", "horizon", "feature_set", "model", "mae"],
            ),
            ExperimentFileDescriptor(
                key="v2_shap_summary",
                name="v2 SHAP Feature Importance",
                category="v2_ablation",
                relative_path="experiments/v2_ablation/shap_summary.csv",
                path=self.experiments_dir / "v2_ablation" / "shap_summary.csv",
                description="Mean absolute SHAP value importance for S4 LightGBM model.",
                required_columns=["feature", "mean_abs_shap"],
            ),
            ExperimentFileDescriptor(
                key="v2_conformal_comparison",
                name="Conformal Prediction Methods Comparison",
                category="v2_uncertainty",
                relative_path="experiments/v2_uncertainty/conformal_methods_comparison.csv",
                path=self.experiments_dir / "v2_uncertainty" / "conformal_methods_comparison.csv",
                description="Constant-width vs Normalized Conformal vs CQR on held-out test servers.",
                required_columns=["method", "nominal_coverage", "empirical_coverage", "mean_interval_width"],
            ),
            ExperimentFileDescriptor(
                key="v2_conformal_regimes",
                name="Conformal Coverage by Regime & Temperature Bin",
                category="v2_uncertainty",
                relative_path="experiments/v2_uncertainty/coverage_by_regime_and_bin.csv",
                path=self.experiments_dir / "v2_uncertainty" / "coverage_by_regime_and_bin.csv",
                description="Conditional coverage breakdowns across Idle/Active/Heavy load and Cool/Moderate/Hot bins.",
                required_columns=["category", "group_bin", "sample_count", "cqr_coverage"],
            ),
            ExperimentFileDescriptor(
                key="v2_weekly_drift",
                name="Weekly Uncertainty Coverage Drift",
                category="v2_uncertainty",
                relative_path="experiments/v2_uncertainty/weekly_coverage_drift.csv",
                path=self.experiments_dir / "v2_uncertainty" / "weekly_coverage_drift.csv",
                description="Coverage tracking over 8 consecutive test weeks.",
                required_columns=["iso_week", "test_samples", "constant_width_coverage", "cqr_coverage"],
            ),
            ExperimentFileDescriptor(
                key="v2_local_recalibration",
                name="Local Conformal Recalibration",
                category="v2_uncertainty",
                relative_path="experiments/v2_uncertainty/local_recalibration_results.csv",
                path=self.experiments_dir / "v2_uncertainty" / "local_recalibration_results.csv",
                description="Recalibrating conformal quantile q_hat using N hours of local target server data.",
            ),
            ExperimentFileDescriptor(
                key="v2_fault_injection",
                name="Thermal Fault Injection Simulation",
                category="v2_fault_injection",
                relative_path="experiments/v2_fault_injection/fault_injection_results.json",
                path=self.experiments_dir / "v2_fault_injection" / "fault_injection_results.json",
                description="Conformal alert safety performance under fan failure, load surge, and sensor drift.",
            ),
        ]
        return {exp.key: exp for exp in exps}

    def list_datasets(self) -> List[str]:
        return list(self._datasets.keys())

    def get_dataset(self, name: str) -> DatasetDescriptor:
        if name not in self._datasets:
            raise KeyError(f"Dataset '{name}' not found in registry. Options: {list(self._datasets.keys())}")
        return self._datasets[name]

    def list_experiments(self, category: Optional[str] = None) -> List[ExperimentFileDescriptor]:
        if category is None:
            return list(self._experiments.values())
        return [exp for exp in self._experiments.values() if exp.category == category]

    def get_experiment(self, key: str) -> ExperimentFileDescriptor:
        if key not in self._experiments:
            raise KeyError(f"Experiment '{key}' not found in registry. Options: {list(self._experiments.keys())}")
        return self._experiments[key]

    def check_dataset_status(self, name: str) -> Dict[str, Any]:
        """Check the physical availability and metadata of a dataset."""
        ds = self.get_dataset(name)
        raw_exists = ds.raw_path.exists()
        raw_size = ds.raw_path.stat().st_size if raw_exists else 0

        combined_exists = ds.combined_parquet.exists()
        combined_size = ds.combined_parquet.stat().st_size if combined_exists else 0

        per_server_files = []
        if ds.interim_dir.exists():
            per_server_files = [
                f for f in ds.interim_dir.glob("*.parquet")
                if not f.name.startswith("_")
            ]

        return {
            "name": ds.name,
            "display_name": ds.display_name,
            "hardware": ds.hardware_description,
            "raw_available": raw_exists,
            "raw_path": str(ds.raw_path),
            "raw_size_bytes": raw_size,
            "interim_available": combined_exists,
            "combined_parquet_path": str(ds.combined_parquet),
            "combined_size_bytes": combined_size,
            "per_server_count": len(per_server_files),
            "expected_servers": ds.expected_servers,
            "has_fan_telemetry": ds.has_fan_telemetry,
            "has_inlet_telemetry": ds.has_inlet_telemetry,
        }

    def list_available_checkpoints(self) -> List[ModelCheckpointDescriptor]:
        """Scan models/ and checkpoints/ for serialized model files.
        Returns empty list if none exist, without inventing any.
        """
        checkpoints: List[ModelCheckpointDescriptor] = []
        search_dirs = [self.models_dir, self.checkpoints_dir]

        seen_stems = set()
        for d in search_dirs:
            if not d.exists():
                continue
            # Sort with joblib first so it takes priority over txt
            for model_file in sorted(d.glob("*.*"), key=lambda p: (0 if p.suffix == ".joblib" else 1)):
                ext = model_file.suffix.lower()
                if ext in [".joblib", ".pkl", ".pickle", ".bin", ".pt", ".onnx", ".model", ".txt"]:
                    if model_file.stem in seen_stems:
                        continue
                    seen_stems.add(model_file.stem)
                    # Attempt to read companion metadata JSON if present
                    meta_file = model_file.with_suffix(".json")
                    meta: Dict[str, Any] = {}
                    if meta_file.exists():
                        try:
                            with open(meta_file, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                        except Exception:
                            meta = {}

                    checkpoints.append(
                        ModelCheckpointDescriptor(
                            name=model_file.stem,
                            path=model_file,
                            format=ext.lstrip("."),
                            feature_set=meta.get("feature_set", "S1"),
                            target_type=meta.get("target_type", "delta"),
                            horizon_steps=meta.get("horizon_steps", 6),
                            horizon_label=meta.get("horizon_label", "1h"),
                            dataset=meta.get("dataset", "dataset1"),
                            feature_names=meta.get("feature_names", []),
                            model_class=meta.get("model_class", "LightGBMModel"),
                            created_at=meta.get("created_at"),
                            git_commit=meta.get("git_commit"),
                        )
                    )

        return checkpoints

    def get_protocol(self) -> Dict[str, Any]:
        """Read frozen protocol configuration from configs/protocol.yaml."""
        protocol_path = self.configs_dir / "protocol.yaml"
        if protocol_path.exists():
            with open(protocol_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_columns_config(self) -> Dict[str, Any]:
        """Read column schema mapping from configs/columns.yaml."""
        columns_path = self.configs_dir / "columns.yaml"
        if columns_path.exists():
            with open(columns_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        return {}
