"""Unit tests for the Research Studio Data Access Layer (src/ttwin/research_studio).

Verifies:
1. DataRegistry discovers real datasets and experiment files.
2. DatasetLoader loads slices and per-server parquets accurately.
3. PreprocessingView tracks verified row drops and transformations.
4. FeatureView extracts causal feature matrices.
5. ModelLoader and ModelInferenceEngine handle missing checkpoints gracefully without fabrication.
6. ExperimentLoader loads committed CSV results with source file annotation.
7. SchemaValidator enforces column presence and data integrity.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from ttwin.research_studio.data_registry import DataRegistry
from ttwin.research_studio.dataset_loader import DatasetLoader
from ttwin.research_studio.preprocessing_view import PreprocessingView
from ttwin.research_studio.feature_view import FeatureView
from ttwin.research_studio.model_loader import ModelLoader
from ttwin.research_studio.inference import ModelInferenceEngine
from ttwin.research_studio.experiment_loader import ExperimentLoader
from ttwin.research_studio.validation import SchemaValidator

# --- data-availability helpers ------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_RAW_D1 = _PROJECT_ROOT / "data" / "raw" / "dataset1_combined.csv"
_RAW_D2 = _PROJECT_ROOT / "data" / "raw" / "dataset2_combined.csv"
_INTERIM_D1 = _PROJECT_ROOT / "data" / "interim" / "dataset1"
_INTERIM_D2 = _PROJECT_ROOT / "data" / "interim" / "dataset2"

_has_raw = _RAW_D1.exists() and _RAW_D2.exists()
_has_interim = _INTERIM_D1.exists() and any(_INTERIM_D1.glob("*.parquet"))

needs_raw = pytest.mark.skipif(not _has_raw, reason="raw data files not present")
needs_interim = pytest.mark.skipif(not _has_interim, reason="interim parquet files not present")
needs_data = pytest.mark.skipif(not (_has_raw and _has_interim), reason="data files not present")


@pytest.fixture
def registry():
    return DataRegistry()


@pytest.fixture
def loader(registry):
    return DatasetLoader(registry)


@needs_data
def test_registry_datasets_exist(registry):
    datasets = registry.list_datasets()
    assert "dataset1" in datasets
    assert "dataset2" in datasets

    d1 = registry.get_dataset("dataset1")
    assert d1.expected_servers == 81
    assert d1.raw_path.exists(), f"Missing {d1.raw_path}"
    assert d1.combined_parquet.exists(), f"Missing {d1.combined_parquet}"

    d2 = registry.get_dataset("dataset2")
    assert d2.expected_servers == 64
    assert d2.raw_path.exists(), f"Missing {d2.raw_path}"
    assert d2.combined_parquet.exists(), f"Missing {d2.combined_parquet}"


@needs_data
def test_registry_experiments_all_exist(registry):
    exps = registry.list_experiments()
    assert len(exps) >= 15
    for exp in exps:
        assert exp.path.exists(), f"Experiment file missing: {exp.path}"


@needs_raw
def test_raw_metadata_loading(loader):
    meta1 = loader.get_raw_metadata("dataset1")
    assert meta1["available"] is True
    assert meta1["total_rows"] == 2_748_525
    assert meta1["server_count"] == 81
    assert "CPU_Temperature" not in meta1["columns"]  # raw has CPU1/CPU2, not combined
    assert "CPU1_Temperature" in meta1["columns"]
    assert "CPU2_Temperature" in meta1["columns"]


@needs_raw
def test_raw_slice_pagination(loader):
    df_slice = loader.load_raw_slice("dataset1", nrows=25, offset=0)
    assert len(df_slice) == 25
    assert "Series" in df_slice.columns
    assert "Time" in df_slice.columns


@needs_interim
def test_interim_server_loading(loader):
    servers = loader.get_interim_servers("dataset1")
    assert len(servers) == 81
    assert "qh2-rcc120" in servers

    df_srv = loader.load_interim_server("dataset1", "qh2-rcc120")
    assert len(df_srv) > 1000
    assert "CPU_Temperature" in df_srv.columns or "CPU1_Temperature" in df_srv.columns


@needs_data
def test_preprocessing_view_dropped_counts(loader):
    view = PreprocessingView(loader)
    comp = view.get_comparison_summary("dataset1")
    assert comp["raw_rows"] == 2_748_525
    assert comp["interim_rows"] == 2_748_039
    assert comp["rows_dropped"] == 486  # 486 duplicate rows removed

    pipeline = view.get_transformation_pipeline_spec("dataset1")
    assert len(pipeline) == 7
    assert pipeline[0]["function"] == "ttwin.clean.strip_host_names"
    assert pipeline[2]["function"] == "ttwin.clean.remove_duplicate_timestamps"


@needs_interim
def test_feature_view_sample_generation(loader):
    fview = FeatureView(loader)
    df_feat = fview.generate_server_feature_sample("dataset1", "qh2-rcc120", n_rows=20)
    assert len(df_feat) == 20
    assert "CPU_Temperature_target_6" in df_feat.columns
    assert "CPU_Temperature_delta_6" in df_feat.columns
    assert "CPU_Temperature_lag_1" in df_feat.columns
    assert "CPU_Temperature_rmean_6" in df_feat.columns


@needs_interim
def test_feature_view_correlations(loader):
    fview = FeatureView(loader)
    df_feat = fview.generate_server_feature_sample("dataset1", "qh2-rcc120", n_rows=100)
    corr = fview.compute_feature_correlations(df_feat, feature_set="S1")
    assert not corr.empty
    assert isinstance(corr, pd.DataFrame)


def test_model_loader_missing_checkpoint_handled_gracefully(registry):
    m_loader = ModelLoader(registry)
    # Checkpoint scanning should return empty without crashing
    checkpoints = m_loader.list_available_models()
    assert isinstance(checkpoints, list)

    model_obj, desc, msg = m_loader.load_checkpoint()
    if not checkpoints:
        assert model_obj is None
        assert desc is None
        assert "Trained checkpoint unavailable" in msg


@needs_data
def test_inference_engine_handles_missing_checkpoint(registry, loader):
    m_loader = ModelLoader(registry)
    engine = ModelInferenceEngine(m_loader, loader)
    t = pd.Timestamp("2018-10-10 00:00:00")
    # Test with non-existent checkpoint
    res = engine.run_inference("dataset1", "qh2-rcc120", t, checkpoint_name="non_existent_model")
    assert res["available"] is False
    assert "not found" in res["error"].lower() or "unavailable" in res["error"].lower()
    assert "python scripts/train_baseline_model.py" in res["reproducible_command"]

    # Also test successful inference when valid checkpoint is present
    if m_loader.has_checkpoints():
        res_real = engine.run_inference("dataset1", "qh2-rcc120", pd.Timestamp("2019-05-15 12:00:00"), horizon_steps=6)
        assert res_real["available"] is True
        assert res_real["model_abs_error"] >= 0.0
        assert "TEST" in res_real["partition"]


@needs_data
def test_experiment_loader_loads_all_benchmarks(registry):
    exp_loader = ExperimentLoader(registry)

    # v3 sensor ablation
    df_sensor, src_sensor = exp_loader.load_sensor_ablation()
    assert len(df_sensor) == 6
    assert "sensor_ablation.csv" in src_sensor

    # v3 leakage study
    df_leak, src_leak = exp_loader.load_leakage_comparison()
    assert len(df_leak) == 4
    assert "random_vs_group_identical_rows.csv" in src_leak

    # v2 split a ablation
    df_a, src_a = exp_loader.load_unified_ablation_benchmarks()
    assert len(df_a) == 48
    assert "results_split_a.csv" in src_a

    # Filtered
    df_d1_s1, _ = exp_loader.load_unified_ablation_benchmarks(dataset="dataset1", feature_set="S1")
    assert len(df_d1_s1) > 0
    assert (df_d1_s1["dataset"] == "dataset1").all()


@needs_raw
def test_schema_validator(loader):
    df_sample = loader.load_raw_slice("dataset1", nrows=50)
    res = SchemaValidator.validate_dataset_schema(
        df_sample,
        expected_columns=["Series", "Time", "CPU_Load", "Power"],
    )
    assert res["valid"] is True
    assert len(res["errors"]) == 0

    # Negative test: missing column
    res_bad = SchemaValidator.validate_dataset_schema(
        df_sample,
        expected_columns=["Series", "Time", "NON_EXISTENT_COLUMN"],
    )
    assert res_bad["valid"] is False
    assert len(res_bad["errors"]) > 0
