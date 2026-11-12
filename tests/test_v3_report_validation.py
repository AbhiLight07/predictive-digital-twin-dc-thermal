"""Test that docs/results_v3.md accurately reflects the committed CSVs."""

from pathlib import Path
import re
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / "docs" / "results_v3.md"
CSV_DIR = ROOT / "experiments" / "v3_experiments"


@pytest.fixture(scope="module")
def report_text():
    assert DOC_PATH.exists(), f"Missing {DOC_PATH}"
    return DOC_PATH.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def csv_data():
    return {
        "e1": pd.read_csv(CSV_DIR / "e1_bootstrap_summary.csv"),
        "sensor": pd.read_csv(CSV_DIR / "sensor_ablation.csv"),
        "splits": pd.read_csv(CSV_DIR / "split_b_and_c_horizons.csv"),
        "planning": pd.read_csv(CSV_DIR / "planning_adaptation_all_folds.csv"),
        "transfer": pd.read_csv(CSV_DIR / "cross_cluster_adaptation.csv"),
        "leakage": pd.read_csv(CSV_DIR / "random_vs_group_identical_rows.csv"),
        "lr": pd.read_csv(CSV_DIR / "s1_learning_rate_study.csv"),
    }


def test_bootstrap_narrative_matches_csv(report_text, csv_data):
    e1 = csv_data["e1"]
    d1 = e1[e1["dataset"] == "Dataset 1"].iloc[0]
    d2 = e1[e1["dataset"] == "Dataset 2"].iloc[0]

    # Verify skill percentages are reported accurately
    assert f"+{d1['overall_skill_pct']:.2f}%" in report_text  # +0.59%
    assert f"+{d2['overall_skill_pct']:.2f}%" in report_text  # +2.83%

    # Verify server CIs crossing zero are explicitly stated
    assert "Server-level 95% confidence intervals cross zero on both datasets" in report_text
    assert f"[{d1['server_ci_95_low_pct']:.2f}%, +{d1['server_ci_95_high_pct']:.2f}%]" in report_text
    assert f"[{d2['server_ci_95_low_pct']:.2f}%, +{d2['server_ci_95_high_pct']:.2f}%]" in report_text
    assert "not statistically significant" in report_text

    # Make sure outdated/invented numbers are NOT in the text
    assert "+0.33% on D2" not in report_text
    assert "[0.03%, 1.15%]" not in report_text


def test_sensor_ablation_matches_csv(report_text, csv_data):
    sensor = csv_data["sensor"]
    s4_full = sensor[sensor["configuration"].str.contains("S4 Full")]["test_mae"].iloc[0]
    s4_nofan = sensor[sensor["configuration"].str.contains("without Fans \\(")] ["test_mae"].iloc[0]
    s4_nopow = sensor[sensor["configuration"].str.contains("without Power")]["test_mae"].iloc[0]
    s1 = sensor[sensor["configuration"].str.contains("S1")]["test_mae"].iloc[0]

    assert f"{s4_full:.4f}°C" in report_text
    assert f"{s4_nofan:.4f}°C" in report_text
    assert f"{s4_nopow:.4f}°C" in report_text
    assert f"{s1:.4f}°C" in report_text


def test_split_b_and_c_matches_csv(report_text, csv_data):
    splits = csv_data["splits"]
    b_10m = splits[(splits["split"].str.contains("Split B")) & (splits["horizon"] == "10min")].iloc[0]
    b_1h = splits[(splits["split"].str.contains("Split B")) & (splits["horizon"] == "1h")].iloc[0]
    b_6h = splits[(splits["split"].str.contains("Split B")) & (splits["horizon"] == "6h")].iloc[0]

    assert f"{b_10m['persistence_mae_mean']:.3f}°C" in report_text
    assert f"{b_1h['persistence_mae_mean']:.3f}°C" in report_text
    assert f"{b_6h['persistence_mae_mean']:.3f}°C" in report_text


def test_planning_adaptation_matches_csv(report_text, csv_data):
    plan = csv_data["planning"]
    fleet = plan[plan["strategy"] == "fleet_mean"].iloc[0]["mean_mae"]
    zero = plan[plan["strategy"] == "zero_shot"].iloc[0]["mean_mae"]
    boost_168 = plan[(plan["strategy"] == "continued_boosting") & (plan["budget_hours"] == 168)].iloc[0]["mean_mae"]

    assert f"{fleet:.2f}°C" in report_text
    assert f"{zero:.2f}°C" in report_text
    assert f"{boost_168:.2f}°C" in report_text
    assert "43.2% error reduction" in report_text


def test_cross_cluster_adaptation_matches_csv(report_text, csv_data):
    transfer = csv_data["transfer"]
    zero = transfer[(transfer["strategy"] == "zero_shot") & (transfer["budget_hours"] == 0)].iloc[0]
    off_6 = transfer[(transfer["strategy"] == "residual_offset") & (transfer["budget_hours"] == 6)].iloc[0]
    off_72 = transfer[(transfer["strategy"] == "residual_offset") & (transfer["budget_hours"] == 72)].iloc[0]

    assert f"{zero['mean_mae']:.2f}°C" in report_text
    assert f"[{zero['ci_95_low']:.2f}°C, {zero['ci_95_high']:.2f}°C]" in report_text
    assert f"{off_6['mean_mae']:.2f}°C" in report_text
    assert f"{off_72['mean_mae']:.2f}°C" in report_text


def test_leakage_matches_csv(report_text, csv_data):
    leak = csv_data["leakage"]
    s1_diff = leak[leak["feature_set"] == "S1"].iloc[0]["leakage_underestimation_pct"]
    s4_diff = leak[leak["feature_set"] == "S4"].iloc[0]["leakage_underestimation_pct"]
    s2_diff = leak[leak["feature_set"] == "S2"].iloc[0]["leakage_underestimation_pct"]
    s3_diff = leak[leak["feature_set"] == "S3"].iloc[0]["leakage_underestimation_pct"]

    assert f"{s1_diff:.2f}%" in report_text
    assert f"{s4_diff:.2f}%" in report_text
    assert f"{s2_diff:.2f}%" in report_text
    assert f"{s3_diff:.2f}%" in report_text


def test_learning_rate_matches_csv(report_text, csv_data):
    lr = csv_data["lr"]
    lr_05 = lr[lr["learning_rate"] == 0.05].iloc[0]
    lr_01 = lr[lr["learning_rate"] == 0.01].iloc[0]
    lr_005 = lr[lr["learning_rate"] == 0.005].iloc[0]

    assert f"{int(lr_05['best_iteration'])}" in report_text
    assert f"{int(lr_01['best_iteration'])}" in report_text
    assert f"{int(lr_005['best_iteration'])}" in report_text
    assert f"{lr_05['test_mae']:.6f}°C" in report_text
    assert f"{lr_01['test_mae']:.6f}°C" in report_text
    assert f"{lr_005['test_mae']:.6f}°C" in report_text
    assert "not practically superior" in report_text
