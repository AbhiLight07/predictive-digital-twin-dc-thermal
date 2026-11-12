"""Tests for v3 experiment runner CLI argument handling and reproducibility controls."""

import json
from pathlib import Path
import pytest
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.run_v3_experiments import parse_args, main, EXPERIMENT_CATALOG, OUT_DIR


def test_parse_args_reuse_existing():
    args = parse_args(["--reuse-existing"])
    assert args.reuse_existing is True
    assert args.force is False
    assert args.dry_run is False


def test_parse_args_force():
    args = parse_args(["--force"])
    assert args.force is True
    assert args.reuse_existing is False
    assert args.dry_run is False


def test_parse_args_dry_run():
    args = parse_args(["--dry-run"])
    assert args.dry_run is True
    assert args.force is False
    assert args.reuse_existing is False


def test_parse_args_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(["--reuse-existing", "--force"])


def test_safe_default_exits_with_error_when_files_exist():
    """Default invocation without flags must exit with 1 when existing outputs are present."""
    # Ensure at least one file exists
    assert any((OUT_DIR / exp["file"]).exists() for exp in EXPERIMENT_CATALOG)
    exit_code = main([])
    assert exit_code == 1


def test_dry_run_exits_zero():
    """Dry run must return 0 without failing."""
    exit_code = main(["--dry-run"])
    assert exit_code == 0


def test_reuse_existing_creates_manifest():
    """--reuse-existing must write execution_manifest.json with complete metadata."""
    exit_code = main(["--reuse-existing"])
    assert exit_code == 0

    manifest_path = OUT_DIR / "execution_manifest.json"
    assert manifest_path.exists()

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert "timestamp_utc" in manifest
    assert "git_commit" in manifest
    assert manifest["cli_mode"] == "reuse-existing"
    assert manifest["protocol_frozen"] is True
    assert "random_seeds" in manifest
    assert manifest["random_seeds"]["bootstrap_seed"] == 42
    assert "experiment_status" in manifest
    for exp in EXPERIMENT_CATALOG:
        assert exp["file"] in manifest["experiment_status"]
        assert manifest["experiment_status"][exp["file"]] == "REUSED"
