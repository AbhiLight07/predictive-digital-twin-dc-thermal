"""Verify every LightGBM model file listed in twin metadata exists on disk."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TWIN_DIR = ROOT / "models" / "twin"


def test_all_twin_models_present():
    """Every model file referenced in d1_metadata.json and d2_metadata.json must exist and be non-empty."""
    for cluster in ("d1", "d2"):
        meta_path = TWIN_DIR / f"{cluster}_metadata.json"
        assert meta_path.exists(), f"Missing metadata file: {meta_path}"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        models = meta.get("models", [])
        assert len(models) > 0, f"No models listed in {meta_path}"
        for m in models:
            kind = m["kind"]
            horizon = int(m["horizon"])
            for role in ("point", "lower", "upper"):
                p = TWIN_DIR / f"{cluster}_{kind}_{horizon}_{role}.txt"
                assert p.exists(), f"Missing model file: {p}"
                assert p.stat().st_size > 0, f"Empty model file: {p}"
