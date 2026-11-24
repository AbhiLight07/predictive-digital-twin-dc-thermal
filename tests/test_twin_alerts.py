"""Unit tests for the deterministic synthetic thermal fault injector (WP4)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from scripts.eval_twin_alerts import SCENARIOS, inject_synthetic_fault  # noqa: E402


def _make_sample_df(n: int = 80) -> pd.DataFrame:
    base = pd.Timestamp("2019-05-10 00:00:00")
    return pd.DataFrame({
        "Series": ["qh2-rcc120"] * n,
        "Time": [base + pd.Timedelta(minutes=10 * i) for i in range(n)],
        "CPU1_Temperature": np.full(n, 55.0, dtype=np.float32),
        "CPU2_Temperature": np.full(n, 54.0, dtype=np.float32),
        "CPU_Temperature": np.full(n, 55.0, dtype=np.float32),
    })


def test_fault_injector_no_pre_onset_leakage():
    df = _make_sample_df(80)
    onset = 30
    for sc in SCENARIOS:
        inj = inject_synthetic_fault(df, sc, onset_idx=onset)
        np.testing.assert_allclose(
            inj["CPU_Temperature"].iloc[:onset].to_numpy(),
            df["CPU_Temperature"].iloc[:onset].to_numpy(),
            err_msg=f"Pre-onset rows modified for scenario {sc}",
        )


def test_fault_injector_determinism_and_magnitudes():
    df = _make_sample_df(90)
    onset = 20

    # Identical repeated calls
    a = inject_synthetic_fault(df, "ramp_6C_2h", onset_idx=onset)
    b = inject_synthetic_fault(df, "ramp_6C_2h", onset_idx=onset)
    np.testing.assert_array_equal(a["CPU_Temperature"].to_numpy(), b["CPU_Temperature"].to_numpy())

    # Step 5C jumps immediately by +5.0°C at onset_idx
    st = inject_synthetic_fault(df, "step_5C", onset_idx=onset)
    assert abs(float(st["CPU_Temperature"].iloc[onset]) - 60.0) < 1e-4

    # Ramp +6C over 12 steps reaches full +6.0°C after 12 steps
    rm = inject_synthetic_fault(df, "ramp_6C_2h", onset_idx=onset)
    assert abs(float(rm["CPU_Temperature"].iloc[onset + 11]) - 61.0) < 1e-4

    # Drift +0.5C/h increases +3.0°C after 36 steps (6 hours)
    dr = inject_synthetic_fault(df, "drift_0.5C_per_h", onset_idx=onset)
    assert abs(float(dr["CPU_Temperature"].iloc[onset + 35]) - 58.0) < 1e-4
