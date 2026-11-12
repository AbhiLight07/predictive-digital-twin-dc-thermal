"""Tests for data cleaning pipeline."""
import sys
import pandas as pd
import pytest

sys.path.insert(0, "d:/mainel5/src")

from ttwin.clean import clean_dataset, strip_host_names


def test_clean_dataset_calls_strip_host_names():
    """Verify clean_dataset strips whitespace from host identifiers."""
    df = pd.DataFrame({
        "Series": [" qh2-rcc120 ", "qh2-rcc121\t", "  qh2-rcc122  "],
        "Time": ["2019-01-01 00:00:00", "2019-01-01 00:10:00", "2019-01-01 00:20:00"],
        "CPU_Temperature": [45.0, 50.0, 55.0],
    })
    cleaned = clean_dataset(df, host_col="Series", time_col="Time")
    expected_hosts = ["qh2-rcc120", "qh2-rcc121", "qh2-rcc122"]
    assert list(cleaned["Series"]) == expected_hosts
