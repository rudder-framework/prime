"""Tests for the Rössler attractor generator."""

import polars as pl
import numpy as np
import pytest
from pathlib import Path

from prime.generators.rossler import generate_rossler


@pytest.fixture
def output_dir(tmp_path):
    return tmp_path / "rossler_test"


def test_generates_correct_shape(output_dir):
    """4 signals × n_samples rows."""
    n = 2000
    path = generate_rossler(output_dir, n_samples=n)
    df = pl.read_parquet(path)
    assert len(df) == 4 * n
    assert df["signal_id"].n_unique() == 4
    assert set(df["signal_id"].unique().to_list()) == {"x", "y", "z", "pulse"}


def test_signals_not_identical(output_dir):
    """x, y, and z must all be different from each other."""
    path = generate_rossler(output_dir, n_samples=2000)
    df = pl.read_parquet(path)

    pivot = df.pivot(on="signal_id", index="signal_0", values="value")
    assert not np.allclose(pivot["x"].to_numpy(), pivot["y"].to_numpy())
    assert not np.allclose(pivot["x"].to_numpy(), pivot["z"].to_numpy())
    assert not np.allclose(pivot["y"].to_numpy(), pivot["z"].to_numpy())


def test_pulse_is_binary(output_dir):
    """pulse must be 0 or 1."""
    path = generate_rossler(output_dir, n_samples=2000)
    df = pl.read_parquet(path)

    pulse = df.filter(pl.col("signal_id") == "pulse")["value"]
    unique_vals = set(pulse.unique().to_list())
    assert unique_vals <= {0.0, 1.0}


def test_pulse_correlates_with_z_spikes(output_dir):
    """pulse=1 should correspond to high z values."""
    path = generate_rossler(output_dir, n_samples=5000)
    df = pl.read_parquet(path)

    pivot = df.pivot(on="signal_id", index="signal_0", values="value")
    z = pivot["z"].to_numpy()
    pulse = pivot["pulse"].to_numpy()

    # Where pulse=1, z should be above the mean
    z_at_pulse = z[pulse == 1.0]
    assert len(z_at_pulse) > 0, "No pulse events generated"
    assert z_at_pulse.mean() > z.mean(), "pulse=1 should correlate with high z"


def test_no_rossler_pulse_signal(output_dir):
    """The 'rossler pulse' bug signal must not exist."""
    path = generate_rossler(output_dir, n_samples=2000)
    df = pl.read_parquet(path)
    assert "rossler pulse" not in df["signal_id"].unique().to_list()


def test_canonical_schema(output_dir):
    """Output must have canonical schema: cohort, signal_0, signal_id, value."""
    path = generate_rossler(output_dir, n_samples=2000)
    df = pl.read_parquet(path)

    assert df.columns == ["cohort", "signal_0", "signal_id", "value"]
    assert df["cohort"].dtype == pl.String
    assert df["signal_0"].dtype == pl.Float64
    assert df["signal_id"].dtype == pl.String
    assert df["value"].dtype == pl.Float64


def test_signal_0_is_integer_index(output_dir):
    """signal_0 should be 0, 1, 2, ..., n_samples-1."""
    n = 500
    path = generate_rossler(output_dir, n_samples=n)
    df = pl.read_parquet(path)

    s0 = df.filter(pl.col("signal_id") == "x")["signal_0"].to_numpy()
    expected = np.arange(n, dtype=np.float64)
    np.testing.assert_array_equal(s0, expected)


def test_single_cohort(output_dir):
    """Single cohort, empty string."""
    path = generate_rossler(output_dir, n_samples=500)
    df = pl.read_parquet(path)
    assert df["cohort"].n_unique() == 1
    assert df["cohort"][0] == ""


def test_custom_parameters(output_dir):
    """Custom Rössler parameters should produce valid output."""
    path = generate_rossler(output_dir, n_samples=1000, dt=0.02, a=0.1, b=0.1, c=14.0)
    df = pl.read_parquet(path)
    assert len(df) == 4 * 1000
