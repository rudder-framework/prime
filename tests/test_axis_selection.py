"""Tests for axis selection (PR-P1).

Verifies that reaxis_observations correctly transforms observations
when a non-time signal is selected as the ordering axis.
"""

import polars as pl
import pytest
from pathlib import Path

from prime.ingest.axis import reaxis_observations


def _make_observations(path: Path, n: int = 20) -> pl.DataFrame:
    """Create test observations.parquet with known values.

    3 signals (x, y, z), single cohort, n timesteps.
    x = 2*i (strictly increasing, easy to verify sorting)
    y = 3*i
    z = 5*i
    """
    rows = []
    for i in range(n):
        rows.extend([
            {"cohort": "", "signal_0": float(i), "signal_id": "x", "value": float(i * 2)},
            {"cohort": "", "signal_0": float(i), "signal_id": "y", "value": float(i * 3)},
            {"cohort": "", "signal_0": float(i), "signal_id": "z", "value": float(i * 5)},
        ])

    df = pl.DataFrame(rows).cast({
        "cohort": pl.String,
        "signal_0": pl.Float64,
        "signal_id": pl.String,
        "value": pl.Float64,
    })
    df.write_parquet(path)
    return df


def _make_observations_nonmonotonic(path: Path) -> pl.DataFrame:
    """Observations where x oscillates (not monotonic).

    Simulates a chaotic signal like Rossler x coordinate.
    """
    # x values: 5, 2, 8, 1, 9, 3, 7, 0, 6, 4 (scrambled)
    x_vals = [5.0, 2.0, 8.0, 1.0, 9.0, 3.0, 7.0, 0.0, 6.0, 4.0]
    y_vals = [float(i) for i in range(10)]
    z_vals = [float(i * 10) for i in range(10)]

    rows = []
    for i in range(10):
        rows.extend([
            {"cohort": "", "signal_0": float(i), "signal_id": "x", "value": x_vals[i]},
            {"cohort": "", "signal_0": float(i), "signal_id": "y", "value": y_vals[i]},
            {"cohort": "", "signal_0": float(i), "signal_id": "z", "value": z_vals[i]},
        ])

    df = pl.DataFrame(rows).cast({
        "cohort": pl.String,
        "signal_0": pl.Float64,
        "signal_id": pl.String,
        "value": pl.Float64,
    })
    df.write_parquet(path)
    return df


def _make_multicohort_observations(path: Path) -> pl.DataFrame:
    """Observations with two cohorts, each with different x ranges."""
    rows = []
    for cohort in ["engine_1", "engine_2"]:
        offset = 0.0 if cohort == "engine_1" else 100.0
        for i in range(10):
            rows.extend([
                {"cohort": cohort, "signal_0": float(i), "signal_id": "x",
                 "value": float(i * 2) + offset},
                {"cohort": cohort, "signal_0": float(i), "signal_id": "y",
                 "value": float(i * 3) + offset},
            ])

    df = pl.DataFrame(rows).cast({
        "cohort": pl.String,
        "signal_0": pl.Float64,
        "signal_id": pl.String,
        "value": pl.Float64,
    })
    df.write_parquet(path)
    return df


class TestReaxisSignals:
    """Verify signal list is correct after reaxis."""

    def test_axis_signal_removed(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations(obs)

        result = reaxis_observations(obs, "x", out)
        signals = result["signal_id"].unique().sort().to_list()
        assert "x" not in signals

    def test_time_signal_added(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations(obs)

        result = reaxis_observations(obs, "x", out)
        signals = result["signal_id"].unique().sort().to_list()
        assert "time" in signals

    def test_other_signals_preserved(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations(obs)

        result = reaxis_observations(obs, "x", out)
        signals = result["signal_id"].unique().sort().to_list()
        assert "y" in signals
        assert "z" in signals

    def test_signal_count_unchanged(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        orig = _make_observations(obs)

        result = reaxis_observations(obs, "x", out)
        assert orig["signal_id"].n_unique() == result["signal_id"].n_unique()


class TestReaxisValues:
    """Verify signal_0 contains axis signal's values."""

    def test_signal_0_is_axis_values(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        orig = _make_observations(obs)

        result = reaxis_observations(obs, "x", out)

        orig_x = orig.filter(pl.col("signal_id") == "x")["value"].sort()
        new_signal_0 = result["signal_0"].unique().sort()
        assert orig_x.to_list() == new_signal_0.to_list()

    def test_signal_0_sorted_ascending(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations(obs)

        result = reaxis_observations(obs, "x", out)

        for sig in result["signal_id"].unique().to_list():
            sig_data = result.filter(pl.col("signal_id") == sig)
            vals = sig_data["signal_0"].to_list()
            assert vals == sorted(vals), f"signal_0 not sorted for {sig}"

    def test_time_signal_has_original_signal_0(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        orig = _make_observations(obs, n=10)

        result = reaxis_observations(obs, "x", out)

        # The "time" signal should contain the original signal_0 values {0..9}
        time_vals = sorted(result.filter(
            pl.col("signal_id") == "time"
        )["value"].to_list())
        expected = [float(i) for i in range(10)]
        assert time_vals == expected

    def test_values_identical_just_reordered(self, tmp_path):
        """All signal values are preserved, just in different order."""
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        orig = _make_observations(obs)

        result = reaxis_observations(obs, "x", out)

        # y values: same set regardless of ordering
        orig_y = sorted(orig.filter(pl.col("signal_id") == "y")["value"].to_list())
        result_y = sorted(result.filter(pl.col("signal_id") == "y")["value"].to_list())
        assert orig_y == result_y

        # z values: same set
        orig_z = sorted(orig.filter(pl.col("signal_id") == "z")["value"].to_list())
        result_z = sorted(result.filter(pl.col("signal_id") == "z")["value"].to_list())
        assert orig_z == result_z


class TestReaxisRowCounts:
    """Verify row counts are preserved."""

    def test_total_rows_unchanged(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        orig = _make_observations(obs)

        result = reaxis_observations(obs, "x", out)
        assert len(orig) == len(result)

    def test_rows_per_signal_unchanged(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        orig = _make_observations(obs, n=15)

        result = reaxis_observations(obs, "x", out)

        orig_counts = orig.group_by("signal_id").len().sort("signal_id")["len"].to_list()
        result_counts = result.group_by("signal_id").len().sort("signal_id")["len"].to_list()
        assert orig_counts == result_counts


class TestReaxisNonMonotonic:
    """Verify reaxis works when axis signal oscillates."""

    def test_nonmonotonic_axis_sorted(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations_nonmonotonic(obs)

        result = reaxis_observations(obs, "x", out)

        # signal_0 should be sorted ascending (0,1,2,...,9)
        s0 = result["signal_0"].unique().sort().to_list()
        assert s0 == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    def test_nonmonotonic_values_preserved(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        orig = _make_observations_nonmonotonic(obs)

        result = reaxis_observations(obs, "x", out)

        orig_y = sorted(orig.filter(pl.col("signal_id") == "y")["value"].to_list())
        result_y = sorted(result.filter(pl.col("signal_id") == "y")["value"].to_list())
        assert orig_y == result_y

    def test_nonmonotonic_time_is_scrambled(self, tmp_path):
        """After reaxis, the time signal values are not in order."""
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations_nonmonotonic(obs)

        result = reaxis_observations(obs, "x", out)

        time_data = result.filter(pl.col("signal_id") == "time").sort("signal_0")
        time_vals = time_data["value"].to_list()
        # x_vals were [5,2,8,1,9,3,7,0,6,4], sorted x = [0,1,2,3,4,5,6,7,8,9]
        # Original index for x=0 was i=7, x=1 was i=3, etc.
        expected_time = [7.0, 3.0, 1.0, 5.0, 9.0, 0.0, 8.0, 6.0, 2.0, 4.0]
        assert time_vals == expected_time


class TestReaxisMultiCohort:
    """Verify reaxis works with multiple cohorts."""

    def test_multicohort_signals_correct(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_multicohort_observations(obs)

        result = reaxis_observations(obs, "x", out)
        signals = result["signal_id"].unique().sort().to_list()
        assert "x" not in signals
        assert "time" in signals
        assert "y" in signals

    def test_multicohort_cohorts_preserved(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_multicohort_observations(obs)

        result = reaxis_observations(obs, "x", out)
        cohorts = result["cohort"].unique().sort().to_list()
        assert cohorts == ["engine_1", "engine_2"]

    def test_multicohort_independent_sorting(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_multicohort_observations(obs)

        result = reaxis_observations(obs, "x", out)

        # engine_1 x values: 0,2,4,...,18
        e1 = result.filter(
            (pl.col("cohort") == "engine_1") & (pl.col("signal_id") == "time")
        ).sort("signal_0")
        assert e1["signal_0"].to_list() == [float(i * 2) for i in range(10)]

        # engine_2 x values: 100,102,104,...,118
        e2 = result.filter(
            (pl.col("cohort") == "engine_2") & (pl.col("signal_id") == "time")
        ).sort("signal_0")
        assert e2["signal_0"].to_list() == [float(i * 2 + 100) for i in range(10)]


class TestReaxisErrors:
    """Verify proper error handling."""

    def test_nonexistent_signal_raises(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "out.parquet"
        _make_observations(obs)

        with pytest.raises(ValueError, match="not found"):
            reaxis_observations(obs, "nonexistent", out)

    def test_time_name_collision_raises(self, tmp_path):
        """If time_name collides with an existing signal, raise."""
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "out.parquet"
        _make_observations(obs)

        # Trying to name old signal_0 "y" when y already exists
        with pytest.raises(ValueError, match="collides"):
            reaxis_observations(obs, "x", out, time_name="y")


class TestReaxisOutputFile:
    """Verify the output file is written correctly."""

    def test_output_file_created(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations(obs)

        reaxis_observations(obs, "x", out)
        assert out.exists()

    def test_output_readable(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations(obs)

        reaxis_observations(obs, "x", out)
        df = pl.read_parquet(out)
        assert "signal_0" in df.columns
        assert "signal_id" in df.columns
        assert "value" in df.columns

    def test_output_schema_types(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "x_observations.parquet"
        _make_observations(obs)

        reaxis_observations(obs, "x", out)
        df = pl.read_parquet(out)
        assert df["signal_0"].dtype == pl.Float64
        assert df["signal_id"].dtype in [pl.String, pl.Utf8]
        assert df["value"].dtype == pl.Float64
