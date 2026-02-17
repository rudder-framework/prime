"""Tests for PR-P2: Typology handles any signal including former axes.

Verifies that typology characterizes every signal_id in observations,
with no assumptions about what signal_0 represents or what signals
should exist.
"""

import numpy as np
import polars as pl
import pytest
from pathlib import Path

from prime.ingest.typology_raw import compute_signal_profile, compute_typology_raw
from prime.ingest.axis import reaxis_observations


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_observations(path: Path, include_constant: bool = False) -> pl.DataFrame:
    """Create test observations with known signal characteristics.

    Signals:
        x: monotonically increasing (2*i) — will be used as axis
        y: sine wave — clearly periodic
        z: random walk — non-stationary
        stuck: constant 4.0 (optional)

    signal_0 = sequential integer (time-like)
    """
    n = 500
    t = np.arange(n, dtype=np.float64)
    x = t * 2.0  # Strictly monotonic
    y = np.sin(2 * np.pi * t / 50.0)  # Period 50
    z = np.cumsum(np.random.RandomState(42).randn(n))  # Random walk

    signals = {"x": x, "y": y, "z": z}
    if include_constant:
        signals["stuck"] = np.full(n, 4.0)

    rows = []
    for i in range(n):
        for name, vals in signals.items():
            rows.append({
                "cohort": "",
                "signal_0": float(i),
                "signal_id": name,
                "value": float(vals[i]),
            })

    df = pl.DataFrame(rows).cast({
        "cohort": pl.String,
        "signal_0": pl.Float64,
        "signal_id": pl.String,
        "value": pl.Float64,
    })
    df.write_parquet(path)
    return df


# ---------------------------------------------------------------------------
# Test: Typology processes every signal_id
# ---------------------------------------------------------------------------

class TestTypologyProcessesAllSignals:
    """Typology characterizes every signal_id, no filtering."""

    def test_all_signals_in_output(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "typology_raw.parquet"
        _make_observations(obs)

        result = compute_typology_raw(str(obs), str(out), verbose=False)
        signal_ids = result["signal_id"].unique().sort().to_list()
        assert signal_ids == ["x", "y", "z"]

    def test_all_signals_including_constant(self, tmp_path):
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "typology_raw.parquet"
        _make_observations(obs, include_constant=True)

        result = compute_typology_raw(str(obs), str(out), verbose=False)
        signal_ids = result["signal_id"].unique().sort().to_list()
        assert signal_ids == ["stuck", "x", "y", "z"]


# ---------------------------------------------------------------------------
# Test: Typology on reaxised observations
# ---------------------------------------------------------------------------

class TestTypologyAfterReaxis:
    """After reaxis (axis=x), typology processes the new signal set."""

    def test_reaxised_signals_correct(self, tmp_path):
        """x removed, time added in reaxised observations."""
        obs = tmp_path / "observations.parquet"
        x_obs = tmp_path / "x_observations.parquet"
        out = tmp_path / "typology_raw.parquet"
        _make_observations(obs)

        reaxis_observations(obs, "x", x_obs)
        result = compute_typology_raw(str(x_obs), str(out), verbose=False)

        signal_ids = result["signal_id"].unique().sort().to_list()
        assert "x" not in signal_ids, "axis signal should not appear in typology"
        assert "time" in signal_ids, "displaced signal_0 should appear as 'time'"
        assert "y" in signal_ids
        assert "z" in signal_ids

    def test_signal_count_matches(self, tmp_path):
        """Same number of signals before and after reaxis."""
        obs = tmp_path / "observations.parquet"
        x_obs = tmp_path / "x_observations.parquet"
        out1 = tmp_path / "typology_raw_1.parquet"
        out2 = tmp_path / "typology_raw_2.parquet"
        _make_observations(obs)

        result1 = compute_typology_raw(str(obs), str(out1), verbose=False)
        reaxis_observations(obs, "x", x_obs)
        result2 = compute_typology_raw(str(x_obs), str(out2), verbose=False)

        assert len(result1) == len(result2)

    def test_monotonic_axis_preserves_typology(self, tmp_path):
        """When axis is monotonic (x=2*i), y and z keep same ordering.

        Since x is strictly monotonic with time, sorting by x gives
        the same value sequence for y and z. Their measures should match.
        """
        obs = tmp_path / "observations.parquet"
        x_obs = tmp_path / "x_observations.parquet"
        out1 = tmp_path / "typology_raw_1.parquet"
        out2 = tmp_path / "typology_raw_2.parquet"
        _make_observations(obs)

        result1 = compute_typology_raw(str(obs), str(out1), verbose=False)
        reaxis_observations(obs, "x", x_obs)
        result2 = compute_typology_raw(str(x_obs), str(out2), verbose=False)

        # y and z should have identical measures since ordering is preserved
        for sig in ["y", "z"]:
            row1 = result1.filter(pl.col("signal_id") == sig).to_dicts()[0]
            row2 = result2.filter(pl.col("signal_id") == sig).to_dicts()[0]

            for col in ["hurst", "perm_entropy", "spectral_flatness",
                         "turning_point_ratio", "determinism_score"]:
                assert row1[col] == pytest.approx(row2[col], abs=1e-6), (
                    f"{sig}.{col}: {row1[col]} != {row2[col]}"
                )


# ---------------------------------------------------------------------------
# Test: Time signal characterization
# ---------------------------------------------------------------------------

class TestTimeSignalCharacterization:
    """A monotonic constant-velocity signal (time) produces expected measures."""

    def test_time_signal_profile(self):
        """time = [0, 1, 2, ..., 999] should be trending/deterministic."""
        values = np.arange(1000, dtype=np.float64)
        profile = compute_signal_profile(values, "time")

        # Hurst ~ 1.0 for monotonic trend
        assert profile.hurst > 0.9, f"hurst={profile.hurst}, expected >0.9"

        # Very low permutation entropy (perfectly ordered)
        assert profile.perm_entropy < 0.1, (
            f"perm_entropy={profile.perm_entropy}, expected <0.1"
        )

        # Spectral flatness near 0 (all energy at low frequency)
        assert profile.spectral_flatness < 0.1, (
            f"spectral_flatness={profile.spectral_flatness}, expected <0.1"
        )

        # Turning point ratio near 0 (no oscillation)
        assert profile.turning_point_ratio < 0.05, (
            f"turning_point_ratio={profile.turning_point_ratio}, expected <0.05"
        )

        # High determinism
        assert profile.determinism_score > 0.8, (
            f"determinism_score={profile.determinism_score}, expected >0.8"
        )

        # Not constant
        assert not profile.is_constant

        # Window factor should be > 1.0 (trending + persistent)
        assert profile.window_factor >= 1.0

    def test_time_signal_in_reaxised_typology(self, tmp_path):
        """After reaxis, the 'time' signal gets characterized."""
        obs = tmp_path / "observations.parquet"
        x_obs = tmp_path / "x_observations.parquet"
        out = tmp_path / "typology_raw.parquet"
        _make_observations(obs)

        reaxis_observations(obs, "x", x_obs)
        result = compute_typology_raw(str(x_obs), str(out), verbose=False)

        time_row = result.filter(pl.col("signal_id") == "time").to_dicts()[0]
        assert time_row["hurst"] > 0.9
        assert time_row["perm_entropy"] < 0.1
        assert time_row["is_constant"] is False
        assert time_row["window_factor"] >= 1.0


# ---------------------------------------------------------------------------
# Test: Constant signal detection
# ---------------------------------------------------------------------------

class TestConstantSignalDetection:
    """is_constant correctly flags stuck sensors."""

    def test_constant_4_detected(self):
        """A signal that's constant 4.0 should be flagged."""
        values = np.full(500, 4.0)
        profile = compute_signal_profile(values, "stuck")

        assert profile.is_constant is True
        assert profile.signal_mean == pytest.approx(4.0)
        assert profile.signal_std == pytest.approx(0.0, abs=1e-10)
        assert profile.n_samples == 500

    def test_constant_gets_default_window(self):
        """Constant signals get window_factor=1.0 (default)."""
        values = np.full(500, 4.0)
        profile = compute_signal_profile(values, "stuck")
        assert profile.window_factor == 1.0

    def test_constant_zero_detected(self):
        """A signal that's constant 0.0 should be flagged."""
        values = np.zeros(500)
        profile = compute_signal_profile(values, "zero")

        assert profile.is_constant is True
        assert profile.signal_mean == pytest.approx(0.0)

    def test_constant_negative_detected(self):
        """A signal that's constant -7.5 should be flagged."""
        values = np.full(500, -7.5)
        profile = compute_signal_profile(values, "neg")

        assert profile.is_constant is True
        assert profile.signal_mean == pytest.approx(-7.5)

    def test_near_constant_not_flagged(self):
        """A signal with tiny variance should NOT be flagged as constant."""
        rng = np.random.RandomState(42)
        values = 100.0 + rng.randn(500) * 0.1  # Small noise around 100
        profile = compute_signal_profile(values, "noisy")

        assert profile.is_constant is False

    def test_constant_in_observations(self, tmp_path):
        """Constant signal in full pipeline is correctly characterized."""
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "typology_raw.parquet"
        _make_observations(obs, include_constant=True)

        result = compute_typology_raw(str(obs), str(out), verbose=False)

        stuck = result.filter(pl.col("signal_id") == "stuck").to_dicts()[0]
        assert stuck["is_constant"] is True
        assert stuck["signal_mean"] == pytest.approx(4.0)
        assert stuck["window_factor"] == 1.0


# ---------------------------------------------------------------------------
# Test: No signal_0 assumptions in typology
# ---------------------------------------------------------------------------

class TestNoSignal0Assumptions:
    """Typology works regardless of what signal_0 represents."""

    def test_noninteger_signal_0(self, tmp_path):
        """signal_0 can be non-integer (e.g., physical x values)."""
        n = 200
        # signal_0 = actual x values from Rossler-like data (non-uniform)
        rng = np.random.RandomState(42)
        x_vals = np.sort(rng.uniform(-10, 10, n))
        y_vals = np.sin(x_vals)

        rows = []
        for i in range(n):
            rows.append({
                "cohort": "",
                "signal_0": float(x_vals[i]),
                "signal_id": "y",
                "value": float(y_vals[i]),
            })
            rows.append({
                "cohort": "",
                "signal_0": float(x_vals[i]),
                "signal_id": "z",
                "value": float(rng.randn()),
            })

        df = pl.DataFrame(rows).cast({
            "cohort": pl.String,
            "signal_0": pl.Float64,
            "signal_id": pl.String,
            "value": pl.Float64,
        })
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "typology_raw.parquet"
        df.write_parquet(obs)

        result = compute_typology_raw(str(obs), str(out), verbose=False)
        assert len(result) == 2
        assert set(result["signal_id"].to_list()) == {"y", "z"}

    def test_signal_0_with_duplicates(self, tmp_path):
        """signal_0 can have duplicate values (from non-monotonic axis)."""
        n = 100
        # Some duplicate signal_0 values
        s0 = np.sort(np.concatenate([
            np.arange(50, dtype=np.float64),
            np.arange(50, dtype=np.float64),  # Duplicates
        ]))

        rows = []
        for i in range(n):
            rows.append({
                "cohort": "",
                "signal_0": float(s0[i]),
                "signal_id": "sig_a",
                "value": float(np.sin(i * 0.1)),
            })

        df = pl.DataFrame(rows).cast({
            "cohort": pl.String,
            "signal_0": pl.Float64,
            "signal_id": pl.String,
            "value": pl.Float64,
        })
        obs = tmp_path / "observations.parquet"
        out = tmp_path / "typology_raw.parquet"
        df.write_parquet(obs)

        # Need at least 2 signals for schema validation, but typology
        # itself doesn't require it — add another signal
        rows2 = [{
            "cohort": "",
            "signal_0": float(s0[i]),
            "signal_id": "sig_b",
            "value": float(np.cos(i * 0.1)),
        } for i in range(n)]

        df2 = pl.concat([df, pl.DataFrame(rows2).cast({
            "cohort": pl.String,
            "signal_0": pl.Float64,
            "signal_id": pl.String,
            "value": pl.Float64,
        })])
        df2.write_parquet(obs)

        result = compute_typology_raw(str(obs), str(out), verbose=False)
        assert len(result) == 2
