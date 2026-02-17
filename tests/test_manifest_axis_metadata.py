"""Tests for PR-P3: Manifest records axis metadata.

Verifies that manifest.yaml correctly records what signal is in signal_0,
includes parameterization section, and preserves windowing consistency.
"""

import pandas as pd
import pytest

from prime.manifest.generator import (
    build_manifest,
    validate_manifest,
    get_window_params,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_typology_df(signals: list[str], cohort: str = "") -> pd.DataFrame:
    """Create minimal typology DataFrame for manifest generation.

    Each signal gets a TRENDING classification with standard measures.
    """
    rows = []
    for sig in signals:
        rows.append({
            "signal_id": sig,
            "cohort": cohort,
            "temporal_pattern": ["TRENDING"],
            "temporal_primary": "TRENDING",
            "temporal_secondary": None,
            "classification_confidence": "clear",
            "spectral": "BROADBAND",
            "n_samples": 1000,
            "hurst": 0.95,
            "perm_entropy": 0.05,
            "acf_half_life": 500.0,
            "is_constant": False,
            "is_discrete_sparse": False,
            "engines": [],
            "window_factor": 1.5,
        })
    return pd.DataFrame(rows)


def _make_mixed_typology_df(
    signals: dict[str, str],
    cohort: str = "",
) -> pd.DataFrame:
    """Create typology with specific temporal_primary per signal.

    Args:
        signals: dict of signal_id → temporal_primary
    """
    rows = []
    for sig, pattern in signals.items():
        is_const = pattern == "CONSTANT"
        rows.append({
            "signal_id": sig,
            "cohort": cohort,
            "temporal_pattern": [pattern],
            "temporal_primary": pattern,
            "temporal_secondary": None,
            "classification_confidence": "clear",
            "spectral": "NONE" if is_const else "BROADBAND",
            "n_samples": 1000,
            "hurst": 0.5 if is_const else 0.7,
            "perm_entropy": 0.0 if is_const else 0.5,
            "acf_half_life": None if is_const else 100.0,
            "is_constant": is_const,
            "is_discrete_sparse": is_const,
            "engines": [],
            "window_factor": 1.0,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test: signal_0 metadata reflects actual axis
# ---------------------------------------------------------------------------

class TestSignal0Metadata:
    """signal_0 section records the actual axis signal."""

    def test_default_axis_is_time(self):
        df = _make_typology_df(["y", "z", "pulse"])
        manifest = build_manifest(df)

        assert manifest["signal_0"]["name"] == "time"

    def test_axis_x(self):
        df = _make_typology_df(["time", "y", "z"])
        manifest = build_manifest(df, axis="x")

        assert manifest["signal_0"]["name"] == "x"

    def test_axis_pressure(self):
        df = _make_typology_df(["temp", "flow"])
        manifest = build_manifest(df, axis="pressure")

        assert manifest["signal_0"]["name"] == "pressure"

    def test_unit_is_arbitrary(self):
        df = _make_typology_df(["y", "z"])
        manifest = build_manifest(df, axis="x")

        assert manifest["signal_0"]["unit"] == "arbitrary"


# ---------------------------------------------------------------------------
# Test: parameterization section
# ---------------------------------------------------------------------------

class TestParameterization:
    """parameterization section tracks run metadata."""

    def test_axis_signal_matches_signal_0(self):
        df = _make_typology_df(["time", "y", "z"])
        manifest = build_manifest(df, axis="x")

        assert manifest["parameterization"]["axis_signal"] == "x"
        assert manifest["parameterization"]["axis_signal"] == manifest["signal_0"]["name"]

    def test_default_run_id(self):
        df = _make_typology_df(["y", "z"])
        manifest = build_manifest(df)

        assert manifest["parameterization"]["run_id"] == 1

    def test_custom_run_id(self):
        df = _make_typology_df(["y", "z"])
        manifest = build_manifest(df, run_id=3)

        assert manifest["parameterization"]["run_id"] == 3

    def test_source_is_observations(self):
        df = _make_typology_df(["y", "z"])
        manifest = build_manifest(df)

        assert manifest["parameterization"]["source"] == "observations.parquet"

    def test_parameterization_for_time_axis(self):
        df = _make_typology_df(["x", "y", "z"])
        manifest = build_manifest(df, axis="time", run_id=1)

        assert manifest["parameterization"]["axis_signal"] == "time"
        assert manifest["parameterization"]["run_id"] == 1

    def test_parameterization_for_x_axis(self):
        df = _make_typology_df(["time", "y", "z"])
        manifest = build_manifest(df, axis="x", run_id=2)

        assert manifest["parameterization"]["axis_signal"] == "x"
        assert manifest["parameterization"]["run_id"] == 2


# ---------------------------------------------------------------------------
# Test: system.mode field
# ---------------------------------------------------------------------------

class TestSystemMode:
    """system section includes mode field."""

    def test_system_mode_auto(self):
        df = _make_typology_df(["y", "z"])
        manifest = build_manifest(df)

        assert manifest["system"]["mode"] == "auto"

    def test_system_has_window_and_stride(self):
        df = _make_typology_df(["y", "z"])
        manifest = build_manifest(df)

        assert "window" in manifest["system"]
        assert "stride" in manifest["system"]


# ---------------------------------------------------------------------------
# Test: Windowing consistency between axis runs
# ---------------------------------------------------------------------------

class TestWindowingConsistency:
    """Window params for shared signals are identical across axis choices."""

    def test_same_windows_for_shared_signals(self):
        """y and z should get the same window regardless of axis."""
        # Run 1: axis=time, signals = x, y, z
        df1 = _make_typology_df(["x", "y", "z"])
        manifest1 = build_manifest(df1, axis="time")

        # Run 2: axis=x, signals = time, y, z
        df2 = _make_typology_df(["time", "y", "z"])
        manifest2 = build_manifest(df2, axis="x")

        cohort = ""
        for sig in ["y", "z"]:
            w1 = manifest1["cohorts"][cohort][sig]["window_size"]
            w2 = manifest2["cohorts"][cohort][sig]["window_size"]
            assert w1 == w2, f"{sig} window: {w1} != {w2}"

            s1 = manifest1["cohorts"][cohort][sig]["stride"]
            s2 = manifest2["cohorts"][cohort][sig]["stride"]
            assert s1 == s2, f"{sig} stride: {s1} != {s2}"

    def test_time_signal_gets_default_window(self):
        """When time is a signal, it gets a valid window (trending default)."""
        df = _make_typology_df(["time", "y", "z"])
        manifest = build_manifest(df, axis="x")

        time_config = manifest["cohorts"][""]["time"]
        assert time_config["window_size"] > 0
        assert time_config["stride"] > 0


# ---------------------------------------------------------------------------
# Test: Signals list correctness
# ---------------------------------------------------------------------------

class TestManifestSignals:
    """Manifest contains exactly the right signals."""

    def test_axis_signal_not_in_manifest_signals(self):
        """When axis=x, x is not a signal (it's in signal_0)."""
        # After reaxis, x is removed from signals. The typology df
        # given to manifest won't contain x — it will have time instead.
        df = _make_typology_df(["time", "y", "z"])
        manifest = build_manifest(df, axis="x")

        cohort_signals = list(manifest["cohorts"][""].keys())
        assert "x" not in cohort_signals
        assert "time" in cohort_signals

    def test_constant_in_skip_signals(self):
        """CONSTANT signals go to skip_signals."""
        df = _make_mixed_typology_df({
            "y": "TRENDING",
            "z": "CHAOTIC",
            "stuck": "CONSTANT",
        })
        manifest = build_manifest(df)

        assert "/stuck" in manifest["skip_signals"]
        assert "stuck" not in manifest["cohorts"][""]


# ---------------------------------------------------------------------------
# Test: Manifest validation still passes
# ---------------------------------------------------------------------------

class TestManifestValidation:
    """Updated manifest still passes validation."""

    def test_valid_with_axis_time(self):
        df = _make_typology_df(["x", "y", "z"])
        manifest = build_manifest(df, axis="time")
        errors = validate_manifest(manifest)
        assert errors == []

    def test_valid_with_axis_x(self):
        df = _make_typology_df(["time", "y", "z"])
        manifest = build_manifest(df, axis="x", run_id=2)
        errors = validate_manifest(manifest)
        assert errors == []
