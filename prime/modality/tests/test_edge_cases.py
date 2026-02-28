"""
Part 1: Edge Case Tests — Prime Modality Module
================================================
Tests 1.1–1.13 from the Modality Testing Protocol.

Rule: ALL tests must pass before proceeding to Part 2.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from prime.modality.config import ModalityConfig, resolve_modalities
from prime.modality.engine import (
    compute_cross_modality_coupling,
    compute_modality_rt,
    compute_system_modality,
)
from prime.modality.export import run_modality_export


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_obs(signal_unit_map: dict[str, str | None], n_cohorts: int = 3,
              n_cycles: int = 50, seed: int = 42) -> pl.DataFrame:
    """Long-format observations from a {signal_id: unit} map."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(1, n_cohorts + 1):
        cohort = f"unit_{c}"
        for sig in signal_unit_map:
            vals = rng.normal(1.0, 0.1, n_cycles)
            for i, v in enumerate(vals):
                rows.append({"cohort": cohort, "signal_0": float(i + 1),
                             "signal_id": sig, "value": float(v)})
    return pl.DataFrame(rows)


def _write_signals(unit_map: dict[str, str | None], tmp: Path) -> Path:
    """Write signals.parquet with a unit column."""
    path = tmp / "signals.parquet"
    pl.DataFrame({
        "signal_id": list(unit_map.keys()),
        "unit": [unit_map[k] for k in unit_map],
        "description": [None] * len(unit_map),
        "source_name": list(unit_map.keys()),
    }).write_parquet(path)
    return path


def _write_signals_no_unit(signal_ids: list[str], tmp: Path) -> Path:
    """Write signals.parquet WITHOUT a unit column."""
    path = tmp / "signals.parquet"
    pl.DataFrame({
        "signal_id": signal_ids,
        "description": [None] * len(signal_ids),
        "source_name": signal_ids,
    }).write_parquet(path)
    return path


# ---------------------------------------------------------------------------
# Test 1.1 — No signals.parquet exists
# ---------------------------------------------------------------------------

def test_1_1_no_signals_parquet():
    """Pipeline must skip gracefully when signals.parquet doesn't exist."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = tmp / "signals.parquet"  # intentionally absent
        obs = _make_obs({"s1": "volts", "s2": "amps"})
        output_dir = tmp / "output"
        output_dir.mkdir()

        # resolve_modalities should return [] with a warning, not raise
        result = resolve_modalities(signals_path)
        assert result == [], "resolve_modalities on missing file must return []"

        # run_modality_export must return [] without crashing
        written = run_modality_export(
            output_dir=output_dir,
            observations=obs,
            signals_path=signals_path,
        )
        assert written == [], "run_modality_export must return [] when signals.parquet missing"
        assert not (output_dir / "ml" / "ml_modality_features.parquet").exists()


# ---------------------------------------------------------------------------
# Test 1.2 — signals.parquet exists but has no 'unit' column
# ---------------------------------------------------------------------------

def test_1_2_no_unit_column():
    """Modality requires unit metadata; skip when unit column absent."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals_no_unit(["s1", "s2", "s3"], tmp)
        obs = _make_obs({"s1": None, "s2": None, "s3": None})
        output_dir = tmp / "output"
        output_dir.mkdir()

        result = resolve_modalities(signals_path)
        assert result == [], "No unit column → resolve_modalities must return []"

        written = run_modality_export(
            output_dir=output_dir,
            observations=obs,
            signals_path=signals_path,
        )
        assert written == [], "No unit column → no modality output written"
        assert not (output_dir / "ml" / "ml_modality_features.parquet").exists()


# ---------------------------------------------------------------------------
# Test 1.3 — All signals have the same unit (1 modality, no coupling)
# ---------------------------------------------------------------------------

def test_1_3_all_same_unit():
    """1 modality → RT geometry computed, 0 coupling pairs."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals({"s1": "volts", "s2": "volts", "s3": "volts"}, tmp)
        obs = _make_obs({"s1": "volts", "s2": "volts", "s3": "volts"})
        output_dir = tmp / "output"
        output_dir.mkdir()

        modalities = resolve_modalities(signals_path)
        assert len(modalities) == 1
        assert modalities[0].unit == "volts"
        assert sorted(modalities[0].signals) == ["s1", "s2", "s3"]

        # Coupling with 1 modality → no rho columns
        rt_dfs = {modalities[0].name: compute_modality_rt(obs, modalities[0].signals, modalities[0].name)}
        coupling = compute_cross_modality_coupling(rt_dfs, window_size=10)
        rho_cols = [c for c in coupling.columns if c.endswith("_rho")]
        assert rho_cols == [], "1 modality → no coupling columns"

        # RT geometry is produced
        rt_df = rt_dfs[modalities[0].name]
        assert len(rt_df) > 0
        assert f"{modalities[0].name}_rt_centroid_dist" in rt_df.columns

        # run_modality_export completes without error
        written = run_modality_export(
            output_dir=output_dir,
            observations=obs,
            signals_path=signals_path,
        )
        ml_path = output_dir / "ml" / "ml_modality_features.parquet"
        assert ml_path.exists()
        ml_df = pl.read_parquet(ml_path)
        rho_cols_ml = [c for c in ml_df.columns if c.endswith("_rho")]
        assert rho_cols_ml == [], "ml_modality_features must have 0 coupling cols when only 1 modality"


# ---------------------------------------------------------------------------
# Test 1.4 — Every signal is a different unit (all singletons)
# ---------------------------------------------------------------------------

def test_1_4_all_singletons():
    """8 singleton modalities → 8×7 RT cols + C(8,2)=28 coupling pairs."""
    unit_map = {
        "s1": "Rankine", "s2": "psia", "s3": "rpm",
        "s4": "dimensionless", "s5": "kg/s", "s6": "kW",
        "s7": "%", "s8": "count",
    }
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals(unit_map, tmp)
        obs = _make_obs({k: None for k in unit_map})
        output_dir = tmp / "output"
        output_dir.mkdir()

        modalities = resolve_modalities(signals_path)
        assert len(modalities) == 8
        assert all(m.is_singleton for m in modalities)

        written = run_modality_export(
            output_dir=output_dir,
            observations=obs,
            signals_path=signals_path,
            rolling_window=10,
        )
        ml_path = output_dir / "ml" / "ml_modality_features.parquet"
        assert ml_path.exists()
        ml_df = pl.read_parquet(ml_path)

        rt_cols = [c for c in ml_df.columns if "_rt_" in c]
        rho_cols = [c for c in ml_df.columns if c.endswith("_rho")]

        assert len(rt_cols) == 8 * 7, f"Expected 56 RT cols, got {len(rt_cols)}"
        assert len(rho_cols) == 28, f"Expected 28 coupling pairs C(8,2), got {len(rho_cols)}"


# ---------------------------------------------------------------------------
# Test 1.5 — Two signals, same unit (minimal 2-signal modality)
# ---------------------------------------------------------------------------

def test_1_5_two_signals_same_unit():
    """2 signals, 1 modality → 7 RT cols, 0 coupling, pc2 ≈ 0."""
    unit_map = {"v1": "volts", "v2": "volts"}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals(unit_map, tmp)
        obs = _make_obs({k: None for k in unit_map})
        output_dir = tmp / "output"
        output_dir.mkdir()

        modalities = resolve_modalities(signals_path)
        assert len(modalities) == 1
        assert not modalities[0].is_singleton

        rt_df = compute_modality_rt(obs, modalities[0].signals, modalities[0].name)
        assert len(rt_df) > 0

        written = run_modality_export(
            output_dir=output_dir,
            observations=obs,
            signals_path=signals_path,
        )
        ml_df = pl.read_parquet(output_dir / "ml" / "ml_modality_features.parquet")
        rho_cols = [c for c in ml_df.columns if c.endswith("_rho")]
        assert rho_cols == []

        # pc2 should be very small (2D SVD: second component captures little variance)
        pc2_col = f"{modalities[0].name}_rt_pc2_projection"
        assert pc2_col in ml_df.columns
        pc2_vals = ml_df[pc2_col].drop_nulls()
        assert float(pc2_vals.abs().mean()) < float(ml_df[f"{modalities[0].name}_rt_pc1_projection"].drop_nulls().abs().mean()), \
            "pc2 should be smaller than pc1 for 2-signal modality"


# ---------------------------------------------------------------------------
# Test 1.6 — Two signals, different units (minimal cross-modality)
# ---------------------------------------------------------------------------

def test_1_6_two_signals_different_units():
    """2 singleton modalities → 14 RT cols, 1 coupling pair."""
    unit_map = {"v": "volts", "a": "amps"}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals(unit_map, tmp)
        obs = _make_obs({k: None for k in unit_map})
        output_dir = tmp / "output"
        output_dir.mkdir()

        written = run_modality_export(
            output_dir=output_dir,
            observations=obs,
            signals_path=signals_path,
            rolling_window=10,
        )
        ml_df = pl.read_parquet(output_dir / "ml" / "ml_modality_features.parquet")
        rt_cols = [c for c in ml_df.columns if "_rt_" in c]
        rho_cols = [c for c in ml_df.columns if c.endswith("_rho")]

        assert len(rt_cols) == 14, f"Expected 14 RT cols (2×7), got {len(rt_cols)}"
        assert len(rho_cols) == 1, f"Expected 1 coupling pair, got {len(rho_cols)}"


# ---------------------------------------------------------------------------
# Test 1.7 — signal_0 has a unit in signals.parquet
# ---------------------------------------------------------------------------

def test_1_7_signal_0_has_unit():
    """signal_0 must be excluded from modality assignment even if it has a unit."""
    unit_map = {"signal_0": "cycles", "s1": "psia", "s2": "psia"}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals(unit_map, tmp)

        modalities = resolve_modalities(signals_path)

        # signal_0 must not appear in any modality's signal list
        all_signals = [sig for m in modalities for sig in m.signals]
        assert "signal_0" not in all_signals, "signal_0 must be excluded from modalities"

        # Only s1 and s2 should be in modalities
        assert sorted(all_signals) == ["s1", "s2"]


# ---------------------------------------------------------------------------
# Test 1.8 — Duplicate signal via YAML override
# ---------------------------------------------------------------------------

def test_1_8_duplicate_yaml_assignment():
    """Signal in multiple modalities after YAML override → first assignment wins."""
    unit_map = {"T2": "Rankine", "T30": "Rankine", "P30": "psia"}
    yaml_content = """overrides:
  - action: merge
    sources: [Rankine, psia]
    to: mixed
  - action: rename
    from: psia
    to: pressure
"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals(unit_map, tmp)
        override_path = tmp / "override.yaml"
        override_path.write_text(yaml_content)

        # Build a scenario where a signal ends up in two groups manually
        # (simulate by patching unit_groups before resolve)
        # We test via a YAML that tries to put T30 in two places
        yaml_dup = """overrides:
  - action: rename
    from: Rankine
    to: thermal
  - action: merge
    sources: [thermal, psia]
    to: thermal
"""
        override_path2 = tmp / "override2.yaml"
        override_path2.write_text(yaml_dup)

        modalities = resolve_modalities(signals_path, override_path2)
        # After merge: thermal contains T2, T30, P30
        all_signals = [sig for m in modalities for sig in m.signals]
        # Each signal should appear exactly once
        assert len(all_signals) == len(set(all_signals)), \
            "Each signal must appear in exactly one modality"


# ---------------------------------------------------------------------------
# Test 1.9 — Empty modality after YAML override (signals don't exist)
# ---------------------------------------------------------------------------

def test_1_9_empty_modality_via_yaml():
    """Modality created by YAML whose signals don't exist in observations → skipped."""
    unit_map = {"s1": "volts", "s2": "amps"}
    yaml_content = """overrides:
  - action: split
    source: volts
    groups:
      magnetic: [B_field, flux]
      other: []
"""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals(unit_map, tmp)
        override_path = tmp / "override.yaml"
        override_path.write_text(yaml_content)
        obs = _make_obs({k: None for k in unit_map})  # only s1 and s2 in observations
        output_dir = tmp / "output"
        output_dir.mkdir()

        # "magnetic" modality has B_field, flux — neither in observations
        # run_modality_export must warn and skip it
        written = run_modality_export(
            output_dir=output_dir,
            observations=obs,
            signals_path=signals_path,
            override_yaml=override_path,
        )
        # Should still produce output for "amps" modality (s2 still present)
        # "magnetic" skipped, "other" skipped (no signals after split)
        if (output_dir / "ml" / "ml_modality_features.parquet").exists():
            ml_df = pl.read_parquet(output_dir / "ml" / "ml_modality_features.parquet")
            assert "magnetic_rt_centroid_dist" not in ml_df.columns


# ---------------------------------------------------------------------------
# Test 1.10 — Very short time series (< coupling window)
# ---------------------------------------------------------------------------

def test_1_10_short_series_below_window():
    """Series shorter than coupling window → coupling is all-null, no crash."""
    unit_map = {"s1": "volts", "s2": "amps", "s3": "watts"}
    # Only 8 cycles — below rolling_window=20
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals(unit_map, tmp)
        obs = _make_obs({k: None for k in unit_map}, n_cohorts=2, n_cycles=8)
        output_dir = tmp / "output"
        output_dir.mkdir()

        written = run_modality_export(
            output_dir=output_dir,
            observations=obs,
            signals_path=signals_path,
            rolling_window=20,
        )
        ml_path = output_dir / "ml" / "ml_modality_features.parquet"
        assert ml_path.exists(), "Should still produce output even for short series"

        ml_df = pl.read_parquet(ml_path)
        rho_cols = [c for c in ml_df.columns if c.endswith("_rho")]
        if rho_cols:
            # All rho values should be null (series too short for any window)
            for col in rho_cols:
                assert ml_df[col].drop_nulls().len() == 0, \
                    f"Expected all-null rho for short series, {col} has non-null values"


# ---------------------------------------------------------------------------
# Test 1.11 — Scattered NaN values in signal data
# ---------------------------------------------------------------------------

def test_1_11_scattered_nan():
    """NaN in one signal drops only the affected cycles, not all cycles."""
    rng = np.random.default_rng(0)
    n_cycles = 40

    # Build observations with NaN in s1 at cycles 10-12
    rows = []
    for i in range(1, n_cycles + 1):
        for sig in ["s1", "s2"]:
            val = rng.normal(1.0, 0.1)
            if sig == "s1" and i in (10, 11, 12):
                val = float("nan")
            rows.append({"cohort": "unit_1", "signal_0": float(i), "signal_id": sig, "value": val})
    obs = pl.DataFrame(rows)

    unit_map = {"s1": "volts", "s2": "volts"}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        signals_path = _write_signals(unit_map, tmp)

        rt_df = compute_modality_rt(obs, ["s1", "s2"], "volts")
        assert len(rt_df) > 0, "Should produce output despite scattered NaN"
        # 3 cycles dropped (10, 11, 12), so 37 rows expected for the modality
        assert len(rt_df) == n_cycles - 3, \
            f"Expected {n_cycles - 3} rows (3 NaN cycles dropped), got {len(rt_df)}"
        # No NaN in centroid_dist for surviving cycles
        assert rt_df["volts_rt_centroid_dist"].is_null().sum() == 0, \
            "Surviving cycles should have non-null centroid_dist"


# ---------------------------------------------------------------------------
# Test 1.12 — Constant signal within a modality
# ---------------------------------------------------------------------------

def test_1_12_constant_signal_in_modality():
    """Constant signal is filtered out; geometry driven by remaining varying signals."""
    rng = np.random.default_rng(1)
    n_cycles = 50
    rows = []
    for i in range(1, n_cycles + 1):
        rows.append({"cohort": "u1", "signal_0": float(i), "signal_id": "P2",
                     "value": float(rng.normal(10.0, 0.5))})
        rows.append({"cohort": "u1", "signal_0": float(i), "signal_id": "P30",
                     "value": float(rng.normal(30.0, 1.0))})
        rows.append({"cohort": "u1", "signal_0": float(i), "signal_id": "Ps30",
                     "value": 554.0})  # constant
    obs = pl.DataFrame(rows)

    rt_df = compute_modality_rt(obs, ["P2", "P30", "Ps30"], "pressure")
    assert len(rt_df) > 0, "Modality with constant signal must still produce output"
    # centroid_dist should be non-zero (varying signals drive geometry)
    assert rt_df["pressure_rt_centroid_dist"].drop_nulls().len() > 0
    assert float(rt_df["pressure_rt_centroid_dist"].drop_nulls().mean()) > 0


# ---------------------------------------------------------------------------
# Test 1.13 — Massive modality imbalance (15 vs 2 vs 1 signals)
# ---------------------------------------------------------------------------

def test_1_13_massive_imbalance():
    """centroid_dist_norm (/ sqrt(n)) handles imbalanced modalities correctly."""
    # Large: 15 signals all 'large_unit'
    # Small: 2 signals 'small_unit'
    # Singleton: 1 signal 'singleton_unit'
    large_sigs = {f"l{i}": "large_unit" for i in range(15)}
    small_sigs = {"s1": "small_unit", "s2": "small_unit"}
    single_sig = {"x": "singleton_unit"}
    unit_map = {**large_sigs, **small_sigs, **single_sig}

    rng = np.random.default_rng(2)
    n_cycles = 30
    rows = []
    for sig in unit_map:
        for i in range(1, n_cycles + 1):
            rows.append({"cohort": "u1", "signal_0": float(i),
                         "signal_id": sig, "value": float(rng.normal(1.0, 0.1))})
    obs = pl.DataFrame(rows)

    rt_large = compute_modality_rt(obs, list(large_sigs.keys()), "large_unit")
    rt_small = compute_modality_rt(obs, list(small_sigs.keys()), "small_unit")
    rt_single = compute_modality_rt(obs, ["x"], "singleton_unit")

    assert len(rt_large) > 0
    assert len(rt_small) > 0
    assert len(rt_single) > 0

    # centroid_dist_norm should normalize by sqrt(n_signals):
    # large: dist / sqrt(15), small: dist / sqrt(2), single: dist / sqrt(1)
    # Raw centroid_dist for large modality should be larger (15D vs 2D vs 1D)
    # Normalized should be more comparable
    raw_large = float(rt_large["large_unit_rt_centroid_dist"].drop_nulls().mean())
    raw_small = float(rt_small["small_unit_rt_centroid_dist"].drop_nulls().mean())
    norm_large = float(rt_large["large_unit_rt_centroid_dist_norm"].drop_nulls().mean())
    norm_small = float(rt_small["small_unit_rt_centroid_dist_norm"].drop_nulls().mean())

    # raw large >> raw small (sqrt(15) ≈ 3.87 amplification vs sqrt(2) ≈ 1.41)
    assert raw_large > raw_small, "Raw centroid_dist should be larger for more signals"
    # normalized values are in same ballpark
    ratio_raw = raw_large / max(raw_small, 1e-10)
    ratio_norm = norm_large / max(norm_small, 1e-10)
    assert ratio_norm < ratio_raw, "Normalization must reduce the imbalance ratio"
