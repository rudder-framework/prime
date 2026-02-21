"""
Regime Normalization

Subtracts fleet-wide per-regime per-sensor means from observations.
Intended for multi-regime datasets (e.g., CMAPSS FD002/FD004) where
operating condition switching dominates sensor variance.

Mean-subtract only — no z-score. Dividing by std would destroy
cross-engine degradation magnitudes.

Output: {parent}/normalized/ containing:
    observations.parquet   — sensor signals only, regime-normalized
    signals.parquet        — sensor metadata (regime indicator signals removed)
    regime_stats.parquet   — fleet means per (signal_id, regime) for provenance
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import polars as pl

from prime.ingest.signal_metadata import write_signal_metadata


@dataclass
class RegimeNormalizationResult:
    """Result of regime normalization."""

    output_dir: Path
    observations_path: Path
    signals_path: Path
    regime_stats_path: Path
    n_regimes: int
    n_sensors: int
    n_rows: int


def identify_regimes(
    df: pl.DataFrame,
    regime_signals: List[str],
) -> pl.DataFrame:
    """
    Identify discrete operating regimes from regime indicator signals.

    Pivots regime signals wide, rounds to 2 decimal places to form
    discrete regime labels, then concatenates into a single regime string.

    Args:
        df: Observations in long format (signal_0, signal_id, value, cohort).
        regime_signals: Signal IDs that define operating regimes (e.g., op1, op2, op3).

    Returns:
        DataFrame with columns (signal_0, cohort, regime) — one row per
        unique (signal_0, cohort) combination.
    """
    regime_df = df.filter(pl.col("signal_id").is_in(regime_signals))

    # Pivot wide: one column per regime signal
    regime_wide = regime_df.pivot(
        on="signal_id",
        index=["signal_0", "cohort"],
        values="value",
    )

    # Round each regime signal to 2 decimal places, concatenate into label
    label_exprs = []
    for sig in regime_signals:
        if sig in regime_wide.columns:
            label_exprs.append(pl.col(sig).round(2).cast(pl.String))

    regime_wide = regime_wide.with_columns(
        pl.concat_str(label_exprs, separator="_").alias("regime")
    )

    return regime_wide.select("signal_0", "cohort", "regime")


def compute_fleet_regime_means(
    df: pl.DataFrame,
    regime_map: pl.DataFrame,
    sensor_signals: List[str],
    min_observations: int = 10,
) -> pl.DataFrame:
    """
    Compute fleet-wide mean per (signal_id, regime).

    For each sensor at each regime, the mean is computed across ALL cohorts
    (engines) and ALL cycles at that regime. If a (sensor, regime) group
    has fewer than min_observations rows, falls back to overall sensor mean.

    Args:
        df: Observations in long format.
        regime_map: DataFrame with (signal_0, cohort, regime) from identify_regimes.
        sensor_signals: Signal IDs to compute means for.
        min_observations: Minimum observations per group; below this, use overall mean.

    Returns:
        DataFrame with columns (signal_id, regime, fleet_mean, n_obs).
    """
    sensors_df = df.filter(pl.col("signal_id").is_in(sensor_signals))

    # Join regime labels onto sensor observations
    joined = sensors_df.join(regime_map, on=["signal_0", "cohort"], how="left")

    # Fleet mean per (signal_id, regime)
    regime_means = (
        joined.group_by(["signal_id", "regime"])
        .agg(
            pl.col("value").mean().alias("fleet_mean"),
            pl.len().alias("n_obs"),
        )
    )

    # Overall mean per signal_id (fallback for sparse regime groups)
    overall_means = (
        joined.group_by("signal_id")
        .agg(pl.col("value").mean().alias("overall_mean"))
    )

    # Use overall mean when regime group is too small
    regime_means = regime_means.join(overall_means, on="signal_id", how="left")
    regime_means = regime_means.with_columns(
        pl.when(pl.col("n_obs") < min_observations)
        .then(pl.col("overall_mean"))
        .otherwise(pl.col("fleet_mean"))
        .alias("fleet_mean")
    ).drop("overall_mean")

    return regime_means


def regime_normalize(
    observations_path: str | Path,
    output_dir: Optional[str | Path] = None,
    regime_signals: Optional[List[str]] = None,
    sensor_signals: Optional[List[str]] = None,
    min_observations: int = 10,
    verbose: bool = True,
) -> RegimeNormalizationResult:
    """
    Regime-normalize observations and write to a self-contained output directory.

    Subtracts fleet-wide per-regime per-sensor means. Regime indicator signals
    are consumed (used to define regimes) and dropped from the output.

    Args:
        observations_path: Path to observations.parquet.
        output_dir: Output directory. Defaults to {parent}/normalized/.
        regime_signals: Signal IDs defining operating regimes.
            If None, auto-detected as signals with <= 10 unique rounded values
            and low variance relative to mean.
        sensor_signals: Signal IDs to normalize. If None, all non-regime signals.
        min_observations: Minimum observations per (sensor, regime) group
            before falling back to overall mean.
        verbose: Print progress.

    Returns:
        RegimeNormalizationResult with paths and summary stats.
    """
    observations_path = Path(observations_path)
    if not observations_path.exists():
        raise FileNotFoundError(f"Not found: {observations_path}")

    if output_dir is None:
        output_dir = observations_path.parent / "normalized"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print("=" * 70)
        print("00: REGIME NORMALIZATION")
        print("Subtracting fleet-wide per-regime per-sensor means")
        print("=" * 70)

    # Load observations
    df = pl.read_parquet(observations_path)
    all_signals = df["signal_id"].unique().sort().to_list()

    if verbose:
        print(f"Loaded: {df.shape[0]:,} rows, {len(all_signals)} signals")

    # Determine regime signals
    if regime_signals is None:
        raise ValueError(
            "regime_signals must be specified. "
            "Pass the signal IDs that define operating regimes "
            "(e.g., --regime-signals op1 op2 op3)."
        )

    missing = set(regime_signals) - set(all_signals)
    if missing:
        raise ValueError(
            f"Regime signals not found in observations: {sorted(missing)}. "
            f"Available signals: {all_signals}"
        )

    # Determine sensor signals (everything except regime signals)
    if sensor_signals is None:
        sensor_signals = [s for s in all_signals if s not in set(regime_signals)]

    if not sensor_signals:
        raise ValueError("No sensor signals remain after excluding regime signals.")

    if verbose:
        print(f"Regime signals: {regime_signals}")
        print(f"Sensor signals: {len(sensor_signals)} signals")

    # Step 1: Identify regimes
    regime_map = identify_regimes(df, regime_signals)
    n_regimes = regime_map["regime"].n_unique()

    if verbose:
        print(f"Identified {n_regimes} discrete regimes")

    # Step 2: Compute fleet-wide regime means
    regime_means = compute_fleet_regime_means(
        df, regime_map, sensor_signals, min_observations
    )

    if verbose:
        print(f"Computed fleet means for {len(regime_means)} (signal, regime) groups")

    # Step 3: Subtract fleet means from sensor observations
    sensors_df = df.filter(pl.col("signal_id").is_in(sensor_signals))
    sensors_df = sensors_df.join(regime_map, on=["signal_0", "cohort"], how="left")
    sensors_df = sensors_df.join(
        regime_means.select("signal_id", "regime", "fleet_mean"),
        on=["signal_id", "regime"],
        how="left",
    )

    # Subtract: value' = value - fleet_mean
    # If fleet_mean is null (shouldn't happen), keep original value
    sensors_df = sensors_df.with_columns(
        pl.when(pl.col("fleet_mean").is_not_null())
        .then(pl.col("value") - pl.col("fleet_mean"))
        .otherwise(pl.col("value"))
        .alias("value")
    )

    # Drop helper columns, keep canonical schema
    output_cols = ["signal_0", "signal_id", "value"]
    if "cohort" in sensors_df.columns:
        output_cols = ["cohort"] + output_cols
    normalized_df = sensors_df.select(output_cols)

    # Sort canonically
    sort_cols = ["signal_id", "signal_0"]
    if "cohort" in normalized_df.columns:
        sort_cols = ["cohort"] + sort_cols
    normalized_df = normalized_df.sort(sort_cols)

    # Step 4: Write outputs
    obs_path = output_dir / "observations.parquet"
    normalized_df.write_parquet(obs_path)

    if verbose:
        print(f"\nWritten: {obs_path}")
        print(f"  {normalized_df.shape[0]:,} rows, {len(sensor_signals)} sensors")

    # Write signals.parquet for the sensor-only subset
    signals_path = write_signal_metadata(normalized_df, output_dir)

    if verbose:
        print(f"Written: {signals_path}")

    # Write regime_stats.parquet for provenance
    stats_path = output_dir / "regime_stats.parquet"
    regime_means.write_parquet(stats_path)

    if verbose:
        print(f"Written: {stats_path}")
        print(f"\nRegime normalization complete.")
        print(f"Run the pipeline on: {output_dir}/")

    return RegimeNormalizationResult(
        output_dir=output_dir,
        observations_path=obs_path,
        signals_path=signals_path,
        regime_stats_path=stats_path,
        n_regimes=n_regimes,
        n_sensors=len(sensor_signals),
        n_rows=normalized_df.shape[0],
    )
