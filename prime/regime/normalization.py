"""
Regime-Aware Normalization (ML path only)
==========================================
Per-regime z-score normalization of signal values.

This is ML preprocessing, NOT system analysis.
Prime's analytical path uses raw observations.
The ML export uses normalized observations.

Each signal is normalized within each regime:
    value_norm = (value - fleet_mean[regime, signal]) / fleet_std[regime, signal]

This removes operating-condition variance so the ML model sees only
degradation signal, not regime-switching noise.

IMPORTANT: For single-regime datasets (FD001, FD003), this is a global z-score.
The normalized parquets will differ from raw parquets only in scale, not structure.

Training vs test:
    - Training: fit stats from training observations via normalize_per_regime()
    - Test: apply training stats via apply_regime_stats()
    Never refit stats on test data — that would cause data leakage.
"""

from __future__ import annotations

import polars as pl


def normalize_per_regime(
    observations: pl.DataFrame,
    regimes: pl.DataFrame,
    min_samples: int = 10,
) -> tuple[pl.DataFrame, dict]:
    """
    Z-score normalize each signal within each regime.

    Computes per-(signal_id, regime_id) mean and std from the provided observations
    (training set only), then applies the normalization.

    Args:
        observations: Long-format DataFrame (signal_0, signal_id, value, cohort).
        regimes: DataFrame with (cohort, signal_0, regime_id) from detect_regimes().
        min_samples: Minimum observations per (signal_id, regime_id) group.
            Groups with fewer samples use global signal mean/std (passthrough normalization).

    Returns:
        (normalized_obs, regime_stats) where:
            normalized_obs: Same schema as observations plus regime_id column.
            regime_stats: dict {regime_id (int): {signal_id (str): {"mean": float, "std": float}}}
                          Serialize this to ml_regime_stats.json for test set application.
    """
    # Join regime_id onto observations
    joined = observations.join(
        regimes.select(["cohort", "signal_0", "regime_id"]),
        on=["cohort", "signal_0"],
        how="left",
    ).with_columns(
        pl.col("regime_id").fill_null(0).cast(pl.Int32),
    )

    # Global (per signal_id) stats — fallback for sparse regime groups
    global_stats = (
        joined
        .group_by("signal_id")
        .agg(
            pl.col("value").mean().alias("_global_mean"),
            pl.col("value").std().alias("_global_std"),
        )
        .with_columns(
            pl.col("_global_std").clip(lower_bound=1e-10).alias("_global_std"),
        )
    )

    # Per-(signal_id, regime_id) stats with fallback to global for sparse groups
    stats_df = (
        joined
        .group_by(["signal_id", "regime_id"])
        .agg(
            pl.col("value").mean().alias("_mean"),
            pl.col("value").std().alias("_std"),
            pl.len().alias("_n"),
        )
        .join(global_stats, on="signal_id", how="left")
        .with_columns(
            # Sparse group: fall back to global signal mean
            pl.when(pl.col("_n") < min_samples)
            .then(pl.col("_global_mean"))
            .otherwise(pl.col("_mean"))
            .alias("_mean"),
            # Sparse or near-constant group: fall back to global signal std
            pl.when(
                (pl.col("_n") < min_samples) | (pl.col("_std") < 1e-10) | pl.col("_std").is_null()
            )
            .then(pl.col("_global_std"))
            .otherwise(pl.col("_std"))
            .alias("_std"),
        )
        .drop(["_global_mean", "_global_std"])
    )

    # Apply normalization
    normalized = (
        joined
        .join(stats_df.select(["signal_id", "regime_id", "_mean", "_std"]), on=["signal_id", "regime_id"], how="left")
        .with_columns(
            pl.when(pl.col("_mean").is_not_null())
            .then((pl.col("value") - pl.col("_mean")) / pl.col("_std"))
            .otherwise(pl.col("value"))
            .alias("value"),
        )
        .drop(["_mean", "_std"])
    )

    # Build regime_stats dict (JSON-serializable, for test set application)
    regime_stats: dict[int, dict[str, dict[str, float]]] = {}
    for row in stats_df.iter_rows(named=True):
        rid = int(row["regime_id"])
        sid = str(row["signal_id"])
        if rid not in regime_stats:
            regime_stats[rid] = {}
        regime_stats[rid][sid] = {
            "mean": float(row["_mean"] if row["_mean"] is not None else 0.0),
            "std": float(row["_std"] if row["_std"] is not None else 1.0),
        }

    n_regimes = len(regime_stats)
    n_signals = len({s for v in regime_stats.values() for s in v})
    print(f"  Regime normalization: {n_regimes} regime(s), {n_signals} signal(s) z-scored")

    return normalized, regime_stats


def apply_regime_stats(
    observations: pl.DataFrame,
    regimes: pl.DataFrame,
    train_regime_stats: dict,
) -> pl.DataFrame:
    """
    Apply training-set regime statistics to normalize test observations.

    CRITICAL: Test data MUST be normalized using TRAINING statistics.
    Never refit on test data — that would leak future information.

    Args:
        observations: Test observations in long format.
        regimes: Regime assignments for test data (cohort, signal_0, regime_id).
        train_regime_stats: Stats dict from normalize_per_regime() on training data.
            Format: {regime_id (int): {signal_id (str): {"mean": float, "std": float}}}

    Returns:
        Normalized observations with regime_id column, same row count as input.
    """
    # Join regime_id
    joined = observations.join(
        regimes.select(["cohort", "signal_0", "regime_id"]),
        on=["cohort", "signal_0"],
        how="left",
    ).with_columns(
        pl.col("regime_id").fill_null(0).cast(pl.Int32),
    )

    # Build stats DataFrame from the dict
    rows = []
    for regime_id, signals in train_regime_stats.items():
        for signal_id, stats in signals.items():
            rows.append({
                "regime_id": int(regime_id),
                "signal_id": str(signal_id),
                "_mean": float(stats["mean"]),
                "_std": float(stats["std"]),
            })

    if not rows:
        return joined

    stats_df = pl.DataFrame(rows).with_columns(
        pl.col("regime_id").cast(pl.Int32),
    )

    normalized = (
        joined
        .join(stats_df, on=["signal_id", "regime_id"], how="left")
        .with_columns(
            pl.when(pl.col("_mean").is_not_null())
            .then((pl.col("value") - pl.col("_mean")) / pl.col("_std"))
            .otherwise(pl.col("value"))
            .alias("value"),
        )
        .drop(["_mean", "_std"])
    )

    return normalized
