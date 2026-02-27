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

import numpy as np
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


def align_regime_ids(
    test_regimes: pl.DataFrame,
    test_obs: pl.DataFrame,
    train_regime_stats: dict,
    op_signals: list[str] | None = None,
) -> pl.DataFrame:
    """
    Remap test regime IDs to match train regime IDs.

    KMeans assigns arbitrary integer labels — train regime 0 and test regime 0
    may represent completely different operating conditions. This function aligns
    test regime IDs to training regime IDs by nearest-centroid matching on the
    operating condition signals (op1, op2, op3 for C-MAPSS FD002/FD004).

    Must be called BEFORE apply_regime_stats() for test splits.

    Algorithm:
        1. Extract train regime centroids from train_regime_stats
           (fleet mean of op signals per training regime_id)
        2. Compute test regime centroids from actual test observations
        3. For each test regime, find the nearest training regime by Euclidean
           distance in op_signals space
        4. Return test_regimes with regime_id column remapped to train IDs

    Args:
        test_regimes:       Regime assignments for test data (cohort, signal_0, regime_id).
        test_obs:           Raw test observations in long format (cohort, signal_0, signal_id, value).
        train_regime_stats: Stats dict from normalize_per_regime() on training data.
                            Format: {regime_id: {signal_id: {"mean": float, "std": float}}}
        op_signals:         Operating condition signal IDs to use for matching.
                            Defaults to ["op1", "op2", "op3"] (C-MAPSS standard).

    Returns:
        test_regimes DataFrame with regime_id remapped to match training regime IDs.
        If op_signals are not found in the stats, returns test_regimes unchanged with a warning.
    """
    if op_signals is None:
        op_signals = ["op1", "op2", "op3"]

    # Verify op_signals are present in train stats
    sample_regime = next(iter(train_regime_stats.values()))
    available_sigs = set(sample_regime.keys())
    op_sigs_present = [s for s in op_signals if s in available_sigs]

    if not op_sigs_present:
        print(
            f"  [regime_align] WARNING: none of {op_signals} found in regime stats "
            f"(available: {sorted(available_sigs)[:8]}...). Skipping alignment."
        )
        return test_regimes

    # Training regime centroids in op_signals space
    train_centroids: dict[int, np.ndarray] = {}
    for rid_key, signals in train_regime_stats.items():
        rid = int(rid_key)
        train_centroids[rid] = np.array([
            float(signals[sig]["mean"]) if sig in signals else 0.0
            for sig in op_sigs_present
        ])

    # Test regime centroids from actual test observations
    op_obs = test_obs.filter(pl.col("signal_id").is_in(op_sigs_present))
    joined = op_obs.join(
        test_regimes.select(["cohort", "signal_0", "regime_id"]),
        on=["cohort", "signal_0"],
        how="left",
    )
    regime_agg = (
        joined
        .group_by(["regime_id", "signal_id"])
        .agg(pl.col("value").mean().alias("mean"))
    )

    test_centroids: dict[int, np.ndarray] = {}
    for rid in test_regimes["regime_id"].unique().sort().to_list():
        centroid = []
        for sig in op_sigs_present:
            rows = regime_agg.filter(
                (pl.col("regime_id") == rid) & (pl.col("signal_id") == sig)
            )
            val = float(rows["mean"][0]) if len(rows) > 0 else 0.0
            centroid.append(val)
        test_centroids[rid] = np.array(centroid)

    # Nearest-centroid matching: each test regime → best-matching train regime
    remap: dict[int, int] = {}
    for test_rid, test_c in test_centroids.items():
        distances = {
            train_rid: float(np.linalg.norm(test_c - train_c))
            for train_rid, train_c in train_centroids.items()
        }
        best_train_rid = min(distances, key=distances.get)
        best_dist = distances[best_train_rid]
        remap[test_rid] = best_train_rid
        print(
            f"  [regime_align] test regime {test_rid} ({test_c.tolist()})"
            f" → train regime {best_train_rid} ({train_centroids[best_train_rid].tolist()})"
            f"  dist={best_dist:.4f}"
        )

    # Warn if multiple test regimes map to the same train regime
    used_train = list(remap.values())
    if len(set(used_train)) < len(used_train):
        print("  [regime_align] WARNING: multiple test regimes mapped to same train regime!")
        for tr in set(used_train):
            conflicts = [te for te, tv in remap.items() if tv == tr]
            if len(conflicts) > 1:
                print(f"    Train {tr} ← test {conflicts}")

    # Apply remapping
    remap_df = pl.DataFrame({
        "regime_id":          list(remap.keys()),
        "regime_id_aligned":  list(remap.values()),
    }).with_columns([
        pl.col("regime_id").cast(pl.Int32),
        pl.col("regime_id_aligned").cast(pl.Int32),
    ])

    aligned = (
        test_regimes
        .join(remap_df, on="regime_id", how="left")
        .drop("regime_id")
        .rename({"regime_id_aligned": "regime_id"})
    )

    return aligned


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
