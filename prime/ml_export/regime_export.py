"""
Regime-Normalized ML Export
============================
Produces regime-aware ML export files after regime detection and normalization.

NEW files written to output_dir/ml/:
    ml_regime_info.parquet        — regime_id + confidence per (cohort, signal_0)
    ml_regime_stats.json          — per-(regime, signal) mean/std for test set
    ml_normalized_csv.parquet     — rolling stats on regime-normalized signals

STUBBED (not yet implemented — require running Manifold on normalized observations):
    ml_normalized_eigendecomp.parquet
    ml_normalized_centroid.parquet
    ml_normalized_rt.parquet

The stubs are logged as SKIP (not FAIL) — they are planned, not broken.

Prime's analytical parquets are NOT affected. This is purely ML preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def run_regime_ml_export(
    output_dir: Path,
    regimes: pl.DataFrame,
    regime_stats: dict,
    normalized_obs: pl.DataFrame,
    windows: list[int] | None = None,
) -> list[Path]:
    """
    Write regime-normalized ML export files to output_dir/ml/.

    Args:
        output_dir: The output_{axis}/ directory.
        regimes: DataFrame from detect_regimes(): cohort, signal_0, regime_id, regime_confidence.
        regime_stats: Dict from normalize_per_regime(): {regime_id: {signal_id: {mean, std}}}.
        normalized_obs: Long-format observations after regime normalization (includes regime_id).
        windows: Rolling window sizes for ml_normalized_csv. Defaults to [5, 10, 20].

    Returns:
        List of Paths that were successfully written.
    """
    if windows is None:
        windows = [5, 10, 20]

    ml_dir = output_dir / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    exported: list[Path] = []

    # 1. Regime info — labels + confidence per observation
    regime_path = ml_dir / "ml_regime_info.parquet"
    regimes.write_parquet(regime_path)
    exported.append(regime_path)
    n_regimes = regimes["regime_id"].n_unique()
    print(f"  [ml_export] OK   ml_regime_info ({n_regimes} regimes)")

    # 2. Regime statistics — for applying to test data without leakage
    stats_path = ml_dir / "ml_regime_stats.json"
    serializable = {str(k): v for k, v in regime_stats.items()}
    with open(stats_path, "w") as f:
        json.dump(serializable, f, indent=2)
    exported.append(stats_path)
    print(f"  [ml_export] OK   ml_regime_stats")

    # 3. Normalized rolling statistics — computed directly in Python
    try:
        normalized_csv = _compute_normalized_rolling_stats(normalized_obs, windows)
        csv_path = ml_dir / "ml_normalized_csv.parquet"
        normalized_csv.write_parquet(csv_path)
        exported.append(csv_path)
        print(f"  [ml_export] OK   ml_normalized_csv (windows={windows})")
    except Exception as e:
        print(f"  [ml_export] FAIL ml_normalized_csv: {e}")

    # 4-6. Stubs: require running Manifold on normalized observations
    # These are planned for a future PR that integrates with manifold_client.
    print("  [ml_export] SKIP ml_normalized_eigendecomp (planned: requires Manifold on normalized obs)")
    print("  [ml_export] SKIP ml_normalized_centroid (planned: requires Manifold on normalized obs)")
    print("  [ml_export] SKIP ml_normalized_rt (planned: requires Manifold on normalized obs)")

    print(f"  [ml_export] Regime-normalized: {len(exported)} file(s) written")
    return exported


def _compute_normalized_rolling_stats(
    normalized_obs: pl.DataFrame,
    windows: list[int],
) -> pl.DataFrame:
    """
    Rolling statistics on regime-normalized signals.

    For each (cohort, signal_id, signal_0), computes:
        - Per window (e.g., 5, 10, 20): rolling_mean, rolling_std, rolling_min, rolling_max
        - d1: backward first difference of normalized value
        - d2: backward second difference of normalized value

    Output schema (one row per observation):
        cohort, signal_id, signal_0, regime_id,
        w5_mean, w5_std, w5_min, w5_max,
        w10_mean, w10_std, w10_min, w10_max,
        w20_mean, w20_std, w20_min, w20_max,
        d1, d2

    cohort column is omitted if not present in normalized_obs.
    """
    has_cohort = "cohort" in normalized_obs.columns
    has_regime = "regime_id" in normalized_obs.columns

    group_cols = (["cohort"] if has_cohort else []) + ["signal_id"]
    sort_cols = group_cols + ["signal_0"]

    df = normalized_obs.sort(sort_cols)

    # Base identity columns
    id_cols = sort_cols[:]
    if has_regime and "regime_id" not in id_cols:
        id_cols.append("regime_id")

    # Build rolling stat expressions for each window
    window_exprs: list[pl.Expr] = []
    for w in windows:
        window_exprs.extend([
            pl.col("value")
            .rolling_mean(window_size=w, min_periods=1)
            .over(group_cols)
            .alias(f"w{w}_mean"),
            pl.col("value")
            .rolling_std(window_size=w, min_periods=1)
            .over(group_cols)
            .alias(f"w{w}_std"),
            pl.col("value")
            .rolling_min(window_size=w, min_periods=1)
            .over(group_cols)
            .alias(f"w{w}_min"),
            pl.col("value")
            .rolling_max(window_size=w, min_periods=1)
            .over(group_cols)
            .alias(f"w{w}_max"),
        ])

    # Backward derivatives of normalized value
    deriv_exprs = [
        pl.col("value").diff(n=1).over(group_cols).alias("d1"),
        pl.col("value").diff(n=1).diff(n=1).over(group_cols).alias("d2"),
    ]

    select_exprs = (
        [pl.col(c) for c in id_cols]
        + [pl.col("value")]
        + window_exprs
        + deriv_exprs
    )

    return df.select(select_exprs)
