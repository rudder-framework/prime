"""
Regime-Normalized ML Export
============================
Produces regime-aware ML export files after regime detection and normalization.

NEW files written to output_dir/ml/:
    ml_regime_info.parquet            — regime_id + confidence per (cohort, signal_0)
    ml_regime_stats.json              — per-(regime, signal) mean/std for test set
    ml_normalized_csv.parquet         — rolling stats on regime-normalized signals
    ml_normalized_observations.parquet — normalized observations in observations.parquet format
    ml_normalized_rt.parquet          — per-cycle RT geometry on normalized observations

STUBBED (requires running full Manifold on normalized observations — not worth the cost):
    ml_normalized_eigendecomp.parquet — use ml_normalized_observations.parquet + Manifold if needed
    ml_normalized_centroid.parquet    — use ml_normalized_observations.parquet + Manifold if needed

Prime's analytical parquets are NOT affected. This is purely ML preprocessing.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

from prime.shared.baseline import find_stable_baseline


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

    # 4. Normalized observations in observations.parquet format.
    # Allows compute_csv_features() and compute_rt_geometry() in the ML package
    # to operate on regime-normalized signals with no code changes.
    try:
        norm_obs_path = ml_dir / "ml_normalized_observations.parquet"
        _write_normalized_observations(normalized_obs, norm_obs_path)
        exported.append(norm_obs_path)
        n_engines = normalized_obs["cohort"].n_unique() if "cohort" in normalized_obs.columns else "?"
        print(f"  [ml_export] OK   ml_normalized_observations ({n_engines} engines)")
    except Exception as e:
        print(f"  [ml_export] FAIL ml_normalized_observations: {e}")

    # 5. Normalized RT geometry — per-cycle degradation distance on normalized signals.
    # Uses the same per-engine baseline algorithm as ml.features.rt_geometry,
    # inlined here to avoid a Prime→ML import boundary violation.
    try:
        rt_df = _compute_normalized_rt_geometry(normalized_obs)
        rt_path = ml_dir / "ml_normalized_rt.parquet"
        rt_df.write_parquet(rt_path)
        exported.append(rt_path)
        print(f"  [ml_export] OK   ml_normalized_rt ({len(rt_df)} rows, adaptive baseline)")
    except Exception as e:
        print(f"  [ml_export] FAIL ml_normalized_rt: {e}")

    # 6-7. Windowed cohort-level eigendecomp/centroid on normalized observations.
    # These require running the full Manifold windowing pipeline (too expensive for
    # an export step). Use ml_normalized_observations.parquet with Manifold directly
    # if cohort-level normalized features are needed. Raw ml_eigendecomp.parquet and
    # ml_centroid.parquet remain available and are unaffected.
    print("  [ml_export] SKIP ml_normalized_eigendecomp "
          "(run Manifold on ml_normalized_observations.parquet if needed)")
    print("  [ml_export] SKIP ml_normalized_centroid "
          "(run Manifold on ml_normalized_observations.parquet if needed)")

    print(f"  [ml_export] Regime-normalized: {len(exported)} file(s) written")
    return exported


def _write_normalized_observations(
    normalized_obs: pl.DataFrame,
    output_path: Path,
) -> None:
    """
    Write regime-normalized observations in observations.parquet format.

    Strips regime_id and other non-standard columns, keeping only:
        cohort, signal_0, signal_id, value (+ unit if present)

    This allows compute_csv_features() and compute_rt_geometry() in the ML package
    to be called on normalized observations with no modifications.
    """
    keep_cols = ["cohort", "signal_0", "signal_id", "value"]
    if "unit" in normalized_obs.columns:
        keep_cols.append("unit")

    obs_clean = normalized_obs.select([c for c in keep_cols if c in normalized_obs.columns])
    obs_clean.write_parquet(output_path)


def _compute_normalized_rt_geometry(
    normalized_obs: pl.DataFrame,
    cohort_col: str = "cohort",
    cycle_col: str = "signal_0",
    value_col: str = "value",
    signal_col: str = "signal_id",
    min_baseline_cycles: int = 3,
) -> pl.DataFrame:
    """
    RT geometry on regime-normalized observations.

    Per-engine distance from healthy baseline, computed on regime-normalized
    signals rather than raw sensor values. Regime normalization removes
    operating-condition bias, so the resulting rt_centroid_dist reflects
    degradation only — not regime switching.

    Baseline discovery uses find_stable_baseline() on the per-cycle L2 norm
    of sensor readings — the most stable window in each engine's lifecycle,
    wherever it falls.

    Algorithm is identical to ml.features.rt_geometry.compute_rt_geometry()
    but accepts a DataFrame instead of a file path, avoiding a Prime→ML
    import boundary violation.

    Returns:
        DataFrame with (cohort, signal_0, rt_centroid_dist, rt_centroid_dist_norm,
                        rt_pc1_projection, rt_pc2_projection, rt_mahalanobis_approx)
    """
    # Use only base observation columns (strip regime_id, unit, etc.)
    base_cols = [c for c in [cohort_col, cycle_col, signal_col, value_col]
                 if c in normalized_obs.columns]
    df = normalized_obs.select(base_cols)

    # Pivot wide: one row per (cohort, cycle), one col per sensor
    wide = df.pivot(
        on=signal_col,
        index=[cohort_col, cycle_col],
        values=value_col,
        aggregate_function="first",
    ).sort([cohort_col, cycle_col])

    id_cols = [cohort_col, cycle_col]
    sensor_cols = [c for c in wide.columns if c not in id_cols]

    # Filter constant sensors (zero variance across fleet in normalized space)
    varying = [
        c for c in sensor_cols
        if wide[c].std() is not None and wide[c].std() > 1e-10
    ]
    n_signals = len(varying)

    if n_signals == 0:
        print("[ml_export] WARNING: no varying sensors — ml_normalized_rt will be empty")
        return pl.DataFrame({
            cohort_col: [], cycle_col: [],
            "rt_centroid_dist": [], "rt_centroid_dist_norm": [],
            "rt_pc1_projection": [], "rt_pc2_projection": [],
            "rt_mahalanobis_approx": [],
        })

    engines = wide[cohort_col].unique().sort().to_list()

    all_cohorts: list = []
    all_cycles: list = []
    all_centroid_dist: list = []
    all_centroid_dist_norm: list = []
    all_pc1: list = []
    all_pc2: list = []
    all_mahal: list = []

    for engine in engines:
        engine_df = wide.filter(pl.col(cohort_col) == engine).sort(cycle_col)
        cycles = engine_df[cycle_col].to_numpy()
        matrix = engine_df.select(varying).to_numpy().astype(np.float64)

        n_cycles = len(matrix)

        # Discover most stable window via per-cycle L2 norm as stability proxy
        signal_norms = np.linalg.norm(matrix, axis=1)
        baseline_result = find_stable_baseline(signal_norms, min_window=min_baseline_cycles)
        baseline_matrix = matrix[baseline_result.start_idx:baseline_result.end_idx]
        n_baseline = len(baseline_matrix)
        centroid = np.mean(baseline_matrix, axis=0)
        baseline_std = np.std(baseline_matrix, axis=0)
        baseline_std[baseline_std < 1e-10] = 1.0  # avoid division by zero

        # Normalize ALL cycles against this engine's early-life state
        normed = (matrix - centroid) / baseline_std
        normed_baseline = normed[baseline_result.start_idx:baseline_result.end_idx]

        # SVD of normalized baseline cycles → principal directions
        try:
            _, S, Vt = np.linalg.svd(normed_baseline, full_matrices=False)
            eigenvalues = (S ** 2) / max(1, n_baseline - 1)
        except np.linalg.LinAlgError:
            Vt = np.eye(min(n_signals, 5), n_signals)
            eigenvalues = np.ones(min(n_signals, 5))

        centroid_dist = np.linalg.norm(normed, axis=1)
        centroid_dist_norm = centroid_dist / np.sqrt(n_signals)

        pc1 = Vt[0] if len(Vt) > 0 else np.zeros(n_signals)
        pc2 = Vt[1] if len(Vt) > 1 else np.zeros(n_signals)
        pc1_proj = normed @ pc1
        pc2_proj = normed @ pc2

        n_components = min(5, len(eigenvalues))
        projections = normed @ Vt[:n_components].T
        eig_safe = np.maximum(eigenvalues[:n_components], 1e-10)
        mahal = np.sqrt(np.sum(projections ** 2 / eig_safe, axis=1))

        all_cohorts.extend([engine] * n_cycles)
        all_cycles.extend(cycles.tolist())
        all_centroid_dist.extend(centroid_dist.tolist())
        all_centroid_dist_norm.extend(centroid_dist_norm.tolist())
        all_pc1.extend(pc1_proj.tolist())
        all_pc2.extend(pc2_proj.tolist())
        all_mahal.extend(mahal.tolist())

    n_engines = len(engines)
    n_rows = len(all_cohorts)
    print(f"[ml_export] normalized RT: {n_engines} engines, {n_rows} cycles")

    return pl.DataFrame({
        cohort_col: all_cohorts,
        cycle_col: all_cycles,
        "rt_centroid_dist": all_centroid_dist,
        "rt_centroid_dist_norm": all_centroid_dist_norm,
        "rt_pc1_projection": all_pc1,
        "rt_pc2_projection": all_pc2,
        "rt_mahalanobis_approx": all_mahal,
    })


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
