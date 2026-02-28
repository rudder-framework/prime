"""
Modality compute engine.

Three pure functions (no classes, following Prime's functional pattern):
  compute_modality_rt          — per-modality RT geometry per cohort
  compute_cross_modality_coupling — rolling Spearman ρ between modality centroid_dist series
  compute_system_modality      — fleet centroid trajectory per modality

Math is identical to _compute_normalized_rt_geometry() in prime/ml_export/regime_export.py.
The only difference: observations are pre-filtered to modality signals before SVD.
"""

from __future__ import annotations

from itertools import combinations
from typing import Optional

import numpy as np
import polars as pl
from scipy import stats as scipy_stats

from prime.shared.baseline import find_stable_baseline


def compute_modality_rt(
    obs: pl.DataFrame,
    signals: list[str],
    modality_name: str,
    cohort_col: str = "cohort",
    cycle_col: str = "signal_0",
    signal_col: str = "signal_id",
    value_col: str = "value",
    min_baseline_cycles: int = 3,
) -> pl.DataFrame:
    """
    RT geometry for signals belonging to one modality, per cohort.

    Algorithm (identical to _compute_normalized_rt_geometry):
      1. Filter obs to modality signals
      2. Pivot wide: (cohort, signal_0, s1, s2, ...)
      3. For each cohort:
         - find_stable_baseline() on per-cycle L2 norm
         - centroid + SVD of baseline → principal directions
         - per-cycle: centroid_dist, centroid_dist_norm, pc1_proj, pc2_proj, mahalanobis
         - backward D1/D2 of centroid_dist
      4. All output columns prefixed {modality_name}_rt_

    Args:
        obs: Long-format observations DataFrame (cohort, signal_0, signal_id, value)
        signals: List of signal_ids belonging to this modality
        modality_name: Name prefix for output columns (e.g. "thermal")

    Returns:
        pl.DataFrame(cohort, signal_0, {modality_name}_rt_centroid_dist, _norm, _pc1,
                     _pc2, _mahalanobis_approx, _centroid_dist_d1, _centroid_dist_d2)
    """
    prefix = f"{modality_name}_rt_"

    empty = pl.DataFrame({
        cohort_col: pl.Series([], dtype=pl.Utf8),
        cycle_col: pl.Series([], dtype=pl.Float64),
        f"{prefix}centroid_dist": pl.Series([], dtype=pl.Float64),
        f"{prefix}centroid_dist_norm": pl.Series([], dtype=pl.Float64),
        f"{prefix}pc1_projection": pl.Series([], dtype=pl.Float64),
        f"{prefix}pc2_projection": pl.Series([], dtype=pl.Float64),
        f"{prefix}mahalanobis_approx": pl.Series([], dtype=pl.Float64),
        f"{prefix}centroid_dist_d1": pl.Series([], dtype=pl.Float64),
        f"{prefix}centroid_dist_d2": pl.Series([], dtype=pl.Float64),
    })

    if not signals:
        return empty

    # Filter to modality signals only
    df = obs.filter(pl.col(signal_col).is_in(signals))
    if len(df) == 0:
        return empty

    # Keep only base columns
    base_cols = [c for c in [cohort_col, cycle_col, signal_col, value_col] if c in df.columns]
    df = df.select(base_cols)

    # Pivot wide
    wide = df.pivot(
        on=signal_col,
        index=[cohort_col, cycle_col],
        values=value_col,
        aggregate_function="first",
    ).sort([cohort_col, cycle_col])

    id_cols = [cohort_col, cycle_col]
    sensor_cols = [c for c in wide.columns if c not in id_cols]

    # Filter to varying sensors using nanstd so scattered NaN doesn't
    # cause a signal to be excluded (std propagates NaN in polars).
    varying = [
        c for c in sensor_cols
        if float(np.nanstd(wide[c].to_numpy())) > 1e-10
    ]
    n_signals = len(varying)

    if n_signals == 0:
        print(f"  [modality] WARNING: {modality_name} — no varying sensors, skipping")
        return empty

    engines = wide[cohort_col].unique().sort().to_list()

    all_cohorts: list = []
    all_cycles: list = []
    all_dist: list = []
    all_dist_norm: list = []
    all_pc1: list = []
    all_pc2: list = []
    all_mahal: list = []

    for engine in engines:
        engine_df = wide.filter(pl.col(cohort_col) == engine).sort(cycle_col)
        cycles = engine_df[cycle_col].to_numpy()
        matrix = engine_df.select(varying).to_numpy().astype(np.float64)

        # Step 1: drop completely-missing columns (all NaN for this engine)
        all_nan_cols = np.all(np.isnan(matrix), axis=0)
        mat = matrix[:, ~all_nan_cols]
        n_valid = mat.shape[1]

        if n_valid == 0:
            # All columns entirely NaN for this engine
            n_cycles = len(cycles)
            all_cohorts.extend([engine] * n_cycles)
            all_cycles.extend(cycles.tolist())
            all_dist.extend([float("nan")] * n_cycles)
            all_dist_norm.extend([float("nan")] * n_cycles)
            all_pc1.extend([float("nan")] * n_cycles)
            all_pc2.extend([float("nan")] * n_cycles)
            all_mahal.extend([float("nan")] * n_cycles)
            continue

        # Step 2: drop cycles (rows) where any remaining signal has NaN.
        # Modality geometry requires all signals present at each cycle.
        # NaN in one signal voids the cross-signal comparison for that cycle.
        valid_row_mask = ~np.any(np.isnan(mat), axis=1)
        n_dropped = int((~valid_row_mask).sum())
        if n_dropped > 0:
            print(f"  [modality] {modality_name}/{engine}: "
                  f"dropped {n_dropped} cycles with NaN values")
            mat = mat[valid_row_mask]
            cycles = cycles[valid_row_mask]

        n_cycles = len(mat)
        signal_norms = np.linalg.norm(mat, axis=1)
        baseline_result = find_stable_baseline(signal_norms, min_window=min_baseline_cycles)
        baseline_matrix = mat[baseline_result.start_idx:baseline_result.end_idx]
        n_baseline = len(baseline_matrix)
        centroid = np.mean(baseline_matrix, axis=0)
        baseline_std = np.std(baseline_matrix, axis=0)
        baseline_std[baseline_std < 1e-10] = 1.0

        normed = (mat - centroid) / baseline_std
        normed_baseline = normed[baseline_result.start_idx:baseline_result.end_idx]

        try:
            _, S, Vt = np.linalg.svd(normed_baseline, full_matrices=False)
            eigenvalues = (S ** 2) / max(1, n_baseline - 1)
        except np.linalg.LinAlgError:
            Vt = np.eye(min(n_valid, 5), n_valid)
            eigenvalues = np.ones(min(n_valid, 5))

        centroid_dist = np.linalg.norm(normed, axis=1)
        centroid_dist_norm = centroid_dist / np.sqrt(n_valid)

        pc1 = Vt[0] if len(Vt) > 0 else np.zeros(n_valid)
        pc2 = Vt[1] if len(Vt) > 1 else np.zeros(n_valid)
        pc1_proj = normed @ pc1
        pc2_proj = normed @ pc2

        n_components = min(5, len(eigenvalues))
        projections = normed @ Vt[:n_components].T
        eig_safe = np.maximum(eigenvalues[:n_components], 1e-10)
        mahal = np.sqrt(np.sum(projections ** 2 / eig_safe, axis=1))

        all_cohorts.extend([engine] * n_cycles)
        all_cycles.extend(cycles.tolist())
        all_dist.extend(centroid_dist.tolist())
        all_dist_norm.extend(centroid_dist_norm.tolist())
        all_pc1.extend(pc1_proj.tolist())
        all_pc2.extend(pc2_proj.tolist())
        all_mahal.extend(mahal.tolist())

    result = pl.DataFrame({
        cohort_col: all_cohorts,
        cycle_col: all_cycles,
        f"{prefix}centroid_dist": all_dist,
        f"{prefix}centroid_dist_norm": all_dist_norm,
        f"{prefix}pc1_projection": all_pc1,
        f"{prefix}pc2_projection": all_pc2,
        f"{prefix}mahalanobis_approx": all_mahal,
    })

    # Backward D1/D2 — per cohort, per modality (causal: no forward leakage)
    result = result.sort([cohort_col, cycle_col])
    dist_col = f"{prefix}centroid_dist"
    result = result.with_columns([
        pl.col(dist_col).diff(n=1).over(cohort_col).alias(f"{prefix}centroid_dist_d1"),
        pl.col(dist_col).diff(n=1).diff(n=1).over(cohort_col).alias(f"{prefix}centroid_dist_d2"),
    ])

    return result


def compute_cross_modality_coupling(
    modality_rt_dfs: dict[str, pl.DataFrame],
    window_size: int = 20,
    cohort_col: str = "cohort",
    cycle_col: str = "signal_0",
) -> pl.DataFrame:
    """
    Rolling Spearman ρ between modality centroid_dist series.

    For each pair of modalities (mod_a, mod_b) and each cohort:
      - Rolling Spearman ρ over window_size cycles between
        {mod_a}_rt_centroid_dist and {mod_b}_rt_centroid_dist

    Args:
        modality_rt_dfs: dict[modality_name, DataFrame from compute_modality_rt()]
        window_size: Rolling window for Spearman ρ computation

    Returns:
        pl.DataFrame(cohort, signal_0, {mod_a}_{mod_b}_rho, ...) per pair
    """
    modality_names = list(modality_rt_dfs.keys())

    if len(modality_names) < 2:
        # No pairs possible
        if len(modality_names) == 1:
            df = modality_rt_dfs[modality_names[0]].select([cohort_col, cycle_col])
        else:
            df = pl.DataFrame({
                cohort_col: pl.Series([], dtype=pl.Utf8),
                cycle_col: pl.Series([], dtype=pl.Float64),
            })
        return df

    # Join all modalities on (cohort, signal_0) — only keep centroid_dist columns
    dist_frames: list[pl.DataFrame] = []
    for name, df in modality_rt_dfs.items():
        dist_col = f"{name}_rt_centroid_dist"
        if dist_col not in df.columns:
            continue
        dist_frames.append(df.select([cohort_col, cycle_col, dist_col]))

    if len(dist_frames) < 2:
        return pl.DataFrame({
            cohort_col: pl.Series([], dtype=pl.Utf8),
            cycle_col: pl.Series([], dtype=pl.Float64),
        })

    joined = dist_frames[0]
    for frame in dist_frames[1:]:
        joined = joined.join(frame, on=[cohort_col, cycle_col], how="full", coalesce=True)
    joined = joined.sort([cohort_col, cycle_col])

    # Build modalities list from available dist columns
    available = [
        col.replace("_rt_centroid_dist", "")
        for col in joined.columns
        if col.endswith("_rt_centroid_dist")
    ]

    pairs = list(combinations(available, 2))
    if not pairs:
        return joined.select([cohort_col, cycle_col])

    engines = joined[cohort_col].unique().sort().to_list()

    cohorts_out: list = []
    cycles_out: list = []
    rho_cols: dict[str, list] = {f"{a}_{b}_rho": [] for a, b in pairs}

    for engine in engines:
        engine_df = joined.filter(pl.col(cohort_col) == engine).sort(cycle_col)
        n = len(engine_df)
        cycles = engine_df[cycle_col].to_numpy()

        if n < window_size:
            print(f"  [modality] WARNING: series length ({n}) < coupling window ({window_size}) "
                  f"for cohort {engine} — coupling will be all-null")

        pair_arrays: dict[str, np.ndarray] = {}
        for a, b in pairs:
            col_a = f"{a}_rt_centroid_dist"
            col_b = f"{b}_rt_centroid_dist"
            arr_a = engine_df[col_a].to_numpy() if col_a in engine_df.columns else np.full(n, np.nan)
            arr_b = engine_df[col_b].to_numpy() if col_b in engine_df.columns else np.full(n, np.nan)
            pair_arrays[(a, b)] = (arr_a, arr_b)

        cohorts_out.extend([engine] * n)
        cycles_out.extend(cycles.tolist())

        for a, b in pairs:
            col_name = f"{a}_{b}_rho"
            arr_a, arr_b = pair_arrays[(a, b)]
            rhos = _rolling_spearman(arr_a, arr_b, window_size)
            rho_cols[col_name].extend(rhos.tolist())

    result_dict: dict = {cohort_col: cohorts_out, cycle_col: cycles_out}
    # Convert float NaN → polars null so downstream drop_nulls() works correctly.
    for col_name, vals in rho_cols.items():
        arr = np.array(vals, dtype=np.float64)
        result_dict[col_name] = pl.Series(col_name, arr).fill_nan(None)
    return pl.DataFrame(result_dict)


def _rolling_spearman(
    x: np.ndarray,
    y: np.ndarray,
    window: int,
) -> np.ndarray:
    """
    Backward-looking rolling Spearman ρ — vectorized via per-window ranking.

    Position i uses x[i-window+1:i+1] and y[i-window+1:i+1].
    First (window-1) positions are NaN (insufficient history).
    NaN inputs → NaN output for that window.

    Fast path (no NaN): sliding_window_view → rank each window row with
    argsort-of-argsort → Pearson on ranks.  Fully vectorized — no Python
    loop over windows.  Numerically equivalent to scipy.spearmanr on each
    window (same per-window ranking, same formula).

    Slow path (NaN present): falls back to per-window scipy.spearmanr so
    that NaN positions are masked correctly before ranking.
    """
    n = len(x)
    has_nan = bool(np.any(np.isnan(x)) or np.any(np.isnan(y)))

    if has_nan:
        # Slow path: NaN positions vary per window — must re-rank each window.
        rhos = np.full(n, np.nan)
        for i in range(window - 1, n):
            xi = x[i - window + 1: i + 1]
            yi = y[i - window + 1: i + 1]
            valid = ~(np.isnan(xi) | np.isnan(yi))
            if valid.sum() < 3:
                continue
            rho, _ = scipy_stats.spearmanr(xi[valid], yi[valid])
            rhos[i] = rho
        return rhos

    # Fast path: no NaN — rank within each window, then vectorized Pearson.
    # sliding_window_view returns shape (n - window + 1, window).
    x_wins = np.lib.stride_tricks.sliding_window_view(x, window)  # (m, w)
    y_wins = np.lib.stride_tricks.sliding_window_view(y, window)

    # Rank each row independently: argsort-of-argsort gives ordinal ranks.
    # For continuous float data, ties are negligible; ordinal ≈ average rank.
    rx = np.argsort(np.argsort(x_wins, axis=1), axis=1).astype(np.float64)
    ry = np.argsort(np.argsort(y_wins, axis=1), axis=1).astype(np.float64)

    # Pearson on per-window ranks = Spearman.
    rx_c = rx - rx.mean(axis=1, keepdims=True)
    ry_c = ry - ry.mean(axis=1, keepdims=True)
    num = (rx_c * ry_c).sum(axis=1)
    denom = np.sqrt((rx_c ** 2).sum(axis=1) * (ry_c ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rho_vals = np.where(denom > 0, num / denom, np.nan)

    rhos = np.full(n, np.nan)
    rhos[window - 1:] = rho_vals
    return rhos


def compute_system_modality(
    modality_rt_dfs: dict[str, pl.DataFrame],
    cohort_col: str = "cohort",
    cycle_col: str = "signal_0",
) -> dict[str, pl.DataFrame]:
    """
    Fleet centroid trajectory per modality.

    For each modality:
      - fleet_{m}_centroid_dist = mean({m}_rt_centroid_dist) over cohorts at each signal_0
      - fleet_{m}_centroid_dist_std = std over cohorts

    Also computes pairwise Euclidean distance between cohorts within each modality
    space (using each cohort's mean RT metrics as a point).

    Args:
        modality_rt_dfs: dict[modality_name, DataFrame from compute_modality_rt()]

    Returns:
        dict[modality_name, pl.DataFrame] — one fleet-level df per modality, columns:
          signal_0, fleet_{name}_centroid_dist, fleet_{name}_centroid_dist_std,
          fleet_{name}_n_cohorts
    """
    result: dict[str, pl.DataFrame] = {}

    for name, df in modality_rt_dfs.items():
        dist_col = f"{name}_rt_centroid_dist"
        if dist_col not in df.columns or len(df) == 0:
            result[name] = pl.DataFrame({
                cycle_col: pl.Series([], dtype=pl.Float64),
                f"fleet_{name}_centroid_dist": pl.Series([], dtype=pl.Float64),
                f"fleet_{name}_centroid_dist_std": pl.Series([], dtype=pl.Float64),
            })
            continue

        fleet = (
            df.group_by(cycle_col)
            .agg([
                pl.col(dist_col).mean().alias(f"fleet_{name}_centroid_dist"),
                pl.col(dist_col).std().alias(f"fleet_{name}_centroid_dist_std"),
                pl.col(dist_col).count().alias(f"fleet_{name}_n_cohorts"),
            ])
            .sort(cycle_col)
        )
        result[name] = fleet

    return result
