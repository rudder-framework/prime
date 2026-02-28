"""
Modality ML Export — Step 5e

Unit-based signal grouping → per-modality RT geometry + cross-modality coupling.

Analytics parquets written to output_dir/analytics/:
    analytics/modality_geometry/{cohort}_{modality}_geometry.parquet
    analytics/modality_coupling/{cohort}_cross_modality_coupling.parquet
    analytics/modality_coupling/fleet_modality_coupling.parquet
    analytics/modality_centroids/fleet_{modality}_centroid.parquet

ML parquets written to output_dir/ml/:
    ml_modality_features.parquet  — wide, per (cohort, signal_0)
    ml_modality_columns.json      — groups: {modality: [cols], coupling: [cols]}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional  # noqa: F401 — used in type hints below

import polars as pl

from prime.modality.config import ModalityConfig, resolve_modalities
from prime.modality.engine import (
    compute_cross_modality_coupling,
    compute_modality_rt,
    compute_system_modality,
)


def run_modality_export(
    output_dir: Path,
    observations: pl.DataFrame,
    signals_path: Path,
    override_yaml: Optional[Path] = None,
    rolling_window: int = 20,
) -> list[Path]:
    """
    Full modality ML export pipeline.

    Steps:
      1. resolve_modalities — discover unit-based signal groups
      2. compute_modality_rt per modality
      3. compute_cross_modality_coupling
      4. compute_system_modality (fleet centroids)
      5. Write analytics/ parquets (per cohort × modality)
      6. Write ml/ parquets (wide ML-ready features)

    Args:
        output_dir: The output_{axis}/ directory.
        observations: Long-format observations DataFrame (cohort, signal_0, signal_id, value).
        signals_path: Path to signals.parquet (unit column).
        override_yaml: Optional YAML override for modality grouping.
        rolling_window: Window size for rolling Spearman ρ.

    Returns:
        List of Paths that were successfully written.
    """
    exported: list[Path] = []

    # Step 1: Discover modalities
    modalities = resolve_modalities(signals_path, override_yaml)
    print(f"  [modality] {len(modalities)} modalities: {[m.name for m in modalities]}")
    for m in modalities:
        if m.is_singleton:
            print(f"  [modality] singleton: {m.name} has 1 signal ({m.signals[0]}) — no pairwise structure")

    if not modalities:
        print("  [modality] No modalities found — skipping modality export")
        return exported

    if len(modalities) == 1:
        print(f"  [modality] Only 1 modality resolved ({modalities[0].name}) — "
              "cross-modality coupling skipped (need ≥ 2 modalities)")

    # Step 2: Compute modality RT geometry
    modality_rt_dfs: dict[str, pl.DataFrame] = {}
    for mod in modalities:
        try:
            rt_df = compute_modality_rt(observations, mod.signals, mod.name)
            n_rows = len(rt_df)
            if n_rows == 0:
                print(f"  [modality] WARNING: {mod.name}: none of its signals "
                      f"({mod.signals}) found in observations — skipped")
                continue
            modality_rt_dfs[mod.name] = rt_df
            n_cohorts = rt_df["cohort"].n_unique() if "cohort" in rt_df.columns else 0
            print(f"  [modality] RT {mod.name}: {n_rows} rows, {n_cohorts} cohorts")
        except Exception as e:
            print(f"  [modality] WARNING: RT failed for {mod.name}: {e}")

    if not modality_rt_dfs:
        print("  [modality] All RT computations failed — skipping modality export")
        return exported

    # Step 3: Cross-modality coupling
    coupling_df = pl.DataFrame()
    try:
        coupling_df = compute_cross_modality_coupling(modality_rt_dfs, window_size=rolling_window)
        print(f"  [modality] coupling: {len(coupling_df)} rows, {len(coupling_df.columns)} columns")
    except Exception as e:
        print(f"  [modality] WARNING: coupling failed: {e}")

    # Step 4: System (fleet) modality centroid trajectories
    system_dfs: dict[str, pl.DataFrame] = {}
    try:
        system_dfs = compute_system_modality(modality_rt_dfs)
    except Exception as e:
        print(f"  [modality] WARNING: system modality failed: {e}")

    # Step 5: Write analytics/ parquets
    analytics_dir = output_dir / "analytics"

    # 5a: Per cohort × modality geometry
    geo_dir = analytics_dir / "modality_geometry"
    geo_dir.mkdir(parents=True, exist_ok=True)
    for mod_name, rt_df in modality_rt_dfs.items():
        if len(rt_df) == 0:
            continue
        try:
            cohort_col = "cohort"
            if cohort_col in rt_df.columns:
                cohorts = rt_df[cohort_col].unique().sort().to_list()
                for cohort in cohorts:
                    cohort_df = rt_df.filter(pl.col(cohort_col) == cohort)
                    safe_cohort = str(cohort).replace("/", "_").replace("\\", "_")
                    path = geo_dir / f"{safe_cohort}_{mod_name}_geometry.parquet"
                    cohort_df.write_parquet(path)
                    exported.append(path)
            else:
                path = geo_dir / f"all_{mod_name}_geometry.parquet"
                rt_df.write_parquet(path)
                exported.append(path)
        except Exception as e:
            print(f"  [modality] WARNING: analytics write failed for {mod_name}: {e}")

    # 5b: Per cohort coupling + fleet coupling
    if len(coupling_df) > 0:
        coupling_dir = analytics_dir / "modality_coupling"
        coupling_dir.mkdir(parents=True, exist_ok=True)
        try:
            cohort_col = "cohort"
            if cohort_col in coupling_df.columns:
                cohorts = coupling_df[cohort_col].unique().sort().to_list()
                for cohort in cohorts:
                    cohort_coupling = coupling_df.filter(pl.col(cohort_col) == cohort)
                    safe_cohort = str(cohort).replace("/", "_").replace("\\", "_")
                    path = coupling_dir / f"{safe_cohort}_cross_modality_coupling.parquet"
                    cohort_coupling.write_parquet(path)
                    exported.append(path)

            # Fleet-level: aggregate coupling over cohorts (mean per signal_0)
            cycle_col = "signal_0"
            rho_cols = [c for c in coupling_df.columns if c.endswith("_rho")]
            if rho_cols:
                fleet_coupling = (
                    coupling_df.group_by(cycle_col)
                    .agg([pl.col(c).mean().alias(c) for c in rho_cols])
                    .sort(cycle_col)
                )
                fleet_path = coupling_dir / "fleet_modality_coupling.parquet"
                fleet_coupling.write_parquet(fleet_path)
                exported.append(fleet_path)
        except Exception as e:
            print(f"  [modality] WARNING: coupling analytics write failed: {e}")

    # 5c: Fleet modality centroid trajectories
    centroid_dir = analytics_dir / "modality_centroids"
    centroid_dir.mkdir(parents=True, exist_ok=True)
    for mod_name, fleet_df in system_dfs.items():
        if len(fleet_df) == 0:
            continue
        try:
            path = centroid_dir / f"fleet_{mod_name}_centroid.parquet"
            fleet_df.write_parquet(path)
            exported.append(path)
        except Exception as e:
            print(f"  [modality] WARNING: centroid write failed for {mod_name}: {e}")

    # Step 6: Write ml/ parquets
    ml_dir = output_dir / "ml"
    ml_dir.mkdir(parents=True, exist_ok=True)

    # 6a: Wide ML features — join all modality RT + coupling on (cohort, signal_0)
    ml_df = _build_ml_features(modality_rt_dfs, coupling_df)
    if ml_df is not None and len(ml_df) > 0:
        try:
            ml_path = ml_dir / "ml_modality_features.parquet"
            ml_df.write_parquet(ml_path)
            exported.append(ml_path)
            print(f"  [modality] ml_modality_features: {len(ml_df)} rows, {len(ml_df.columns)} columns")
        except Exception as e:
            print(f"  [modality] WARNING: ml_modality_features write failed: {e}")

        # 6b: Column groups JSON
        try:
            groups = _build_column_groups(ml_df, list(modality_rt_dfs.keys()))
            json_path = ml_dir / "ml_modality_columns.json"
            with open(json_path, "w") as f:
                json.dump(groups, f, indent=2)
            exported.append(json_path)
        except Exception as e:
            print(f"  [modality] WARNING: column groups JSON failed: {e}")

    print(f"  [modality] {len(exported)} file(s) written")
    return exported


def _build_ml_features(
    modality_rt_dfs: dict[str, pl.DataFrame],
    coupling_df: pl.DataFrame,
    cohort_col: str = "cohort",
    cycle_col: str = "signal_0",
) -> Optional[pl.DataFrame]:
    """
    Join all modality RT DataFrames + coupling into one wide ML-ready DataFrame.
    Grain: per (cohort, signal_0).
    """
    dfs = []
    for rt_df in modality_rt_dfs.values():
        if len(rt_df) > 0 and cohort_col in rt_df.columns:
            dfs.append(rt_df)

    if not dfs:
        return None

    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.join(df, on=[cohort_col, cycle_col], how="full", coalesce=True)

    if len(coupling_df) > 0 and cohort_col in coupling_df.columns:
        rho_cols = [c for c in coupling_df.columns if c.endswith("_rho")]
        if rho_cols:
            coupling_slim = coupling_df.select([cohort_col, cycle_col] + rho_cols)
            merged = merged.join(coupling_slim, on=[cohort_col, cycle_col], how="full", coalesce=True)

    return merged.sort([cohort_col, cycle_col])


def _build_column_groups(
    ml_df: pl.DataFrame,
    modality_names: list[str],
) -> dict:
    """
    Build column group metadata JSON.

    Structure:
      {
        "modality_name": ["col1", "col2", ...],
        ...
        "coupling": ["mod_a_mod_b_rho", ...]
      }
    """
    groups: dict = {}
    for name in modality_names:
        prefix = f"{name}_rt_"
        cols = [c for c in ml_df.columns if c.startswith(prefix)]
        if cols:
            groups[name] = cols

    coupling_cols = [c for c in ml_df.columns if c.endswith("_rho")]
    if coupling_cols:
        groups["coupling"] = coupling_cols

    return groups
