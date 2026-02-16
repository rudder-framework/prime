"""
Build ML Feature Matrix
=======================
Joins Manifold pipeline outputs into a single machine_learning.parquet
with RUL target and lifecycle position.

Usage:
    python -m prime.ml.entry_points.features --data ~/data/FD001/output --obs ~/data/FD001/observations.parquet

Input:  Pipeline parquets from Manifold (cohort_vector, geometry_dynamics, etc.)
Output: machine_learning.parquet — one row per (cohort, I) window.

Columns:
    cohort          — engine ID
    signal_0        — canonical time index
    RUL             — remaining useful life (target)
    lifecycle       — total lifecycle length
    lifecycle_pct   — normalized position (0→1)
    cv_*            — cohort vector features (geometry per engine group)
    gd_*            — geometry dynamics (velocity, acceleration, jerk, curvature)
    sv_*            — state vector (centroid distances)
    tp_*            — topology (network density, degree)
    vf_*            — cohort velocity field (speed, acceleration in state space)
    gfp_*           — gaussian fingerprint (static per-cohort signal characterization)
    gsim_*          — gaussian similarity (static per-cohort pairwise signal coupling)
"""

import argparse
import polars as pl
import numpy as np
from pathlib import Path


def load_if_exists(data_dir: Path, filename: str) -> pl.DataFrame | None:
    path = data_dir / filename
    if path.exists():
        df = pl.read_parquet(str(path))
        if len(df) > 0:
            return df
    return None


def pivot_by_engine(df: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Pivot a (cohort, signal_0, engine) table to (cohort, signal_0) with engine-prefixed columns."""
    feat_cols = [c for c in df.columns if c not in ['signal_0', 'cohort', 'engine']]

    result = None
    for engine in df['engine'].unique().sort().to_list():
        engine_data = df.filter(pl.col('engine') == engine).drop('engine')
        renamed = engine_data.rename({c: f'{prefix}_{engine}_{c}' for c in feat_cols})
        if result is None:
            result = renamed
        else:
            result = result.join(renamed, on=['cohort', 'signal_0'], how='full', coalesce=True)

    return result


def prefix_and_clean(df: pl.DataFrame, prefix: str, drop_cols: list = None) -> pl.DataFrame:
    """Add prefix to feature columns, drop specified columns."""
    drop_cols = drop_cols or []
    feat_cols = [c for c in df.columns if c not in ['signal_0', 'cohort'] + drop_cols]
    result = df.select(['cohort', 'signal_0'] + feat_cols)
    return result.rename({c: f'{prefix}_{c}' for c in feat_cols})


def build_rul(observations: pl.DataFrame) -> dict:
    """Get lifecycle per cohort from observations."""
    first_sig = observations['signal_id'].unique().sort()[0]
    return dict(
        observations
        .filter(pl.col('signal_id') == first_sig)
        .group_by('cohort')
        .agg(pl.col('signal_0').max().alias('max_signal_0'))
        .iter_rows()
    )


def run(data: str | Path, obs: str | Path, output: str | Path = None) -> Path:
    """
    Build ML feature matrix from Manifold outputs.

    Parameters
    ----------
    data : path to directory with pipeline parquets
    obs : path to observations.parquet
    output : output path (default: data/machine_learning.parquet)

    Returns
    -------
    Path to the written machine_learning.parquet
    """
    data = Path(data)
    obs_path = Path(obs)
    out_path = Path(output) if output else data / 'machine_learning.parquet'

    print("=" * 60)
    print("  BUILD ML FEATURE MATRIX")
    print("=" * 60)

    # ──────────────────────────────────────────
    # Load
    # ──────────────────────────────────────────
    observations = pl.read_parquet(str(obs_path))
    lifecycles = build_rul(observations)
    print(f"\n  Cohorts: {len(lifecycles)}")
    print(f"  Lifecycles: {min(lifecycles.values())+1}–{max(lifecycles.values())+1}")

    # Tier 1: Core
    cv = load_if_exists(data, 'cohort_vector.parquet')
    gd = load_if_exists(data, 'geometry_dynamics.parquet')
    sv = load_if_exists(data, 'state_vector.parquet')

    # Tier 2: Supplemental
    tp = load_if_exists(data, 'topology.parquet')
    vf = load_if_exists(data, 'cohort_velocity_field.parquet')

    # ──────────────────────────────────────────
    # Build base from cohort_vector
    # ──────────────────────────────────────────
    if cv is None:
        raise FileNotFoundError("cohort_vector.parquet is required")

    cv_feat = [c for c in cv.columns if c not in ['signal_0', 'cohort']]
    ml = cv.rename({c: f'cv_{c}' for c in cv_feat})
    print(f"\n  Base: cohort_vector → {len(cv_feat)} features, {len(ml)} rows")

    # ──────────────────────────────────────────
    # Pivot + join geometry_dynamics
    # ──────────────────────────────────────────
    if gd is not None:
        gd_pivoted = pivot_by_engine(gd, 'gd')
        before = ml.shape[1]
        ml = ml.join(gd_pivoted, on=['cohort', 'signal_0'], how='left', coalesce=True)
        print(f"  + geometry_dynamics: {ml.shape[1] - before} features")

    # ──────────────────────────────────────────
    # Join state_vector
    # ──────────────────────────────────────────
    if sv is not None:
        sv_clean = prefix_and_clean(sv, 'sv', drop_cols=['n_signals'])
        before = ml.shape[1]
        ml = ml.join(sv_clean, on=['cohort', 'signal_0'], how='left', coalesce=True)
        print(f"  + state_vector: {ml.shape[1] - before} features")

    # ──────────────────────────────────────────
    # Join topology
    # ──────────────────────────────────────────
    if tp is not None:
        tp_clean = prefix_and_clean(tp, 'tp', drop_cols=['topology_computed'])
        before = ml.shape[1]
        ml = ml.join(tp_clean, on=['cohort', 'signal_0'], how='left', coalesce=True)
        print(f"  + topology: {ml.shape[1] - before} features")

    # ──────────────────────────────────────────
    # Join cohort velocity field
    # ──────────────────────────────────────────
    if vf is not None:
        # Drop string columns
        vf_num = vf.select(['cohort', 'signal_0'] + [c for c in vf.columns
                    if c not in ['cohort', 'signal_0'] and vf[c].dtype in [pl.Float64, pl.Float32, pl.Int64]])
        vf_feat = [c for c in vf_num.columns if c not in ['cohort', 'signal_0']]
        vf_renamed = vf_num.rename({c: f'vf_{c}' for c in vf_feat})
        before = ml.shape[1]
        ml = ml.join(vf_renamed, on=['cohort', 'signal_0'], how='left', coalesce=True)
        print(f"  + cohort_velocity_field: {ml.shape[1] - before} features")

    # ──────────────────────────────────────────
    # Gaussian fingerprint + similarity (static per-cohort features)
    # ──────────────────────────────────────────
    gf = load_if_exists(data, 'gaussian_fingerprint.parquet')
    gs = load_if_exists(data, 'gaussian_similarity.parquet')

    if gf is not None:
        gf_feat = [c for c in gf.columns if c not in ['cohort', 'signal_id', 'n_windows']]
        gf_exprs = []
        for c in gf_feat:
            gf_exprs.extend([
                pl.col(c).mean().alias(f'gfp_mean_{c}'),
                pl.col(c).std().alias(f'gfp_std_{c}'),
                pl.col(c).min().alias(f'gfp_min_{c}'),
                pl.col(c).max().alias(f'gfp_max_{c}'),
            ])
        gf_agg = gf.group_by('cohort').agg(gf_exprs)
        before = ml.shape[1]
        ml = ml.join(gf_agg, on='cohort', how='left', coalesce=True)
        print(f"  + gaussian_fingerprint: {ml.shape[1] - before} features (static per-cohort)")

    if gs is not None:
        gs_feat = [c for c in gs.columns if c not in ['cohort', 'signal_a', 'signal_b', 'n_features']]
        gs_exprs = []
        for c in gs_feat:
            gs_exprs.extend([
                pl.col(c).mean().alias(f'gsim_mean_{c}'),
                pl.col(c).std().alias(f'gsim_std_{c}'),
                pl.col(c).min().alias(f'gsim_min_{c}'),
                pl.col(c).max().alias(f'gsim_max_{c}'),
            ])
        gs_agg = gs.group_by('cohort').agg(gs_exprs)
        before = ml.shape[1]
        ml = ml.join(gs_agg, on='cohort', how='left', coalesce=True)
        print(f"  + gaussian_similarity: {ml.shape[1] - before} features (static per-cohort)")

    # ──────────────────────────────────────────
    # Add RUL target + lifecycle position
    # ──────────────────────────────────────────
    rul_records = []
    for cohort, max_signal_0 in lifecycles.items():
        cohort_rows = ml.filter(pl.col('cohort') == cohort)
        for row in cohort_rows.select(['cohort', 'signal_0']).iter_rows(named=True):
            rul_records.append({
                'cohort': cohort,
                'signal_0': row['signal_0'],
                'RUL': max_signal_0 - row['signal_0'],
                'lifecycle': max_signal_0 + 1,
                'lifecycle_pct': row['signal_0'] / max_signal_0,
            })

    rul_df = pl.DataFrame(rul_records)
    ml = ml.join(rul_df, on=['cohort', 'signal_0'], how='left', coalesce=True)

    # ──────────────────────────────────────────
    # Clean up
    # ──────────────────────────────────────────

    # Drop constant columns
    feat_cols = [c for c in ml.columns if c not in ['cohort', 'signal_0', 'RUL', 'lifecycle', 'lifecycle_pct']]
    drop_const = []
    for c in feat_cols:
        if ml[c].dtype in [pl.Float64, pl.Float32, pl.Int64]:
            std = ml[c].drop_nulls().std()
            if std is not None and std < 1e-10:
                drop_const.append(c)

    if drop_const:
        ml = ml.drop(drop_const)
        print(f"\n  Dropped {len(drop_const)} constant columns")

    # Drop remaining string columns (except cohort)
    string_cols = [c for c in ml.columns if ml[c].dtype == pl.Utf8 and c != 'cohort']
    if string_cols:
        ml = ml.drop(string_cols)
        print(f"  Dropped {len(string_cols)} string columns: {string_cols}")

    # ──────────────────────────────────────────
    # Report
    # ──────────────────────────────────────────
    feat_cols = [c for c in ml.columns if c not in ['cohort', 'signal_0', 'RUL', 'lifecycle', 'lifecycle_pct']]

    null_counts = {c: ml[c].null_count() + ml[c].is_nan().sum() if ml[c].dtype in [pl.Float64, pl.Float32]
                   else ml[c].null_count() for c in feat_cols}
    high_null = {c: v / len(ml) for c, v in null_counts.items() if v / len(ml) > 0.20}

    print(f"\n{'=' * 60}")
    print(f"  machine_learning.parquet")
    print(f"{'=' * 60}")
    print(f"  Rows:      {ml.shape[0]}")
    print(f"  Features:  {len(feat_cols)}")
    print(f"  Target:    RUL ({ml['RUL'].min()} – {ml['RUL'].max()})")
    print(f"  Cohorts:   {ml['cohort'].n_unique()}")

    # Feature groups
    groups = {}
    for c in feat_cols:
        prefix = c.split('_')[0]
        groups.setdefault(prefix, []).append(c)

    print(f"\n  Feature groups:")
    for prefix in sorted(groups.keys()):
        n_null = sum(1 for c in groups[prefix] if c in high_null)
        null_note = f" ({n_null} >20% null)" if n_null > 0 else ""
        print(f"    {prefix:>4s}: {len(groups[prefix]):>3d} features{null_note}")

    if high_null:
        print(f"\n  Warning: {len(high_null)} features >20% null (first window + velocity field)")
        print(f"  These are structural nulls from derivative computation — impute or drop.")

    print(f"\n  → {out_path}")

    ml.write_parquet(str(out_path))

    # Also write column manifest for reference
    col_manifest = []
    for c in feat_cols:
        col_manifest.append({
            'column': c,
            'dtype': str(ml[c].dtype),
            'null_pct': null_counts.get(c, 0) / len(ml),
            'mean': float(ml[c].mean()) if ml[c].dtype in [pl.Float64, pl.Float32] and ml[c].mean() is not None else None,
            'std': float(ml[c].std()) if ml[c].dtype in [pl.Float64, pl.Float32] and ml[c].std() is not None else None,
        })

    col_df = pl.DataFrame(col_manifest)
    col_path = out_path.parent / 'ml_column_manifest.parquet'
    col_df.write_parquet(str(col_path))
    print(f"  → {col_path}")
    print()

    return out_path


def main():
    parser = argparse.ArgumentParser(description='Build ML feature matrix from Manifold outputs')
    parser.add_argument('--data', required=True, help='Directory with pipeline parquets')
    parser.add_argument('--obs', required=True, help='Path to observations.parquet')
    parser.add_argument('--output', default=None, help='Output path (default: data/machine_learning.parquet)')
    args = parser.parse_args()

    run(data=args.data, obs=args.obs, output=args.output)


if __name__ == '__main__':
    main()
