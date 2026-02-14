#!/usr/bin/env python3
"""
Prime ML v2: Cycle-Level Hybrid Features + Stacking Ensemble
=============================================================

The Özcan (2025) approach that achieves RMSE=6.62 on FD001:
  - Per-cycle predictions (not per-window)
  - Raw sensor values + cycle number
  - LightGBM + CatBoost stacking with ridge meta-learner
  - ~20,000 training rows vs our 860

This script builds a HYBRID feature matrix:
  Layer 1: Per-cycle raw sensor values (what they have)
  Layer 2: Per-cycle geometry context (what only WE have)
  Layer 3: Per-cycle real-time geometric features (no window needed)
  Layer 4: Static gaussian fingerprint per engine

Directory structure expected:
    data/FD001/
    ├── train/
    │   ├── observations.parquet       ← raw ingested observations
    │   └── output/                    ← Manifold pipeline output
    │       ├── state_vector.parquet
    │       ├── state_geometry.parquet  (or geometry_dynamics.parquet)
    │       ├── cohort_vector.parquet
    │       ├── gaussian_fingerprint.parquet
    │       └── gaussian_similarity.parquet
    ├── test/
    │   ├── observations.parquet
    │   └── output/
    │       └── (same files)
    └── RUL_FD001.txt

Usage:
    # Step 1: Build features
    python build_cycle_features.py --obs data/FD001/train/observations.parquet \
                                   --manifold data/FD001/train/output \
                                   --output data/FD001/train/cycle_features.parquet

    python build_cycle_features.py --obs data/FD001/test/observations.parquet \
                                   --manifold data/FD001/test/output \
                                   --output data/FD001/test/cycle_features.parquet

    # Step 2: Train + evaluate
    python train_cycle_model.py --train data/FD001/train/cycle_features.parquet \
                                --test  data/FD001/test/cycle_features.parquet \
                                --rul   data/FD001/RUL_FD001.txt
"""

import argparse
import numpy as np
import polars as pl
from pathlib import Path


# ═══════════════════════════════════════════════
#  PART 1: BUILD CYCLE-LEVEL FEATURE MATRIX
# ═══════════════════════════════════════════════

MAX_RUL_CAP = 125


def load_if_exists(data_dir: Path, filename: str) -> pl.DataFrame | None:
    path = data_dir / filename
    if path.exists():
        df = pl.read_parquet(str(path))
        return df if len(df) > 0 else None
    return None


def pivot_observations(obs: pl.DataFrame) -> pl.DataFrame:
    """
    Pivot long-format observations to wide: one row per (cohort, I).
    Columns: cohort, I, sensor_1, sensor_2, ..., op1, op2, op3
    """
    # Pivot signal_id → columns
    wide = obs.pivot(
        on='signal_id',
        index=['cohort', 'I'],
        values='value',
        aggregate_function='first',
    ).sort(['cohort', 'I'])
    
    return wide


def compute_lifecycle(obs: pl.DataFrame) -> pl.DataFrame:
    """Get max_I per cohort from observations."""
    first_sig = obs['signal_id'].unique().sort()[0]
    lifecycle = (
        obs.filter(pl.col('signal_id') == first_sig)
        .group_by('cohort')
        .agg(pl.col('I').max().alias('max_I'))
    )
    return lifecycle


def add_rul_and_cycle(wide: pl.DataFrame, lifecycle: pl.DataFrame) -> pl.DataFrame:
    """Add RUL (capped), lifecycle, lifecycle_pct, and cycle features."""
    df = wide.join(lifecycle, on='cohort', how='left')
    df = df.with_columns([
        (pl.col('max_I') - pl.col('I')).alias('RUL_raw'),
        (pl.col('max_I') + 1).alias('lifecycle'),
        (pl.col('I') / pl.col('max_I')).alias('lifecycle_pct'),
        pl.col('I').alias('cycle'),  # cycle number = I (canonical index)
    ])
    # Cap RUL
    df = df.with_columns(
        pl.when(pl.col('RUL_raw') > MAX_RUL_CAP)
        .then(MAX_RUL_CAP)
        .otherwise(pl.col('RUL_raw'))
        .alias('RUL')
    )
    df = df.drop(['max_I', 'RUL_raw'])
    return df


def interpolate_geometry_to_cycles(
    wide: pl.DataFrame, 
    manifold_dir: Path,
) -> pl.DataFrame:
    """
    Interpolate windowed geometry features to every cycle.
    
    Geometry is computed at window centers (e.g., I=5,15,25,...).
    For each cycle, we linearly interpolate between the nearest
    geometry windows to get continuous geometry context.
    """
    # Try geometry_dynamics first (has velocity/acceleration)
    gd = load_if_exists(manifold_dir, 'geometry_dynamics.parquet')
    sg = load_if_exists(manifold_dir, 'state_geometry.parquet')
    
    if gd is None and sg is None:
        print("  [WARN] No geometry files found — skipping geometry interpolation")
        return wide
    
    # Use geometry_dynamics if available (richer), else state_geometry
    geom_source = gd if gd is not None else sg
    
    # Identify geometry feature columns
    meta_cols = {'cohort', 'I', 'engine', 'signal_id', 'n_signals'}
    
    # If geometry_dynamics has 'engine' column, we need to handle pivoting
    if 'engine' in geom_source.columns:
        engines = geom_source['engine'].unique().sort().to_list()
        geom_feat_cols = [c for c in geom_source.columns if c not in meta_cols]
        
        # For each engine, create prefixed columns
        geom_wide = None
        for eng in engines:
            eng_data = geom_source.filter(pl.col('engine') == eng).drop('engine')
            rename_map = {c: f'geom_{eng}_{c}' for c in geom_feat_cols}
            eng_data = eng_data.rename(rename_map)
            if geom_wide is None:
                geom_wide = eng_data
            else:
                geom_wide = geom_wide.join(eng_data, on=['cohort', 'I'], how='full', coalesce=True)
        geom_source = geom_wide
    else:
        geom_feat_cols = [c for c in geom_source.columns if c not in meta_cols]
        geom_source = geom_source.rename({c: f'geom_{c}' for c in geom_feat_cols})
    
    geom_cols = [c for c in geom_source.columns if c.startswith('geom_')]
    
    print(f"  Interpolating {len(geom_cols)} geometry features to every cycle...")
    
    # For each cohort, interpolate geometry to cycle-level
    result_frames = []
    cohorts = wide['cohort'].unique().sort().to_list()
    
    for cohort in cohorts:
        # Get this cohort's cycles
        cohort_cycles = wide.filter(pl.col('cohort') == cohort).sort('I')
        cycle_Is = cohort_cycles['I'].to_numpy()
        
        # Get this cohort's geometry windows
        cohort_geom = geom_source.filter(pl.col('cohort') == cohort).sort('I')
        
        if len(cohort_geom) == 0:
            # No geometry — fill with nulls
            null_df = pl.DataFrame({
                'cohort': [cohort] * len(cycle_Is),
                'I': cycle_Is.tolist(),
            })
            for gc in geom_cols:
                null_df = null_df.with_columns(pl.lit(None).cast(pl.Float64).alias(gc))
            result_frames.append(null_df)
            continue
        
        geom_Is = cohort_geom['I'].to_numpy().astype(float)
        
        # Interpolate each geometry column
        interp_data = {
            'cohort': [cohort] * len(cycle_Is),
            'I': cycle_Is.tolist(),
        }
        
        for gc in geom_cols:
            geom_vals = cohort_geom[gc].to_numpy().astype(float)
            
            # Handle NaN in geometry
            valid_mask = ~np.isnan(geom_vals)
            if valid_mask.sum() < 2:
                interp_data[gc] = [None] * len(cycle_Is)
                continue
            
            valid_Is = geom_Is[valid_mask]
            valid_vals = geom_vals[valid_mask]
            
            # Linear interpolation (extrapolate with nearest value)
            interp_vals = np.interp(cycle_Is.astype(float), valid_Is, valid_vals)
            interp_data[gc] = interp_vals.tolist()
        
        result_frames.append(pl.DataFrame(interp_data))
    
    if not result_frames:
        return wide
    
    geom_interp = pl.concat(result_frames)
    
    # Join to wide
    before = wide.shape[1]
    wide = wide.join(geom_interp, on=['cohort', 'I'], how='left', coalesce=True)
    print(f"  + interpolated geometry: {wide.shape[1] - before} features")
    
    return wide


def add_realtime_geometry(wide: pl.DataFrame, manifold_dir: Path) -> pl.DataFrame:
    """
    Add per-cycle geometric features that need NO window.
    
    For each cycle, given the engine's centroid and principal components:
      - centroid_distance: ||sensor_vector - centroid||
      - pc1_projection: sensor_vector · PC1 (where along primary axis)
      - pc2_projection: sensor_vector · PC2
      - mahalanobis_approx: weighted distance using eigenvalues
    
    These are TRUE real-time features — computable from a single observation.
    """
    sv = load_if_exists(manifold_dir, 'state_vector.parquet')
    
    if sv is None:
        print("  [WARN] No state_vector — skipping real-time geometry")
        return wide
    
    # state_vector has centroid per (cohort, I) at window level
    # We need the per-cohort MEAN centroid as the engine's "home position"
    # Then measure distance from each cycle to that centroid
    
    # Get sensor columns from wide (not meta columns)
    meta = {'cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct', 'cycle'}
    geom_prefixed = {c for c in wide.columns if c.startswith('geom_') or c.startswith('gfp_') or c.startswith('gsim_')}
    sensor_cols = sorted([c for c in wide.columns if c not in meta and c not in geom_prefixed])
    
    if not sensor_cols:
        return wide
    
    print(f"  Computing real-time geometry from {len(sensor_cols)} sensors...")
    
    # For each cohort, compute centroid from early-life cycles (first 20%)
    # This represents "healthy baseline" — no window needed
    result_frames = []
    cohorts = wide['cohort'].unique().sort().to_list()
    
    for cohort in cohorts:
        cohort_data = wide.filter(pl.col('cohort') == cohort).sort('I')
        n_cycles = len(cohort_data)
        
        if n_cycles < 5:
            # Too few cycles
            null_df = cohort_data.select(['cohort', 'I']).with_columns([
                pl.lit(None).cast(pl.Float64).alias('rt_centroid_dist'),
                pl.lit(None).cast(pl.Float64).alias('rt_centroid_dist_norm'),
                pl.lit(None).cast(pl.Float64).alias('rt_pc1_projection'),
                pl.lit(None).cast(pl.Float64).alias('rt_pc2_projection'),
                pl.lit(None).cast(pl.Float64).alias('rt_sensor_norm'),
            ])
            result_frames.append(null_df)
            continue
        
        # Sensor matrix: (n_cycles, n_sensors)
        sensor_matrix = cohort_data.select(sensor_cols).to_numpy().astype(np.float64)
        
        # Replace NaN with column median
        for col_idx in range(sensor_matrix.shape[1]):
            col = sensor_matrix[:, col_idx]
            nan_mask = np.isnan(col)
            if nan_mask.any():
                med = np.nanmedian(col)
                sensor_matrix[nan_mask, col_idx] = med if not np.isnan(med) else 0.0
        
        # Baseline centroid: mean of first 20% of cycles
        n_baseline = max(3, int(n_cycles * 0.20))
        centroid = np.mean(sensor_matrix[:n_baseline], axis=0)
        
        # Z-score normalize using baseline stats
        baseline_std = np.std(sensor_matrix[:n_baseline], axis=0)
        baseline_std[baseline_std < 1e-12] = 1.0  # avoid div by zero
        
        sensor_normed = (sensor_matrix - centroid) / baseline_std
        
        # Centroid distance (Euclidean in normalized space)
        dists = np.sqrt(np.sum(sensor_normed ** 2, axis=1))
        
        # Normalized by sqrt(n_sensors) for comparability
        dists_norm = dists / np.sqrt(sensor_normed.shape[1])
        
        # PCA on baseline to get principal directions
        # (This is the engine's "shape" — computed once from early life)
        baseline_centered = sensor_matrix[:n_baseline] - centroid
        baseline_normed = baseline_centered / baseline_std
        
        if baseline_normed.shape[0] >= 3 and baseline_normed.shape[1] >= 2:
            try:
                U, S, Vt = np.linalg.svd(baseline_normed, full_matrices=False)
                pc1 = Vt[0]  # first principal direction
                pc2 = Vt[1] if Vt.shape[0] > 1 else np.zeros_like(pc1)
                
                # Project ALL cycles onto PC1, PC2
                pc1_proj = sensor_normed @ pc1
                pc2_proj = sensor_normed @ pc2
            except np.linalg.LinAlgError:
                pc1_proj = np.zeros(n_cycles)
                pc2_proj = np.zeros(n_cycles)
        else:
            pc1_proj = np.zeros(n_cycles)
            pc2_proj = np.zeros(n_cycles)
        
        # Sensor norm (total magnitude — proxy for total_variance at cycle level)
        sensor_norm = np.sqrt(np.sum(sensor_matrix ** 2, axis=1))
        
        rt_df = pl.DataFrame({
            'cohort': [cohort] * n_cycles,
            'I': cohort_data['I'].to_list(),
            'rt_centroid_dist': dists.tolist(),
            'rt_centroid_dist_norm': dists_norm.tolist(),
            'rt_pc1_projection': pc1_proj.tolist(),
            'rt_pc2_projection': pc2_proj.tolist(),
            'rt_sensor_norm': sensor_norm.tolist(),
        })
        result_frames.append(rt_df)
    
    if not result_frames:
        return wide
    
    rt_all = pl.concat(result_frames)
    
    before = wide.shape[1]
    wide = wide.join(rt_all, on=['cohort', 'I'], how='left', coalesce=True)
    print(f"  + real-time geometry: {wide.shape[1] - before} features")
    
    return wide


def add_gaussian_static(wide: pl.DataFrame, manifold_dir: Path) -> pl.DataFrame:
    """Add static per-cohort gaussian fingerprint + similarity features."""
    gf = load_if_exists(manifold_dir, 'gaussian_fingerprint.parquet')
    gs = load_if_exists(manifold_dir, 'gaussian_similarity.parquet')
    
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
        before = wide.shape[1]
        wide = wide.join(gf_agg, on='cohort', how='left', coalesce=True)
        print(f"  + gaussian fingerprint: {wide.shape[1] - before} static features")
    
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
        before = wide.shape[1]
        wide = wide.join(gs_agg, on='cohort', how='left', coalesce=True)
        print(f"  + gaussian similarity: {wide.shape[1] - before} static features")
    
    return wide


def add_rolling_features(wide: pl.DataFrame) -> pl.DataFrame:
    """
    Add rolling window statistics computed on ALL non-constant sensor columns.
    Matches Özcan feature depth:
      Windows: 5, 10, 15, 20, 30 cycles
      Stats: mean, std, min, max, range
    """
    meta = {'cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct', 'cycle'}
    derived = {c for c in wide.columns if any(c.startswith(p) for p in
               ['geom_', 'gfp_', 'gsim_', 'rt_', 'roll_'])}
    ops = {'op1', 'op2', 'op3'}

    sensor_cols = sorted([c for c in wide.columns
                          if c not in meta and c not in derived and c not in ops])

    if not sensor_cols:
        return wide

    # Drop constant sensors (zero variance)
    varying = []
    for c in sensor_cols:
        v = wide[c].drop_nulls().var()
        if v is not None and v > 1e-10:
            varying.append(c)

    print(f"  Rolling features on {len(varying)} varying sensors: {varying[:5]}...")

    windows = [5, 10, 15, 20, 30]
    new_cols = []

    for w in windows:
        for s in varying:
            new_cols.extend([
                pl.col(s).rolling_mean(window_size=w, min_periods=1)
                .over('cohort').alias(f'roll_{s}_mean_{w}'),
                pl.col(s).rolling_std(window_size=w, min_periods=2)
                .over('cohort').alias(f'roll_{s}_std_{w}'),
                pl.col(s).rolling_min(window_size=w, min_periods=1)
                .over('cohort').alias(f'roll_{s}_min_{w}'),
                pl.col(s).rolling_max(window_size=w, min_periods=1)
                .over('cohort').alias(f'roll_{s}_max_{w}'),
            ])

    before = wide.shape[1]
    wide = wide.sort(['cohort', 'I']).with_columns(new_cols)

    # Add range = max - min
    for w in windows:
        for s in varying:
            wide = wide.with_columns(
                (pl.col(f'roll_{s}_max_{w}') - pl.col(f'roll_{s}_min_{w}'))
                .alias(f'roll_{s}_range_{w}')
            )

    print(f"  + rolling features: {wide.shape[1] - before} features")

    return wide


def add_delta_features(wide: pl.DataFrame) -> pl.DataFrame:
    """
    Add per-cycle deltas (first differences) for ALL varying sensor columns.
    Captures instantaneous rate of change — velocity at cycle level.
    """
    meta = {'cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct', 'cycle'}
    derived = {c for c in wide.columns if any(c.startswith(p) for p in
               ['geom_', 'gfp_', 'gsim_', 'rt_', 'roll_', 'delta_'])}
    ops = {'op1', 'op2', 'op3'}

    sensor_cols = sorted([c for c in wide.columns
                          if c not in meta and c not in derived and c not in ops])

    # All varying sensors
    varying = []
    for c in sensor_cols:
        v = wide[c].drop_nulls().var()
        if v is not None and v > 1e-10:
            varying.append(c)

    delta_cols = []
    for s in varying:
        delta_cols.append(
            (pl.col(s) - pl.col(s).shift(1)).over('cohort').alias(f'delta_{s}')
        )

    before = wide.shape[1]
    wide = wide.sort(['cohort', 'I']).with_columns(delta_cols)
    print(f"  + delta features: {wide.shape[1] - before} features")

    return wide


def build_cycle_features(obs_path: str, manifold_dir: str, output_path: str):
    """Main: build cycle-level hybrid feature matrix."""
    obs_path = Path(obs_path)
    manifold_dir = Path(manifold_dir)
    output_path = Path(output_path)
    
    print("=" * 60)
    print("  BUILD CYCLE-LEVEL HYBRID FEATURES")
    print("=" * 60)
    
    # Load observations
    obs = pl.read_parquet(str(obs_path))
    print(f"\n  Observations: {len(obs):,} rows")
    print(f"  Signals: {obs['signal_id'].n_unique()}")
    print(f"  Cohorts: {obs['cohort'].n_unique()}")
    
    # Step 1: Pivot to wide (one row per cycle)
    print("\n  [1/7] Pivoting observations to per-cycle wide format...")
    wide = pivot_observations(obs)
    print(f"    → {wide.shape[0]:,} rows × {wide.shape[1]} columns")
    
    # Step 2: Add RUL target
    print("\n  [2/7] Adding RUL + cycle features...")
    lifecycle = compute_lifecycle(obs)
    wide = add_rul_and_cycle(wide, lifecycle)
    
    # Step 3: Interpolate geometry
    print("\n  [3/7] Interpolating manifold geometry to every cycle...")
    wide = interpolate_geometry_to_cycles(wide, manifold_dir)
    
    # Step 4: Real-time geometry (no window needed)
    print("\n  [4/7] Computing real-time per-cycle geometry...")
    wide = add_realtime_geometry(wide, manifold_dir)
    
    # Step 5: Static gaussian features
    print("\n  [5/7] Adding static gaussian fingerprint + similarity...")
    wide = add_gaussian_static(wide, manifold_dir)
    
    # Step 6: Rolling features
    print("\n  [6/7] Computing rolling window statistics...")
    wide = add_rolling_features(wide)
    
    # Step 7: Delta features (cycle-to-cycle velocity)
    print("\n  [7/7] Computing per-cycle deltas...")
    wide = add_delta_features(wide)
    
    # ─── Clean up ───
    # Drop constant columns
    feat_cols = [c for c in wide.columns 
                 if c not in ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']]
    
    drop_const = []
    for c in feat_cols:
        if wide[c].dtype in [pl.Float64, pl.Float32, pl.Int64]:
            vals = wide[c].drop_nulls()
            if len(vals) > 0:
                std = vals.std()
                if std is not None and std < 1e-10:
                    drop_const.append(c)
    
    if drop_const:
        wide = wide.drop(drop_const)
        print(f"\n  Dropped {len(drop_const)} constant columns")
    
    # Drop string columns except cohort
    string_cols = [c for c in wide.columns if wide[c].dtype == pl.Utf8 and c != 'cohort']
    if string_cols:
        wide = wide.drop(string_cols)
    
    # ─── Report ───
    feat_cols = [c for c in wide.columns 
                 if c not in ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']]
    
    # Feature groups
    groups = {}
    for c in feat_cols:
        if c.startswith('geom_'):
            groups.setdefault('geom (interpolated)', []).append(c)
        elif c.startswith('rt_'):
            groups.setdefault('rt (real-time)', []).append(c)
        elif c.startswith('gfp_'):
            groups.setdefault('gfp (fingerprint)', []).append(c)
        elif c.startswith('gsim_'):
            groups.setdefault('gsim (similarity)', []).append(c)
        elif c.startswith('roll_'):
            groups.setdefault('roll (rolling)', []).append(c)
        elif c.startswith('delta_'):
            groups.setdefault('delta (velocity)', []).append(c)
        elif c == 'cycle':
            groups.setdefault('cycle', []).append(c)
        else:
            groups.setdefault('sensor (raw)', []).append(c)
    
    print(f"\n{'=' * 60}")
    print(f"  cycle_features.parquet")
    print(f"{'=' * 60}")
    print(f"  Rows:      {wide.shape[0]:,}")
    print(f"  Features:  {len(feat_cols)}")
    print(f"  Target:    RUL ({wide['RUL'].min()} – {wide['RUL'].max()})")
    print(f"  Cohorts:   {wide['cohort'].n_unique()}")
    print(f"\n  Feature groups:")
    for g in sorted(groups.keys()):
        print(f"    {g:>25s}: {len(groups[g]):>4d} features")
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    wide.write_parquet(str(output_path))
    print(f"\n  → {output_path}")
    print()


# ═══════════════════════════════════════════════
#  PART 2: TRAIN STACKING ENSEMBLE
# ═══════════════════════════════════════════════

def train_and_evaluate(train_path: str, test_path: str, rul_path: str, output_dir: str = None):
    """
    Train LightGBM + CatBoost + GB stacking ensemble.
    Evaluate on official test set.
    
    Matches Özcan approach:
      - Base learners: LightGBM, CatBoost, GradientBoosting
      - Meta-learner: Ridge regression on out-of-fold predictions
      - Prediction: last cycle per test engine
    """
    import json
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer
        from sklearn.model_selection import GroupKFold
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    except ImportError:
        print("ERROR: scikit-learn required")
        raise SystemExit(1)
    
    try:
        import lightgbm as lgb
        HAS_LGBM = True
    except ImportError:
        print("WARNING: lightgbm not available, using sklearn GBR as substitute")
        HAS_LGBM = False
    
    try:
        import catboost as cb
        HAS_CATBOOST = True
    except ImportError:
        print("WARNING: catboost not available, using sklearn GBR as substitute")
        HAS_CATBOOST = False
    
    # ─── Load data ───
    train_df = pl.read_parquet(train_path)
    test_df = pl.read_parquet(test_path)
    
    # Ground truth
    with open(rul_path) as f:
        y_true = np.array([int(line.strip()) for line in f if line.strip()], dtype=float)
    y_true_capped = np.minimum(y_true, MAX_RUL_CAP)
    
    print("=" * 60)
    print("  TRAIN STACKING ENSEMBLE — CYCLE LEVEL")
    print("=" * 60)
    print(f"\n  Train: {train_df.shape[0]:,} rows × {train_df.shape[1]} cols")
    print(f"  Test:  {test_df.shape[0]:,} rows × {test_df.shape[1]} cols")
    print(f"  Ground truth: {len(y_true)} engines")
    
    # ─── Prepare features ───
    meta_cols = ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']
    feat_cols = sorted([c for c in train_df.columns if c not in meta_cols])
    
    # Ensure test has same columns
    for c in feat_cols:
        if c not in test_df.columns:
            test_df = test_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))
    
    test_feat_cols = [c for c in feat_cols if c in test_df.columns]
    
    X_train = train_df.select(feat_cols).to_numpy().astype(np.float64)
    y_train = train_df['RUL'].to_numpy().astype(np.float64)
    groups = train_df['cohort'].to_numpy()
    
    # Get last cycle per test engine
    def cohort_sort_key(c):
        num = ''.join(filter(str.isdigit, str(c)))
        return int(num) if num else 0
    
    test_last = (test_df
        .with_columns(pl.col('cohort').map_elements(cohort_sort_key, return_dtype=pl.Int64).alias('_sort'))
        .group_by('cohort')
        .agg([pl.all().sort_by('I').last()])
        .sort('_sort')
        .drop('_sort')
    )
    
    sorted_cohorts = test_last['cohort'].to_list()
    X_test = test_last.select(feat_cols).to_numpy().astype(np.float64)
    
    print(f"  Feature columns: {len(feat_cols)}")
    print(f"  Test engines (last cycle): {len(sorted_cohorts)}")
    
    # ─── Impute + scale ───
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    X_train = np.where(np.isinf(X_train), 0, X_train)
    X_test = np.where(np.isinf(X_test), 0, X_test)
    
    # Drop high-null columns (>30%)
    null_pcts = np.isnan(X_train).mean(axis=0)
    keep_mask = null_pcts <= 0.30
    if keep_mask.sum() < X_train.shape[1]:
        print(f"  Dropping {(~keep_mask).sum()} high-null features")
        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    print(f"  Final feature matrix: {X_train_s.shape}")
    
    # ─── Define base learners ───
    base_models = {}
    
    if HAS_LGBM:
        base_models['lightgbm'] = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1,
        )
    
    if HAS_CATBOOST:
        base_models['catboost'] = cb.CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, random_seed=42, verbose=0,
        )
    
    base_models['gradient_boosting'] = GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        min_samples_leaf=5, subsample=0.8, random_state=42,
    )
    
    print(f"\n  Base learners: {list(base_models.keys())}")
    
    # ─── Generate out-of-fold predictions for stacking ───
    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    
    oof_preds = {name: np.zeros(len(y_train)) for name in base_models}
    test_preds = {name: np.zeros(len(X_test_s)) for name in base_models}
    fold_scores = {name: [] for name in base_models}
    
    print(f"\n  Generating OOF predictions ({n_folds}-fold GroupKFold)...")
    
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_s, y_train, groups)):
        X_tr, X_val = X_train_s[train_idx], X_train_s[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        
        for name, model in base_models.items():
            import copy
            m = copy.deepcopy(model) if not HAS_CATBOOST or name != 'catboost' else cb.CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, random_seed=42, verbose=0,
            )
            
            if name == 'lightgbm' and HAS_LGBM:
                m = lgb.LGBMRegressor(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    num_leaves=31, min_child_samples=10, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    random_state=42, verbose=-1, n_jobs=-1,
                )
            
            m.fit(X_tr, y_tr)
            
            val_pred = m.predict(X_val)
            oof_preds[name][val_idx] = val_pred
            
            # Accumulate test predictions (average across folds)
            test_preds[name] += m.predict(X_test_s) / n_folds
            
            rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            fold_scores[name].append(rmse)
        
        print(f"    Fold {fold_idx}: " + 
              "  ".join(f"{n}={fold_scores[n][-1]:.2f}" for n in base_models))
    
    # ─── Report base learner performance ───
    print(f"\n  Base learner CV results:")
    for name in base_models:
        scores = fold_scores[name]
        print(f"    {name:<20s}: RMSE = {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    
    # ─── Stack with Ridge meta-learner ───
    print(f"\n  Training Ridge meta-learner on OOF predictions...")
    
    # Stack OOF predictions as features for meta-learner
    oof_stack = np.column_stack([oof_preds[name] for name in base_models])
    test_stack = np.column_stack([test_preds[name] for name in base_models])
    
    meta = Ridge(alpha=1.0)
    meta.fit(oof_stack, y_train)
    
    print(f"    Meta weights: {dict(zip(base_models.keys(), meta.coef_.round(3).tolist()))}")
    print(f"    Meta intercept: {meta.intercept_:.3f}")
    
    # ─── Final predictions ───
    # Individual base learners (retrained on full training set)
    final_preds = {}
    
    for name, model_template in base_models.items():
        # Re-create fresh model
        if name == 'lightgbm' and HAS_LGBM:
            m = lgb.LGBMRegressor(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                num_leaves=31, min_child_samples=10, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                random_state=42, verbose=-1, n_jobs=-1,
            )
        elif name == 'catboost' and HAS_CATBOOST:
            m = cb.CatBoostRegressor(
                iterations=500, depth=6, learning_rate=0.05,
                l2_leaf_reg=3.0, random_seed=42, verbose=0,
            )
        else:
            m = GradientBoostingRegressor(
                n_estimators=500, max_depth=5, learning_rate=0.05,
                min_samples_leaf=5, subsample=0.8, random_state=42,
            )
        
        m.fit(X_train_s, y_train)
        pred = np.clip(m.predict(X_test_s), 0, MAX_RUL_CAP)
        final_preds[name] = pred
    
    # Stacking prediction
    final_stack = np.column_stack([final_preds[name] for name in base_models])
    y_stacked = np.clip(meta.predict(final_stack), 0, MAX_RUL_CAP)
    final_preds['stacking_ensemble'] = y_stacked
    
    # Also add simple average
    avg_pred = np.clip(np.mean([final_preds[n] for n in base_models], axis=0), 0, MAX_RUL_CAP)
    final_preds['simple_average'] = avg_pred
    
    # ─── PHM08 Score ───
    def phm08_score(y_true, y_pred):
        d = y_pred - y_true
        s = 0.0
        for di in d:
            s += np.exp(-di / 13.0) - 1.0 if di < 0 else np.exp(di / 10.0) - 1.0
        return s
    
    # ─── Evaluate all models ───
    n_test = min(len(y_true_capped), len(sorted_cohorts))
    
    print(f"\n{'=' * 70}")
    print(f"  OFFICIAL TEST RESULTS — FD001")
    print(f"{'=' * 70}")
    print(f"\n  {'Model':<25s} {'RMSE':>8s} {'MAE':>8s} {'R²':>8s} {'PHM08':>10s} {'Bias':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
    
    results = {}
    best_rmse = 999
    best_name = ''
    
    for name in sorted(final_preds.keys()):
        pred = final_preds[name][:n_test]
        true = y_true_capped[:n_test]
        
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        score = phm08_score(true, pred)
        bias = float(np.mean(pred - true))
        
        results[name] = {
            'rmse': float(rmse), 'mae': float(mae), 'r2': float(r2),
            'phm08_score': float(score), 'bias': float(bias),
        }
        
        marker = ''
        if rmse < best_rmse:
            best_rmse = rmse
            best_name = name
            marker = ' ◄'
        
        print(f"  {name:<25s} {rmse:>8.2f} {mae:>8.2f} {r2:>8.4f} "
              f"{score:>10.1f} {bias:>+8.2f}{marker}")
    
    # Mark best
    print(f"\n  Best: {best_name} (RMSE={best_rmse:.2f})")
    print(f"\n  Published benchmark:")
    print(f"    Özcan LightGBM+CatBoost (2025): RMSE = 6.62, PHM08 = 2,951")
    
    # ─── Save ───
    if output_dir:
        out = Path(output_dir)
    else:
        out = Path(test_path).parent / 'ml_results_v2'
    out.mkdir(parents=True, exist_ok=True)
    
    # Summary
    summary = {
        'dataset': 'FD001',
        'evaluation': 'official_test_split',
        'approach': 'cycle_level_hybrid_stacking',
        'n_train_rows': int(X_train_s.shape[0]),
        'n_features': int(X_train_s.shape[1]),
        'n_test_engines': n_test,
        'best_model': best_name,
        'best_rmse': float(best_rmse),
        'models': results,
    }
    with open(out / 'test_summary_v2.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Per-engine predictions
    pred_rows = []
    for i in range(n_test):
        row = {
            'cohort': sorted_cohorts[i],
            'RUL_true': float(y_true[i]),
            'RUL_true_capped': float(y_true_capped[i]),
        }
        for name, preds in final_preds.items():
            row[f'pred_{name}'] = float(preds[i])
        pred_rows.append(row)
    
    pl.DataFrame(pred_rows).write_parquet(str(out / 'test_predictions_v2.parquet'))
    
    print(f"\n  Outputs: {out}")
    print()


# ═══════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Prime ML v2 — Cycle-Level Hybrid Features')
    sub = parser.add_subparsers(dest='command')

    # Build features
    build_parser = sub.add_parser('build', help='Build cycle-level feature matrix')
    build_parser.add_argument('--obs', required=True, help='Path to observations.parquet')
    build_parser.add_argument('--manifold', required=True, help='Manifold output directory')
    build_parser.add_argument('--output', required=True, help='Output parquet path')

    # Train + evaluate
    train_parser = sub.add_parser('train', help='Train stacking ensemble and evaluate')
    train_parser.add_argument('--train', required=True, help='Training cycle_features.parquet')
    train_parser.add_argument('--test', required=True, help='Test cycle_features.parquet')
    train_parser.add_argument('--rul', required=True, help='Path to RUL_FD001.txt')
    train_parser.add_argument('--output', default=None, help='Output directory')

    args = parser.parse_args()

    if args.command == 'build':
        build_cycle_features(args.obs, args.manifold, args.output)
    elif args.command == 'train':
        train_and_evaluate(args.train, args.test, args.rul, args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
