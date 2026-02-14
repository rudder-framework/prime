#!/usr/bin/env python3
"""
FD002 Pipeline — Same as FD001 11.72 result, multi-regime dataset
=================================================================

Same pipeline:
  - Raw sensors + rolling stats + deltas + RT geometry
  - LGB + XGB + HistGBR → RidgeCV
  - 5-fold GroupKFold (3-fold if OOM)

FD002 adaptation:
  - 6 operating regimes (clustered from op1, op2, op3)
  - Per-regime fleet baselines (from training data first 20%)
  - Observations scored against their regime's baseline
  - regime_id included as feature (Option A: let trees handle regime jumps)

No normalization of observations. No new feature types.
"""

import numpy as np
import polars as pl
import json
import copy
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import xgboost as xgb

# ─── Paths ───
DATA_DIR = Path('/Users/jasonrudder/data/FD_002')
TRAIN_OBS = DATA_DIR / 'train' / 'observations.parquet'
TEST_OBS = DATA_DIR / 'test' / 'observations.parquet'
RUL_FILE = DATA_DIR / 'RUL_FD002.txt'
FLEET_BASELINE = DATA_DIR / 'fleet_baseline.parquet'
TRAIN_FEATURES = DATA_DIR / 'train' / 'cycle_features.parquet'
TEST_FEATURES = DATA_DIR / 'test' / 'cycle_features.parquet'

MAX_RUL_CAP = 125


def phm08_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    return sum(np.exp(-di / 13.0) - 1 if di < 0 else np.exp(di / 10.0) - 1 for di in d)


def cohort_sort_key(c):
    num = ''.join(filter(str.isdigit, str(c)))
    return int(num) if num else 0


# ═══════════════════════════════════════════════
#  STEP 2: CLUSTER OPERATING REGIMES
# ═══════════════════════════════════════════════

def cluster_regimes(obs: pl.DataFrame, n_regimes: int = 6) -> tuple:
    """
    Cluster operating conditions into regimes using KMeans.
    Returns (regime_map: dict mapping (op1_r, op2_r, op3_r) -> regime_id,
             kmeans_model: fitted KMeans for scoring test data).
    """
    print("\n  [STEP 2] Clustering operating regimes...")

    # Get op values per (cohort, I) — pivot to wide
    ops = obs.filter(pl.col('signal_id').is_in(['op1', 'op2', 'op3']))
    ops_wide = ops.pivot(
        on='signal_id', index=['cohort', 'I'], values='value',
        aggregate_function='first',
    ).sort(['cohort', 'I'])

    op_matrix = ops_wide.select(['op1', 'op2', 'op3']).to_numpy().astype(np.float64)
    print(f"    Operating conditions: {op_matrix.shape[0]:,} observations")

    # KMeans clustering
    km = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    labels = km.fit_predict(op_matrix)

    # Report
    unique, counts = np.unique(labels, return_counts=True)
    print(f"    {n_regimes} regimes found:")
    for r, c in zip(unique, counts):
        center = km.cluster_centers_[r]
        print(f"      regime {r}: {c:,} cycles  (op1={center[0]:.2f}, op2={center[1]:.4f}, op3={center[2]:.1f})")

    # Add regime_id to ops_wide
    ops_wide = ops_wide.with_columns(pl.Series('regime_id', labels.astype(np.int32)))

    return ops_wide.select(['cohort', 'I', 'regime_id']), km


def assign_regimes(obs: pl.DataFrame, km: KMeans) -> pl.DataFrame:
    """Assign regime labels to any observation set using fitted KMeans."""
    ops = obs.filter(pl.col('signal_id').is_in(['op1', 'op2', 'op3']))
    ops_wide = ops.pivot(
        on='signal_id', index=['cohort', 'I'], values='value',
        aggregate_function='first',
    ).sort(['cohort', 'I'])

    op_matrix = ops_wide.select(['op1', 'op2', 'op3']).to_numpy().astype(np.float64)
    labels = km.predict(op_matrix)
    ops_wide = ops_wide.with_columns(pl.Series('regime_id', labels.astype(np.int32)))
    return ops_wide.select(['cohort', 'I', 'regime_id'])


# ═══════════════════════════════════════════════
#  STEP 2+3: PER-REGIME FLEET BASELINES + SCORING
# ═══════════════════════════════════════════════

def pivot_observations(obs: pl.DataFrame) -> pl.DataFrame:
    """Pivot long-format observations to wide: one row per (cohort, I)."""
    wide = obs.pivot(
        on='signal_id', index=['cohort', 'I'], values='value',
        aggregate_function='first',
    ).sort(['cohort', 'I'])
    return wide


def compute_regime_baselines(
    train_wide: pl.DataFrame,
    train_regimes: pl.DataFrame,
    sensor_cols: list,
    baseline_fraction: float = 0.20,
) -> dict:
    """
    Compute per-regime fleet baselines from training data.

    For each regime:
      1. Collect first 20% of cycles from ALL training engines operating in that regime
      2. Compute centroid, std, SVD → principal directions
      3. Store baseline dict

    Returns: {regime_id: {centroid, std, pcs, eigenvalues}}
    """
    print("\n  [STEP 2b] Computing per-regime fleet baselines...")

    # Join regime labels to wide data
    train_with_regime = train_wide.join(train_regimes, on=['cohort', 'I'], how='left')

    regimes = sorted(train_regimes['regime_id'].unique().to_list())
    baselines = {}

    for regime in regimes:
        regime_data = train_with_regime.filter(pl.col('regime_id') == regime)

        # For each cohort, take first 20% of cycles in this regime
        pooled = []
        cohorts = regime_data['cohort'].unique().sort().to_list()

        for cohort in cohorts:
            cohort_regime = regime_data.filter(pl.col('cohort') == cohort).sort('I')
            n = len(cohort_regime)
            if n < 3:
                continue
            n_baseline = max(3, int(n * baseline_fraction))
            x = cohort_regime.select(sensor_cols).to_numpy().astype(np.float64)

            # Handle NaN
            for j in range(x.shape[1]):
                col = x[:, j]
                nans = np.isnan(col)
                if nans.any() and not nans.all():
                    col[nans] = np.nanmedian(col)
                elif nans.all():
                    col[:] = 0.0
                x[:, j] = col

            pooled.append(x[:n_baseline])

        if not pooled:
            print(f"    regime {regime}: no valid data, skipping")
            continue

        x_pooled = np.vstack(pooled)
        centroid = np.mean(x_pooled, axis=0)
        baseline_std = np.std(x_pooled, axis=0)
        baseline_std[baseline_std < 1e-12] = 1.0

        x_normed = (x_pooled - centroid) / baseline_std

        try:
            U, S, Vt = np.linalg.svd(x_normed, full_matrices=False)
            n_samples = x_pooled.shape[0]
            eigenvalues = (S ** 2) / max(1, n_samples - 1)
            k = min(5, len(S))
            pcs = Vt[:k]
        except np.linalg.LinAlgError:
            eigenvalues = np.ones(min(5, x_pooled.shape[1]))
            pcs = np.eye(min(5, x_pooled.shape[1]), x_pooled.shape[1])

        total_var = float(np.sum(eigenvalues))
        eff_dim = float(total_var ** 2 / np.sum(eigenvalues ** 2)) if total_var > 0 else 0.0

        baselines[regime] = {
            'centroid': centroid,
            'std': baseline_std,
            'pcs': pcs,
            'eigenvalues': eigenvalues[:min(5, len(eigenvalues))],
            'n_pooled': x_pooled.shape[0],
            'n_cohorts': len(pooled),
            'effective_dim': eff_dim,
            'total_variance': total_var,
        }

        print(f"    regime {regime}: {x_pooled.shape[0]:,} pooled cycles from {len(pooled)} engines, "
              f"eff_dim={eff_dim:.2f}, total_var={total_var:.2f}")

    return baselines


def score_observations(
    wide: pl.DataFrame,
    regime_labels: pl.DataFrame,
    baselines: dict,
    sensor_cols: list,
) -> pl.DataFrame:
    """
    Score every observation against its regime's fleet baseline.

    Returns DataFrame with: cohort, I, rt_centroid_dist, rt_centroid_dist_norm,
                            rt_pc1_projection, rt_pc2_projection, rt_pc3_projection
    """
    print("\n  [STEP 3] Scoring observations against regime baselines...")

    # Join regime
    with_regime = wide.join(regime_labels, on=['cohort', 'I'], how='left')

    results = []
    n_scored = 0

    cohorts = with_regime['cohort'].unique().sort().to_list()

    for cohort in cohorts:
        cohort_data = with_regime.filter(pl.col('cohort') == cohort).sort('I')
        x = cohort_data.select(sensor_cols).to_numpy().astype(np.float64)
        i_values = cohort_data['I'].to_numpy()
        regimes = cohort_data['regime_id'].to_numpy()

        # Handle NaN in sensor data
        for j in range(x.shape[1]):
            col = x[:, j]
            nans = np.isnan(col)
            if nans.any() and not nans.all():
                col[nans] = np.nanmedian(col)
            elif nans.all():
                col[:] = 0.0
            x[:, j] = col

        for i in range(len(x)):
            regime = int(regimes[i]) if not np.isnan(regimes[i]) else 0
            bl = baselines.get(regime)
            if bl is None:
                # Fallback to first available baseline
                bl = next(iter(baselines.values()))

            x_norm = (x[i] - bl['centroid']) / bl['std']

            dist = float(np.sqrt(np.sum(x_norm ** 2)))
            dist_norm = dist / np.sqrt(len(sensor_cols))

            pcs = bl['pcs']
            pc1_proj = float(x_norm @ pcs[0]) if len(pcs) > 0 else 0.0
            pc2_proj = float(x_norm @ pcs[1]) if len(pcs) > 1 else 0.0
            pc3_proj = float(x_norm @ pcs[2]) if len(pcs) > 2 else 0.0

            results.append({
                'cohort': cohort,
                'I': int(i_values[i]),
                'rt_centroid_dist': dist,
                'rt_centroid_dist_norm': dist_norm,
                'rt_pc1_projection': pc1_proj,
                'rt_pc2_projection': pc2_proj,
                'rt_pc3_projection': pc3_proj,
            })
            n_scored += 1

    print(f"    Scored {n_scored:,} observations across {len(cohorts)} engines")

    return pl.DataFrame(results)


# ═══════════════════════════════════════════════
#  STEP 4: BUILD FEATURE MATRIX
# ═══════════════════════════════════════════════

def build_features(
    obs: pl.DataFrame,
    regime_labels: pl.DataFrame,
    rt_geometry: pl.DataFrame,
    is_train: bool = True,
) -> pl.DataFrame:
    """
    Build cycle-level feature matrix — same features as FD001 11.72 result:
      - Raw sensor values (21 sensors + op1/op2/op3)
      - cycle number
      - regime_id (Option A: let trees handle regime jumps)
      - Rolling statistics (mean, std, min, max, range) at windows [5, 10, 15, 20, 30]
      - Delta features (cycle-to-cycle change)
      - RT geometry: rt_centroid_dist, rt_centroid_dist_norm, rt_pc1/2/3_projection
      - RUL target (capped at 125)
    """
    split = "TRAIN" if is_train else "TEST"
    print(f"\n  [STEP 4] Building {split} feature matrix...")

    # 1. Pivot to wide
    wide = pivot_observations(obs)
    print(f"    Pivoted: {wide.shape[0]:,} rows x {wide.shape[1]} cols")

    # 2. Add regime_id
    wide = wide.join(regime_labels, on=['cohort', 'I'], how='left')

    # 3. Add RUL + cycle
    first_sig = obs['signal_id'].unique().sort()[0]
    lifecycle = (
        obs.filter(pl.col('signal_id') == first_sig)
        .group_by('cohort')
        .agg(pl.col('I').max().alias('max_I'))
    )
    wide = wide.join(lifecycle, on='cohort', how='left')
    wide = wide.with_columns([
        (pl.col('max_I') - pl.col('I')).alias('RUL_raw'),
        (pl.col('max_I') + 1).alias('lifecycle'),
        (pl.col('I') / pl.col('max_I')).alias('lifecycle_pct'),
        pl.col('I').alias('cycle'),
    ])
    wide = wide.with_columns(
        pl.when(pl.col('RUL_raw') > MAX_RUL_CAP)
        .then(MAX_RUL_CAP)
        .otherwise(pl.col('RUL_raw'))
        .alias('RUL')
    )
    wide = wide.drop(['max_I', 'RUL_raw'])

    # 4. Join RT geometry from stage 35 scoring
    wide = wide.join(rt_geometry, on=['cohort', 'I'], how='left', coalesce=True)
    print(f"    + RT geometry: 5 features")

    # 5. Rolling features on varying sensors
    meta = {'cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct', 'cycle', 'regime_id'}
    derived = {c for c in wide.columns if c.startswith('rt_') or c.startswith('roll_') or c.startswith('delta_')}
    ops = {'op1', 'op2', 'op3'}
    sensor_cols = sorted([c for c in wide.columns
                          if c not in meta and c not in derived and c not in ops])

    # Filter to varying sensors
    varying = []
    for c in sensor_cols:
        v = wide[c].drop_nulls().var()
        if v is not None and v > 1e-10:
            varying.append(c)

    print(f"    Rolling features on {len(varying)} varying sensors...")

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

    wide = wide.sort(['cohort', 'I']).with_columns(new_cols)

    # Add range = max - min
    for w in windows:
        for s in varying:
            wide = wide.with_columns(
                (pl.col(f'roll_{s}_max_{w}') - pl.col(f'roll_{s}_min_{w}'))
                .alias(f'roll_{s}_range_{w}')
            )

    n_rolling = len(varying) * len(windows) * 5
    print(f"    + rolling features: {n_rolling}")

    # 6. Delta features
    delta_cols = []
    for s in varying:
        delta_cols.append(
            (pl.col(s) - pl.col(s).shift(1)).over('cohort').alias(f'delta_{s}')
        )
    wide = wide.sort(['cohort', 'I']).with_columns(delta_cols)
    print(f"    + delta features: {len(varying)}")

    # 7. Drop constant columns
    feat_cols = [c for c in wide.columns
                 if c not in ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']]
    drop_const = []
    for c in feat_cols:
        if wide[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]:
            vals = wide[c].drop_nulls()
            if len(vals) > 0:
                std = vals.std()
                if std is not None and std < 1e-10:
                    drop_const.append(c)

    if drop_const:
        wide = wide.drop(drop_const)
        print(f"    Dropped {len(drop_const)} constant columns")

    # Drop string columns except cohort
    string_cols = [c for c in wide.columns if wide[c].dtype == pl.Utf8 and c != 'cohort']
    if string_cols:
        wide = wide.drop(string_cols)

    feat_cols = [c for c in wide.columns
                 if c not in ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']]
    print(f"    Final: {wide.shape[0]:,} rows x {len(feat_cols)} features")

    return wide


# ═══════════════════════════════════════════════
#  STEP 5+6: TRAIN + EVALUATE ENSEMBLE
# ═══════════════════════════════════════════════

def train_and_evaluate(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_true: np.ndarray,
):
    """
    Same ensemble as FD001 11.72:
      LGB + XGB + HistGBR → RidgeCV meta-learner
      5-fold GroupKFold (fallback to 3-fold if needed)
      Same hyperparameters, n_estimators=300 for memory safety
    """
    y_true_capped = np.minimum(y_true, MAX_RUL_CAP)

    # ─── Feature selection: ONLY generalizing features ───
    meta_cols = {'cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct'}
    leak_prefixes = ('geom_', 'gfp_', 'gsim_')  # per-engine identity encoders

    feat_cols = sorted([c for c in train_df.columns
                        if c not in meta_cols
                        and not any(c.startswith(p) for p in leak_prefixes)])

    print(f"\n  [STEP 5] Training ensemble...")
    print(f"    Train: {train_df.shape[0]:,} rows")
    print(f"    Test:  {test_df.shape[0]:,} rows")
    print(f"    Features (clean): {len(feat_cols)}")

    # Feature groups
    groups = {}
    for c in feat_cols:
        if c.startswith('roll_'):
            groups.setdefault('rolling', []).append(c)
        elif c.startswith('delta_'):
            groups.setdefault('delta', []).append(c)
        elif c.startswith('rt_'):
            groups.setdefault('rt_geometry', []).append(c)
        elif c == 'cycle':
            groups.setdefault('cycle', []).append(c)
        elif c == 'regime_id':
            groups.setdefault('regime', []).append(c)
        else:
            groups.setdefault('sensor', []).append(c)

    for p in sorted(groups):
        print(f"      {p}: {len(groups[p])}")

    # ─── Prepare ───
    for c in feat_cols:
        if c not in test_df.columns:
            test_df = test_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

    X_train = train_df.select(feat_cols).to_numpy().astype(np.float32)
    y_train = train_df['RUL'].to_numpy().astype(np.float32)
    groups_arr = train_df['cohort'].to_numpy()

    # Last cycle per test engine (sorted numerically)
    test_last = (test_df
        .with_columns(pl.col('cohort').map_elements(cohort_sort_key, return_dtype=pl.Int64).alias('_sort'))
        .group_by('cohort')
        .agg([pl.all().sort_by('I').last()])
        .sort('_sort')
        .drop('_sort')
    )
    X_test = test_last.select(feat_cols).to_numpy().astype(np.float32)

    print(f"    Test engines (last cycle): {X_test.shape[0]}")

    # ─── Impute + scale ───
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = np.where(np.isinf(X_train), 0, X_train)
    X_test = np.where(np.isinf(X_test), 0, X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print(f"    Final matrix: {X_train_s.shape}")

    # ─── Base learners (n_estimators=300 for 16GB memory safety, n_jobs=2) ───
    base_models = {
        'lgb': lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            num_leaves=15, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=2),
        'xgb': xgb.XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=10, random_state=42, verbosity=0, n_jobs=2),
        'hist': HistGradientBoostingRegressor(
            max_iter=300, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, random_state=42),
    }

    # ─── 5-fold GroupKFold ───
    n_folds = 5
    gkf = GroupKFold(n_splits=n_folds)
    oof_preds = {n: np.zeros(len(y_train)) for n in base_models}
    test_preds = {n: np.zeros(len(X_test_s)) for n in base_models}

    print(f"\n    {n_folds}-fold GroupKFold stacking...")
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_s, y_train, groups_arr)):
        X_tr, X_val = X_train_s[train_idx], X_train_s[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        for name, model in base_models.items():
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = m.predict(X_val)
            test_preds[name] += m.predict(X_test_s) / n_folds
        oof_scores = {n: np.sqrt(mean_squared_error(y_train[val_idx], oof_preds[n][val_idx]))
                      for n in base_models}
        print(f"      Fold {fold_idx+1}: " + ", ".join(f"{n}={v:.2f}" for n, v in oof_scores.items()))

    # ─── OOF ───
    print(f"\n    OOF Scores:")
    for name in base_models:
        r = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        print(f"      {name}: RMSE={r:.2f}")

    # ─── Stack ───
    stack_train = np.column_stack([oof_preds[n] for n in base_models])
    stack_test = np.column_stack([test_preds[n] for n in base_models])
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta.fit(stack_train, y_train)
    pred = np.clip(meta.predict(stack_test), 0, 125)

    # ─── OOF ensemble RMSE ───
    oof_ensemble = np.clip(meta.predict(stack_train), 0, 125)
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_ensemble))

    # ─── Test metrics ───
    n_test = min(len(y_true_capped), len(pred))
    pred = pred[:n_test]
    y_tc = y_true_capped[:n_test]

    rmse = np.sqrt(mean_squared_error(y_tc, pred))
    mae = mean_absolute_error(y_tc, pred)
    r2 = r2_score(y_tc, pred)
    score = phm08_score(y_tc, pred)
    gap = rmse - oof_rmse

    # ─── Feature importance ───
    lgb_full = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=15, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=2)
    lgb_full.fit(X_train_s, y_train)
    imp = dict(zip(feat_cols, lgb_full.feature_importances_))
    total_imp = sum(imp.values())

    # ─── Report ───
    print(f"\n{'='*60}")
    print(f"FD002 Results")
    print(f"{'='*60}")
    print(f"OOF RMSE:    {oof_rmse:.2f}")
    print(f"Test RMSE:   {rmse:.2f}")
    print(f"Gap:         {gap:+.2f} (test - oof, negative = good)")
    print(f"PHM08 Score: {score:,.0f}")
    print(f"R2:          {r2:.4f}")
    print(f"MAE:         {mae:.2f}")
    print(f"Meta alpha:  {meta.alpha_:.2f}")
    print(f"Meta weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(base_models.keys(), meta.coef_)))

    print(f"\nTop 10 Feature Importances:")
    sorted_imp = sorted(imp.items(), key=lambda x: x[1], reverse=True)
    for i, (k, v) in enumerate(sorted_imp[:10]):
        pct = 100 * v / total_imp if total_imp > 0 else 0
        print(f"  {i+1:2d}. {k}: {pct:.1f}%")

    print(f"\nComparison:")
    print(f"| Dataset | OOF   | Test  | Gap    | PHM08 |")
    print(f"|---------|-------|-------|--------|-------|")
    print(f"| FD001   | 12.41 | 11.72 | -0.69  | 188   |")
    print(f"| FD002   | {oof_rmse:.2f} | {rmse:.2f} | {gap:+.2f} | {score:,.0f}   |")

    print(f"\nReference: Ozcan FD002 = 10.15 RMSE")

    # Base learner test scores
    print(f"\nBase learner test:")
    for name in base_models:
        p = np.clip(test_preds[name][:n_test], 0, 125)
        r = np.sqrt(mean_squared_error(y_tc, p))
        s = phm08_score(y_tc, p)
        print(f"  {name}: RMSE={r:.2f}, PHM08={s:,.0f}")

    # Worst 10
    errors = pred - y_tc
    worst = np.argsort(np.abs(errors))[-10:][::-1]
    print(f"\nWorst 10 predictions:")
    for i in worst:
        print(f"  Engine {i+1}: pred={pred[i]:.1f}, true={y_tc[i]:.0f}, err={errors[i]:+.1f}")

    return rmse, oof_rmse, score, r2


# ═══════════════════════════════════════════════
#  MAIN PIPELINE
# ═══════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  FD002 PIPELINE — Same as FD001 11.72, multi-regime")
    print("=" * 60)

    # ─── Load data ───
    print("\n  [STEP 1] Loading observations...")
    train_obs = pl.read_parquet(str(TRAIN_OBS))
    test_obs = pl.read_parquet(str(TEST_OBS))
    with open(str(RUL_FILE)) as f:
        y_true = np.array([int(line.strip()) for line in f if line.strip()], dtype=float)

    print(f"    Train: {train_obs.shape[0]:,} rows, {train_obs['cohort'].n_unique()} engines")
    print(f"    Test:  {test_obs.shape[0]:,} rows, {test_obs['cohort'].n_unique()} engines")
    print(f"    RUL:   {len(y_true)} test engines")

    # ─── Step 2: Cluster regimes from training data ───
    train_regimes, km = cluster_regimes(train_obs, n_regimes=6)
    test_regimes = assign_regimes(test_obs, km)

    # ─── Identify sensor columns (excluding ops) ───
    all_signals = sorted(train_obs['signal_id'].unique().to_list())
    sensor_cols = [s for s in all_signals if s not in ['op1', 'op2', 'op3']]
    print(f"\n    Sensor columns for baseline: {len(sensor_cols)}")

    # ─── Pivot ONCE for baseline computation ───
    train_wide = pivot_observations(train_obs)

    # ─── Step 2b: Per-regime fleet baselines ───
    baselines = compute_regime_baselines(
        train_wide, train_regimes, sensor_cols, baseline_fraction=0.20,
    )

    # ─── Step 3: Score ALL observations ───
    # Score train
    train_rt = score_observations(train_wide, train_regimes, baselines, sensor_cols)

    # Score test
    test_wide = pivot_observations(test_obs)
    test_rt = score_observations(test_wide, test_regimes, baselines, sensor_cols)

    # ─── Step 4: Build feature matrices ───
    train_features = build_features(train_obs, train_regimes, train_rt, is_train=True)
    test_features = build_features(test_obs, test_regimes, test_rt, is_train=False)

    # Save features
    TRAIN_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    TEST_FEATURES.parent.mkdir(parents=True, exist_ok=True)
    train_features.write_parquet(str(TRAIN_FEATURES))
    test_features.write_parquet(str(TEST_FEATURES))
    print(f"\n    Saved: {TRAIN_FEATURES}")
    print(f"    Saved: {TEST_FEATURES}")

    # ─── Step 5+6: Train + Evaluate ───
    train_and_evaluate(train_features, test_features, y_true)


if __name__ == '__main__':
    main()
