#!/usr/bin/env python3
"""
Run Cycle-Level ML Pipeline for any C-MAPSS / PHM08 Dataset
=============================================================

One command: ingest → manifold → features → train → evaluate.
NO dataset-specific code. Same pipeline, same features, same model.

Usage:
    python run_cycle_ml.py ~/data/FD_001/
    python run_cycle_ml.py ~/data/FD_002/
    python run_cycle_ml.py ~/data/FD_003/
    python run_cycle_ml.py ~/data/FD_004/
    python run_cycle_ml.py --all          # Run all datasets

Directory structure expected:
    ~/data/FD_XXX/
    ├── train_FDXXX.txt          ← raw NASA file
    ├── test_FDXXX.txt           ← raw NASA file
    ├── RUL_FDXXX.txt            ← ground truth
    ├── train/
    │   ├── observations.parquet ← created by convert_cmapss.py
    │   └── output/              ← created by Manifold pipeline
    └── test/
        ├── observations.parquet
        └── output/
"""

import argparse
import subprocess
import sys
import time
import numpy as np
import polars as pl
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import copy
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    import xgboost as xgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

PRIME_DIR = Path(__file__).parent.resolve()
MANIFOLD_DIR = Path.home() / 'manifold'
PRIME_VENV = PRIME_DIR / 'venv' / 'bin' / 'python'
MANIFOLD_VENV = MANIFOLD_DIR / 'venv' / 'bin' / 'python'
MAX_RUL_CAP = 125


def run_cmd(cmd, cwd=None, timeout=1800):
    """Run a command, stream output, raise on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(
        [str(c) for c in cmd],
        cwd=str(cwd) if cwd else None,
        timeout=timeout,
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}")


def phm08_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    return sum(np.exp(-di / 13.0) - 1 if di < 0 else np.exp(di / 10.0) - 1 for di in d)


def cohort_sort_key(c):
    num = ''.join(filter(str.isdigit, str(c)))
    return int(num) if num else 0


def detect_regimes(obs_path):
    """Detect operating condition regimes from observations."""
    obs = pl.read_parquet(str(obs_path))

    # Get op1, op2, op3 for all cycles
    ops_df = obs.filter(pl.col('signal_id').is_in(['op1', 'op2', 'op3']))
    wide = ops_df.pivot(on='signal_id', index=['cohort', 'I'], values='value')

    if 'op1' not in wide.columns:
        return 1, None  # No op settings

    ops = wide.select(['op1', 'op2', 'op3']).to_numpy()
    # Round to find distinct regimes
    ops_rounded = np.round(ops, 0)
    unique_regimes = np.unique(ops_rounded, axis=0)
    return len(unique_regimes), wide


def normalize_by_regime(data_dir):
    """
    Normalize sensors per operating condition for multi-regime datasets.
    Computes regime stats from TRAINING data, applies to both train and test.
    Saves normalized observations alongside originals.
    """
    train_obs_path = data_dir / 'train' / 'observations.parquet'
    test_obs_path = data_dir / 'test' / 'observations.parquet'

    n_regimes, _ = detect_regimes(train_obs_path)
    if n_regimes <= 3:
        print(f"  {n_regimes} regimes — no normalization needed")
        return  # FD001/FD003: single regime, skip

    print(f"  {n_regimes} regimes detected — normalizing per operating condition")

    train_obs = pl.read_parquet(str(train_obs_path))
    test_obs = pl.read_parquet(str(test_obs_path))

    # Pivot to wide for regime detection
    sensor_sigs = sorted([s for s in train_obs['signal_id'].unique().to_list()
                          if s not in ['op1', 'op2', 'op3']])
    op_sigs = ['op1', 'op2', 'op3']

    # Build regime labels using op1 rounding (dominant regime indicator)
    def assign_regime(obs_df):
        """Assign regime label to each (cohort, I) based on op1."""
        op1_df = obs_df.filter(pl.col('signal_id') == 'op1').select(['cohort', 'I', 'value'])
        op1_df = op1_df.with_columns(
            pl.col('value').round(0).cast(pl.Int64).alias('regime')
        )
        return op1_df.select(['cohort', 'I', 'regime'])

    train_regimes = assign_regime(train_obs)
    test_regimes = assign_regime(test_obs)

    # Compute per-regime, per-signal mean/std from TRAINING data
    train_with_regime = train_obs.join(train_regimes, on=['cohort', 'I'], how='left')
    regime_stats = (
        train_with_regime
        .filter(~pl.col('signal_id').is_in(op_sigs))
        .group_by(['regime', 'signal_id'])
        .agg([
            pl.col('value').mean().alias('regime_mean'),
            pl.col('value').std().alias('regime_std'),
        ])
    )
    # Replace zero std with 1
    regime_stats = regime_stats.with_columns(
        pl.when(pl.col('regime_std').abs() < 1e-12)
        .then(1.0)
        .otherwise(pl.col('regime_std'))
        .alias('regime_std')
    )

    def normalize_obs(obs_df, regime_labels):
        """Normalize sensor values by regime-specific mean/std."""
        obs_with_regime = obs_df.join(regime_labels, on=['cohort', 'I'], how='left')
        # Join regime stats
        obs_with_stats = obs_with_regime.join(
            regime_stats, on=['regime', 'signal_id'], how='left'
        )
        # Normalize: (value - regime_mean) / regime_std for sensors only
        normalized = obs_with_stats.with_columns(
            pl.when(pl.col('regime_mean').is_not_null())
            .then((pl.col('value') - pl.col('regime_mean')) / pl.col('regime_std'))
            .otherwise(pl.col('value'))
            .alias('value')
        )
        return normalized.select(['cohort', 'signal_id', 'I', 'value'])

    train_norm = normalize_obs(train_obs, train_regimes)
    test_norm = normalize_obs(test_obs, test_regimes)

    # Overwrite observations with normalized versions
    train_norm.write_parquet(str(train_obs_path))
    test_norm.write_parquet(str(test_obs_path))
    print(f"  Normalized: train={len(train_norm):,}, test={len(test_norm):,}")


def step_1_ingest(data_dir):
    """Convert raw text to observations.parquet if needed."""
    print("\n" + "=" * 60)
    print("  STEP 1: INGEST")
    print("=" * 60)

    train_obs = data_dir / 'train' / 'observations.parquet'
    test_obs = data_dir / 'test' / 'observations.parquet'

    if train_obs.exists() and test_obs.exists():
        train_df = pl.read_parquet(str(train_obs))
        test_df = pl.read_parquet(str(test_obs))
        print(f"  Train observations exist: {len(train_df):,} rows, {train_df['cohort'].n_unique()} engines")
        print(f"  Test observations exist:  {len(test_df):,} rows, {test_df['cohort'].n_unique()} engines")
        return

    run_cmd([PRIME_VENV, PRIME_DIR / 'convert_cmapss.py', '--dir', data_dir])


def step_2_normalize(data_dir):
    """Normalize per operating condition for multi-regime datasets."""
    print("\n" + "=" * 60)
    print("  STEP 2: REGIME NORMALIZATION")
    print("=" * 60)
    normalize_by_regime(data_dir)


def step_3_manifold(data_dir, split):
    """Run full Manifold pipeline on train or test."""
    print("\n" + "=" * 60)
    print(f"  STEP 3: MANIFOLD PIPELINE ({split.upper()})")
    print("=" * 60)

    split_dir = data_dir / split
    output_dir = split_dir / 'output'

    # Check if already complete
    if (output_dir / 'state_geometry.parquet').exists() and (output_dir / 'signal_vector.parquet').exists():
        n_files = len(list(output_dir.glob('*.parquet')))
        print(f"  Output exists: {n_files} parquet files in {output_dir}")
        # Still run if missing key files
        if (output_dir / 'gaussian_fingerprint.parquet').exists():
            print("  Skipping — already complete")
            return

    # Run Manifold: `python -m engines run <split_dir>`
    # This auto-generates typology + manifest if not present.
    # Default output goes to <split_dir>/output/ — which is what we want.
    # Do NOT pass -o flag, or it nests output/output/.
    run_cmd(
        [MANIFOLD_VENV, '-m', 'engines', 'run', str(split_dir)],
        cwd=MANIFOLD_DIR,
        timeout=1800,
    )


def step_4_fleet_baseline(data_dir):
    """Compute fleet baseline from TRAINING data, score both train and test."""
    print("\n" + "=" * 60)
    print("  STEP 4: FLEET BASELINE + OBSERVATION GEOMETRY")
    print("=" * 60)

    train_obs = data_dir / 'train' / 'observations.parquet'
    test_obs = data_dir / 'test' / 'observations.parquet'
    baseline_path = data_dir / 'fleet_baseline.parquet'
    train_geom = data_dir / 'train' / 'output' / 'observation_geometry.parquet'
    test_geom = data_dir / 'test' / 'output' / 'observation_geometry.parquet'

    # Stage 34: Fleet baseline from TRAINING data only
    if not baseline_path.exists():
        print("  Computing fleet baseline from training fleet...")
        run_cmd(
            [MANIFOLD_VENV, '-c', f"""
import sys; sys.path.insert(0, '.')
from engines.entry_points.stage_34_cohort_baseline import run
run('{train_obs}', '{baseline_path}', mode='fleet')
"""],
            cwd=MANIFOLD_DIR,
        )
    else:
        print(f"  Fleet baseline exists: {baseline_path}")

    # Stage 35: Score train against fleet baseline
    if not train_geom.exists():
        print("  Scoring training data...")
        train_geom.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [MANIFOLD_VENV, '-c', f"""
import sys; sys.path.insert(0, '.')
from engines.entry_points.stage_35_observation_geometry import run
run('{train_obs}', '{baseline_path}', '{train_geom}')
"""],
            cwd=MANIFOLD_DIR,
        )
    else:
        print(f"  Train observation geometry exists: {train_geom}")

    # Stage 35: Score test against SAME fleet baseline
    if not test_geom.exists():
        print("  Scoring test data...")
        test_geom.parent.mkdir(parents=True, exist_ok=True)
        run_cmd(
            [MANIFOLD_VENV, '-c', f"""
import sys; sys.path.insert(0, '.')
from engines.entry_points.stage_35_observation_geometry import run
run('{test_obs}', '{baseline_path}', '{test_geom}')
"""],
            cwd=MANIFOLD_DIR,
        )
    else:
        print(f"  Test observation geometry exists: {test_geom}")


def step_5_features(data_dir):
    """Build cycle-level feature matrix for train and test."""
    print("\n" + "=" * 60)
    print("  STEP 5: BUILD CYCLE FEATURES")
    print("=" * 60)

    train_feats = data_dir / 'train' / 'cycle_features.parquet'
    test_feats = data_dir / 'test' / 'cycle_features.parquet'

    for split, feats_path in [('train', train_feats), ('test', test_feats)]:
        if feats_path.exists():
            df = pl.read_parquet(str(feats_path))
            print(f"  {split} features exist: {df.shape}")
            continue

        obs_path = data_dir / split / 'observations.parquet'
        manifold_dir = data_dir / split / 'output'

        run_cmd([
            PRIME_VENV,
            PRIME_DIR / 'build_cycle_features.py', 'build',
            '--obs', str(obs_path),
            '--manifold', str(manifold_dir),
            '--output', str(feats_path),
        ])


def step_6_train_evaluate(data_dir):
    """Train clean ensemble and evaluate on test set."""
    print("\n" + "=" * 60)
    print("  STEP 6: TRAIN + EVALUATE")
    print("=" * 60)

    train_df = pl.read_parquet(str(data_dir / 'train' / 'cycle_features.parquet'))
    test_df = pl.read_parquet(str(data_dir / 'test' / 'cycle_features.parquet'))

    # Load ground truth RUL
    rul_files = list(data_dir.glob('RUL_*.txt'))
    if not rul_files:
        print("  No RUL file found — skipping evaluation")
        return None
    with open(rul_files[0]) as f:
        y_true = np.array([int(line.strip()) for line in f if line.strip()], dtype=float)
    y_true_capped = np.minimum(y_true, MAX_RUL_CAP)

    # Feature selection: ONLY generalizing features
    meta_cols = {'cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct'}
    leak_prefixes = ('geom_', 'gfp_', 'gsim_')

    feat_cols = sorted([c for c in train_df.columns
                        if c not in meta_cols
                        and not any(c.startswith(p) for p in leak_prefixes)])

    print(f"  Train: {train_df.shape[0]:,} rows")
    print(f"  Test:  {test_df.shape[0]:,} rows")
    print(f"  Features (clean): {len(feat_cols)}")
    print(f"  Excluded: {sum(1 for c in train_df.columns if any(c.startswith(p) for p in leak_prefixes))} leaking features")

    # Ensure test has all feature columns
    for c in feat_cols:
        if c not in test_df.columns:
            test_df = test_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

    X_train = train_df.select(feat_cols).to_numpy().astype(np.float64)
    y_train = train_df['RUL'].to_numpy().astype(np.float64)
    groups_arr = train_df['cohort'].to_numpy()

    # Test: last cycle per engine
    test_last = (test_df
        .with_columns(pl.col('cohort').map_elements(cohort_sort_key, return_dtype=pl.Int64).alias('_sort'))
        .group_by('cohort')
        .agg([pl.all().sort_by('I').last()])
        .sort('_sort')
        .drop('_sort')
    )
    X_test = test_last.select(feat_cols).to_numpy().astype(np.float64)

    n_test_engines = len(X_test)
    if n_test_engines != len(y_true):
        print(f"  WARNING: {n_test_engines} test engines vs {len(y_true)} RUL values")
        y_true = y_true[:n_test_engines]
        y_true_capped = y_true_capped[:n_test_engines]

    # Impute + scale
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    X_train = np.where(np.isinf(X_train), 0, X_train)
    X_test = np.where(np.isinf(X_test), 0, X_test)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Base learners
    base_models = {}
    if HAS_LGB:
        base_models['lgb'] = lgb.LGBMRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1, n_jobs=-1)
        base_models['xgb'] = xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=10, random_state=42, verbosity=0, n_jobs=-1)
    base_models['hist'] = HistGradientBoostingRegressor(
        max_iter=500, max_depth=6, learning_rate=0.05,
        min_samples_leaf=10, random_state=42)

    # 5-fold GroupKFold stacking
    gkf = GroupKFold(n_splits=5)
    oof_preds = {n: np.zeros(len(y_train)) for n in base_models}
    test_preds = {n: np.zeros(len(X_test_s)) for n in base_models}

    print(f"\n  5-fold GroupKFold stacking ({len(base_models)} models)...")
    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_s, y_train, groups_arr)):
        X_tr, X_val = X_train_s[train_idx], X_train_s[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        for name, model in base_models.items():
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = m.predict(X_val)
            test_preds[name] += m.predict(X_test_s) / 5
        oof_scores = {n: np.sqrt(mean_squared_error(y_train[val_idx], oof_preds[n][val_idx])) for n in base_models}
        print(f"    Fold {fold_idx+1}: " + ", ".join(f"{n}={v:.2f}" for n, v in oof_scores.items()))

    # OOF scores
    print(f"\n  OOF Scores:")
    for name in base_models:
        r = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
        print(f"    {name}: RMSE={r:.2f}")

    # Meta-learner
    stack_train = np.column_stack([oof_preds[n] for n in base_models])
    stack_test = np.column_stack([test_preds[n] for n in base_models])
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta.fit(stack_train, y_train)
    pred = np.clip(meta.predict(stack_test), 0, MAX_RUL_CAP)

    rmse = np.sqrt(mean_squared_error(y_true_capped, pred))
    mae = mean_absolute_error(y_true_capped, pred)
    r2 = r2_score(y_true_capped, pred)
    score = phm08_score(y_true_capped, pred)

    oof_avg = np.mean([np.sqrt(mean_squared_error(y_train, oof_preds[n])) for n in base_models])

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {data_dir.name}")
    print(f"{'=' * 60}")
    print(f"  Features:    {len(feat_cols)}")
    print(f"  OOF RMSE:    {oof_avg:.2f}")
    print(f"  Test RMSE:   {rmse:.2f}  (gap: {rmse - oof_avg:+.1f})")
    print(f"  Test MAE:    {mae:.2f}")
    print(f"  Test R²:     {r2:.4f}")
    print(f"  PHM08 Score: {score:,.0f}")
    print(f"  Meta alpha:  {meta.alpha_:.2f}")
    print(f"  Meta weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(base_models.keys(), meta.coef_)))

    # Base learner individual scores
    print(f"\n  Base learner test scores:")
    for name in base_models:
        p = np.clip(test_preds[name], 0, MAX_RUL_CAP)
        r = np.sqrt(mean_squared_error(y_true_capped, p))
        s = phm08_score(y_true_capped, p)
        print(f"    {name}: RMSE={r:.2f}, PHM08={s:,.0f}")

    return {
        'dataset': data_dir.name,
        'features': len(feat_cols),
        'oof_rmse': oof_avg,
        'test_rmse': rmse,
        'mae': mae,
        'r2': r2,
        'phm08': score,
        'gap': rmse - oof_avg,
    }


def run_dataset(data_dir):
    """Run full pipeline on one dataset."""
    data_dir = Path(data_dir).expanduser().resolve()
    print("\n" + "#" * 60)
    print(f"  DATASET: {data_dir.name}")
    print("#" * 60)

    t0 = time.time()

    step_1_ingest(data_dir)
    step_2_normalize(data_dir)
    step_3_manifold(data_dir, 'train')
    step_3_manifold(data_dir, 'test')
    step_4_fleet_baseline(data_dir)
    step_5_features(data_dir)
    result = step_6_train_evaluate(data_dir)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} minutes")

    return result


def main():
    parser = argparse.ArgumentParser(description='Run cycle-level ML pipeline')
    parser.add_argument('data_dir', nargs='?', help='Dataset directory (e.g., ~/data/FD_001/)')
    parser.add_argument('--all', action='store_true', help='Run all FD001-FD004')
    parser.add_argument('--skip-manifold', action='store_true', help='Skip Manifold pipeline (use existing output)')
    args = parser.parse_args()

    if args.all:
        data_root = Path.home() / 'data'
        datasets = sorted([d for d in data_root.iterdir()
                          if d.is_dir() and d.name.startswith('FD_')])
    elif args.data_dir:
        datasets = [Path(args.data_dir).expanduser()]
    else:
        parser.error("Provide a dataset directory or --all")

    results = []
    for ds in datasets:
        result = run_dataset(ds)
        if result:
            results.append(result)

    if len(results) > 1:
        print("\n" + "=" * 70)
        print("  SUMMARY")
        print("=" * 70)
        print(f"  {'Dataset':<12s} {'Features':>8s} {'OOF':>8s} {'Test':>8s} {'Gap':>6s} {'PHM08':>8s} {'R²':>7s}")
        print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*7}")
        for r in results:
            print(f"  {r['dataset']:<12s} {r['features']:>8d} {r['oof_rmse']:>8.2f} {r['test_rmse']:>8.2f} {r['gap']:>+6.1f} {r['phm08']:>8,.0f} {r['r2']:>7.4f}")


if __name__ == '__main__':
    main()
