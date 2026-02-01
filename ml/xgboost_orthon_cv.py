#!/usr/bin/env python3
"""
XGBoost Cross-Validation: With vs Without ORTHON Features
==========================================================

More rigorous comparison using K-fold cross-validation.
Ensures results aren't due to lucky split variance.
"""

import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# Paths
CMAPSS_DIR = Path('/Users/jasonrudder/prism-mac copy/data/CMAPSSData')
ML_FEATURES_DIR = Path('/Users/jasonrudder/Domains/domains/cmapss/notebooks/ml_accelerator')
ORTHON_DIR = Path('/Users/jasonrudder/prism_engines-prism/data/cmapss')


def load_data():
    """Load and prepare all features."""
    # Standard trajectory features
    traj = pl.read_parquet(ML_FEATURES_DIR / 'train_trajectory_features.parquet')

    # ORTHON physics features
    orthon = pl.read_parquet(ORTHON_DIR / 'ml_features_dense.parquet')

    # Ground truth: lifespan
    train_raw = pl.read_parquet(CMAPSS_DIR / 'train_FD001.parquet')
    lifespan = train_raw.group_by('unit').agg(pl.col('cycle').max().alias('total_life'))

    # Prepare baseline features
    baseline_cols = [c for c in traj.columns
                     if c not in ['unit', 'n_cycles']
                     and traj[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    baseline_df = traj.select(['unit'] + baseline_cols)
    for col in baseline_cols:
        baseline_df = baseline_df.with_columns(pl.col(col).fill_null(0))

    # Prepare ORTHON features
    orthon_cols = [c for c in orthon.columns
                   if c not in ['entity_id', 'risk_level']
                   and orthon[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]
    orthon_df = orthon.select(['entity_id'] + orthon_cols)
    orthon_df = orthon_df.with_columns(pl.col('entity_id').cast(pl.Int64).alias('unit'))
    orthon_df = orthon_df.drop('entity_id')

    for col in orthon_cols:
        orthon_df = orthon_df.rename({col: f'orthon_{col}'})
    orthon_feature_cols = [f'orthon_{c}' for c in orthon_cols]

    # Join everything
    df = baseline_df.join(lifespan, on='unit')
    df = df.join(orthon_df, on='unit', how='left')
    for col in orthon_feature_cols:
        df = df.with_columns(pl.col(col).fill_null(0))

    return df, baseline_cols, orthon_feature_cols


def run_cv(X, y, n_splits=5, model_name='model'):
    """Run K-fold cross-validation."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    rmse_scores = []
    mae_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)

        rmse_scores.append(rmse)
        mae_scores.append(mae)

    return {
        'rmse_mean': np.mean(rmse_scores),
        'rmse_std': np.std(rmse_scores),
        'mae_mean': np.mean(mae_scores),
        'mae_std': np.std(mae_scores),
        'rmse_scores': rmse_scores,
        'mae_scores': mae_scores
    }


def main():
    print("=" * 70)
    print("XGBoost 5-FOLD CROSS-VALIDATION: With vs Without ORTHON")
    print("=" * 70)

    # Load data
    print("\n[1] Loading data...")
    df, baseline_cols, orthon_cols = load_data()

    X_baseline = df.select(baseline_cols).to_numpy()
    X_enhanced = df.select(baseline_cols + orthon_cols).to_numpy()
    y = df['total_life'].to_numpy()

    print(f"  Baseline features: {len(baseline_cols)}")
    print(f"  ORTHON features:   {len(orthon_cols)}")
    print(f"  Total samples:     {len(y)}")

    # Run CV for baseline
    print("\n[2] Running 5-fold CV on BASELINE model...")
    baseline_results = run_cv(X_baseline, y, n_splits=5, model_name='BASELINE')

    print(f"  RMSE: {baseline_results['rmse_mean']:.2f} ± {baseline_results['rmse_std']:.2f}")
    print(f"  MAE:  {baseline_results['mae_mean']:.2f} ± {baseline_results['mae_std']:.2f}")
    print(f"  Per-fold RMSE: {[f'{x:.2f}' for x in baseline_results['rmse_scores']]}")

    # Run CV for enhanced
    print("\n[3] Running 5-fold CV on ENHANCED (+ ORTHON) model...")
    enhanced_results = run_cv(X_enhanced, y, n_splits=5, model_name='ENHANCED')

    print(f"  RMSE: {enhanced_results['rmse_mean']:.2f} ± {enhanced_results['rmse_std']:.2f}")
    print(f"  MAE:  {enhanced_results['mae_mean']:.2f} ± {enhanced_results['mae_std']:.2f}")
    print(f"  Per-fold RMSE: {[f'{x:.2f}' for x in enhanced_results['rmse_scores']]}")

    # Comparison
    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPARISON")
    print("=" * 70)

    rmse_improvement = baseline_results['rmse_mean'] - enhanced_results['rmse_mean']
    rmse_pct = (rmse_improvement / baseline_results['rmse_mean']) * 100

    mae_improvement = baseline_results['mae_mean'] - enhanced_results['mae_mean']
    mae_pct = (mae_improvement / baseline_results['mae_mean']) * 100

    print(f"\n{'Metric':<10} {'Baseline':>15} {'Enhanced':>15} {'Improvement':>15}")
    print("-" * 60)
    print(f"{'RMSE':<10} {baseline_results['rmse_mean']:>8.2f} ± {baseline_results['rmse_std']:.2f}   {enhanced_results['rmse_mean']:>8.2f} ± {enhanced_results['rmse_std']:.2f}   {rmse_improvement:+.2f} ({rmse_pct:+.1f}%)")
    print(f"{'MAE':<10} {baseline_results['mae_mean']:>8.2f} ± {baseline_results['mae_std']:.2f}   {enhanced_results['mae_mean']:>8.2f} ± {enhanced_results['mae_std']:.2f}   {mae_improvement:+.2f} ({mae_pct:+.1f}%)")

    # Fold-by-fold comparison
    print("\n" + "-" * 60)
    print("Per-Fold Comparison:")
    print(f"{'Fold':<6} {'Baseline RMSE':>15} {'Enhanced RMSE':>15} {'Winner':>12}")
    print("-" * 60)

    baseline_wins = 0
    enhanced_wins = 0
    for i in range(5):
        b = baseline_results['rmse_scores'][i]
        e = enhanced_results['rmse_scores'][i]
        winner = "ENHANCED" if e < b else "BASELINE"
        if e < b:
            enhanced_wins += 1
        else:
            baseline_wins += 1
        print(f"{i+1:<6} {b:>15.2f} {e:>15.2f} {winner:>12}")

    print("-" * 60)
    print(f"ENHANCED wins: {enhanced_wins}/5 folds")

    # Statistical significance (paired t-test)
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(
        baseline_results['rmse_scores'],
        enhanced_results['rmse_scores']
    )

    print(f"\nPaired t-test: t={t_stat:.3f}, p={p_value:.4f}")
    if p_value < 0.05:
        print("  → Result is STATISTICALLY SIGNIFICANT (p < 0.05)")
    else:
        print("  → Result is NOT statistically significant (p >= 0.05)")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if rmse_improvement > 0 and enhanced_wins >= 3:
        print(f"\n✓ ORTHON features provide CONSISTENT improvement")
        print(f"  - Average RMSE reduction: {rmse_pct:.1f}%")
        print(f"  - Won {enhanced_wins}/5 folds")
        if p_value < 0.05:
            print(f"  - Statistically significant (p={p_value:.4f})")
    else:
        print(f"\n? ORTHON features show MIXED results")
        print(f"  - Average RMSE change: {rmse_pct:+.1f}%")
        print(f"  - Won {enhanced_wins}/5 folds")

    return baseline_results, enhanced_results


if __name__ == '__main__':
    main()
