#!/usr/bin/env python3
"""
XGBoost Comparison: With vs Without ORTHON Features
====================================================

Compares predictive performance for RUL (Remaining Useful Life) prediction:
1. Baseline: Standard trajectory features only
2. Enhanced: Standard + ORTHON physics features

Uses CMAPSS FD001 turbofan dataset.
Ensures no data leakage between train/test.
"""

import polars as pl
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Try to import XGBoost
try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("XGBoost not installed. Using sklearn GradientBoosting instead.")
    from sklearn.ensemble import GradientBoostingRegressor

# Paths
CMAPSS_DIR = Path('/Users/jasonrudder/prism-mac copy/data/CMAPSSData')
ML_FEATURES_DIR = Path('/Users/jasonrudder/Domains/domains/cmapss/notebooks/ml_accelerator')
ORTHON_DIR = Path('/Users/jasonrudder/prism_engines-prism/data/cmapss')


def load_train_data():
    """Load training features and targets."""
    # Standard trajectory features
    traj = pl.read_parquet(ML_FEATURES_DIR / 'train_trajectory_features.parquet')

    # ORTHON physics features
    orthon = pl.read_parquet(ORTHON_DIR / 'ml_features_dense.parquet')

    # Ground truth: lifespan from train data
    train_raw = pl.read_parquet(CMAPSS_DIR / 'train_FD001.parquet')
    lifespan = train_raw.group_by('unit').agg(pl.col('cycle').max().alias('total_life'))

    return traj, orthon, lifespan


def load_test_data():
    """Load test features and RUL ground truth."""
    # Standard trajectory features for test
    traj = pl.read_parquet(ML_FEATURES_DIR / 'test_trajectory_features.parquet')

    # RUL ground truth (already has 'unit' column)
    rul = pl.read_parquet(CMAPSS_DIR / 'RUL_FD001.parquet')

    # Test data cycles
    test_raw = pl.read_parquet(CMAPSS_DIR / 'test_FD001.parquet')
    test_cycles = test_raw.group_by('unit').agg(pl.col('cycle').max().alias('observed_cycles'))

    # Compute total life for test (for consistent target)
    test_life = test_cycles.join(rul, on='unit')
    test_life = test_life.with_columns(
        (pl.col('observed_cycles') + pl.col('RUL')).alias('total_life')
    )

    return traj, rul, test_life


def prepare_baseline_features(traj_df):
    """Select baseline (non-ORTHON) features."""
    # Get numeric columns only
    feature_cols = [c for c in traj_df.columns
                    if c not in ['unit', 'n_cycles']
                    and traj_df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    # Fill nulls with 0
    df = traj_df.select(['unit'] + feature_cols)
    for col in feature_cols:
        df = df.with_columns(pl.col(col).fill_null(0))

    return df, feature_cols


def prepare_orthon_features(orthon_df):
    """Select ORTHON features for joining."""
    # Key ORTHON features (exclude entity_id and string columns)
    orthon_cols = [c for c in orthon_df.columns
                   if c not in ['entity_id', 'risk_level']
                   and orthon_df[c].dtype in [pl.Float64, pl.Float32, pl.Int64, pl.Int32]]

    # Rename entity_id to unit for joining
    df = orthon_df.select(['entity_id'] + orthon_cols)
    df = df.with_columns(pl.col('entity_id').cast(pl.Int64).alias('unit'))
    df = df.drop('entity_id')

    # Prefix ORTHON columns
    for col in orthon_cols:
        df = df.rename({col: f'orthon_{col}'})

    orthon_feature_cols = [f'orthon_{c}' for c in orthon_cols]

    return df, orthon_feature_cols


def train_model(X_train, y_train, X_val, y_val, model_name='model'):
    """Train XGBoost (or fallback) model."""
    if HAS_XGBOOST:
        model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    else:
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )

    model.fit(X_train, y_train)

    # Validation metrics
    y_pred_val = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    val_mae = mean_absolute_error(y_val, y_pred_val)
    val_r2 = r2_score(y_val, y_pred_val)

    print(f"\n{model_name} Validation Metrics:")
    print(f"  RMSE: {val_rmse:.2f}")
    print(f"  MAE:  {val_mae:.2f}")
    print(f"  R2:   {val_r2:.3f}")

    return model, {'rmse': val_rmse, 'mae': val_mae, 'r2': val_r2}


def main():
    print("=" * 70)
    print("XGBoost COMPARISON: With vs Without ORTHON Features")
    print("=" * 70)
    print(f"\nUsing: {'XGBoost' if HAS_XGBOOST else 'sklearn GradientBoosting'}")

    # Load data
    print("\n[1] Loading data...")
    traj_train, orthon_train, lifespan_train = load_train_data()
    traj_test, rul_test, test_life = load_test_data()

    print(f"  Train units: {len(lifespan_train)}")
    print(f"  Test units:  {len(rul_test)}")

    # Prepare baseline features
    print("\n[2] Preparing features...")
    baseline_df, baseline_cols = prepare_baseline_features(traj_train)
    print(f"  Baseline features: {len(baseline_cols)}")

    # Prepare ORTHON features
    orthon_df, orthon_cols = prepare_orthon_features(orthon_train)
    print(f"  ORTHON features:   {len(orthon_cols)}")

    # Join baseline with lifespan (target)
    train_baseline = baseline_df.join(lifespan_train, on='unit')

    # Join with ORTHON
    train_enhanced = train_baseline.join(orthon_df, on='unit', how='left')

    # Fill any remaining nulls
    for col in orthon_cols:
        train_enhanced = train_enhanced.with_columns(pl.col(col).fill_null(0))

    # Prepare numpy arrays
    X_baseline = train_baseline.select(baseline_cols).to_numpy()
    X_enhanced = train_enhanced.select(baseline_cols + orthon_cols).to_numpy()
    y = train_baseline['total_life'].to_numpy()

    print(f"\n  X_baseline shape: {X_baseline.shape}")
    print(f"  X_enhanced shape: {X_enhanced.shape}")
    print(f"  y shape: {y.shape}")

    # Train/validation split (no leakage - within train only)
    print("\n[3] Creating train/validation split (80/20)...")
    X_base_train, X_base_val, y_train, y_val, idx_train, idx_val = train_test_split(
        X_baseline, y, np.arange(len(y)), test_size=0.2, random_state=42
    )
    X_enh_train = X_enhanced[idx_train]
    X_enh_val = X_enhanced[idx_val]

    print(f"  Train size: {len(y_train)}")
    print(f"  Val size:   {len(y_val)}")

    # Train baseline model
    print("\n[4] Training BASELINE model (no ORTHON features)...")
    model_baseline, metrics_baseline = train_model(
        X_base_train, y_train, X_base_val, y_val,
        model_name="BASELINE"
    )

    # Train enhanced model
    print("\n[5] Training ENHANCED model (with ORTHON features)...")
    model_enhanced, metrics_enhanced = train_model(
        X_enh_train, y_train, X_enh_val, y_val,
        model_name="ENHANCED (+ ORTHON)"
    )

    # Comparison
    print("\n" + "=" * 70)
    print("COMPARISON: ORTHON UPLIFT")
    print("=" * 70)

    rmse_improvement = metrics_baseline['rmse'] - metrics_enhanced['rmse']
    rmse_pct = (rmse_improvement / metrics_baseline['rmse']) * 100

    mae_improvement = metrics_baseline['mae'] - metrics_enhanced['mae']
    mae_pct = (mae_improvement / metrics_baseline['mae']) * 100

    r2_improvement = metrics_enhanced['r2'] - metrics_baseline['r2']

    print(f"\nMetric          Baseline    Enhanced    Improvement")
    print(f"-" * 55)
    print(f"RMSE            {metrics_baseline['rmse']:8.2f}    {metrics_enhanced['rmse']:8.2f}    {rmse_improvement:+.2f} ({rmse_pct:+.1f}%)")
    print(f"MAE             {metrics_baseline['mae']:8.2f}    {metrics_enhanced['mae']:8.2f}    {mae_improvement:+.2f} ({mae_pct:+.1f}%)")
    print(f"R2              {metrics_baseline['r2']:8.3f}    {metrics_enhanced['r2']:8.3f}    {r2_improvement:+.3f}")

    # Feature importance for enhanced model
    if HAS_XGBOOST:
        print("\n[6] Top 10 Most Important Features (Enhanced Model):")
        importance = model_enhanced.feature_importances_
        all_cols = baseline_cols + orthon_cols
        feat_imp = sorted(zip(all_cols, importance), key=lambda x: x[1], reverse=True)

        print(f"\n{'Feature':<45} {'Importance':>12} {'Type':>10}")
        print("-" * 70)
        for feat, imp in feat_imp[:10]:
            feat_type = "ORTHON" if feat.startswith('orthon_') else "baseline"
            print(f"{feat:<45} {imp:>12.4f} {feat_type:>10}")

        # Count ORTHON features in top 20
        orthon_in_top20 = sum(1 for f, _ in feat_imp[:20] if f.startswith('orthon_'))
        print(f"\nORTHON features in top 20: {orthon_in_top20}/20")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    if rmse_improvement > 0:
        print(f"\nORTHON features IMPROVED prediction by {rmse_pct:.1f}% (RMSE)")
        print("The physics-based features capture degradation patterns")
        print("that complement standard statistical features.")
    else:
        print(f"\nORTHON features did not improve prediction ({rmse_pct:.1f}%)")
        print("This may indicate the standard features already capture")
        print("the relevant patterns for this dataset.")

    return {
        'baseline': metrics_baseline,
        'enhanced': metrics_enhanced,
        'improvement': {
            'rmse': rmse_improvement,
            'rmse_pct': rmse_pct,
            'mae': mae_improvement,
            'mae_pct': mae_pct,
            'r2': r2_improvement
        }
    }


if __name__ == '__main__':
    results = main()
