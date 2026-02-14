#!/usr/bin/env python3
"""
Geometry Ladder Experiments
===========================
4 experiments building from simplest to full ensemble.

Exp 1: RUL = β₀ + β₁·centroid_distance                         → simplest possible
Exp 2: RUL = β₀ + β₁·centroid_distance + β₂·cycle              → add time context
Exp 3: RUL = β₀ + β₁·cd + β₂·cycle + β₃·pc1 + β₄·pc2          → add drift direction
Exp 4: Full ensemble with all features                           → chase the number
"""

import polars as pl
import numpy as np
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# Try optional imports
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import catboost as cb
    HAS_CB = True
except ImportError:
    HAS_CB = False


def load_train_data():
    """Load train observation_geometry + compute RUL."""
    og = pl.read_parquet('/Users/jasonrudder/data/FD001/train/output/observation_geometry.parquet')
    obs = pl.read_parquet('/Users/jasonrudder/data/FD001/train/observations.parquet')

    first_sig = obs['signal_id'].unique().sort()[0]
    lifecycle = (obs.filter(pl.col('signal_id') == first_sig)
        .group_by('cohort').agg(pl.col('I').max().alias('max_I')))

    df = og.join(lifecycle, on='cohort', how='left')
    df = df.with_columns([
        (pl.col('max_I') - pl.col('I')).alias('RUL'),
        pl.col('I').alias('cycle'),
    ])
    # Cap RUL at 125
    df = df.with_columns(
        pl.when(pl.col('RUL') > 125).then(125).otherwise(pl.col('RUL')).alias('RUL')
    )
    return df


def load_test_data():
    """Load test observation_geometry + ground truth RUL."""
    og = pl.read_parquet('/Users/jasonrudder/data/FD001/test/output/observation_geometry.parquet')

    # Get last cycle per engine
    last_cycle = og.group_by('cohort').agg(pl.col('I').max().alias('last_I'))
    og_last = og.join(last_cycle, on='cohort', how='inner')
    og_last = og_last.filter(pl.col('I') == pl.col('last_I'))
    og_last = og_last.with_columns(pl.col('I').alias('cycle'))

    # Sort by engine number
    og_last = og_last.with_columns(
        pl.col('cohort').str.extract(r'(\d+)').cast(pl.Int64).alias('engine_num')
    ).sort('engine_num')

    # Load ground truth
    with open('/Users/jasonrudder/data/FD001/RUL_FD001.txt') as f:
        true_ruls = [int(x.strip()) for x in f.readlines() if x.strip()]

    # Cap at 125 for fair comparison
    true_ruls_capped = [min(r, 125) for r in true_ruls]

    return og_last, true_ruls, true_ruls_capped


def phm08_score(y_true, y_pred):
    """PHM08 asymmetric scoring function."""
    d = np.array(y_pred) - np.array(y_true)
    score = 0.0
    for di in d:
        if di < 0:  # early prediction
            score += np.exp(-di / 13.0) - 1
        else:  # late prediction
            score += np.exp(di / 10.0) - 1
    return score


def report(name, y_true, y_pred, y_true_uncapped=None):
    """Print metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    score = phm08_score(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  RMSE:        {rmse:.2f}")
    print(f"  MAE:         {mae:.2f}")
    print(f"  R²:          {r2:.4f}")
    print(f"  PHM08 Score: {score:,.0f}")

    if y_true_uncapped is not None:
        rmse_uc = np.sqrt(mean_squared_error(y_true_uncapped, y_pred))
        print(f"  RMSE (uncapped): {rmse_uc:.2f}")

    # Show worst predictions
    errors = np.array(y_pred) - np.array(y_true)
    worst_idx = np.argsort(np.abs(errors))[-5:][::-1]
    print(f"\n  Worst 5 predictions:")
    for i in worst_idx:
        print(f"    Engine {i+1}: pred={y_pred[i]:.1f}, true={y_true[i]}, error={errors[i]:+.1f}")

    return rmse


def main():
    print("Loading data...")
    train = load_train_data()
    test_last, true_ruls, true_ruls_capped = load_test_data()

    print(f"Train: {train.shape[0]} rows, {train['cohort'].n_unique()} engines")
    print(f"Test:  {test_last.shape[0]} engines")

    results = {}

    # ── Experiment 1: centroid_distance only ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: RUL ~ centroid_distance")
    print("=" * 60)

    X_train_1 = train.select('centroid_distance').to_numpy()
    y_train = train['RUL'].to_numpy().astype(float)
    X_test_1 = test_last.select('centroid_distance').to_numpy()

    model1 = Ridge(alpha=1.0)
    model1.fit(X_train_1, y_train)
    pred1 = model1.predict(X_test_1)
    pred1 = np.clip(pred1, 0, 125)

    print(f"  Coefficients: intercept={model1.intercept_:.2f}, centroid_distance={model1.coef_[0]:.4f}")
    results['Exp1'] = report("Exp 1: RUL ~ centroid_distance", true_ruls_capped, pred1, true_ruls)

    # ── Experiment 2: centroid_distance + cycle ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: RUL ~ centroid_distance + cycle")
    print("=" * 60)

    X_train_2 = train.select('centroid_distance', 'cycle').to_numpy()
    X_test_2 = test_last.select('centroid_distance', 'cycle').to_numpy()

    model2 = Ridge(alpha=1.0)
    model2.fit(X_train_2, y_train)
    pred2 = model2.predict(X_test_2)
    pred2 = np.clip(pred2, 0, 125)

    print(f"  Coefficients: intercept={model2.intercept_:.2f}, cd={model2.coef_[0]:.4f}, cycle={model2.coef_[1]:.4f}")
    results['Exp2'] = report("Exp 2: RUL ~ centroid_distance + cycle", true_ruls_capped, pred2, true_ruls)

    # ── Experiment 3: cd + cycle + pc1 + pc2 ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: RUL ~ cd + cycle + pc1 + pc2")
    print("=" * 60)

    feat3 = ['centroid_distance', 'cycle', 'pc1_projection', 'pc2_projection']
    X_train_3 = train.select(feat3).to_numpy()
    X_test_3 = test_last.select(feat3).to_numpy()

    model3 = Ridge(alpha=1.0)
    model3.fit(X_train_3, y_train)
    pred3 = model3.predict(X_test_3)
    pred3 = np.clip(pred3, 0, 125)

    coefs3 = dict(zip(feat3, model3.coef_))
    print(f"  Coefficients: intercept={model3.intercept_:.2f}")
    for k, v in coefs3.items():
        print(f"    {k}: {v:.4f}")
    results['Exp3'] = report("Exp 3: RUL ~ cd + cycle + pc1 + pc2", true_ruls_capped, pred3, true_ruls)

    # ── Experiment 3b: cd + cycle + pc1 + pc2 + mahalanobis (all geometry) ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3b: RUL ~ all geometry features")
    print("=" * 60)

    feat3b = ['centroid_distance', 'centroid_distance_norm', 'cycle',
              'pc1_projection', 'pc2_projection', 'mahalanobis_approx', 'sensor_norm']
    X_train_3b = train.select(feat3b).to_numpy()
    X_test_3b = test_last.select(feat3b).to_numpy()

    scaler_3b = StandardScaler()
    X_train_3b_s = scaler_3b.fit_transform(X_train_3b)
    X_test_3b_s = scaler_3b.transform(X_test_3b)

    model3b = Ridge(alpha=1.0)
    model3b.fit(X_train_3b_s, y_train)
    pred3b = model3b.predict(X_test_3b_s)
    pred3b = np.clip(pred3b, 0, 125)

    coefs3b = dict(zip(feat3b, model3b.coef_))
    print(f"  Coefficients (standardized): intercept={model3b.intercept_:.2f}")
    for k, v in sorted(coefs3b.items(), key=lambda x: abs(x[1]), reverse=True):
        print(f"    {k}: {v:.4f}")
    results['Exp3b'] = report("Exp 3b: All geometry features", true_ruls_capped, pred3b, true_ruls)

    # ── Experiment 4: Full ensemble with GBR ──
    print("\n" + "=" * 60)
    print("  EXPERIMENT 4: Full ensemble (all geometry → GBR)")
    print("=" * 60)

    # Also train on ALL test cycle data (not just last), but predict on last
    feat4 = ['centroid_distance', 'centroid_distance_norm', 'cycle',
             'pc1_projection', 'pc2_projection', 'mahalanobis_approx', 'sensor_norm']

    X_train_4 = train.select(feat4).to_numpy()
    X_test_4 = test_last.select(feat4).to_numpy()

    # Handle any NaN
    X_train_4 = np.nan_to_num(X_train_4, nan=0.0)
    X_test_4 = np.nan_to_num(X_test_4, nan=0.0)

    # GBR
    gbr = GradientBoostingRegressor(
        n_estimators=500, max_depth=5, learning_rate=0.05,
        subsample=0.8, min_samples_leaf=10, random_state=42
    )
    gbr.fit(X_train_4, y_train)
    pred_gbr = gbr.predict(X_test_4)
    pred_gbr = np.clip(pred_gbr, 0, 125)

    results['Exp4_GBR'] = report("Exp 4a: GBR (all geometry)", true_ruls_capped, pred_gbr, true_ruls)

    # Feature importance
    imp = dict(zip(feat4, gbr.feature_importances_))
    print(f"\n  Feature importance (GBR):")
    for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True):
        print(f"    {k}: {v:.4f}")

    # LightGBM if available
    if HAS_LGB:
        lgb_model = lgb.LGBMRegressor(
            n_estimators=500, max_depth=5, learning_rate=0.05,
            subsample=0.8, min_child_samples=10, random_state=42,
            verbose=-1
        )
        lgb_model.fit(X_train_4, y_train)
        pred_lgb = lgb_model.predict(X_test_4)
        pred_lgb = np.clip(pred_lgb, 0, 125)
        results['Exp4_LGB'] = report("Exp 4b: LightGBM (all geometry)", true_ruls_capped, pred_lgb, true_ruls)

    # CatBoost if available
    if HAS_CB:
        cb_model = cb.CatBoostRegressor(
            iterations=500, depth=5, learning_rate=0.05,
            subsample=0.8, min_data_in_leaf=10, random_seed=42,
            verbose=0
        )
        cb_model.fit(X_train_4, y_train)
        pred_cb = cb_model.predict(X_test_4)
        pred_cb = np.clip(pred_cb, 0, 125)
        results['Exp4_CB'] = report("Exp 4c: CatBoost (all geometry)", true_ruls_capped, pred_cb, true_ruls)

    # Stacking ensemble
    if HAS_LGB and HAS_CB:
        print("\n" + "=" * 60)
        print("  EXPERIMENT 4d: Stacking Ensemble (GBR + LGB + CB → Ridge)")
        print("=" * 60)

        # Stack predictions
        stack_train = np.column_stack([
            gbr.predict(X_train_4),
            lgb_model.predict(X_train_4),
            cb_model.predict(X_train_4),
        ])
        stack_test = np.column_stack([pred_gbr, pred_lgb, pred_cb])

        meta = Ridge(alpha=1.0)
        meta.fit(stack_train, y_train)
        pred_stack = meta.predict(stack_test)
        pred_stack = np.clip(pred_stack, 0, 125)

        print(f"  Meta weights: GBR={meta.coef_[0]:.3f}, LGB={meta.coef_[1]:.3f}, CB={meta.coef_[2]:.3f}")
        results['Exp4_Stack'] = report("Exp 4d: Stacking Ensemble", true_ruls_capped, pred_stack, true_ruls)

    # ── Summary ──
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  {'Experiment':<30s} {'RMSE':>8s}")
    print(f"  {'-'*30} {'-'*8}")
    for name, rmse in results.items():
        marker = " ← best" if rmse == min(results.values()) else ""
        print(f"  {name:<30s} {rmse:>8.2f}{marker}")
    print(f"\n  Target: RMSE ≤ 6.62 (Özcan 2025)")
    print(f"  Published SOTA: 11.28 (Transformer 2024)")


if __name__ == '__main__':
    main()
