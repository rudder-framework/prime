#!/usr/bin/env python3
"""
Feature ablation — find what's causing overfit.
"""

import polars as pl
import numpy as np
import copy
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')


def phm08_score(y_true, y_pred):
    d = np.array(y_pred) - np.array(y_true)
    return sum(np.exp(-di / 13.0) - 1 if di < 0 else np.exp(di / 10.0) - 1 for di in d)


def cohort_sort_key(c):
    num = ''.join(filter(str.isdigit, str(c)))
    return int(num) if num else 0


# ─── Load ───
train_df = pl.read_parquet('/Users/jasonrudder/data/FD001/train/cycle_features.parquet')
test_df = pl.read_parquet('/Users/jasonrudder/data/FD001/test/cycle_features.parquet')

with open('/Users/jasonrudder/data/FD001/RUL_FD001.txt') as f:
    y_true = np.array([int(line.strip()) for line in f if line.strip()], dtype=float)
y_true_capped = np.minimum(y_true, 125)

meta_cols = ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']
all_feat_cols = sorted([c for c in train_df.columns if c not in meta_cols])

# Ensure test has all columns
for c in all_feat_cols:
    if c not in test_df.columns:
        test_df = test_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

y_train = train_df['RUL'].to_numpy().astype(np.float64)
groups = train_df['cohort'].to_numpy()

# Get last cycle per test engine
test_last = (test_df
    .with_columns(pl.col('cohort').map_elements(cohort_sort_key, return_dtype=pl.Int64).alias('_sort'))
    .group_by('cohort')
    .agg([pl.all().sort_by('I').last()])
    .sort('_sort')
    .drop('_sort')
)


def run_ensemble(feat_cols, label):
    """Run LGB+XGB+Hist stacking with given features."""
    X_train = train_df.select(feat_cols).to_numpy().astype(np.float64)
    X_test = test_last.select(feat_cols).to_numpy().astype(np.float64)

    imp = SimpleImputer(strategy='median')
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)
    X_train = np.where(np.isinf(X_train), 0, X_train)
    X_test = np.where(np.isinf(X_test), 0, X_test)

    sc = StandardScaler()
    X_train_s = sc.fit_transform(X_train)
    X_test_s = sc.transform(X_test)

    base_models = {
        'lgb': lgb.LGBMRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
            num_leaves=31, min_child_samples=10, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=-1),
        'xgb': xgb.XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            min_child_weight=10, random_state=42, verbosity=0, n_jobs=-1),
        'hist': HistGradientBoostingRegressor(max_iter=500, max_depth=6,
            learning_rate=0.05, min_samples_leaf=10, random_state=42),
    }

    gkf = GroupKFold(n_splits=5)
    oof_preds = {n: np.zeros(len(y_train)) for n in base_models}
    test_preds = {n: np.zeros(len(X_test_s)) for n in base_models}

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_s, y_train, groups)):
        X_tr, X_val = X_train_s[train_idx], X_train_s[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        for name, model in base_models.items():
            m = copy.deepcopy(model)
            m.fit(X_tr, y_tr)
            oof_preds[name][val_idx] = m.predict(X_val)
            test_preds[name] += m.predict(X_test_s) / 5

    # OOF scores
    oof_rmses = {n: np.sqrt(mean_squared_error(y_train, oof_preds[n])) for n in base_models}

    # Stack
    stack_train = np.column_stack([oof_preds[n] for n in base_models])
    stack_test = np.column_stack([test_preds[n] for n in base_models])
    meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
    meta.fit(stack_train, y_train)
    pred = np.clip(meta.predict(stack_test), 0, 125)

    rmse = np.sqrt(mean_squared_error(y_true_capped, pred))
    score = phm08_score(y_true_capped, pred)
    oof_avg = np.mean(list(oof_rmses.values()))

    print(f"  {label:<45s} | {len(feat_cols):>4d} feat | OOF={oof_avg:.2f} | Test RMSE={rmse:.2f} | PHM08={score:,.0f}")
    return rmse, oof_avg, pred


# ─── Define feature groups ───
sensor_cols = [c for c in all_feat_cols if not c.startswith(('geom_', 'gfp_', 'gsim_', 'roll_', 'delta_', 'rt_', 'lifecycle'))]
geom_cols = [c for c in all_feat_cols if c.startswith('geom_')]
rt_cols = [c for c in all_feat_cols if c.startswith('rt_')]
roll_cols = [c for c in all_feat_cols if c.startswith('roll_')]
delta_cols = [c for c in all_feat_cols if c.startswith('delta_')]
gfp_cols = [c for c in all_feat_cols if c.startswith('gfp_')]
gsim_cols = [c for c in all_feat_cols if c.startswith('gsim_')]
life_cols = [c for c in all_feat_cols if c.startswith('lifecycle')]

print(f"Feature groups:")
print(f"  sensors:  {len(sensor_cols)} (raw sensor values + ops + cycle)")
print(f"  geom:     {len(geom_cols)} (interpolated geometry dynamics)")
print(f"  rt:       {len(rt_cols)} (real-time geometry from stage 35)")
print(f"  roll:     {len(roll_cols)} (rolling stats)")
print(f"  delta:    {len(delta_cols)} (cycle-to-cycle deltas)")
print(f"  gfp:      {len(gfp_cols)} (gaussian fingerprint - STATIC)")
print(f"  gsim:     {len(gsim_cols)} (gaussian similarity - STATIC)")
print(f"  life:     {len(life_cols)} (lifecycle)")
print(f"  total:    {len(all_feat_cols)}")

print(f"\n{'='*100}")
print(f"  ABLATION STUDY")
print(f"{'='*100}")

# Run ablations
results = {}

# 1. Sensors only (Özcan baseline — 21 sensors + 3 ops + cycle)
results['sensors'] = run_ensemble(sensor_cols, "1. Sensors only (Özcan baseline)")

# 2. Sensors + rolling stats
results['sensors+roll'] = run_ensemble(sensor_cols + roll_cols, "2. Sensors + rolling stats")

# 3. Sensors + rolling + deltas
results['sensors+roll+delta'] = run_ensemble(sensor_cols + roll_cols + delta_cols, "3. Sensors + rolling + deltas")

# 4. Sensors + rolling + deltas + RT geometry
results['sensors+roll+delta+rt'] = run_ensemble(sensor_cols + roll_cols + delta_cols + rt_cols, "4. + real-time geometry (stage 35)")

# 5. Sensors + rolling + deltas + RT geometry + interp geometry
results['sensors+roll+delta+rt+geom'] = run_ensemble(sensor_cols + roll_cols + delta_cols + rt_cols + geom_cols, "5. + interpolated geometry dynamics")

# 6. Add gaussian fingerprint
results['all_no_gsim'] = run_ensemble(sensor_cols + roll_cols + delta_cols + rt_cols + geom_cols + gfp_cols, "6. + gaussian fingerprint (STATIC)")

# 7. Full (add gaussian similarity)
results['full'] = run_ensemble(all_feat_cols, "7. Full (all 204 features)")

print(f"\n{'='*100}")
print(f"  SUMMARY")
print(f"{'='*100}")
print(f"  {'Config':<45s} | {'Feat':>4s} | {'OOF':>6s} | {'Test':>6s} | {'Gap':>5s}")
print(f"  {'-'*45} | {'-'*4} | {'-'*6} | {'-'*6} | {'-'*5}")
for name, (test_rmse, oof_rmse, _) in results.items():
    gap = test_rmse - oof_rmse
    print(f"  {name:<45s} |      | {oof_rmse:>6.2f} | {test_rmse:>6.2f} | {gap:>+5.1f}")

print(f"\n  Target: RMSE ≤ 6.62 (Özcan 2025)")
print(f"  Key: If OOF << Test, we're overfitting to engine identity")
