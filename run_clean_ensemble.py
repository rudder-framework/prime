#!/usr/bin/env python3
"""
Clean Ensemble — ONLY features that generalize
================================================
sensors + rolling + delta + RT geometry (no geom_, no gaussian)
LGB + XGB + Hist → RidgeCV
"""

import polars as pl
import numpy as np
import copy
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# ─── Feature selection: ONLY generalizing features ───
meta_cols = {'cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct'}
leak_prefixes = ('geom_', 'gfp_', 'gsim_')  # per-engine identity encoders

feat_cols = sorted([c for c in train_df.columns
                    if c not in meta_cols
                    and not any(c.startswith(p) for p in leak_prefixes)])

print(f"Train: {train_df.shape[0]:,} rows")
print(f"Test:  {test_df.shape[0]:,} rows")
print(f"Features (clean): {len(feat_cols)}")
print(f"Dropped: {sum(1 for c in train_df.columns if any(c.startswith(p) for p in leak_prefixes))} leaking features")

# Feature groups
groups = {}
for c in feat_cols:
    prefix = c.split('_')[0] if '_' in c else c
    groups.setdefault(prefix, []).append(c)
for p in sorted(groups):
    print(f"  {p}: {len(groups[p])}")

# ─── Prepare ───
for c in feat_cols:
    if c not in test_df.columns:
        test_df = test_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

X_train = train_df.select(feat_cols).to_numpy().astype(np.float64)
y_train = train_df['RUL'].to_numpy().astype(np.float64)
groups_arr = train_df['cohort'].to_numpy()

test_last = (test_df
    .with_columns(pl.col('cohort').map_elements(cohort_sort_key, return_dtype=pl.Int64).alias('_sort'))
    .group_by('cohort')
    .agg([pl.all().sort_by('I').last()])
    .sort('_sort')
    .drop('_sort')
)
X_test = test_last.select(feat_cols).to_numpy().astype(np.float64)

# ─── Impute + scale ───
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)
X_train = np.where(np.isinf(X_train), 0, X_train)
X_test = np.where(np.isinf(X_test), 0, X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"\nFinal: {X_train_s.shape}")

# ─── Base learners ───
base_models = {
    'lgb': lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1),
    'xgb': xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=10, random_state=42, verbosity=0, n_jobs=-1),
    'hist': HistGradientBoostingRegressor(
        max_iter=500, max_depth=6, learning_rate=0.05,
        min_samples_leaf=10, random_state=42),
}

# ─── 5-fold GroupKFold ───
gkf = GroupKFold(n_splits=5)
oof_preds = {n: np.zeros(len(y_train)) for n in base_models}
test_preds = {n: np.zeros(len(X_test_s)) for n in base_models}

print(f"\n5-fold GroupKFold stacking...")
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_s, y_train, groups_arr)):
    X_tr, X_val = X_train_s[train_idx], X_train_s[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]
    for name, model in base_models.items():
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)
        oof_preds[name][val_idx] = m.predict(X_val)
        test_preds[name] += m.predict(X_test_s) / 5
    oof_scores = {n: np.sqrt(mean_squared_error(y_train[val_idx], oof_preds[n][val_idx])) for n in base_models}
    print(f"  Fold {fold_idx+1}: " + ", ".join(f"{n}={v:.2f}" for n, v in oof_scores.items()))

# ─── OOF ───
print(f"\nOOF Scores:")
for name in base_models:
    r = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
    print(f"  {name}: RMSE={r:.2f}")

# ─── Stack ───
stack_train = np.column_stack([oof_preds[n] for n in base_models])
stack_test = np.column_stack([test_preds[n] for n in base_models])
meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
meta.fit(stack_train, y_train)
pred = np.clip(meta.predict(stack_test), 0, 125)

rmse = np.sqrt(mean_squared_error(y_true_capped, pred))
mae = mean_absolute_error(y_true_capped, pred)
r2 = r2_score(y_true_capped, pred)
score = phm08_score(y_true_capped, pred)

print(f"\n{'='*60}")
print(f"  CLEAN ENSEMBLE (no geom_, no gaussian)")
print(f"  {len(feat_cols)} features: sensors + rolling + delta + RT")
print(f"{'='*60}")
print(f"  RMSE:        {rmse:.2f}")
print(f"  MAE:         {mae:.2f}")
print(f"  R²:          {r2:.4f}")
print(f"  PHM08 Score: {score:,.0f}")
print(f"  Meta alpha:  {meta.alpha_:.2f}")
print(f"  Meta weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(base_models.keys(), meta.coef_)))

# Base learner scores
print(f"\nBase learner test:")
for name in base_models:
    p = np.clip(test_preds[name], 0, 125)
    r = np.sqrt(mean_squared_error(y_true_capped, p))
    s = phm08_score(y_true_capped, p)
    print(f"  {name}: RMSE={r:.2f}, PHM08={s:,.0f}")

# Worst 10
errors = pred - y_true_capped
worst = np.argsort(np.abs(errors))[-10:][::-1]
print(f"\nWorst 10:")
for i in worst:
    print(f"  Engine {i+1}: pred={pred[i]:.1f}, true={y_true_capped[i]:.0f}, err={errors[i]:+.1f}")

# Feature importance
lgb_full = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=10, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1)
lgb_full.fit(X_train_s, y_train)
imp = dict(zip(feat_cols, lgb_full.feature_importances_))
print(f"\nTop 20 features (LGB):")
for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {k}: {v}")

# Comparison
print(f"\n{'='*60}")
print(f"  COMPARISON")
print(f"{'='*60}")
print(f"  {'Method':<45s} {'RMSE':>8s} {'Gap':>6s}")
print(f"  {'-'*45} {'-'*8} {'-'*6}")
oof_avg = np.mean([np.sqrt(mean_squared_error(y_train, oof_preds[n])) for n in base_models])
print(f"  {'Our OOF (cross-validation)':<45s} {oof_avg:>8.2f}")
print(f"  {'Our Test (clean ensemble)':<45s} {rmse:>8.2f} {rmse-oof_avg:>+6.1f}")
print(f"  {'─'*60}")
print(f"  {'Özcan LGB+CB+XGB (2025)':<45s} {'6.62':>8s}")
print(f"  {'Transformer SOTA (2024)':<45s} {'11.28':>8s}")
