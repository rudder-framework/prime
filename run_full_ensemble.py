#!/usr/bin/env python3
"""
Full Stacking Ensemble — Özcan-style
=====================================
Base learners: LightGBM + XGBoost + GBR (CatBoost substitute — no Py3.14 wheel)
Meta-learner: Ridge on 5-fold GroupKFold OOF predictions
Features: 209-column cycle_features.parquet (sensors + geometry + rolling + gaussian)
"""

import polars as pl
import numpy as np
import copy
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, RidgeCV
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

print(f"Train: {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")
print(f"Test:  {test_df.shape[0]:,} rows x {test_df.shape[1]} cols")
print(f"Ground truth: {len(y_true)} engines")

# ─── Features ───
meta_cols = ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']
feat_cols = sorted([c for c in train_df.columns if c not in meta_cols])

for c in feat_cols:
    if c not in test_df.columns:
        test_df = test_df.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

X_train = train_df.select(feat_cols).to_numpy().astype(np.float64)
y_train = train_df['RUL'].to_numpy().astype(np.float64)
groups = train_df['cohort'].to_numpy()

test_last = (test_df
    .with_columns(pl.col('cohort').map_elements(cohort_sort_key, return_dtype=pl.Int64).alias('_sort'))
    .group_by('cohort')
    .agg([pl.all().sort_by('I').last()])
    .sort('_sort')
    .drop('_sort')
)

X_test = test_last.select(feat_cols).to_numpy().astype(np.float64)
print(f"Features: {len(feat_cols)}")
print(f"Test engines (last cycle): {X_test.shape[0]}")

# ─── Impute + scale ───
imputer = SimpleImputer(strategy='median')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

X_train = np.where(np.isinf(X_train), 0, X_train)
X_test = np.where(np.isinf(X_test), 0, X_test)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Final: {X_train_s.shape}")

# ─── Base learners ───
base_models = {
    'lightgbm': lgb.LGBMRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        num_leaves=31, min_child_samples=10, subsample=0.8,
        colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=42, verbose=-1, n_jobs=-1,
    ),
    'xgboost': xgb.XGBRegressor(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        min_child_weight=10, random_state=42, verbosity=0, n_jobs=-1,
    ),
    'hist': HistGradientBoostingRegressor(
        max_iter=500, max_depth=6, learning_rate=0.05,
        min_samples_leaf=10, random_state=42,
    ),
}

print(f"\nBase learners: {list(base_models.keys())}")

# ─── 5-fold GroupKFold OOF ───
n_folds = 5
gkf = GroupKFold(n_splits=n_folds)

oof_preds = {name: np.zeros(len(y_train)) for name in base_models}
test_preds = {name: np.zeros(len(X_test_s)) for name in base_models}
fold_scores = {name: [] for name in base_models}

print(f"\n5-fold GroupKFold stacking...")
for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_train_s, y_train, groups)):
    X_tr, X_val = X_train_s[train_idx], X_train_s[val_idx]
    y_tr, y_val = y_train[train_idx], y_train[val_idx]

    for name, model in base_models.items():
        m = copy.deepcopy(model)
        m.fit(X_tr, y_tr)

        val_pred = m.predict(X_val)
        oof_preds[name][val_idx] = val_pred

        test_pred = m.predict(X_test_s)
        test_preds[name] += test_pred / n_folds

        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        fold_scores[name].append(rmse)

    print(f"  Fold {fold_idx+1}: " + ", ".join(f"{n}={s[-1]:.2f}" for n, s in fold_scores.items()))

# ─── OOF scores ───
print(f"\nOOF Scores:")
for name in base_models:
    oof_rmse = np.sqrt(mean_squared_error(y_train, oof_preds[name]))
    print(f"  {name}: RMSE={oof_rmse:.2f}")

# ─── Meta-learner ───
stack_train = np.column_stack([oof_preds[n] for n in base_models])
stack_test = np.column_stack([test_preds[n] for n in base_models])

meta = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0])
meta.fit(stack_train, y_train)
pred_stack = np.clip(meta.predict(stack_test), 0, 125)

# ─── Results ───
rmse = np.sqrt(mean_squared_error(y_true_capped, pred_stack))
mae = mean_absolute_error(y_true_capped, pred_stack)
r2 = r2_score(y_true_capped, pred_stack)
score = phm08_score(y_true_capped, pred_stack)
rmse_uc = np.sqrt(mean_squared_error(y_true, pred_stack))

print(f"\n{'='*60}")
print(f"  STACKING ENSEMBLE: LGB + XGB + Hist → RidgeCV")
print(f"  209 features (sensors + geometry + rolling + gaussian)")
print(f"{'='*60}")
print(f"  RMSE (capped):   {rmse:.2f}")
print(f"  RMSE (uncapped): {rmse_uc:.2f}")
print(f"  MAE:             {mae:.2f}")
print(f"  R²:              {r2:.4f}")
print(f"  PHM08 Score:     {score:,.0f}")

# Individual base learner test results
print(f"\nBase learner test scores:")
for name in base_models:
    pred = np.clip(test_preds[name], 0, 125)
    r = np.sqrt(mean_squared_error(y_true_capped, pred))
    s = phm08_score(y_true_capped, pred)
    print(f"  {name}: RMSE={r:.2f}, PHM08={s:,.0f}")

print(f"\nMeta: alpha={meta.alpha_:.2f}, weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(base_models.keys(), meta.coef_)))

# Worst predictions
errors = pred_stack - y_true_capped
worst_idx = np.argsort(np.abs(errors))[-10:][::-1]
print(f"\nWorst 10 predictions:")
for i in worst_idx:
    print(f"  Engine {i+1}: pred={pred_stack[i]:.1f}, true={y_true_capped[i]:.0f}, error={errors[i]:+.1f}")

# Feature importance (LightGBM full retrain)
lgb_full = lgb.LGBMRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    num_leaves=31, min_child_samples=10, subsample=0.8,
    colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=-1,
)
lgb_full.fit(X_train_s, y_train)
imp = dict(zip(feat_cols, lgb_full.feature_importances_))
print(f"\nTop 20 features (LightGBM importance):")
for k, v in sorted(imp.items(), key=lambda x: x[1], reverse=True)[:20]:
    print(f"  {k}: {v}")

# ─── Comparison table ───
print(f"\n{'='*60}")
print(f"  COMPARISON")
print(f"{'='*60}")
print(f"  {'Method':<40s} {'RMSE':>8s}")
print(f"  {'-'*40} {'-'*8}")
print(f"  {'Geometry only (LGB, 7 feat)':<40s} {'16.66':>8s}")
print(f"  {'Full ensemble (LGB+XGB+Hist, 209 feat)':<40s} {rmse:>8.2f}")
print(f"  {'─'*48}")
print(f"  {'Özcan LGB+CB+XGB (2025)':<40s} {'6.62':>8s}")
print(f"  {'Transformer SOTA (2024)':<40s} {'11.28':>8s}")
print(f"  {'Attention DCNN (2021)':<40s} {'11.81':>8s}")
