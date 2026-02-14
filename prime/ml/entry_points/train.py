"""
Prime ML: RUL Prediction from Manifold Geometry
================================================
Trains and evaluates ML models on machine_learning.parquet.

Usage:
    python -m prime.ml.entry_points.train --data ~/data/FD001/output

Expects:
    machine_learning.parquet  (from build_ml_features.py)

Produces in --data/ml_results/:
    predictions.parquet       — per-row predictions vs actual RUL
    feature_importance.parquet — ranked feature importance
    model_summary.json        — RMSE, MAE, R² for each model
    cv_results.parquet        — per-fold cross-validation results
    residuals.parquet         — prediction errors for analysis

Models:
    1. Ridge regression (baseline)
    2. Random forest
    3. Gradient boosting (best expected)
    4. Top-5 feature linear model (interpretable)

Validation:
    GroupKFold on cohort — no data leakage between engines.
    Train on 80 engines, test on 20. Repeat 5 folds.
"""

import argparse
import json
import warnings
import numpy as np
import polars as pl
from pathlib import Path
from dataclasses import dataclass

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Lazy imports — fail fast with clear message
# ──────────────────────────────────────────────
try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.impute import SimpleImputer
except ImportError:
    print("ERROR: scikit-learn required. Install with:")
    print("  pip install scikit-learn")
    raise SystemExit(1)


# ──────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────
N_FOLDS = 5
RANDOM_STATE = 42
MAX_RUL_CAP = 125  # Standard C-MAPSS cap — RUL above this is clipped


@dataclass
class ModelResult:
    name: str
    rmse: float
    mae: float
    r2: float
    rmse_std: float
    mae_std: float
    n_features: int


# ──────────────────────────────────────────────
# Preprocessing
# ──────────────────────────────────────────────

def prepare_features(ml: pl.DataFrame, cap_rul: int = MAX_RUL_CAP):
    """Prepare X, y, groups from machine_learning.parquet."""

    meta_cols = ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']
    feat_cols = [c for c in ml.columns if c not in meta_cols]

    # Drop features with >30% null
    null_pcts = {c: (ml[c].null_count() + (ml[c].is_nan().sum() if ml[c].dtype in [pl.Float64, pl.Float32] else 0)) / len(ml)
                 for c in feat_cols}
    keep_cols = [c for c in feat_cols if null_pcts[c] <= 0.30]
    dropped = [c for c in feat_cols if null_pcts[c] > 0.30]

    if dropped:
        print(f"  Dropped {len(dropped)} high-null features: {dropped[:5]}...")

    X = ml.select(keep_cols).to_numpy().astype(np.float64)
    y = ml['RUL'].to_numpy().astype(np.float64)
    groups = ml['cohort'].to_numpy()

    # Cap RUL (standard C-MAPSS practice)
    y = np.minimum(y, cap_rul)

    # Impute remaining NaN/null with median
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)

    # Replace any remaining inf
    X = np.where(np.isinf(X), 0, X)

    return X, y, groups, keep_cols, imputer


# ──────────────────────────────────────────────
# Training + evaluation
# ──────────────────────────────────────────────

def evaluate_model(model, X, y, groups, model_name, n_folds=N_FOLDS):
    """GroupKFold cross-validation. Returns per-fold and aggregate metrics."""

    gkf = GroupKFold(n_splits=n_folds)

    fold_results = []
    all_preds = np.full_like(y, np.nan)

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Scale
        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Train
        model_clone = clone_model(model)
        model_clone.fit(X_train_s, y_train)

        # Predict
        y_pred = model_clone.predict(X_test_s)
        y_pred = np.clip(y_pred, 0, MAX_RUL_CAP)

        all_preds[test_idx] = y_pred

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        test_cohorts = np.unique(groups[test_idx])

        fold_results.append({
            'model': model_name,
            'fold': fold,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
            'test_cohorts': ','.join(str(c) for c in test_cohorts[:5]),
        })

        print(f"    Fold {fold}: RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.4f}  (test={len(test_idx)})")

    rmses = [f['rmse'] for f in fold_results]
    maes = [f['mae'] for f in fold_results]
    r2s = [f['r2'] for f in fold_results]

    result = ModelResult(
        name=model_name,
        rmse=float(np.mean(rmses)),
        mae=float(np.mean(maes)),
        r2=float(np.mean(r2s)),
        rmse_std=float(np.std(rmses)),
        mae_std=float(np.std(maes)),
        n_features=X.shape[1],
    )

    return result, fold_results, all_preds


def clone_model(model):
    """Simple model cloning."""
    params = model.get_params()
    return model.__class__(**params)


def get_feature_importance(model, feature_names, model_name):
    """Extract feature importance from fitted model."""
    if hasattr(model, 'feature_importances_'):
        imp = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imp = np.abs(model.coef_)
    else:
        return None

    rows = []
    for name, importance in zip(feature_names, imp):
        rows.append({
            'feature': name,
            'importance': float(importance),
            'model': model_name,
        })

    return pl.DataFrame(rows).sort('importance', descending=True)


# ──────────────────────────────────────────────
# Run (programmatic entry point)
# ──────────────────────────────────────────────

def run(data: str | Path, output: str | Path = None, cap_rul: int = MAX_RUL_CAP) -> Path:
    """
    Train and evaluate ML models on machine_learning.parquet.

    Parameters
    ----------
    data : path to directory with machine_learning.parquet
    output : output directory (default: data/ml_results)
    cap_rul : RUL cap (default: 125)

    Returns
    -------
    Path to the output directory containing all results
    """
    data = Path(data)
    out = Path(output) if output else data / 'ml_results'
    out.mkdir(parents=True, exist_ok=True)

    ml_path = data / 'machine_learning.parquet'
    if not ml_path.exists():
        print(f"ERROR: {ml_path} not found. Run build_ml_features.py first.")
        raise SystemExit(1)

    ml = pl.read_parquet(str(ml_path))

    print("=" * 60)
    print("  PRIME ML — RUL PREDICTION")
    print("=" * 60)
    print(f"\n  Data: {ml.shape[0]} rows × {ml.shape[1]} columns")
    print(f"  RUL cap: {cap_rul}")
    print(f"  Validation: {N_FOLDS}-fold GroupKFold (no cohort leakage)")

    # ──────────────────────────────────────────
    # Prepare
    # ──────────────────────────────────────────
    X, y, groups, feature_names, imputer = prepare_features(ml, cap_rul=cap_rul)
    print(f"  Features after cleaning: {X.shape[1]}")
    print(f"  RUL range (capped): {y.min():.0f}–{y.max():.0f}")

    # ──────────────────────────────────────────
    # Define models
    # ──────────────────────────────────────────
    models = {
        'ridge': Ridge(alpha=1.0),
        'random_forest': RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=RANDOM_STATE,
        ),
    }

    # ──────────────────────────────────────────
    # Train + evaluate each model
    # ──────────────────────────────────────────
    all_results = []
    all_cv = []
    all_predictions = {}
    all_importance = []
    best_rmse = float('inf')
    best_model_name = None

    for name, model in models.items():
        print(f"\n  [{name}]")
        result, cv_folds, preds = evaluate_model(model, X, y, groups, name)

        all_results.append(result)
        all_cv.extend(cv_folds)
        all_predictions[name] = preds

        print(f"    → RMSE={result.rmse:.2f} ± {result.rmse_std:.2f}  "
              f"MAE={result.mae:.2f}  R²={result.r2:.4f}")

        if result.rmse < best_rmse:
            best_rmse = result.rmse
            best_model_name = name

    # ──────────────────────────────────────────
    # Top-5 interpretable model
    # ──────────────────────────────────────────
    print(f"\n  [top5_linear] (interpretable baseline)")

    # Train full GB to get importance, then pick top 5
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)
    gb_full = GradientBoostingRegressor(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        min_samples_leaf=5, subsample=0.8, random_state=RANDOM_STATE
    )
    gb_full.fit(X_scaled, y)

    imp_idx = np.argsort(gb_full.feature_importances_)[::-1][:5]
    top5_names = [feature_names[i] for i in imp_idx]
    top5_imp = [float(gb_full.feature_importances_[i]) for i in imp_idx]

    print(f"    Top 5 features:")
    for fname, fimp in zip(top5_names, top5_imp):
        print(f"      {fname}: {fimp:.4f}")

    X_top5 = X[:, imp_idx]
    result_top5, cv_top5, preds_top5 = evaluate_model(
        LinearRegression(), X_top5, y, groups, 'top5_linear'
    )

    all_results.append(result_top5)
    all_cv.extend(cv_top5)
    all_predictions['top5_linear'] = preds_top5

    print(f"    → RMSE={result_top5.rmse:.2f} ± {result_top5.rmse_std:.2f}  "
          f"MAE={result_top5.mae:.2f}  R²={result_top5.r2:.4f}")

    # ──────────────────────────────────────────
    # Feature importance from best model
    # ──────────────────────────────────────────
    print(f"\n  Feature importance (from {best_model_name}):")

    # Retrain best model on full data for importance
    best_model = clone_model(models[best_model_name])
    best_model.fit(X_scaled, y)

    imp_df = get_feature_importance(best_model, feature_names, best_model_name)
    if imp_df is not None:
        for row in imp_df.head(15).iter_rows(named=True):
            print(f"    {row['feature']:>50s}  {row['importance']:.4f}")
        all_importance.append(imp_df)

    # Also get GB importance
    gb_imp = get_feature_importance(gb_full, feature_names, 'gradient_boosting')
    if gb_imp is not None:
        all_importance.append(gb_imp)

    # ──────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────

    # 1. Predictions
    pred_I = ml['I'].to_numpy()
    pred_lp = ml['lifecycle_pct'].to_numpy()

    pred_rows = []
    for i in range(len(y)):
        row = {
            'cohort': str(groups[i]),
            'I': int(pred_I[i]),
            'RUL_actual': float(y[i]),
            'lifecycle_pct': float(pred_lp[i]),
        }
        for name, preds in all_predictions.items():
            if not np.isnan(preds[i]):
                row[f'RUL_pred_{name}'] = float(preds[i])
                row[f'error_{name}'] = float(preds[i] - y[i])
        pred_rows.append(row)

    pred_df = pl.DataFrame(pred_rows)
    pred_df.write_parquet(str(out / 'predictions.parquet'))
    print(f"\n  → {out / 'predictions.parquet'} ({len(pred_df)} rows)")

    # 2. Feature importance
    if all_importance:
        imp_all = pl.concat(all_importance)
        imp_all.write_parquet(str(out / 'feature_importance.parquet'))
        print(f"  → {out / 'feature_importance.parquet'}")

    # 3. CV results
    cv_df = pl.DataFrame(all_cv)
    cv_df.write_parquet(str(out / 'cv_results.parquet'))
    print(f"  → {out / 'cv_results.parquet'}")

    # 4. Residuals analysis
    best_preds = all_predictions[best_model_name]
    valid = ~np.isnan(best_preds)
    residuals = best_preds[valid] - y[valid]

    I_arr = ml['I'].to_numpy()
    lp_arr = ml['lifecycle_pct'].to_numpy()

    resid_rows = []
    for i in np.where(valid)[0]:
        i_int = int(i)
        resid_rows.append({
            'cohort': str(groups[i_int]),
            'I': int(I_arr[i_int]),
            'RUL_actual': float(y[i_int]),
            'RUL_pred': float(best_preds[i_int]),
            'error': float(best_preds[i_int] - y[i_int]),
            'abs_error': float(abs(best_preds[i_int] - y[i_int])),
            'lifecycle_pct': float(lp_arr[i_int]),
        })

    resid_df = pl.DataFrame(resid_rows)
    resid_df.write_parquet(str(out / 'residuals.parquet'))
    print(f"  → {out / 'residuals.parquet'}")

    # 5. Model summary
    summary = {
        'dataset': 'FD001',
        'n_rows': int(ml.shape[0]),
        'n_features_input': int(X.shape[1]),
        'rul_cap': int(cap_rul),
        'n_folds': N_FOLDS,
        'validation': 'GroupKFold on cohort',
        'best_model': best_model_name,
        'models': {},
        'top5_features': list(zip(top5_names, top5_imp)),
    }

    for r in all_results:
        summary['models'][r.name] = {
            'rmse': r.rmse,
            'rmse_std': r.rmse_std,
            'mae': r.mae,
            'mae_std': r.mae_std,
            'r2': r.r2,
            'n_features': r.n_features,
        }

    # Error by lifecycle phase
    early = resid_df.filter(pl.col('lifecycle_pct') < 0.33)
    mid = resid_df.filter((pl.col('lifecycle_pct') >= 0.33) & (pl.col('lifecycle_pct') < 0.66))
    late = resid_df.filter(pl.col('lifecycle_pct') >= 0.66)

    summary['error_by_phase'] = {
        'early_0_33': {
            'rmse': float(np.sqrt(np.mean(early['error'].to_numpy()**2))) if len(early) > 0 else None,
            'mae': float(np.mean(np.abs(early['error'].to_numpy()))) if len(early) > 0 else None,
            'n': len(early),
        },
        'mid_33_66': {
            'rmse': float(np.sqrt(np.mean(mid['error'].to_numpy()**2))) if len(mid) > 0 else None,
            'mae': float(np.mean(np.abs(mid['error'].to_numpy()))) if len(mid) > 0 else None,
            'n': len(mid),
        },
        'late_66_100': {
            'rmse': float(np.sqrt(np.mean(late['error'].to_numpy()**2))) if len(late) > 0 else None,
            'mae': float(np.mean(np.abs(late['error'].to_numpy()))) if len(late) > 0 else None,
            'n': len(late),
        },
    }

    summary_path = out / 'model_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"  → {summary_path}")

    # ──────────────────────────────────────────
    # Final report
    # ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print()
    print(f"  {'Model':<20s} {'RMSE':>8s} {'± std':>8s} {'MAE':>8s} {'R²':>8s} {'Feats':>6s}")
    print(f"  {'-'*20:<20s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*8:>8s} {'-'*6:>6s}")

    for r in sorted(all_results, key=lambda x: x.rmse):
        marker = ' ◄' if r.name == best_model_name else ''
        print(f"  {r.name:<20s} {r.rmse:>8.2f} {r.rmse_std:>8.2f} {r.mae:>8.2f} {r.r2:>8.4f} {r.n_features:>6d}{marker}")

    print()
    print(f"  Error by lifecycle phase ({best_model_name}):")
    for phase, phase_data in summary['error_by_phase'].items():
        if phase_data['rmse'] is not None:
            print(f"    {phase:<15s}  RMSE={phase_data['rmse']:.2f}  MAE={phase_data['mae']:.2f}  n={phase_data['n']}")

    print()
    print(f"  All outputs in: {out}")
    print()

    return out


def main():
    parser = argparse.ArgumentParser(description='Prime ML — RUL prediction from Manifold geometry')
    parser.add_argument('--data', required=True, help='Directory with machine_learning.parquet')
    parser.add_argument('--output', default=None, help='Output directory (default: data/ml_results)')
    parser.add_argument('--cap-rul', type=int, default=MAX_RUL_CAP, help='RUL cap (default: 125)')
    args = parser.parse_args()

    run(data=args.data, output=args.output, cap_rul=args.cap_rul)


if __name__ == '__main__':
    main()
