"""
Prime ML: Official C-MAPSS Test Evaluation
==========================================
Trains on ALL training data, predicts on official test set,
scores against RUL ground truth.

Directory structure expected:
    data/FD001/
    ├── train/
    │   └── output/           ← Manifold output from training engines
    │       └── machine_learning.parquet  ← from build_ml_features.py
    │
    ├── test/
    │   └── output/           ← Manifold output from test engines
    │       └── machine_learning.parquet
    │
    └── RUL_FD001.txt         ← Ground truth (one RUL per test engine)

Usage:
    python evaluate_test.py --train data/FD001/train/output \
                            --test  data/FD001/test/output \
                            --rul   data/FD001/RUL_FD001.txt

Metrics:
    - RMSE (standard)
    - MAE
    - PHM08 Score (asymmetric: late predictions penalized more)
    - R²
"""

import argparse
import json
import warnings
import numpy as np
import polars as pl
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
except ImportError:
    print("ERROR: scikit-learn required.")
    print("  pip install scikit-learn")
    raise SystemExit(1)


# ──────────────────────────────────────────────
# C-MAPSS metrics
# ──────────────────────────────────────────────

MAX_RUL_CAP = 125


def phm08_score(y_true, y_pred):
    """
    PHM08 asymmetric scoring function.
    Late predictions (positive error) penalized exponentially more
    than early predictions (negative error).

    d = predicted - actual
    s = sum( exp(-d/13) - 1  if d < 0   [early]
             exp( d/10) - 1  if d >= 0   [late]  )
    """
    d = y_pred - y_true
    s = 0.0
    for di in d:
        if di < 0:
            s += np.exp(-di / 13.0) - 1.0
        else:
            s += np.exp(di / 10.0) - 1.0
    return s


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_ml_data(data_dir: Path):
    """Load machine_learning.parquet from a Manifold output directory."""
    ml_path = data_dir / 'machine_learning.parquet'
    if not ml_path.exists():
        print(f"ERROR: {ml_path} not found.")
        print("Run build_ml_features.py on this output directory first.")
        raise SystemExit(1)
    return pl.read_parquet(str(ml_path))


def load_rul_truth(rul_path: Path):
    """
    Load official RUL ground truth.
    Format: one integer per line, one per test engine.
    Engine numbering is 1-indexed in order.
    """
    with open(rul_path) as f:
        ruls = [int(line.strip()) for line in f if line.strip()]
    return np.array(ruls, dtype=float)


def prepare_features(ml: pl.DataFrame, cap_rul: int = MAX_RUL_CAP):
    """Prepare X, y, groups, feature_names from ML parquet."""
    meta_cols = ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']
    feat_cols = [c for c in ml.columns if c not in meta_cols]

    # Drop high-null features
    null_pcts = {}
    for c in feat_cols:
        null_count = ml[c].null_count()
        if ml[c].dtype in [pl.Float64, pl.Float32]:
            null_count += ml[c].is_nan().sum()
        null_pcts[c] = null_count / len(ml)

    keep_cols = [c for c in feat_cols if null_pcts[c] <= 0.30]

    X = ml.select(keep_cols).to_numpy().astype(np.float64)

    has_rul = 'RUL' in ml.columns
    y = ml['RUL'].to_numpy().astype(np.float64) if has_rul else None
    if y is not None:
        y = np.minimum(y, cap_rul)

    groups = ml['cohort'].to_numpy() if 'cohort' in ml.columns else None

    # Impute + clean
    imputer = SimpleImputer(strategy='median')
    X = imputer.fit_transform(X)
    X = np.where(np.isinf(X), 0, X)

    return X, y, groups, keep_cols, imputer


def get_last_window_per_engine(ml: pl.DataFrame):
    """
    For test evaluation: get the LAST window per engine.
    The official test asks for ONE prediction per engine at the cutoff point.
    """
    # Get the maximum I per cohort (last observation window)
    last_I = (ml.group_by('cohort')
              .agg(pl.col('I').max().alias('max_I')))

    # Join to get only last-window rows
    last_rows = ml.join(last_I, on='cohort').filter(pl.col('I') == pl.col('max_I'))

    return last_rows


def align_features(train_cols, test_ml, imputer_train):
    """Ensure test has same columns as train, in same order."""
    meta_cols = ['cohort', 'I', 'RUL', 'lifecycle', 'lifecycle_pct']
    feat_cols = [c for c in train_cols if c not in meta_cols]

    # Add missing columns as NaN
    for c in feat_cols:
        if c not in test_ml.columns:
            test_ml = test_ml.with_columns(pl.lit(None).cast(pl.Float64).alias(c))

    X_test = test_ml.select(feat_cols).to_numpy().astype(np.float64)

    # Impute with training medians
    X_test = imputer_train.transform(X_test)
    X_test = np.where(np.isinf(X_test), 0, X_test)

    return X_test


# ──────────────────────────────────────────────
# Run (programmatic entry point)
# ──────────────────────────────────────────────

def run(train: str | Path, test: str | Path, rul: str | Path,
        output: str | Path = None, cap_rul: int = MAX_RUL_CAP) -> Path:
    """
    Official C-MAPSS test evaluation.

    Parameters
    ----------
    train : path to training output directory (with machine_learning.parquet)
    test : path to test output directory (with machine_learning.parquet)
    rul : path to RUL ground truth file (RUL_FD001.txt)
    output : output directory (default: test/ml_results)
    cap_rul : RUL cap (default: 125)

    Returns
    -------
    Path to the output directory containing results
    """
    train_dir = Path(train)
    test_dir = Path(test)
    rul_path = Path(rul)
    out_dir = Path(output) if output else test_dir / 'ml_results'
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  PRIME ML — OFFICIAL TEST EVALUATION")
    print("=" * 60)

    # ──────────────────────────────────────────
    # Load training data
    # ──────────────────────────────────────────
    print(f"\n  Loading training data: {train_dir}")
    train_ml = load_ml_data(train_dir)
    X_train, y_train, groups_train, feat_cols, imputer = prepare_features(train_ml, cap_rul)
    print(f"    {X_train.shape[0]} rows × {X_train.shape[1]} features")
    print(f"    RUL range (capped): {y_train.min():.0f}–{y_train.max():.0f}")

    # ──────────────────────────────────────────
    # Load test data
    # ──────────────────────────────────────────
    print(f"\n  Loading test data: {test_dir}")
    test_ml = load_ml_data(test_dir)
    print(f"    {test_ml.shape[0]} rows × {test_ml.shape[1]} columns")

    # Get last window per engine (official eval = 1 prediction per engine)
    test_last = get_last_window_per_engine(test_ml)
    print(f"    Last-window rows: {test_last.shape[0]} engines")

    # Sort cohorts numerically: engine_1, engine_2, ..., engine_100
    def cohort_sort_key(c):
        num = ''.join(filter(str.isdigit, c))
        return int(num) if num else 0

    test_last_sorted = test_last.with_columns(
        pl.col('cohort').map_elements(cohort_sort_key, return_dtype=pl.Int64).alias('_sort')
    ).sort('_sort').drop('_sort')

    sorted_cohorts = test_last_sorted['cohort'].to_list()
    print(f"    Cohort order: {sorted_cohorts[:5]}... → {sorted_cohorts[-3:]}")

    # Align features
    X_test = align_features(feat_cols, test_last_sorted, imputer)
    print(f"    Test feature matrix: {X_test.shape}")

    # ──────────────────────────────────────────
    # Load ground truth
    # ──────────────────────────────────────────
    print(f"\n  Loading ground truth: {rul_path}")
    y_true = load_rul_truth(rul_path)
    y_true_capped = np.minimum(y_true, cap_rul)
    print(f"    {len(y_true)} engines, RUL range: {y_true.min():.0f}–{y_true.max():.0f}")
    print(f"    RUL range (capped): {y_true_capped.min():.0f}–{y_true_capped.max():.0f}")

    # Align ground truth by engine ID — test may not have all engines
    # RUL_FD001.txt line 1 = engine_1, line 2 = engine_2, etc.
    rul_by_engine = {f'engine_{i+1}': y_true[i] for i in range(len(y_true))}
    rul_capped_by_engine = {f'engine_{i+1}': y_true_capped[i] for i in range(len(y_true))}

    # Filter to only engines we have predictions for
    matched_engines = [c for c in sorted_cohorts if c in rul_by_engine]
    missing_engines = [c for c in sorted_cohorts if c not in rul_by_engine]
    no_features_engines = [f'engine_{i+1}' for i in range(len(y_true))
                           if f'engine_{i+1}' not in set(sorted_cohorts)]

    if missing_engines:
        print(f"\n  WARNING: {len(missing_engines)} test engines not in ground truth: {missing_engines[:5]}...")

    if no_features_engines:
        print(f"\n  NOTE: {len(no_features_engines)} engines had too few cycles for features (< window size)")
        print(f"    Evaluating on {len(matched_engines)}/{len(y_true)} engines")

    # Rebuild aligned arrays
    y_true = np.array([rul_by_engine[c] for c in matched_engines])
    y_true_capped = np.array([rul_capped_by_engine[c] for c in matched_engines])

    # Filter X_test to matched engines only
    engine_idx = [sorted_cohorts.index(c) for c in matched_engines]
    X_test = X_test[engine_idx]
    sorted_cohorts = matched_engines

    # ──────────────────────────────────────────
    # Train models on ALL training data
    # ──────────────────────────────────────────
    print(f"\n  Training on full training set ({X_train.shape[0]} rows)...")

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        'ridge': Ridge(alpha=1.0),
        'gradient_boosting': GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            min_samples_leaf=5, subsample=0.8, random_state=42,
        ),
        'random_forest': RandomForestRegressor(
            n_estimators=200, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
    }

    results = {}
    all_preds = {}

    for name, model in models.items():
        print(f"\n  [{name}]")
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_pred = np.clip(y_pred, 0, cap_rul)

        # Metrics against capped truth
        rmse = np.sqrt(mean_squared_error(y_true_capped, y_pred))
        mae = mean_absolute_error(y_true_capped, y_pred)
        r2 = r2_score(y_true_capped, y_pred)
        score = phm08_score(y_true_capped, y_pred)

        # Also compute against uncapped truth (for comparison)
        rmse_uncapped = np.sqrt(mean_squared_error(y_true, y_pred))

        # Bias (mean error)
        bias = float(np.mean(y_pred - y_true_capped))

        # Early vs late predictions
        d = y_pred - y_true_capped
        n_early = int(np.sum(d < 0))
        n_late = int(np.sum(d >= 0))

        results[name] = {
            'rmse': float(rmse),
            'rmse_uncapped': float(rmse_uncapped),
            'mae': float(mae),
            'r2': float(r2),
            'phm08_score': float(score),
            'bias': bias,
            'n_early': n_early,
            'n_late': n_late,
        }
        all_preds[name] = y_pred

        print(f"    RMSE = {rmse:.2f}  (uncapped: {rmse_uncapped:.2f})")
        print(f"    MAE  = {mae:.2f}")
        print(f"    R²   = {r2:.4f}")
        print(f"    PHM08 Score = {score:.1f}")
        print(f"    Bias = {bias:+.2f}  ({n_early} early, {n_late} late)")

    # ──────────────────────────────────────────
    # Feature importance from best model
    # ──────────────────────────────────────────
    best_name = min(results, key=lambda k: results[k]['rmse'])
    best_model = models[best_name]

    if hasattr(best_model, 'feature_importances_'):
        imp = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        imp = np.abs(best_model.coef_)
    else:
        imp = None

    if imp is not None:
        imp_df = pl.DataFrame({
            'feature': feat_cols,
            'importance': imp.tolist(),
        }).sort('importance', descending=True)

        imp_df.write_parquet(str(out_dir / 'test_feature_importance.parquet'))

        print(f"\n  Top 15 features ({best_name}):")
        for row in imp_df.head(15).iter_rows(named=True):
            print(f"    {row['feature']:>50s}  {row['importance']:.4f}")

    # ──────────────────────────────────────────
    # Save results
    # ──────────────────────────────────────────

    # Per-engine predictions
    pred_rows = []
    for i in range(len(sorted_cohorts)):
        row = {
            'cohort': sorted_cohorts[i],
            'RUL_true': float(y_true[i]),
            'RUL_true_capped': float(y_true_capped[i]),
        }
        for name, preds in all_preds.items():
            row[f'RUL_pred_{name}'] = float(preds[i])
            row[f'error_{name}'] = float(preds[i] - y_true_capped[i])
        pred_rows.append(row)

    pred_df = pl.DataFrame(pred_rows)
    pred_df.write_parquet(str(out_dir / 'test_predictions.parquet'))

    # Summary JSON
    summary = {
        'dataset': 'FD001',
        'evaluation': 'official_test_split',
        'n_train_rows': int(X_train.shape[0]),
        'n_train_features': int(X_train.shape[1]),
        'n_test_engines': int(len(sorted_cohorts)),
        'rul_cap': int(cap_rul),
        'best_model': best_name,
        'models': results,
    }

    with open(out_dir / 'test_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # ──────────────────────────────────────────
    # Final report
    # ──────────────────────────────────────────
    print()
    print("=" * 60)
    print("  OFFICIAL TEST RESULTS — FD001")
    print("=" * 60)
    print()
    print(f"  {'Model':<20s} {'RMSE':>8s} {'MAE':>8s} {'R²':>8s} {'PHM08':>10s} {'Bias':>8s}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")

    for name in sorted(results, key=lambda k: results[k]['rmse']):
        r = results[name]
        marker = ' ◄' if name == best_name else ''
        print(f"  {name:<20s} {r['rmse']:>8.2f} {r['mae']:>8.2f} {r['r2']:>8.4f} "
              f"{r['phm08_score']:>10.1f} {r['bias']:>+8.2f}{marker}")

    print()
    print(f"  Published benchmarks (FD001):")
    print(f"    LightGBM+CatBoost ensemble (2025): RMSE = 6.62")
    print(f"    Transformer SOTA (2024):            RMSE = 11.28")
    print(f"    Attention DCNN (2021):               RMSE = 11.81")
    print(f"    CAELSTM (2025):                      RMSE = 14.44")
    print()
    print(f"  Outputs: {out_dir}")
    print()

    return out_dir


def main():
    parser = argparse.ArgumentParser(description='Prime ML — Official C-MAPSS Test Evaluation')
    parser.add_argument('--train', required=True, help='Training output directory (with machine_learning.parquet)')
    parser.add_argument('--test', required=True, help='Test output directory (with machine_learning.parquet)')
    parser.add_argument('--rul', required=True, help='Path to RUL_FD001.txt (ground truth)')
    parser.add_argument('--output', default=None, help='Output directory for results')
    parser.add_argument('--cap-rul', type=int, default=MAX_RUL_CAP, help='RUL cap (default: 125)')
    args = parser.parse_args()

    run(train=args.train, test=args.test, rul=args.rul,
        output=args.output, cap_rul=args.cap_rul)


if __name__ == '__main__':
    main()
