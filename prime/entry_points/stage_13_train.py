"""
13: Train Entry Point
======================

Pure orchestration - calls ML training pipeline.
Trains supervised models on Manifold features for RUL/health prediction.

Stages: ml_features.parquet → trained model + results

Supports XGBoost, CatBoost, LightGBM, RandomForest, GradientBoosting.
Requires Manifold features (run Manifold pipeline first).
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any


def run(
    data_dir: str = "data",
    model_name: str = "xgboost",
    tune: bool = False,
    cv_folds: int = 5,
    test_split: float = 0.2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Train ML model on Manifold features.

    Args:
        data_dir: Directory containing ml_features.parquet
        model_name: Model framework (xgboost, catboost, lightgbm, randomforest, gradientboosting)
        tune: Run hyperparameter tuning
        cv_folds: Cross-validation folds
        test_split: Test set fraction
        verbose: Print progress

    Returns:
        Dict with training metrics and model info
    """
    if verbose:
        print("=" * 70)
        print(f"13: TRAIN - {model_name.upper()}")
        print("=" * 70)

    # Defer import — requires manifold package for data paths
    try:
        from prime.ml.entry_points.train import get_model, tune_model, compute_feature_importance
    except ImportError as e:
        raise ImportError(
            f"ML training requires the manifold package: {e}\n"
            "Install manifold or run from the manifold environment."
        ) from e

    import numpy as np
    import polars as pl
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import pickle

    features_path = Path(data_dir) / "ml_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features not found: {features_path}\n"
            "Run Manifold feature extraction first."
        )

    df = pl.read_parquet(features_path)
    if verbose:
        print(f"  Features: {len(df)} samples, {len(df.columns)} columns")

    # Separate features and target
    target_col = "rul" if "rul" in df.columns else df.columns[-1]
    feature_cols = [c for c in df.columns if c != target_col and c != "unit_id"]

    X = df.select(feature_cols).to_numpy()
    y = df[target_col].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_split, random_state=42
    )

    if verbose:
        print(f"  Target: {target_col}")
        print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

    # Get model
    if tune:
        if verbose:
            print(f"  Tuning {model_name}...")
        model = tune_model(model_name, X_train, y_train)
    else:
        model = get_model(model_name)
        model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred)),
    }

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2')
    metrics["cv_r2_mean"] = float(np.mean(cv_scores))
    metrics["cv_r2_std"] = float(np.std(cv_scores))

    if verbose:
        print(f"\n  Results:")
        print(f"    RMSE: {metrics['rmse']:.4f}")
        print(f"    MAE:  {metrics['mae']:.4f}")
        print(f"    R²:   {metrics['r2']:.4f}")
        print(f"    CV R²: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']:.4f}")

    # Save model
    model_path = Path(data_dir) / "ml_model.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    if verbose:
        print(f"\n  Model saved: {model_path}")

    return {"model": model_name, "metrics": metrics}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="13: Train ML Model")
    parser.add_argument('--data-dir', default='data', help='Data directory')
    parser.add_argument('--model', default='xgboost',
                        choices=['xgboost', 'catboost', 'lightgbm',
                                 'randomforest', 'gradientboosting'],
                        help='Model framework')
    parser.add_argument('--tune', action='store_true', help='Hyperparameter tuning')
    parser.add_argument('--cv', type=int, default=5, help='CV folds')
    parser.add_argument('--split', type=float, default=0.2, help='Test split')
    parser.add_argument('--quiet', '-q', action='store_true')

    args = parser.parse_args()

    results = run(
        data_dir=args.data_dir,
        model_name=args.model,
        tune=args.tune,
        cv_folds=args.cv,
        test_split=args.split,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\n" + json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
