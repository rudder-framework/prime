"""
Remaining Useful Life (RUL) Predictor.

SUPERVISED model that learns RUL from PRISM features.
Requires training data with known RUL labels.
"""

from pathlib import Path
from typing import Any, Optional, Union
import pickle

import numpy as np
import polars as pl

from .base import BasePredictor, PredictionResult


class RULPredictor:
    """
    Supervised RUL predictor using PRISM features.

    This is a TRAINABLE model, not a heuristic.
    Requires:
    - Training data with PRISM features
    - RUL labels (e.g., max_cycle - current_cycle for run-to-failure)

    Usage:
        # Train
        predictor = RULPredictor()
        predictor.fit(train_features, train_rul)

        # Evaluate
        metrics = predictor.evaluate(test_features, test_rul)
        print(f"RMSE: {metrics['rmse']:.2f}")

        # Predict
        predictions = predictor.predict(new_features)
    """

    # Default feature columns from PRISM outputs
    DEFAULT_FEATURES = [
        # Physics features (from physics.parquet)
        "effective_dim",
        "state_velocity",
        "entropy",
        "free_energy",
        # Primitives features
        "rms",
        "kurtosis",
        "skewness",
        "crest_factor",
        "hurst",
        "sample_entropy",
        # Dynamics features
        "lyapunov_exponent",
        "correlation_dim",
        "recurrence_rate",
        "determinism",
    ]

    def __init__(
        self,
        model_type: str = "random_forest",
        feature_cols: Optional[list[str]] = None,
    ):
        """
        Initialize RUL predictor.

        Args:
            model_type: "random_forest", "gradient_boosting", or "linear"
            feature_cols: Feature columns to use (default: DEFAULT_FEATURES)
        """
        self.model_type = model_type
        self.feature_cols = feature_cols or self.DEFAULT_FEATURES
        self.model = None
        self.scaler = None
        self._fitted = False
        self._feature_importance: dict[str, float] = {}
        self._train_metrics: dict[str, float] = {}

    def _create_model(self):
        """Create the underlying ML model."""
        if self.model_type == "random_forest":
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            )
        elif self.model_type == "gradient_boosting":
            from sklearn.ensemble import GradientBoostingRegressor
            return GradientBoostingRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
            )
        elif self.model_type == "linear":
            from sklearn.linear_model import Ridge
            return Ridge(alpha=1.0)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _prepare_features(
        self,
        data: Union[pl.DataFrame, np.ndarray],
        fit_scaler: bool = False,
    ) -> np.ndarray:
        """
        Prepare feature matrix from data.

        Args:
            data: DataFrame with feature columns, or numpy array
            fit_scaler: Whether to fit the scaler (True for training)

        Returns:
            Numpy array of features
        """
        if isinstance(data, np.ndarray):
            X = data
        else:
            # Select available feature columns
            available = [c for c in self.feature_cols if c in data.columns]
            if not available:
                raise ValueError(
                    f"No feature columns found. Expected: {self.feature_cols[:5]}..."
                )
            X = data.select(available).to_numpy()

        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        # Scale features
        from sklearn.preprocessing import StandardScaler
        if fit_scaler:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            X = self.scaler.transform(X)

        return X

    def fit(
        self,
        features: Union[pl.DataFrame, np.ndarray],
        target: Union[pl.Series, np.ndarray],
        validation_split: float = 0.2,
    ) -> "RULPredictor":
        """
        Train the RUL predictor.

        Args:
            features: DataFrame with feature columns, or numpy array
            target: RUL values (cycles remaining)
            validation_split: Fraction of data for validation

        Returns:
            self (for chaining)
        """
        # Prepare features
        X = self._prepare_features(features, fit_scaler=True)

        # Prepare target
        if isinstance(target, pl.Series):
            y = target.to_numpy()
        else:
            y = np.asarray(target)

        # Handle NaN in target
        valid_mask = ~np.isnan(y) & ~np.isinf(y)
        X = X[valid_mask]
        y = y[valid_mask]

        if len(y) < 100:
            raise ValueError(f"Insufficient training data: {len(y)} samples")

        # Split for validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )

        # Create and fit model
        self.model = self._create_model()
        self.model.fit(X_train, y_train)
        self._fitted = True

        # Compute training metrics
        y_train_pred = self.model.predict(X_train)
        y_val_pred = self.model.predict(X_val)

        self._train_metrics = {
            "train_rmse": float(np.sqrt(np.mean((y_train - y_train_pred) ** 2))),
            "train_r2": float(1 - np.sum((y_train - y_train_pred) ** 2) / np.sum((y_train - np.mean(y_train)) ** 2)),
            "val_rmse": float(np.sqrt(np.mean((y_val - y_val_pred) ** 2))),
            "val_r2": float(1 - np.sum((y_val - y_val_pred) ** 2) / np.sum((y_val - np.mean(y_val)) ** 2)),
            "n_train": len(y_train),
            "n_val": len(y_val),
        }

        # Feature importance
        if hasattr(self.model, "feature_importances_"):
            if isinstance(features, pl.DataFrame):
                available = [c for c in self.feature_cols if c in features.columns]
                self._feature_importance = dict(
                    zip(available, self.model.feature_importances_)
                )
            else:
                self._feature_importance = dict(
                    zip(range(X.shape[1]), self.model.feature_importances_)
                )

        return self

    def predict(
        self,
        features: Union[pl.DataFrame, np.ndarray],
    ) -> np.ndarray:
        """
        Predict RUL for new data.

        Args:
            features: DataFrame with feature columns, or numpy array

        Returns:
            Predicted RUL values
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X = self._prepare_features(features, fit_scaler=False)
        predictions = self.model.predict(X)

        # Clip to non-negative
        predictions = np.maximum(predictions, 0)

        return predictions

    def evaluate(
        self,
        features: Union[pl.DataFrame, np.ndarray],
        target: Union[pl.Series, np.ndarray],
    ) -> dict[str, float]:
        """
        Evaluate model on test data.

        Args:
            features: Test features
            target: True RUL values

        Returns:
            Dictionary with RMSE, MAE, R², and scoring metrics
        """
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        predictions = self.predict(features)

        if isinstance(target, pl.Series):
            y_true = target.to_numpy()
        else:
            y_true = np.asarray(target)

        # Handle NaN
        valid_mask = ~np.isnan(y_true) & ~np.isinf(y_true)
        y_true = y_true[valid_mask]
        predictions = predictions[valid_mask]

        # Compute metrics
        rmse = float(np.sqrt(np.mean((y_true - predictions) ** 2)))
        mae = float(np.mean(np.abs(y_true - predictions)))
        ss_res = np.sum((y_true - predictions) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # Scoring function (NASA-style)
        # s_i = exp(-d_i/13) - 1 if d_i < 0 (early prediction)
        # s_i = exp(d_i/10) - 1 if d_i >= 0 (late prediction)
        d = predictions - y_true
        s = np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)
        score = float(np.sum(s))

        return {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "nasa_score": score,
            "n_samples": len(y_true),
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained model."""
        return self._feature_importance.copy()

    def get_train_metrics(self) -> dict[str, float]:
        """Get metrics from training."""
        return self._train_metrics.copy()

    def save(self, path: Union[str, Path]) -> None:
        """Save trained model to file."""
        if not self._fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.model,
                "scaler": self.scaler,
                "model_type": self.model_type,
                "feature_cols": self.feature_cols,
                "feature_importance": self._feature_importance,
                "train_metrics": self._train_metrics,
            }, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "RULPredictor":
        """Load trained model from file."""
        path = Path(path)
        with open(path, "rb") as f:
            data = pickle.load(f)

        predictor = cls(
            model_type=data["model_type"],
            feature_cols=data["feature_cols"],
        )
        predictor.model = data["model"]
        predictor.scaler = data["scaler"]
        predictor._feature_importance = data["feature_importance"]
        predictor._train_metrics = data["train_metrics"]
        predictor._fitted = True

        return predictor


def create_rul_labels(
    data: pl.DataFrame,
    unit_col: str = "unit_id",
    cycle_col: str = "I",
) -> pl.DataFrame:
    """
    Create RUL labels for run-to-failure data.

    For each unit, RUL = max_cycle - current_cycle.

    Args:
        data: DataFrame with unit and cycle columns
        unit_col: Column name for unit identifier
        cycle_col: Column name for cycle/time index

    Returns:
        DataFrame with added "RUL" column
    """
    # Get max cycle per unit
    max_cycles = data.group_by(unit_col).agg(
        pl.col(cycle_col).max().alias("max_cycle")
    )

    # Join and compute RUL
    result = data.join(max_cycles, on=unit_col)
    result = result.with_columns(
        (pl.col("max_cycle") - pl.col(cycle_col)).alias("RUL")
    )
    result = result.drop("max_cycle")

    return result


def load_prism_features(
    prism_dir: Union[str, Path],
    feature_sources: list[str] = ["physics", "primitives", "dynamics"],
) -> pl.DataFrame:
    """
    Load and merge PRISM feature parquets.

    Args:
        prism_dir: Directory containing PRISM outputs
        feature_sources: Which parquet files to load

    Returns:
        Merged DataFrame with all features
    """
    prism_dir = Path(prism_dir)
    dfs = []

    for source in feature_sources:
        path = prism_dir / f"{source}.parquet"
        if path.exists():
            df = pl.read_parquet(path)
            dfs.append((source, df))

    if not dfs:
        raise FileNotFoundError(f"No feature files found in {prism_dir}")

    # Start with first dataframe
    result = dfs[0][1]

    # Merge others
    for name, df in dfs[1:]:
        # Find common columns for join
        common = set(result.columns) & set(df.columns)
        join_cols = [c for c in ["unit_id", "signal_id", "I"] if c in common]

        if join_cols:
            # Get non-join columns from df
            new_cols = [c for c in df.columns if c not in result.columns]
            if new_cols:
                result = result.join(
                    df.select(join_cols + new_cols),
                    on=join_cols,
                    how="left",
                )

    return result
