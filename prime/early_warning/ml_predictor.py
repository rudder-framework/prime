"""
ML-based Early Failure and RUL Prediction

Uses early-life features to predict:
1. Early failure classification (will this engine fail before expected?)
2. Lifecycle regression (how many cycles until failure?)
3. Atypical failure mode detection (does this engine degrade differently?)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score


@dataclass
class MLPrediction:
    """Prediction result for a single engine."""
    engine_id: str

    # Classification
    early_failure_prob: float
    early_failure_pred: bool

    # Regression
    predicted_cycles: float
    prediction_interval: Tuple[float, float]  # 80% confidence

    # Atypical mode
    atypical_score: float
    atypical_pred: bool

    # Risk summary
    risk_score: float  # 0-1 combined risk
    risk_category: str  # 'low', 'medium', 'high', 'critical'


class EarlyLifeFeatureExtractor:
    """Extract ML features from early-life engine data."""

    def __init__(self, early_pct: float = 10.0):
        self.early_pct = early_pct
        self.feature_names_: List[str] = []

    def extract_features(
        self,
        obs_df: pd.DataFrame,
        engine_id: str
    ) -> Dict[str, float]:
        """Extract features for a single engine."""
        engine_data = obs_df[obs_df['cohort'] == engine_id]

        if engine_data.empty:
            return {}

        total_cycles = engine_data['signal_0'].max()
        early_cutoff = int(total_cycles * (self.early_pct / 100.0))

        features = {}
        signals = engine_data['signal_id'].unique()

        for signal in signals:
            sig_data = engine_data[engine_data['signal_id'] == signal].sort_values('signal_0')
            early = sig_data[sig_data['signal_0'] <= early_cutoff]['value'].values

            if len(early) < 5:
                continue

            # Basic statistics
            features[f'{signal}_mean'] = float(np.mean(early))
            features[f'{signal}_std'] = float(np.std(early))
            features[f'{signal}_range'] = float(np.max(early) - np.min(early))

            # Derivatives
            d1 = np.gradient(early)
            d2 = np.gradient(d1)
            features[f'{signal}_d1_mean'] = float(np.mean(d1))
            features[f'{signal}_d1_std'] = float(np.std(d1))
            features[f'{signal}_d2_mean'] = float(np.mean(d2))

            # Trend (linear slope)
            if len(early) >= 3:
                features[f'{signal}_trend'] = float(np.polyfit(range(len(early)), early, 1)[0])

            # Volatility ratio
            mean_abs_d1 = np.abs(np.mean(d1))
            if mean_abs_d1 > 1e-10:
                features[f'{signal}_volatility'] = float(np.std(d1) / mean_abs_d1)

        # Cross-signal interaction features (discovered discriminators)
        self._add_interaction_features(features)

        return features

    def _add_interaction_features(self, features: Dict[str, float]):
        """Add cross-signal interaction features."""
        # sensor_12 vs sensor_08 (smoking gun)
        if 'sensor_12_d1_mean' in features and 'sensor_08_d1_mean' in features:
            s08 = features['sensor_08_d1_mean']
            features['s12_s08_d1_diff'] = features['sensor_12_d1_mean'] - s08
            if abs(s08) > 1e-10:
                features['s12_s08_d1_ratio'] = features['sensor_12_d1_mean'] / s08

        # sensor_11 vs sensor_14
        if 'sensor_11_d1_mean' in features and 'sensor_14_d1_mean' in features:
            s14 = features['sensor_14_d1_mean']
            features['s11_s14_d1_diff'] = features['sensor_11_d1_mean'] - s14
            if abs(s14) > 1e-10:
                features['s11_s14_d1_ratio'] = features['sensor_11_d1_mean'] / s14

        # op2 vs sensor_17 (perfect discriminator)
        if 'op2_d1_mean' in features and 'sensor_17_d1_mean' in features:
            s17 = features['sensor_17_d1_mean']
            features['op2_s17_d1_diff'] = features['op2_d1_mean'] - s17
            if abs(s17) > 1e-10:
                features['op2_s17_d1_ratio'] = features['op2_d1_mean'] / s17

    def extract_batch(self, obs_df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for all engines in the dataset."""
        engines = obs_df['cohort'].unique()

        all_features = []
        for engine in engines:
            features = self.extract_features(obs_df, engine)
            features['engine_id'] = engine
            features['total_cycles'] = obs_df[obs_df['cohort'] == engine]['signal_0'].max()
            all_features.append(features)

        df = pd.DataFrame(all_features)

        # Store feature names (excluding metadata)
        self.feature_names_ = [c for c in df.columns
                               if c not in ['engine_id', 'total_cycles']]

        return df


class MLFailurePredictor:
    """
    ML-based predictor for early failure and RUL.

    Combines:
    - Classification: Will this engine fail early?
    - Regression: How many cycles until failure?
    - Atypical detection: Is this engine degrading unusually?
    """

    def __init__(
        self,
        early_pct: float = 10.0,
        early_failure_threshold: float = 0.3,
        atypical_threshold: float = 2.0,
    ):
        self.early_pct = early_pct
        self.early_failure_threshold = early_failure_threshold
        self.atypical_threshold = atypical_threshold

        self.extractor = EarlyLifeFeatureExtractor(early_pct=early_pct)
        self.scaler = StandardScaler()

        self.classifier = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=42
        )

        self.regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )

        # Population statistics for atypical detection
        self._population_mean: Optional[np.ndarray] = None
        self._population_cov_inv: Optional[np.ndarray] = None
        self._lifecycle_p25: float = 0.0

        self.feature_names_: List[str] = []
        self.is_fitted_: bool = False

    def fit(
        self,
        obs_df: pd.DataFrame,
        early_failure_percentile: float = 25.0
    ) -> 'MLFailurePredictor':
        """
        Fit the predictor on training data.

        Args:
            obs_df: Observations dataframe with [cohort, signal_id, I, value]
            early_failure_percentile: Percentile threshold for "early" failure
        """
        # Extract features
        feature_df = self.extractor.extract_batch(obs_df)
        self.feature_names_ = self.extractor.feature_names_

        # Prepare data
        X = feature_df[self.feature_names_].fillna(0)
        y_cycles = feature_df['total_cycles']

        # Define early failure threshold
        self._lifecycle_p25 = y_cycles.quantile(early_failure_percentile / 100.0)
        y_early = (y_cycles <= self._lifecycle_p25).astype(int)

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Fit classifier
        self.classifier.fit(X_scaled, y_early)

        # Fit regressor
        self.regressor.fit(X_scaled, y_cycles)

        # Compute population statistics for atypical detection (Mahalanobis)
        self._population_mean = np.mean(X_scaled, axis=0)
        cov = np.cov(X_scaled.T)
        # Regularize to avoid singularity
        cov += np.eye(cov.shape[0]) * 1e-6
        self._population_cov_inv = np.linalg.inv(cov)

        self.is_fitted_ = True
        return self

    def predict(
        self,
        obs_df: pd.DataFrame,
        engine_ids: Optional[List[str]] = None
    ) -> List[MLPrediction]:
        """
        Make predictions for engines.

        Args:
            obs_df: Observations dataframe
            engine_ids: Specific engines to predict (if None, all engines)

        Returns:
            List of MLPrediction objects
        """
        if not self.is_fitted_:
            raise RuntimeError("Predictor not fitted. Call fit() first.")

        if engine_ids is None:
            engine_ids = obs_df['cohort'].unique().tolist()

        predictions = []

        for engine_id in engine_ids:
            features = self.extractor.extract_features(obs_df, engine_id)

            if not features:
                continue

            # Create feature vector
            X = np.array([[features.get(f, 0) for f in self.feature_names_]])
            X_scaled = self.scaler.transform(X)

            # Classification
            early_prob = float(self.classifier.predict_proba(X_scaled)[0, 1])
            early_pred = early_prob > self.early_failure_threshold

            # Regression
            pred_cycles = float(self.regressor.predict(X_scaled)[0])

            # Prediction interval (using training residuals as proxy)
            # Simple approximation: +/- 20% for 80% CI
            interval = (pred_cycles * 0.85, pred_cycles * 1.15)

            # Atypical detection (Mahalanobis distance)
            diff = X_scaled[0] - self._population_mean
            mahal_dist = np.sqrt(np.dot(np.dot(diff, self._population_cov_inv), diff))
            atypical_score = float(mahal_dist)
            atypical_pred = atypical_score > self.atypical_threshold

            # Combined risk score
            risk_score = 0.0
            if early_pred:
                risk_score += 0.4 * early_prob
            if atypical_pred:
                risk_score += 0.3 * min(atypical_score / 4.0, 1.0)
            # Add penalty for low predicted cycles
            if pred_cycles < self._lifecycle_p25:
                risk_score += 0.3 * (1 - pred_cycles / self._lifecycle_p25)

            risk_score = min(risk_score, 1.0)

            # Risk category
            if risk_score > 0.7:
                risk_category = 'critical'
            elif risk_score > 0.5:
                risk_category = 'high'
            elif risk_score > 0.3:
                risk_category = 'medium'
            else:
                risk_category = 'low'

            predictions.append(MLPrediction(
                engine_id=engine_id,
                early_failure_prob=early_prob,
                early_failure_pred=early_pred,
                predicted_cycles=pred_cycles,
                prediction_interval=interval,
                atypical_score=atypical_score,
                atypical_pred=atypical_pred,
                risk_score=risk_score,
                risk_category=risk_category,
            ))

        return predictions

    def predict_df(
        self,
        obs_df: pd.DataFrame,
        engine_ids: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Return predictions as a DataFrame."""
        predictions = self.predict(obs_df, engine_ids)

        return pd.DataFrame([
            {
                'engine_id': p.engine_id,
                'early_failure_prob': p.early_failure_prob,
                'early_failure_pred': p.early_failure_pred,
                'predicted_cycles': p.predicted_cycles,
                'pred_interval_low': p.prediction_interval[0],
                'pred_interval_high': p.prediction_interval[1],
                'atypical_score': p.atypical_score,
                'atypical_pred': p.atypical_pred,
                'risk_score': p.risk_score,
                'risk_category': p.risk_category,
            }
            for p in predictions
        ])

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """Get feature importance from classifier and regressor."""
        if not self.is_fitted_:
            raise RuntimeError("Predictor not fitted.")

        clf_imp = self.classifier.feature_importances_
        reg_imp = self.regressor.feature_importances_

        df = pd.DataFrame({
            'feature': self.feature_names_,
            'classifier_importance': clf_imp,
            'regressor_importance': reg_imp,
            'combined': (clf_imp + reg_imp) / 2
        })

        return df.sort_values('combined', ascending=False).head(top_n)

    def evaluate(
        self,
        obs_df: pd.DataFrame,
        test_engines: List[str]
    ) -> Dict[str, float]:
        """
        Evaluate model on test engines.

        Returns metrics dict with classification and regression scores.
        """
        # Get ground truth
        engine_cycles = obs_df.groupby('cohort')['signal_0'].max()

        y_true_cycles = [engine_cycles[e] for e in test_engines]
        y_true_early = [1 if c <= self._lifecycle_p25 else 0 for c in y_true_cycles]

        # Get predictions
        preds = self.predict(obs_df, test_engines)

        y_pred_prob = [p.early_failure_prob for p in preds]
        y_pred_early = [int(p.early_failure_pred) for p in preds]
        y_pred_cycles = [p.predicted_cycles for p in preds]

        return {
            'roc_auc': roc_auc_score(y_true_early, y_pred_prob) if sum(y_true_early) > 0 else 0.0,
            'mae': mean_absolute_error(y_true_cycles, y_pred_cycles),
            'r2': r2_score(y_true_cycles, y_pred_cycles),
            'n_test': len(test_engines),
            'n_early_true': sum(y_true_early),
            'n_early_pred': sum(y_pred_early),
        }


def train_and_evaluate(
    obs_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[MLFailurePredictor, Dict[str, float], pd.DataFrame]:
    """
    Convenience function to train and evaluate a predictor.

    Returns:
        predictor: Fitted MLFailurePredictor
        metrics: Evaluation metrics
        predictions: DataFrame of test predictions
    """
    from sklearn.model_selection import train_test_split

    # Split engines - convert to list for sklearn compatibility
    engines = list(obs_df['cohort'].unique())
    engine_cycles = obs_df.groupby('cohort')['signal_0'].max()

    # Stratify by lifecycle quartile - must align with engines list
    quartiles = [pd.qcut(engine_cycles, 4, labels=False)[e] for e in engines]

    train_engines, test_engines = train_test_split(
        engines,
        test_size=test_size,
        random_state=random_state,
        stratify=quartiles
    )

    # Filter to training data
    train_df = obs_df[obs_df['cohort'].isin(train_engines)]

    # Fit predictor
    predictor = MLFailurePredictor()
    predictor.fit(train_df)

    # Evaluate on test
    metrics = predictor.evaluate(obs_df, test_engines)

    # Get test predictions with ground truth
    predictions = predictor.predict_df(obs_df, test_engines)
    predictions['actual_cycles'] = [engine_cycles[e] for e in predictions['engine_id']]
    predictions['error'] = predictions['predicted_cycles'] - predictions['actual_cycles']
    predictions['actual_early'] = predictions['actual_cycles'] <= predictor._lifecycle_p25

    return predictor, metrics, predictions
