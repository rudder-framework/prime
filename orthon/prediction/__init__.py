"""
ORTHON Prediction Module.

Provides prediction capabilities built on PRISM-computed features:
- RUL (Remaining Useful Life) prediction - SUPERVISED model
- Health scoring (0-100)
- Anomaly detection

RUL Prediction requires training:
    from orthon.prediction import RULPredictor, create_rul_labels

    # Load features and create labels
    physics = pl.read_parquet('physics.parquet')
    physics = create_rul_labels(physics)

    # Train
    predictor = RULPredictor()
    predictor.fit(physics, physics['RUL'])

    # Evaluate
    metrics = predictor.evaluate(test_data, test_data['RUL'])
    print(f"RMSE: {metrics['rmse']:.2f} cycles")

    # Predict
    predictions = predictor.predict(new_data)

Health and Anomaly detection work without training:
    from orthon.prediction import score_health, detect_anomalies

    health = score_health("/path/to/prism/output")
    anomalies = detect_anomalies("/path/to/prism/output")
"""

from pathlib import Path
from typing import Optional, Union

from .base import BasePredictor, EnsemblePredictor, PredictionResult
from .rul import RULPredictor, create_rul_labels, load_prism_features
from .health import HealthScorer
from .anomaly import AnomalyDetector, AnomalyMethod


def score_health(
    prism_output_dir: Union[str, Path],
    unit_id: Optional[str] = None,
    baseline_mode: str = "first_10_percent",
) -> PredictionResult:
    """
    Compute health score (0-100) from PRISM outputs.

    Args:
        prism_output_dir: Directory containing PRISM output parquets
        unit_id: Specific unit to score (None for all units)
        baseline_mode: How to determine healthy baseline

    Returns:
        PredictionResult with health score(s)

    Example:
        >>> result = score_health("/path/to/prism/output")
        >>> print(f"Health: {result.prediction:.0f}%")
    """
    scorer = HealthScorer(
        prism_output_dir,
        baseline_mode=baseline_mode,
    )
    return scorer.predict(unit_id)


def detect_anomalies(
    prism_output_dir: Union[str, Path],
    unit_id: Optional[str] = None,
    method: str = "zscore",
    threshold: float = 3.0,
) -> PredictionResult:
    """
    Detect anomalies in PRISM features.

    Args:
        prism_output_dir: Directory containing PRISM output parquets
        unit_id: Specific unit to analyze (None for all)
        method: Detection method ("zscore", "isolation_forest", "lof", "combined")
        threshold: Z-score threshold for zscore method

    Returns:
        PredictionResult with anomaly rate and details

    Example:
        >>> result = detect_anomalies("/path/to/prism/output")
        >>> print(f"Anomaly rate: {result.prediction:.1%}")
        >>> indices = result.raw_scores["anomaly_labels"]
    """
    detector = AnomalyDetector(
        prism_output_dir,
        method=method,
        threshold=threshold,
    )
    return detector.predict(unit_id)


# Export all public symbols
__all__ = [
    # Base classes
    "BasePredictor",
    "EnsemblePredictor",
    "PredictionResult",
    # RUL Predictor (supervised - requires training)
    "RULPredictor",
    "create_rul_labels",
    "load_prism_features",
    # Health and Anomaly (unsupervised)
    "HealthScorer",
    "AnomalyDetector",
    "AnomalyMethod",
    # Simple API for unsupervised methods
    "score_health",
    "detect_anomalies",
]
