"""
RUDDER Early Warning Module

Early failure fingerprint detection - identifying engines that will fail differently
before degradation patterns become apparent.
"""

from .failure_fingerprint_detector import (
    RiskLevel,
    FailureMode,
    EarlyLifeFingerprint,
    EarlyFailurePredictor,
    FailurePopulationAnalyzer,
    SmokingGunReportGenerator,
)

from .ml_predictor import (
    MLPrediction,
    EarlyLifeFeatureExtractor,
    MLFailurePredictor,
    train_and_evaluate,
)

__all__ = [
    # Fingerprint detection
    'RiskLevel',
    'FailureMode',
    'EarlyLifeFingerprint',
    'EarlyFailurePredictor',
    'FailurePopulationAnalyzer',
    'SmokingGunReportGenerator',
    # ML prediction
    'MLPrediction',
    'EarlyLifeFeatureExtractor',
    'MLFailurePredictor',
    'train_and_evaluate',
]
