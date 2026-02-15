"""
07: Predict Entry Point
========================

Pure orchestration - calls prediction modules.
Supports RUL prediction, health scoring, and anomaly detection.

Stages: Manifold output dir → predictions

Requires Manifold outputs (signal_vector, state_vector, etc.)
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from prime.prediction.rul import RULPredictor
from prime.prediction.health import HealthScorer
from prime.prediction.anomaly import AnomalyDetector, AnomalyMethod


def run(
    manifold_dir: str,
    mode: str = "health",
    unit: Optional[str] = None,
    threshold: float = 0.8,
    method: str = "percentile",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run prediction on Manifold outputs.

    Args:
        manifold_dir: Path to Manifold output directory
        mode: 'rul', 'health', or 'anomaly'
        unit: Specific unit to predict (or all)
        threshold: Failure threshold for RUL
        method: Anomaly detection method (percentile, isolation_forest, lof, combined)
        verbose: Print progress

    Returns:
        Dict with prediction results
    """
    if verbose:
        print("=" * 70)
        print(f"07: PREDICT - {mode.upper()}")
        print("=" * 70)

    if mode == "rul":
        predictor = RULPredictor(manifold_dir, failure_threshold=threshold)
        result = predictor.predict(unit)
    elif mode == "health":
        scorer = HealthScorer(manifold_dir)
        result = scorer.predict(unit)
    elif mode == "anomaly":
        detector = AnomalyDetector(manifold_dir, method=AnomalyMethod(method))
        result = detector.predict(unit)
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'rul', 'health', or 'anomaly'.")

    output = result.to_dict()

    if verbose:
        print(json.dumps(output, indent=2, default=str))

    return output


def main():
    import argparse

    parser = argparse.ArgumentParser(description="07: Predict (RUL/Health/Anomaly)")
    parser.add_argument('manifold_dir', help='Path to Manifold output directory')
    parser.add_argument('--mode', choices=['rul', 'health', 'anomaly'],
                        default='health', help='Prediction mode')
    parser.add_argument('--unit', '-u', help='Specific unit to predict')
    parser.add_argument('--threshold', type=float, default=0.8,
                        help='Failure threshold for RUL')
    parser.add_argument('--method', default='percentile',
                        choices=['percentile', 'isolation_forest', 'lof', 'combined'],
                        help='Anomaly detection method')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.manifold_dir,
        mode=args.mode,
        unit=args.unit,
        threshold=args.threshold,
        method=args.method,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
