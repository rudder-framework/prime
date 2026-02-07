"""
08: Alert Entry Point
======================

Pure orchestration - calls early warning modules.
Detects early failure fingerprints and risk scoring.

Stages: observations.parquet → early warning predictions

Uses ML-based failure prediction and fingerprint detection.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any, List

from orthon.early_warning import (
    MLFailurePredictor,
    EarlyFailurePredictor,
    FailurePopulationAnalyzer,
    train_and_evaluate,
)


def run(
    observations_path: str,
    mode: str = "predict",
    physics_path: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run early warning analysis.

    Args:
        observations_path: Path to observations.parquet
        mode: 'predict' (ML), 'fingerprint' (heuristic), or 'train' (fit model)
        physics_path: Path to physics.parquet (for fingerprint mode)
        verbose: Print progress

    Returns:
        Dict with early warning results
    """
    if verbose:
        print("=" * 70)
        print(f"08: ALERT - Early Warning ({mode})")
        print("=" * 70)

    obs_df = pd.read_parquet(observations_path)

    if mode == "predict":
        predictor = MLFailurePredictor()
        predictor.fit(obs_df)
        predictions = predictor.predict_df(obs_df)

        if verbose:
            n_at_risk = len(predictions[predictions['risk_level'] != 'low']) if 'risk_level' in predictions.columns else 0
            print(f"  Analyzed: {len(predictions)} entities")
            print(f"  At risk: {n_at_risk}")

        return {"predictions": predictions.to_dict(orient='records')}

    elif mode == "fingerprint":
        if not physics_path:
            raise ValueError("fingerprint mode requires --physics-path")
        predictor = EarlyFailurePredictor(physics_path, observations_path)
        at_risk = predictor.predict_early_failures()
        analyzer = FailurePopulationAnalyzer(physics_path, observations_path)
        populations = analyzer.analyze_populations()

        if verbose:
            print(f"  At-risk entities: {len(at_risk)}")
            for entity in at_risk[:5]:
                print(f"    - {entity}")

        return {"at_risk": at_risk, "populations": populations}

    elif mode == "train":
        predictor, metrics, results_df = train_and_evaluate(obs_df)

        if verbose:
            print(f"  Training metrics:")
            for k, v in metrics.items():
                print(f"    {k}: {v:.4f}")

        return {"metrics": metrics}

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'predict', 'fingerprint', or 'train'.")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="08: Early Warning / Alert")
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--mode', choices=['predict', 'fingerprint', 'train'],
                        default='predict', help='Alert mode')
    parser.add_argument('--physics-path', help='Path to physics.parquet (for fingerprint)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    results = run(
        args.observations,
        mode=args.mode,
        physics_path=args.physics_path,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
