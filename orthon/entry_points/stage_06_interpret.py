"""
06: Interpret Entry Point
=========================

Pure orchestration - calls interpreters on PRISM outputs.
Runs dynamics and physics interpretation on PRISM results.

Stages: PRISM output dir → interpretation report

PRISM computes numbers. ORTHON classifies.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from orthon.services.dynamics_interpreter import (
    DynamicsInterpreter,
    StabilityDiagnosis,
    generate_stability_story,
)
from orthon.services.physics_interpreter import (
    PhysicsInterpreter,
    SystemDiagnosis,
)


def run(
    prism_dir: str,
    unit: Optional[str] = None,
    mode: str = "both",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run interpretation on PRISM outputs.

    Args:
        prism_dir: Path to PRISM output directory
        unit: Specific unit/entity to analyze (or all)
        mode: 'dynamics', 'physics', or 'both'
        verbose: Print progress

    Returns:
        Dict with interpretation results
    """
    if verbose:
        print("=" * 70)
        print("06: INTERPRET - PRISM Output Analysis")
        print("=" * 70)

    prism_dir = Path(prism_dir)
    results = {}

    # Dynamics interpretation
    if mode in ("dynamics", "both"):
        if verbose:
            print("\nDynamics interpretation...")

        interp = DynamicsInterpreter(prism_output=str(prism_dir))

        if unit:
            diagnosis = interp.analyze_stability(unit)
            results["dynamics"] = {
                "unit": unit,
                "stability_class": diagnosis.stability_class,
                "basin_score": diagnosis.basin_score,
                "lyapunov_max": diagnosis.lyapunov_max,
                "regime": diagnosis.regime,
                "birth_grade": diagnosis.birth_grade,
                "prognosis": diagnosis.prognosis,
            }
            if verbose:
                print(f"  {unit}: {diagnosis.stability_class} "
                      f"(basin={diagnosis.basin_score:.2f})")
                print(f"  Prognosis: {diagnosis.prognosis}")
        else:
            summary = interp.get_fleet_stability_summary()
            results["dynamics"] = summary
            if verbose:
                print(f"  Fleet summary: {len(summary.get('units', {}))} units")

    # Physics interpretation
    if mode in ("physics", "both"):
        if verbose:
            print("\nPhysics interpretation...")

        interp = PhysicsInterpreter(
            physics_path=prism_dir / "physics.parquet"
            if (prism_dir / "physics.parquet").exists() else None,
        )

        if unit:
            analysis = interp.analyze_system(unit)
            results["physics"] = analysis
            if verbose:
                diag = analysis.get("diagnosis", {})
                print(f"  {unit}: severity={diag.get('severity', 'unknown')}")
        else:
            fleet = interp.analyze_fleet()
            results["physics"] = fleet
            if verbose:
                print(f"  Fleet analysis complete")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="06: Interpret PRISM Outputs")
    parser.add_argument('prism_dir', help='Path to PRISM output directory')
    parser.add_argument('--unit', '-u', help='Specific unit to analyze')
    parser.add_argument('--mode', choices=['dynamics', 'physics', 'both'],
                        default='both', help='Interpretation mode')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    results = run(
        args.prism_dir,
        unit=args.unit,
        mode=args.mode,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
