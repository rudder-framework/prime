"""
06: Interpret Entry Point
=========================

Pure orchestration - calls interpreters on Manifold outputs.
Runs dynamics and physics interpretation on Manifold results.

Stages: Manifold output dir → interpretation report

Manifold computes numbers. Prime classifies.
"""

import json
from pathlib import Path
from typing import Optional, Dict, Any

from prime.services.dynamics_interpreter import (
    DynamicsInterpreter,
    StabilityDiagnosis,
    generate_stability_story,
)
from prime.services.physics_interpreter import (
    PhysicsInterpreter,
    SystemDiagnosis,
)
from prime.io.readme_writer import generate_manifold_readmes


def run(
    manifold_dir: str,
    unit: Optional[str] = None,
    mode: str = "both",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Run interpretation on Manifold outputs.

    Args:
        manifold_dir: Path to Manifold output directory
        unit: Specific unit/entity to analyze (or all)
        mode: 'dynamics', 'physics', or 'both'
        verbose: Print progress

    Returns:
        Dict with interpretation results
    """
    if verbose:
        print("=" * 70)
        print("06: INTERPRET - Manifold Output Analysis")
        print("=" * 70)

    manifold_dir = Path(manifold_dir)
    results = {}

    # Dynamics interpretation
    if mode in ("dynamics", "both"):
        if verbose:
            print("\nDynamics interpretation...")

        interp = DynamicsInterpreter(manifold_output=str(manifold_dir))

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

    # Generate Manifold output READMEs
    try:
        generate_manifold_readmes(manifold_dir)
        if verbose:
            print("\nManifold output READMEs generated.")
    except Exception:
        pass  # README generation is non-critical

    # Physics interpretation
    if mode in ("physics", "both"):
        if verbose:
            print("\nPhysics interpretation...")

        interp = PhysicsInterpreter(
            physics_path=manifold_dir / "physics.parquet"
            if (manifold_dir / "physics.parquet").exists() else None,
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

    parser = argparse.ArgumentParser(description="06: Interpret Manifold Outputs")
    parser.add_argument('manifold_dir', help='Path to Manifold output directory')
    parser.add_argument('--unit', '-u', help='Specific unit to analyze')
    parser.add_argument('--mode', choices=['dynamics', 'physics', 'both'],
                        default='both', help='Interpretation mode')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    results = run(
        args.manifold_dir,
        unit=args.unit,
        mode=args.mode,
        verbose=not args.quiet,
    )

    if not args.quiet:
        print("\n" + json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    main()
