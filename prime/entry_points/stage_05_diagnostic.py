"""
05: Diagnostic Entry Point
==========================

Pure orchestration - calls Prime engines for diagnostic assessment.
Runs the full diagnostic pipeline: Typology → Spin Glass.

Stages: observations.parquet → diagnostic_report.txt

Uses engines from prime/engines/:
- typology_engine.py (Level 0)
- stationarity_engine.py (Level 1)
- classification_engine.py (Level 2)
- signal_geometry.py
- mass_engine.py
- structure_engine.py
- stability_engine.py
- tipping_engine.py
- spin_glass.py
"""

import numpy as np
import polars as pl
from pathlib import Path
from typing import Optional

# Import diagnostic from engines
from prime.engines.diagnostic_report import (
    run_diagnostic,
    generate_report,
    DiagnosticResult,
)


def run(
    observations_path: str,
    output_path: Optional[str] = None,
    domain: str = "general",
    window_size: int = 100,
    verbose: bool = True,
) -> DiagnosticResult:
    """
    Run full Prime diagnostic on observations.

    Args:
        observations_path: Path to observations.parquet
        output_path: Output path for report (optional)
        domain: Domain identifier for interpretation
        window_size: Window size for rolling analysis
        verbose: Print progress

    Returns:
        DiagnosticResult with complete assessment
    """
    if verbose:
        print("=" * 70)
        print("05: DIAGNOSTIC ASSESSMENT")
        print("=" * 70)

    # Load data
    df = pl.read_parquet(observations_path)
    if verbose:
        n_signals = df['signal_id'].n_unique() if 'signal_id' in df.columns else 0
        print(f"Loaded: {len(df):,} rows, {n_signals} signals")

    # Pivot to matrix
    if 'signal_id' in df.columns and 'value' in df.columns and 'signal_0' in df.columns:
        matrix = df.pivot(on='signal_id', index='signal_0', values='value')
        signals = [c for c in matrix.columns if c != 'signal_0']
        X = matrix.select(signals).to_numpy()
    else:
        raise ValueError("Expected columns: signal_id, value, signal_0")

    if verbose:
        print(f"Signal matrix: {X.shape[0]} timepoints × {X.shape[1]} signals")
        print(f"Window size: {window_size}")
        print(f"Domain: {domain}")

    # Run diagnostic
    if verbose:
        print("\nRunning diagnostic...")

    result = run_diagnostic(
        X,
        domain=domain,
        window_size=window_size,
    )

    # Generate report
    report = generate_report(result)

    if verbose:
        print(report)

    # Save report
    if output_path:
        with open(output_path, 'w') as f:
            f.write(report)
        if verbose:
            print(f"\nSaved: {output_path}")

    return result


def main():
    import argparse

    parser = argparse.ArgumentParser(description="05: Prime Diagnostic")
    parser.add_argument('observations', help='Path to observations.parquet')
    parser.add_argument('--output', '-o', help='Output path for report')
    parser.add_argument('--domain', default='general', help='Domain for interpretation')
    parser.add_argument('--window-size', type=int, default=100, help='Window size')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress output')

    args = parser.parse_args()

    run(
        args.observations,
        output_path=args.output,
        domain=args.domain,
        window_size=args.window_size,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
