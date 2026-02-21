"""
00: Regime Normalization Entry Point
====================================

Subtracts fleet-wide per-regime per-sensor means from observations.
Writes a self-contained normalized/ directory that can be passed
directly to the standard pipeline.

Usage:
    python -m prime.entry_points.stage_00_regime_normalize \
        ~/domains/cmapss/FD_004/train/observations.parquet \
        --regime-signals op1 op2 op3
"""

from prime.ingest.regime_normalize import regime_normalize


def run(
    observations_path: str,
    regime_signals: list[str],
    sensor_signals: list[str] | None = None,
    output_dir: str | None = None,
    min_observations: int = 10,
    verbose: bool = True,
):
    """
    Run regime normalization.

    Args:
        observations_path: Path to observations.parquet.
        regime_signals: Signal IDs defining operating regimes.
        sensor_signals: Signal IDs to normalize (default: all non-regime signals).
        output_dir: Output directory (default: {parent}/normalized/).
        min_observations: Min observations per (sensor, regime) before fallback.
        verbose: Print progress.

    Returns:
        RegimeNormalizationResult.
    """
    return regime_normalize(
        observations_path=observations_path,
        output_dir=output_dir,
        regime_signals=regime_signals,
        sensor_signals=sensor_signals,
        min_observations=min_observations,
        verbose=verbose,
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="00: Regime Normalization — subtract fleet-wide regime means"
    )
    parser.add_argument(
        "observations",
        help="Path to observations.parquet",
    )
    parser.add_argument(
        "--regime-signals",
        nargs="+",
        required=True,
        help="Signal IDs that define operating regimes (e.g., op1 op2 op3)",
    )
    parser.add_argument(
        "--sensor-signals",
        nargs="+",
        default=None,
        help="Signal IDs to normalize (default: all non-regime signals)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory (default: {parent}/normalized/)",
    )
    parser.add_argument(
        "--min-observations",
        type=int,
        default=10,
        help="Min observations per (sensor, regime) group (default: 10)",
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    run(
        observations_path=args.observations,
        regime_signals=args.regime_signals,
        sensor_signals=args.sensor_signals,
        output_dir=args.output_dir,
        min_observations=args.min_observations,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
