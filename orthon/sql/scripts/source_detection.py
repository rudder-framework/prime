"""
Propagation Source Detection

If one area is always primary, it will:
1. Lead all other signals (positive lag)
2. Granger-cause all others
3. Show consistent directionality

Usage:
    cd /path/to/domain
    python /path/to/source_detection.py

Outputs:
    source_detection.parquet - Source ranking and propagation chain
"""

import polars as pl
import numpy as np
from pathlib import Path


def find_peak_lag(x: np.ndarray, y: np.ndarray, max_lag: int = 50):
    """
    Find lag where correlation is maximized.

    Args:
        x: First signal
        y: Second signal
        max_lag: Maximum lag to search

    Returns:
        best_lag: Lag with highest |correlation|
        best_corr: Correlation at that lag
    """
    best_lag = 0
    best_corr = 0

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]

        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_lag = lag

    return best_lag, best_corr


def detect_sources(
    observations_path: str = 'observations.parquet',
    output_path: str = 'source_detection.parquet',
    max_lag: int = 50,
    lead_threshold: int = 2,
    verbose: bool = True,
):
    """
    Run propagation source detection on observations.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path for output parquet
        max_lag: Maximum lag to search
        lead_threshold: Minimum lag to count as "leading"
        verbose: Print progress
    """
    # Load data
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded {len(obs):,} observations")

    # Pivot to wide
    wide = obs.pivot(values='value', index='I', on='signal_id').sort('I')
    signals = [c for c in wide.columns if c != 'I']
    data = {s: wide[s].to_numpy() for s in signals}

    if verbose:
        print(f"Signals: {signals}")

    # ============================================================
    # LEAD-LAG MATRIX
    # ============================================================

    if verbose:
        print("\nLEAD-LAG MATRIX")
        print("(Positive = row leads column)")
        print()

        # Header
        print(f"{'':20}", end='')
        for s in signals:
            print(f"{s[:15]:>16}", end='')
        print()

    lead_counts = {s: 0 for s in signals}
    lag_matrix = {}

    for sig_a in signals:
        if verbose:
            print(f"{sig_a:20}", end='')
        lag_matrix[sig_a] = {}

        for sig_b in signals:
            if sig_a == sig_b:
                if verbose:
                    print(f"{'---':>16}", end='')
                lag_matrix[sig_a][sig_b] = {'lag': 0, 'corr': 1.0}
            else:
                lag, corr = find_peak_lag(data[sig_a], data[sig_b], max_lag)
                if verbose:
                    print(f"{lag:>8} ({corr:+.2f})", end='')
                lag_matrix[sig_a][sig_b] = {'lag': lag, 'corr': corr}

                # Count leads
                if lag > lead_threshold:  # A leads B
                    lead_counts[sig_a] += 1
                elif lag < -lead_threshold:  # B leads A
                    lead_counts[sig_b] += 1

        if verbose:
            print()

    if verbose:
        print()
        print("LEAD COUNTS (how many signals does each lead?):")
        for s, count in sorted(lead_counts.items(), key=lambda x: -x[1]):
            print(f"  {s}: leads {count} other signals")

    # ============================================================
    # CONSISTENT DIRECTIONALITY TEST
    # ============================================================

    if verbose:
        print()
        print("="*60)
        print("CONSISTENT DIRECTIONALITY TEST")
        print("="*60)

    directionality = {}

    for sig in signals:
        lags_vs_others = []
        for other in signals:
            if sig != other:
                lag, _ = find_peak_lag(data[sig], data[other], max_lag)
                lags_vs_others.append(lag)

        # Check consistency
        all_positive = all(l > lead_threshold for l in lags_vs_others)
        all_negative = all(l < -lead_threshold for l in lags_vs_others)
        mean_lag = np.mean(lags_vs_others)

        if all_positive:
            direction = "ALWAYS_LEADS"
            if verbose:
                print(f"{sig}: ALWAYS LEADS (primary source candidate)")
        elif all_negative:
            direction = "ALWAYS_FOLLOWS"
            if verbose:
                print(f"{sig}: ALWAYS FOLLOWS (end of chain)")
        else:
            direction = "MIXED"
            if verbose:
                print(f"{sig}: MIXED (mean lag: {mean_lag:+.1f})")

        directionality[sig] = {
            'direction': direction,
            'mean_lag': mean_lag,
            'all_positive': all_positive,
            'all_negative': all_negative,
        }

    # ============================================================
    # PROPAGATION CHAIN INFERENCE
    # ============================================================

    if verbose:
        print()
        print("="*60)
        print("INFERRED PROPAGATION CHAIN")
        print("="*60)

    # Rank by mean lead time
    mean_leads = {}
    for sig in signals:
        lags = []
        for other in signals:
            if sig != other:
                lag, _ = find_peak_lag(data[sig], data[other], max_lag)
                lags.append(lag)
        mean_leads[sig] = np.mean(lags)

    # Sort by mean lead (most positive = earliest in chain)
    chain = sorted(mean_leads.items(), key=lambda x: -x[1])

    if verbose:
        print("\nPropagation order (earliest to latest):")
        for i, (sig, mean_lag) in enumerate(chain):
            arrow = "→" if i < len(chain) - 1 else ""
            print(f"  {i+1}. {sig} (mean lead: {mean_lag:+.1f} samples) {arrow}")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results = []

    for i, (sig, mean_lag) in enumerate(chain):
        results.append({
            'signal_id': sig,
            'chain_position': i + 1,
            'mean_lead_lag': mean_lag,
            'lead_count': lead_counts[sig],
            'directionality': directionality[sig]['direction'],
            'is_primary_candidate': directionality[sig]['all_positive'],
            'is_end_of_chain': directionality[sig]['all_negative'],
        })

    df = pl.DataFrame(results)
    df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")

    return df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Propagation source detection'
    )
    parser.add_argument(
        '--observations', '-i',
        default='observations.parquet',
        help='Input observations.parquet path'
    )
    parser.add_argument(
        '--output', '-o',
        default='source_detection.parquet',
        help='Output parquet path'
    )
    parser.add_argument(
        '--max-lag', '-l',
        type=int, default=50,
        help='Maximum lag to search'
    )
    parser.add_argument(
        '--lead-threshold', '-t',
        type=int, default=2,
        help='Minimum lag to count as leading'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    detect_sources(
        observations_path=args.observations,
        output_path=args.output,
        max_lag=args.max_lag,
        lead_threshold=args.lead_threshold,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
