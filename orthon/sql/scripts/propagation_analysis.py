"""
Vibration Propagation Analysis

Detects causal relationships between signals:
1. Cross-correlation with lag (who leads?)
2. Granger causality (predictive power)
3. Coherence (frequency-dependent coupling)
4. Event-triggered coupling (do signals couple during excitation?)

Usage:
    cd /path/to/domain
    python /path/to/propagation_analysis.py

Outputs:
    propagation_analysis.parquet - Summary of causal relationships
    cross_correlation_lags.png - Cross-correlation functions
    coherence_spectra.png - Frequency-domain coupling
"""

import polars as pl
import numpy as np
from scipy import signal as sig
from pathlib import Path
import matplotlib.pyplot as plt


def cross_corr_with_lag(x: np.ndarray, y: np.ndarray, max_lag: int) -> np.ndarray:
    """
    Compute cross-correlation for range of lags.

    Args:
        x: First signal
        y: Second signal
        max_lag: Maximum lag in samples

    Returns:
        Array of correlations for lags from -max_lag to +max_lag
    """
    correlations = []

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            corr = np.corrcoef(x[:lag], y[-lag:])[0, 1]
        elif lag > 0:
            corr = np.corrcoef(x[lag:], y[:-lag])[0, 1]
        else:
            corr = np.corrcoef(x, y)[0, 1]
        correlations.append(corr)

    return np.array(correlations)


def granger_causality_simple(x: np.ndarray, y: np.ndarray, max_lag: int = 10):
    """
    Simplified Granger causality test.
    Tests if past values of x help predict y.

    Args:
        x: Potential cause signal
        y: Potential effect signal
        max_lag: Number of lags to include

    Returns:
        f_stat: F-statistic (higher = stronger causality)
        ssr_r: Restricted model sum of squared residuals
        ssr_u: Unrestricted model sum of squared residuals
    """
    from numpy.linalg import lstsq

    n = len(y) - max_lag

    # Restricted model: y predicted by its own past
    Y = y[max_lag:]
    X_restricted = np.column_stack([y[max_lag-i-1:-i-1] for i in range(max_lag)])

    # Unrestricted model: y predicted by own past + x's past
    X_unrestricted = np.column_stack([
        X_restricted,
        *[x[max_lag-i-1:-i-1] for i in range(max_lag)]
    ])

    # Fit models
    ssr_r = np.sum((Y - X_restricted @ lstsq(X_restricted, Y, rcond=None)[0])**2)
    ssr_u = np.sum((Y - X_unrestricted @ lstsq(X_unrestricted, Y, rcond=None)[0])**2)

    # F-statistic
    df1 = max_lag
    df2 = n - 2 * max_lag

    if ssr_u > 0 and df2 > 0:
        f_stat = ((ssr_r - ssr_u) / df1) / (ssr_u / df2)
    else:
        f_stat = 0

    return f_stat, ssr_r, ssr_u


def analyze_propagation(
    observations_path: str = 'observations.parquet',
    output_path: str = 'propagation_analysis.parquet',
    max_lag: int = 100,
    granger_lag: int = 20,
    coherence_nperseg: int = 256,
    event_window: int = 64,
    event_stride: int = 16,
    event_threshold_sigma: float = 3.0,
    verbose: bool = True,
    save_plots: bool = True,
):
    """
    Run propagation analysis on observations.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path for output parquet
        max_lag: Maximum lag for cross-correlation
        granger_lag: Number of lags for Granger causality
        coherence_nperseg: Segment length for coherence estimation
        event_window: Window size for event detection
        event_stride: Stride for event detection
        event_threshold_sigma: Z-score threshold for event detection
        verbose: Print progress
        save_plots: Generate visualization plots
    """
    # Load data
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded {len(obs):,} observations")

    # Pivot to wide
    wide = obs.pivot(values='value', index='I', on='signal_id').sort('I')
    signals = [c for c in wide.columns if c != 'I']
    n_signals = len(signals)

    if verbose:
        print(f"Signals: {signals}")

    # Extract as numpy
    data = {s: wide[s].to_numpy() for s in signals}

    # ============================================================
    # 1. CROSS-CORRELATION WITH LAG
    # ============================================================

    if verbose:
        print("\n" + "="*60)
        print("CROSS-CORRELATION WITH LAG")
        print("="*60)

    lag_range = np.arange(-max_lag, max_lag + 1)
    cross_corr_results = {}

    for i, sig_a in enumerate(signals):
        for j, sig_b in enumerate(signals):
            if i >= j:
                continue

            cc = cross_corr_with_lag(data[sig_a], data[sig_b], max_lag)
            peak_lag = lag_range[np.argmax(np.abs(cc))]
            peak_corr = cc[np.argmax(np.abs(cc))]

            cross_corr_results[(sig_a, sig_b)] = {
                'correlations': cc,
                'peak_lag': peak_lag,
                'peak_corr': peak_corr,
            }

            if verbose and (abs(peak_lag) > 5 or abs(peak_corr) > 0.1):
                direction = f"{sig_a} leads" if peak_lag > 0 else f"{sig_b} leads"
                print(f"\n{sig_a} ↔ {sig_b}:")
                print(f"  Peak correlation: {peak_corr:.4f} at lag={peak_lag}")
                print(f"  Direction: {direction if peak_lag != 0 else 'simultaneous'}")

    # ============================================================
    # 2. GRANGER CAUSALITY
    # ============================================================

    if verbose:
        print("\n" + "="*60)
        print("GRANGER CAUSALITY TEST")
        print("="*60)

    granger_results = []

    for i, sig_a in enumerate(signals):
        for j, sig_b in enumerate(signals):
            if i == j:
                continue

            f_stat, ssr_r, ssr_u = granger_causality_simple(
                data[sig_a], data[sig_b], max_lag=granger_lag
            )

            granger_results.append({
                'cause': sig_a,
                'effect': sig_b,
                'f_stat': f_stat,
                'variance_reduction': (ssr_r - ssr_u) / ssr_r if ssr_r > 0 else 0
            })

    # Sort by F-statistic
    granger_results.sort(key=lambda x: x['f_stat'], reverse=True)

    if verbose:
        print(f"\n{'Cause':<20} → {'Effect':<20} F-stat    Var Reduction")
        print("-" * 70)
        for r in granger_results[:10]:
            print(f"{r['cause']:<20} → {r['effect']:<20} {r['f_stat']:8.2f}  {r['variance_reduction']*100:6.2f}%")

    # ============================================================
    # 3. COHERENCE (frequency-domain coupling)
    # ============================================================

    if verbose:
        print("\n" + "="*60)
        print("COHERENCE ANALYSIS (frequency-domain coupling)")
        print("="*60)

    fs = 1.0  # Assume unit sampling rate
    coherence_results = {}

    for i, sig_a in enumerate(signals):
        for j, sig_b in enumerate(signals):
            if i >= j:
                continue

            # Compute coherence
            f, coh = sig.coherence(
                data[sig_a], data[sig_b], fs=fs, nperseg=coherence_nperseg
            )

            # Find peak coherence and its frequency
            peak_idx = np.argmax(coh)
            peak_freq = f[peak_idx]
            peak_coh = coh[peak_idx]
            mean_coh = np.mean(coh)

            coherence_results[(sig_a, sig_b)] = {
                'frequencies': f,
                'coherence': coh,
                'peak_freq': peak_freq,
                'peak_coherence': peak_coh,
                'mean_coherence': mean_coh,
            }

            if verbose:
                print(f"\n{sig_a} ↔ {sig_b}:")
                print(f"  Mean coherence: {mean_coh:.4f}")
                print(f"  Peak coherence: {peak_coh:.4f} at freq={peak_freq:.4f}")

    # ============================================================
    # 4. EVENT-TRIGGERED COUPLING
    # ============================================================

    if verbose:
        print("\n" + "="*60)
        print("EVENT-TRIGGERED COUPLING")
        print("="*60)
        print("(Do signals couple during high-energy events?)")

    event_correlations = []
    quiet_correlations = []

    for start in range(0, len(data[signals[0]]) - event_window, event_stride):
        end = start + event_window

        # Check if any signal is excited (> threshold σ from mean)
        is_event = False
        for s in signals:
            window_data = data[s][start:end]
            z_score = np.abs(window_data - np.mean(data[s])) / np.std(data[s])
            if np.any(z_score > event_threshold_sigma):
                is_event = True
                break

        # Compute correlation matrix for this window
        window_matrix = np.column_stack([data[s][start:end] for s in signals])
        corr_matrix = np.corrcoef(window_matrix, rowvar=False)

        # Extract upper triangle (excluding diagonal)
        upper_tri = corr_matrix[np.triu_indices(n_signals, k=1)]
        mean_abs_corr = np.mean(np.abs(upper_tri))

        if is_event:
            event_correlations.append(mean_abs_corr)
        else:
            quiet_correlations.append(mean_abs_corr)

    coupling_increase = 1.0
    if verbose:
        print(f"\nQuiet periods: {len(quiet_correlations)} windows")
        if quiet_correlations:
            print(f"  Mean |correlation|: {np.mean(quiet_correlations):.4f}")

        print(f"\nEvent periods: {len(event_correlations)} windows")
        if event_correlations:
            print(f"  Mean |correlation|: {np.mean(event_correlations):.4f}")

        if event_correlations and quiet_correlations:
            coupling_increase = np.mean(event_correlations) / np.mean(quiet_correlations)
            print(f"\nCoupling increase during events: {coupling_increase:.2f}x")

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    # Build summary DataFrame
    summary_rows = []

    # Cross-correlation results
    for (sig_a, sig_b), result in cross_corr_results.items():
        leader = sig_a if result['peak_lag'] > 0 else sig_b
        follower = sig_b if result['peak_lag'] > 0 else sig_a

        summary_rows.append({
            'analysis': 'cross_correlation',
            'signal_a': sig_a,
            'signal_b': sig_b,
            'metric': 'peak_correlation',
            'value': result['peak_corr'],
        })
        summary_rows.append({
            'analysis': 'cross_correlation',
            'signal_a': sig_a,
            'signal_b': sig_b,
            'metric': 'peak_lag',
            'value': float(result['peak_lag']),
        })

    # Granger results
    for r in granger_results:
        summary_rows.append({
            'analysis': 'granger_causality',
            'signal_a': r['cause'],
            'signal_b': r['effect'],
            'metric': 'f_statistic',
            'value': r['f_stat'],
        })
        summary_rows.append({
            'analysis': 'granger_causality',
            'signal_a': r['cause'],
            'signal_b': r['effect'],
            'metric': 'variance_reduction',
            'value': r['variance_reduction'],
        })

    # Coherence results
    for (sig_a, sig_b), result in coherence_results.items():
        summary_rows.append({
            'analysis': 'coherence',
            'signal_a': sig_a,
            'signal_b': sig_b,
            'metric': 'mean_coherence',
            'value': result['mean_coherence'],
        })
        summary_rows.append({
            'analysis': 'coherence',
            'signal_a': sig_a,
            'signal_b': sig_b,
            'metric': 'peak_coherence',
            'value': result['peak_coherence'],
        })
        summary_rows.append({
            'analysis': 'coherence',
            'signal_a': sig_a,
            'signal_b': sig_b,
            'metric': 'peak_frequency',
            'value': result['peak_freq'],
        })

    # Event coupling
    summary_rows.append({
        'analysis': 'event_coupling',
        'signal_a': 'all',
        'signal_b': 'all',
        'metric': 'coupling_increase',
        'value': coupling_increase,
    })
    if quiet_correlations:
        summary_rows.append({
            'analysis': 'event_coupling',
            'signal_a': 'all',
            'signal_b': 'all',
            'metric': 'quiet_mean_correlation',
            'value': np.mean(quiet_correlations),
        })
    if event_correlations:
        summary_rows.append({
            'analysis': 'event_coupling',
            'signal_a': 'all',
            'signal_b': 'all',
            'metric': 'event_mean_correlation',
            'value': np.mean(event_correlations),
        })

    df = pl.DataFrame(summary_rows)
    df.write_parquet(output_path)

    if verbose:
        print(f"\nSaved: {output_path}")

    # ============================================================
    # PLOTS
    # ============================================================

    if save_plots:
        output_dir = Path(output_path).parent

        # Cross-correlation plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, ((sig_a, sig_b), result) in enumerate(cross_corr_results.items()):
            if idx >= 6:
                break

            ax = axes[idx]
            ax.plot(lag_range, result['correlations'], 'b-', linewidth=1.5)
            ax.axvline(x=0, color='k', linestyle='--', alpha=0.5)
            ax.axvline(x=result['peak_lag'], color='r', linestyle='-',
                       label=f"Peak at lag={result['peak_lag']}")
            ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            ax.set_xlabel('Lag (samples)')
            ax.set_ylabel('Correlation')
            ax.set_title(f"{sig_a} vs {sig_b}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / 'cross_correlation_lags.png', dpi=150)
        if verbose:
            print(f"Saved: {output_dir / 'cross_correlation_lags.png'}")
        plt.close()

        # Coherence plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()

        for idx, ((sig_a, sig_b), result) in enumerate(coherence_results.items()):
            if idx >= 6:
                break

            ax = axes[idx]
            ax.semilogy(result['frequencies'], result['coherence'], 'b-', linewidth=1.5)
            ax.axhline(y=result['mean_coherence'], color='r', linestyle='--',
                       label=f"Mean: {result['mean_coherence']:.3f}")
            ax.set_xlabel('Frequency')
            ax.set_ylabel('Coherence')
            ax.set_title(f"{sig_a} vs {sig_b}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.001, 1)

        plt.tight_layout()
        plt.savefig(output_dir / 'coherence_spectra.png', dpi=150)
        if verbose:
            print(f"Saved: {output_dir / 'coherence_spectra.png'}")
        plt.close()

    # ============================================================
    # SUMMARY
    # ============================================================

    if verbose:
        print("\n" + "="*60)
        print("PROPAGATION SUMMARY")
        print("="*60)

        # Find strongest causal relationships
        print("\nStrongest causal relationships (Granger):")
        for r in granger_results[:3]:
            if r['f_stat'] > 10:
                print(f"  {r['cause']} → {r['effect']}: F={r['f_stat']:.1f}")

        # Find lead-lag relationships
        print("\nLead-lag relationships:")
        for (sig_a, sig_b), result in cross_corr_results.items():
            if abs(result['peak_lag']) > 5 and abs(result['peak_corr']) > 0.05:
                leader = sig_a if result['peak_lag'] > 0 else sig_b
                follower = sig_b if result['peak_lag'] > 0 else sig_a
                print(f"  {leader} leads {follower} by {abs(result['peak_lag'])} samples")

        # Event coupling
        if event_correlations:
            print(f"\nEvent-triggered coupling: {coupling_increase:.2f}x increase during excitation")

    return df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Propagation analysis of observations'
    )
    parser.add_argument(
        '--observations', '-i',
        default='observations.parquet',
        help='Input observations.parquet path'
    )
    parser.add_argument(
        '--output', '-o',
        default='propagation_analysis.parquet',
        help='Output parquet path'
    )
    parser.add_argument(
        '--max-lag', '-l',
        type=int, default=100,
        help='Maximum lag for cross-correlation'
    )
    parser.add_argument(
        '--granger-lag', '-g',
        type=int, default=20,
        help='Number of lags for Granger causality'
    )
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plot generation'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    analyze_propagation(
        observations_path=args.observations,
        output_path=args.output,
        max_lag=args.max_lag,
        granger_lag=args.granger_lag,
        verbose=not args.quiet,
        save_plots=not args.no_plots,
    )


if __name__ == '__main__':
    main()
