"""
Driving Vector Analysis

Identifies which signal drives structural changes by tracking:
1. Eigenvector loading changes per signal
2. Innovation (residual) magnitude per signal
3. Lead-lag of loading changes

Usage:
    cd /path/to/domain
    python /path/to/driving_vector_analysis.py

Outputs:
    driving_vector_analysis.parquet - Per-window driver identification
"""

import polars as pl
import numpy as np
from pathlib import Path
import sys


def analyze_drivers(
    observations_path: str = 'observations.parquet',
    output_path: str = 'driving_vector_analysis.parquet',
    window_size: int = 128,
    stride: int = 32,
    verbose: bool = True,
):
    """
    Run driving vector analysis on observations.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path for output parquet
        window_size: Samples per window
        stride: Samples between window starts
        verbose: Print progress
    """
    # Load data
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded {len(obs):,} observations")

    # Pivot to wide format
    wide = obs.pivot(
        values='value',
        index='I',
        on='signal_id'
    ).sort('I')

    signals = [c for c in wide.columns if c != 'I']
    n_signals = len(signals)

    if verbose:
        print(f"Signals: {signals}")

    # Extract and normalize
    I_values = wide['I'].to_numpy()
    X = wide.select(signals).to_numpy()
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    # Storage
    results = []
    prev_eigenvectors = None
    prev_eigenvalues = None
    prev_loadings = None

    for start in range(0, len(X_norm) - window_size + 1, stride):
        end = start + window_size
        window_I = I_values[end - 1]

        W = X_norm[start:end]
        if np.any(np.isnan(W)):
            continue

        # Covariance and eigen decomposition
        cov = np.cov(W, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        eigenvalues = np.maximum(eigenvalues, 0)

        # Loadings: how much each signal contributes to each PC
        loadings = eigenvectors  # Shape: (n_signals, n_pcs)

        # Effective dimension
        total_var = np.sum(eigenvalues)
        if total_var > 0:
            p = eigenvalues / total_var
            effective_dim = 1.0 / np.sum(p ** 2)
        else:
            effective_dim = 0

        # === DRIVING VECTOR ANALYSIS ===

        if prev_loadings is not None:
            # 1. Loading change per signal on PC1
            # Handle sign flip (eigenvector sign is arbitrary)
            sign_flip = np.sign(np.dot(eigenvectors[:, 0], prev_eigenvectors[:, 0]))
            adjusted_loadings = loadings * sign_flip

            loading_change_pc1 = np.abs(adjusted_loadings[:, 0] - prev_loadings[:, 0])
            loading_change_pc2 = np.abs(adjusted_loadings[:, 1] - prev_loadings[:, 1])

            # 2. Innovation: project last point onto previous eigenstructure
            last_point = W[-1]

            # Expected position based on previous structure
            projection = np.dot(last_point, prev_eigenvectors)
            reconstructed = np.dot(projection, prev_eigenvectors.T)
            innovation = np.abs(last_point - reconstructed)

            # 3. Which signal leads?
            driver_by_loading = signals[np.argmax(loading_change_pc1)]
            driver_by_innovation = signals[np.argmax(innovation)]

            row = {
                'I': int(window_I),
                'effective_dim': round(effective_dim, 4),
                'driver_by_loading': driver_by_loading,
                'driver_by_innovation': driver_by_innovation,
            }

            # Add loading change per signal
            for i, sig in enumerate(signals):
                row[f'loading_change_{sig}'] = round(float(loading_change_pc1[i]), 4)
                row[f'innovation_{sig}'] = round(float(innovation[i]), 4)

            # Total structural change
            row['total_loading_change'] = round(float(np.sum(loading_change_pc1)), 4)
            row['total_innovation'] = round(float(np.sum(innovation)), 4)

            results.append(row)

            prev_loadings = adjusted_loadings.copy()
        else:
            prev_loadings = loadings.copy()

        prev_eigenvectors = eigenvectors.copy()
        prev_eigenvalues = eigenvalues.copy()

    # Create DataFrame
    df = pl.DataFrame(results)

    if verbose:
        print(f"\nComputed {len(df)} windows")

    # Save
    df.write_parquet(output_path)

    if verbose:
        print(f"Saved: {output_path}")

        # === ANALYSIS ===
        print("\n" + "=" * 60)
        print("DRIVING VECTOR ANALYSIS")
        print("=" * 60)

        # Who drives most often?
        print("\n=== DRIVER FREQUENCY (by loading change) ===")
        driver_counts = df.group_by('driver_by_loading').agg(
            pl.count().alias('count')
        ).sort('count', descending=True)
        print(driver_counts)

        print("\n=== DRIVER FREQUENCY (by innovation) ===")
        innov_counts = df.group_by('driver_by_innovation').agg(
            pl.count().alias('count')
        ).sort('count', descending=True)
        print(innov_counts)

        # Find windows with largest structural change
        print("\n=== TOP 10 STRUCTURAL CHANGE WINDOWS ===")
        top_changes = df.sort('total_loading_change', descending=True).head(10)
        print(top_changes.select(['I', 'effective_dim', 'driver_by_loading', 'total_loading_change']))

        # Innovation spike detection
        print("\n=== INNOVATION SPIKES (total > 2σ) ===")
        mean_innov = df['total_innovation'].mean()
        std_innov = df['total_innovation'].std()
        threshold = mean_innov + 2 * std_innov

        spikes = df.filter(pl.col('total_innovation') > threshold)
        print(f"Threshold: {threshold:.4f}")
        print(spikes.select(['I', 'driver_by_innovation', 'total_innovation']))

    return df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Driving vector analysis of observations'
    )
    parser.add_argument(
        '--observations', '-i',
        default='observations.parquet',
        help='Input observations.parquet path'
    )
    parser.add_argument(
        '--output', '-o',
        default='driving_vector_analysis.parquet',
        help='Output parquet path'
    )
    parser.add_argument(
        '--window', '-w',
        type=int, default=128,
        help='Window size in samples'
    )
    parser.add_argument(
        '--stride', '-s',
        type=int, default=32,
        help='Stride between windows'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    analyze_drivers(
        observations_path=args.observations,
        output_path=args.output,
        window_size=args.window,
        stride=args.stride,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
