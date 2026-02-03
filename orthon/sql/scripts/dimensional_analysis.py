"""
Windowed Dimensional Analysis

Computes per-window:
- Covariance matrix across all signals
- Eigenvalues (λ₁, λ₂, ..., λₙ)
- Effective dimension (participation ratio)
- Eigenvector alignment with previous window
- Condition number (λ_max / λ_min)

Usage:
    cd /path/to/domain
    python /path/to/dimensional_analysis.py

Outputs:
    dimensional_analysis.parquet - Per-window eigenvalue decomposition
"""

import polars as pl
import numpy as np
from pathlib import Path
import sys


def compute_participation_ratio(eigenvalues: np.ndarray) -> float:
    """
    Compute participation ratio (effective dimension).

    PR = (Σλ)² / Σλ²

    Ranges from 1 (single dominant mode) to n (uniform distribution).
    """
    eigenvalues = eigenvalues[eigenvalues > 0]  # Filter zeros
    if len(eigenvalues) == 0:
        return 0.0
    total = np.sum(eigenvalues)
    sum_sq = np.sum(eigenvalues ** 2)
    if sum_sq == 0:
        return 0.0
    return (total ** 2) / sum_sq


def compute_eigenvector_alignment(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute alignment between eigenvector matrices.

    Uses absolute value of dot product (eigenvectors can flip sign).
    Returns mean alignment across corresponding eigenvectors.
    """
    if v1 is None or v2 is None:
        return np.nan

    alignments = []
    for i in range(min(v1.shape[1], v2.shape[1])):
        alignment = abs(np.dot(v1[:, i], v2[:, i]))
        alignments.append(alignment)

    return np.mean(alignments)


def analyze_dimensions(
    observations_path: str = 'observations.parquet',
    output_path: str = 'dimensional_analysis.parquet',
    window_size: int = 128,
    stride: int = 32,
    max_eigenvalues: int = 8,
    verbose: bool = True,
):
    """
    Run windowed dimensional analysis on observations.

    Args:
        observations_path: Path to observations.parquet
        output_path: Path for output parquet
        window_size: Samples per window
        stride: Samples between window starts
        max_eigenvalues: Maximum eigenvalues to store (top-k)
        verbose: Print progress
    """
    # Load data
    obs = pl.read_parquet(observations_path)

    if verbose:
        print(f"Loaded {len(obs):,} observations")

    # Pivot to wide format: rows = I, columns = signals
    wide = obs.pivot(
        values='value',
        index='I',
        columns='signal_id'
    ).sort('I')

    signals = [c for c in wide.columns if c != 'I']
    n_signals = len(signals)

    if verbose:
        print(f"Signals ({n_signals}): {signals}")

    # Extract values matrix
    I_values = wide['I'].to_numpy()
    X = wide.select(signals).to_numpy()  # Shape: (n_timepoints, n_signals)

    # Z-score normalize each signal (important for comparable eigenvalues)
    X_mean = np.nanmean(X, axis=0)
    X_std = np.nanstd(X, axis=0)
    X_std[X_std == 0] = 1.0  # Avoid division by zero for constant signals
    X_norm = (X - X_mean) / X_std

    # Storage
    results = []
    prev_eigenvectors = None

    n_windows = (len(X_norm) - window_size) // stride + 1

    if verbose:
        print(f"Computing {n_windows} windows (size={window_size}, stride={stride})...")

    for start in range(0, len(X_norm) - window_size + 1, stride):
        end = start + window_size
        window_I = int(I_values[end - 1])  # Window end index

        # Extract window
        W = X_norm[start:end]

        # Skip if NaN
        if np.any(np.isnan(W)):
            continue

        # Compute covariance matrix
        cov = np.cov(W.T)  # Shape: (n_signals, n_signals)

        # Eigendecomposition (sorted descending)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Compute metrics
        total_variance = np.sum(eigenvalues)
        participation_ratio = compute_participation_ratio(eigenvalues)

        # Condition number (ratio of largest to smallest positive eigenvalue)
        pos_eig = eigenvalues[eigenvalues > 1e-10]
        condition_number = pos_eig[0] / pos_eig[-1] if len(pos_eig) > 1 else 1.0

        # Alignment with previous window
        alignment = compute_eigenvector_alignment(eigenvectors, prev_eigenvectors)
        prev_eigenvectors = eigenvectors.copy()

        # Variance explained by top-k
        cumvar = np.cumsum(eigenvalues) / total_variance

        # Build result row
        row = {
            'I': window_I,
            'total_variance': float(total_variance),
            'participation_ratio': float(participation_ratio),
            'effective_dim': float(participation_ratio),
            'condition_number': float(condition_number),
            'eigenvector_alignment': float(alignment) if not np.isnan(alignment) else None,
        }

        # Add individual eigenvalues
        for i in range(min(max_eigenvalues, len(eigenvalues))):
            row[f'lambda_{i+1}'] = float(eigenvalues[i])
            row[f'var_explained_{i+1}'] = float(cumvar[i])

        # Pad with NaN if fewer eigenvalues than max
        for i in range(len(eigenvalues), max_eigenvalues):
            row[f'lambda_{i+1}'] = np.nan
            row[f'var_explained_{i+1}'] = np.nan

        results.append(row)

    # Create DataFrame
    df = pl.DataFrame(results)

    # Write output
    df.write_parquet(output_path)

    if verbose:
        print(f"\nWrote {len(df)} windows to {output_path}")
        print(f"\nSummary:")
        print(f"  Participation ratio: {df['participation_ratio'].mean():.2f} "
              f"(range: {df['participation_ratio'].min():.2f} - {df['participation_ratio'].max():.2f})")
        print(f"  Condition number: {df['condition_number'].median():.1f} median")
        print(f"  Top eigenvalue captures: {df['var_explained_1'].mean()*100:.1f}% variance (mean)")

        # Show eigenvalue spectrum (mean across windows)
        print(f"\n  Mean eigenvalue spectrum:")
        for i in range(min(max_eigenvalues, n_signals)):
            col = f'lambda_{i+1}'
            if col in df.columns:
                mean_val = df[col].mean()
                var_col = f'var_explained_{i+1}'
                var_pct = df[var_col].mean() * 100 if i == 0 else (df[var_col].mean() - df[f'var_explained_{i}'].mean()) * 100
                print(f"    λ_{i+1} = {mean_val:.4f} ({var_pct:.1f}% variance)")

    return df


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Windowed dimensional analysis of observations'
    )
    parser.add_argument(
        '--observations', '-i',
        default='observations.parquet',
        help='Input observations.parquet path'
    )
    parser.add_argument(
        '--output', '-o',
        default='dimensional_analysis.parquet',
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
        '--max-eigenvalues', '-k',
        type=int, default=8,
        help='Maximum eigenvalues to store'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Suppress output'
    )

    args = parser.parse_args()

    analyze_dimensions(
        observations_path=args.observations,
        output_path=args.output,
        window_size=args.window,
        stride=args.stride,
        max_eigenvalues=args.max_eigenvalues,
        verbose=not args.quiet,
    )


if __name__ == '__main__':
    main()
