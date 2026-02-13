"""
Geometry Engine

Compute eigenstructure of signal covariance matrix.
This is the GEOMETRY in Structure = Geometry × Mass.

Core metrics:
- eff_dim: Effective dimension (participation ratio)
- alignment: PC1 dominance
- mean_abs_correlation: Average coupling strength
- condition_number: Matrix conditioning
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Dict


@dataclass
class GeometryResult:
    """Result of geometry computation."""
    eff_dim: float                  # Effective dimension
    eff_dim_ratio: float           # eff_dim / n_signals
    alignment: float                # PC1 dominance (λ₁ / Σλ)
    mean_abs_correlation: float    # Mean |ρᵢⱼ|
    coupling_fraction: float        # Fraction |ρ| > 0.5
    condition_number: float         # λ_max / λ_min
    eigenvalues: List[float]        # Sorted eigenvalues
    n_signals: int
    n_timepoints: int


def compute_geometry(
    X: np.ndarray,
    use_correlation: bool = True,
) -> GeometryResult:
    """
    Compute eigenstructure of signal matrix.

    Args:
        X: Signal matrix (n_timepoints × n_signals)
        use_correlation: If True, use correlation matrix; else covariance

    Returns:
        GeometryResult with eigenstructure metrics
    """
    n_timepoints, n_signals = X.shape

    if n_signals < 2 or n_timepoints < n_signals:
        return _empty_geometry_result(n_signals, n_timepoints)

    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0)

    # Z-score normalize for correlation-based analysis
    if use_correlation:
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        cov_matrix = np.corrcoef(X_norm.T)
    else:
        X_centered = X - X.mean(axis=0)
        cov_matrix = np.cov(X_centered.T)

    # Handle numerical issues
    cov_matrix = np.nan_to_num(cov_matrix, nan=0.0)

    # Eigendecomposition
    try:
        eigenvalues, _ = np.linalg.eigh(cov_matrix)
    except np.linalg.LinAlgError:
        return _empty_geometry_result(n_signals, n_timepoints)

    # Sort descending
    eigenvalues = np.sort(eigenvalues)[::-1]

    # Clip small/negative eigenvalues (numerical noise)
    eigenvalues = np.maximum(eigenvalues, 1e-10)

    # === EFFECTIVE DIMENSION ===
    # Participation ratio via Shannon entropy
    total = eigenvalues.sum()
    p = eigenvalues / total
    entropy = -np.sum(p * np.log(p + 1e-10))
    eff_dim = np.exp(entropy)

    # === ALIGNMENT (PC1 dominance) ===
    alignment = eigenvalues[0] / total

    # === CORRELATION METRICS ===
    if use_correlation:
        corr_matrix = cov_matrix
    else:
        # Compute correlation from covariance
        stds = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / (np.outer(stds, stds) + 1e-10)

    upper_idx = np.triu_indices(n_signals, k=1)
    correlations = corr_matrix[upper_idx]

    mean_abs_corr = np.mean(np.abs(correlations))
    coupling_fraction = np.mean(np.abs(correlations) > 0.5)

    # === CONDITION NUMBER ===
    condition_number = eigenvalues[0] / eigenvalues[-1] if eigenvalues[-1] > 1e-10 else np.inf

    return GeometryResult(
        eff_dim=float(eff_dim),
        eff_dim_ratio=float(eff_dim / n_signals),
        alignment=float(alignment),
        mean_abs_correlation=float(mean_abs_corr),
        coupling_fraction=float(coupling_fraction),
        condition_number=float(min(condition_number, 1e10)),
        eigenvalues=eigenvalues.tolist(),
        n_signals=n_signals,
        n_timepoints=n_timepoints,
    )


def _empty_geometry_result(n_signals: int, n_timepoints: int) -> GeometryResult:
    """Return empty geometry result for degenerate cases."""
    return GeometryResult(
        eff_dim=0.0,
        eff_dim_ratio=0.0,
        alignment=0.0,
        mean_abs_correlation=0.0,
        coupling_fraction=0.0,
        condition_number=np.inf,
        eigenvalues=[],
        n_signals=n_signals,
        n_timepoints=n_timepoints,
    )


def compute_geometry_trajectory(
    geometries: List[GeometryResult],
) -> Dict:
    """
    Analyze geometry evolution over time.

    Args:
        geometries: List of GeometryResult from sequential windows

    Returns:
        Dict with trajectory metrics
    """
    if len(geometries) < 2:
        return {
            'eff_dim_trend': 0.0,
            'alignment_trend': 0.0,
            'coupling_trend': 0.0,
            'is_collapsing': False,
            'collapse_rate': 0.0,
        }

    eff_dims = np.array([g.eff_dim for g in geometries])
    alignments = np.array([g.alignment for g in geometries])
    couplings = np.array([g.mean_abs_correlation for g in geometries])

    t = np.arange(len(geometries))

    # Compute trends
    eff_dim_trend = np.polyfit(t, eff_dims, 1)[0] if len(t) > 1 else 0.0
    alignment_trend = np.polyfit(t, alignments, 1)[0] if len(t) > 1 else 0.0
    coupling_trend = np.polyfit(t, couplings, 1)[0] if len(t) > 1 else 0.0

    # Normalize trends
    eff_dim_mean = np.mean(eff_dims)
    if eff_dim_mean > 1:
        eff_dim_trend_norm = eff_dim_trend / eff_dim_mean
    else:
        eff_dim_trend_norm = eff_dim_trend

    # Collapse detection
    is_collapsing = eff_dim_trend_norm < -0.01  # Declining by >1% per window
    collapse_rate = -eff_dim_trend_norm if is_collapsing else 0.0

    return {
        'eff_dim_trend': float(eff_dim_trend_norm),
        'alignment_trend': float(alignment_trend),
        'coupling_trend': float(coupling_trend),
        'is_collapsing': bool(is_collapsing),
        'collapse_rate': float(collapse_rate),
        'eff_dim_min': float(np.min(eff_dims)),
        'eff_dim_max': float(np.max(eff_dims)),
        'eff_dim_std': float(np.std(eff_dims)),
    }


def classify_geometry_state(geometry: GeometryResult) -> Dict:
    """
    Classify the current geometry state.

    Args:
        geometry: GeometryResult from compute_geometry

    Returns:
        Dict with state classification
    """
    ratio = geometry.eff_dim_ratio

    if ratio > 0.8:
        zone = 'healthy'
        color = '🟢'
        description = 'High dimensional: signals independent'
    elif ratio > 0.6:
        zone = 'caution'
        color = '🟡'
        description = 'Some coupling: moderate compression'
    elif ratio > 0.4:
        zone = 'stress'
        color = '🟠'
        description = 'Significant coupling: geometry compressed'
    else:
        zone = 'crisis'
        color = '🔴'
        description = 'Severe collapse: dimensional crisis'

    return {
        'zone': zone,
        'color': color,
        'description': description,
        'eff_dim_ratio': ratio,
        'alignment': geometry.alignment,
    }
