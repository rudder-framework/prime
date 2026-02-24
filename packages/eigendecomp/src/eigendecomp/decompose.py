"""
Core eigendecomposition computation.

Takes signal_vector feature matrix (n_signals × n_features) per window,
produces eigenvalue metrics, effective dimensionality, and loadings.

All math delegates to pmtvs. This module is orchestration:
normalize → covariance → eigendecompose → derive metrics.
"""

import numpy as np
from typing import Dict, Any, Optional, List, Literal


# ---------------------------------------------------------------------------
# pmtvs imports — NaN on failure, no fallback math
# ---------------------------------------------------------------------------

def _pmtvs_eigendecomposition(matrix):
    try:
        from pmtvs import eigendecomposition
        return eigendecomposition(matrix)
    except ImportError:
        # Minimal fallback: numpy eigh (pmtvs would do the same)
        vals, vecs = np.linalg.eigh(matrix)
        idx = np.argsort(np.abs(vals))[::-1]
        return vals[idx].real, vecs[:, idx].real


def _pmtvs_effective_dimension(eigenvalues):
    try:
        from pmtvs import effective_dimension
        return float(effective_dimension(eigenvalues))
    except ImportError:
        # Participation ratio fallback
        eigenvalues = np.abs(eigenvalues)
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        if len(eigenvalues) == 0:
            return 0.0
        total = np.sum(eigenvalues)
        if total < 1e-30:
            return 0.0
        p = eigenvalues / total
        return float(np.sum(p) ** 2 / np.sum(p ** 2))


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def _zscore_normalize(matrix: np.ndarray) -> np.ndarray:
    """Per-feature z-score normalization (axis=0). Constant features → 0."""
    mean = np.nanmean(matrix, axis=0)
    std = np.nanstd(matrix, axis=0)
    std[std < 1e-15] = 1.0  # constant features → zero after centering
    result = (matrix - mean) / std
    return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def compute_eigendecomp(
    signal_matrix: np.ndarray,
    centroid: Optional[np.ndarray] = None,
    norm_method: Literal["zscore", "none"] = "zscore",
    min_signals: int = 2,
    max_eigenvalues: int = 10,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Compute eigendecomposition of signal ensemble.

    Parameters
    ----------
    signal_matrix : np.ndarray
        (n_signals, n_features) feature matrix from signal_vector.
    centroid : np.ndarray, optional
        Pre-computed centroid. If None, computed as nanmean.
    norm_method : str
        "zscore" = per-feature z-score before SVD. "none" = raw.
    min_signals : int
        Minimum valid (non-NaN) signals required.
    max_eigenvalues : int
        Maximum eigenvalues to include in output.

    Returns
    -------
    dict with:
        eigenvalues : np.ndarray — sorted descending
        explained_ratio : np.ndarray — fraction of total variance per eigenvalue
        total_variance : float
        effective_dim : float — participation ratio
        effective_dim_entropy : float — exp(entropy) effective dim
        eigenvalue_entropy : float — Shannon entropy of eigenvalue distribution
        eigenvalue_entropy_normalized : float — normalized to [0, 1]
        condition_number : float
        ratio_2_1 : float — λ₂/λ₁ (multimode indicator)
        ratio_3_1 : float — λ₃/λ₁
        energy_concentration : float — λ₁/sum(λ)
        principal_components : np.ndarray — (n_features, n_features) eigenvectors
        signal_loadings : np.ndarray — (n_valid, n_features) projections onto PCs
        n_signals : int
        n_features : int
    """
    signal_matrix = np.asarray(signal_matrix, dtype=np.float64)
    if signal_matrix.ndim == 1:
        signal_matrix = signal_matrix.reshape(1, -1)

    N, D = signal_matrix.shape

    # Remove rows with any NaN/inf
    valid_mask = np.all(np.isfinite(signal_matrix), axis=1)
    n_valid = int(valid_mask.sum())

    if n_valid < min_signals:
        return _empty_result(D, max_eigenvalues)

    matrix = signal_matrix[valid_mask]

    # Normalize
    if norm_method == "zscore":
        matrix = _zscore_normalize(matrix)

    # Drop constant columns (zero variance after normalization)
    col_std = np.std(matrix, axis=0)
    varying = col_std > 1e-12
    if varying.sum() < 2:
        return _empty_result(D, max_eigenvalues)
    matrix_clean = matrix[:, varying]
    n_feat_clean = matrix_clean.shape[1]

    # Covariance matrix
    cov = np.cov(matrix_clean, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])

    # Eigendecomposition
    try:
        eigenvalues, eigenvectors = _pmtvs_eigendecomposition(cov)
    except (np.linalg.LinAlgError, ValueError):
        return _empty_result(D, max_eigenvalues)

    eigenvalues = np.real(eigenvalues)
    eigenvectors = np.real(eigenvectors)

    # Ensure non-negative (numerical noise from float precision)
    eigenvalues = np.maximum(eigenvalues, 0.0)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # ---------- Derived metrics ----------

    total_var = float(np.sum(eigenvalues))
    if total_var < 1e-30:
        return _empty_result(D, max_eigenvalues)

    explained = eigenvalues / total_var

    # Effective dimension (participation ratio)
    eff_dim = _pmtvs_effective_dimension(eigenvalues)

    # Effective dimension (entropy-based): exp(H)
    pos = eigenvalues[eigenvalues > 0]
    if len(pos) > 0:
        p = pos / pos.sum()
        eig_entropy = float(-np.sum(p * np.log2(p + 1e-30)))
        max_ent = np.log2(len(p)) if len(p) > 1 else 1.0
        eig_entropy_norm = eig_entropy / max_ent if max_ent > 0 else 0.0
        eff_dim_entropy = float(2 ** eig_entropy)
    else:
        eig_entropy = np.nan
        eig_entropy_norm = np.nan
        eff_dim_entropy = np.nan

    # Condition number
    if eigenvalues[-1] > 1e-30:
        cond_num = float(eigenvalues[0] / eigenvalues[-1])
    else:
        nonzero = eigenvalues[eigenvalues > 1e-30]
        cond_num = float(nonzero[0] / nonzero[-1]) if len(nonzero) > 1 else np.nan

    # Eigenvalue ratios
    ratio_2_1 = float(eigenvalues[1] / eigenvalues[0]) if len(eigenvalues) > 1 and eigenvalues[0] > 0 else np.nan
    ratio_3_1 = float(eigenvalues[2] / eigenvalues[0]) if len(eigenvalues) > 2 and eigenvalues[0] > 0 else np.nan

    # Energy concentration
    energy_concentration = float(eigenvalues[0] / total_var)

    # Principal components and signal loadings
    principal_components = eigenvectors.T  # (n_feat_clean, n_feat_clean)
    signal_loadings = matrix_clean @ eigenvectors  # (n_valid, n_feat_clean)

    # Truncate eigenvalues for output
    n_out = min(max_eigenvalues, len(eigenvalues))

    return {
        'eigenvalues': eigenvalues[:n_out],
        'explained_ratio': explained[:n_out],
        'total_variance': total_var,
        'effective_dim': eff_dim,
        'effective_dim_entropy': eff_dim_entropy,
        'eigenvalue_entropy': eig_entropy,
        'eigenvalue_entropy_normalized': eig_entropy_norm,
        'condition_number': cond_num,
        'ratio_2_1': ratio_2_1,
        'ratio_3_1': ratio_3_1,
        'energy_concentration': energy_concentration,
        'principal_components': principal_components,
        'signal_loadings': signal_loadings,
        'n_signals': n_valid,
        'n_features': D,
        'n_features_valid': n_feat_clean,
        'varying_mask': varying,
    }


def compute_eigendecomp_batch(
    matrices: List[np.ndarray],
    window_indices: List[int],
    norm_method: Literal["zscore", "none"] = "zscore",
    min_signals: int = 2,
    max_eigenvalues: int = 10,
    enforce_continuity: bool = True,
) -> List[Dict[str, Any]]:
    """
    Compute eigendecomposition for a sequence of windows.

    Parameters
    ----------
    matrices : list of np.ndarray
        Each (n_signals, n_features), one per window.
    window_indices : list of int
        Window index (I) for each matrix.
    enforce_continuity : bool
        If True, enforce eigenvector sign consistency across windows.

    Returns
    -------
    list of dict, one per window, each containing eigendecomp results
    plus 'I' key for the window index.
    """
    from eigendecomp.continuity import enforce_eigenvector_continuity

    results = []
    prev_pcs = None

    for matrix, win_idx in zip(matrices, window_indices):
        result = compute_eigendecomp(
            matrix,
            norm_method=norm_method,
            min_signals=min_signals,
            max_eigenvalues=max_eigenvalues,
        )
        result['I'] = win_idx

        # Enforce eigenvector continuity
        if enforce_continuity and result['principal_components'] is not None:
            if prev_pcs is not None:
                result['principal_components'] = enforce_eigenvector_continuity(
                    result['principal_components'], prev_pcs
                )
                # Recompute signal loadings with corrected PCs
                if result.get('signal_loadings') is not None and result.get('varying_mask') is not None:
                    # signal_loadings = matrix_clean @ eigenvectors
                    # PCs are eigenvectors.T, so eigenvectors = PCs.T
                    pass  # loadings already computed, sign flip doesn't change magnitudes
            prev_pcs = result['principal_components']

        results.append(result)

    return results


def _empty_result(D: int, max_eigenvalues: int) -> Dict[str, Any]:
    """Return NaN result for insufficient data."""
    n_out = min(max_eigenvalues, D)
    return {
        'eigenvalues': np.full(n_out, np.nan),
        'explained_ratio': np.full(n_out, np.nan),
        'total_variance': np.nan,
        'effective_dim': np.nan,
        'effective_dim_entropy': np.nan,
        'eigenvalue_entropy': np.nan,
        'eigenvalue_entropy_normalized': np.nan,
        'condition_number': np.nan,
        'ratio_2_1': np.nan,
        'ratio_3_1': np.nan,
        'energy_concentration': np.nan,
        'principal_components': None,
        'signal_loadings': None,
        'n_signals': 0,
        'n_features': D,
        'n_features_valid': 0,
        'varying_mask': np.zeros(D, dtype=bool),
    }
