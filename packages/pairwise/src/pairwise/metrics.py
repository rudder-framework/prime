"""
Core pairwise metrics between two vectors.

All math delegates to pmtvs where available.
These are feature-space metrics (comparing signal_vector rows),
not time-series metrics (comparing raw waveforms).
"""

import numpy as np
from typing import Dict


# ---------------------------------------------------------------------------
# pmtvs imports with inline fallbacks for basic operations
# ---------------------------------------------------------------------------

def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation."""
    try:
        from pmtvs import correlation
        return float(correlation(x, y))
    except ImportError:
        pass
    # Fallback
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    denom = np.linalg.norm(xc) * np.linalg.norm(yc)
    if denom < 1e-15:
        return 0.0
    return float(np.dot(xc, yc) / denom)


def _euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Euclidean distance."""
    try:
        from pmtvs import euclidean_distance
        return float(euclidean_distance(x, y))
    except ImportError:
        return float(np.sqrt(np.sum((x - y) ** 2)))


def _cosine_similarity(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine similarity."""
    try:
        from pmtvs import cosine_similarity
        return float(cosine_similarity(x, y))
    except ImportError:
        nx = np.linalg.norm(x)
        ny = np.linalg.norm(y)
        if nx < 1e-15 or ny < 1e-15:
            return 0.0
        return float(np.dot(x, y) / (nx * ny))


def _mutual_information(x: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """Mutual information (histogram-based)."""
    try:
        from pmtvs import mutual_information
        return float(mutual_information(x, y, n_bins=n_bins))
    except ImportError:
        pass
    # Fallback
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    hist_xy, _, _ = np.histogram2d(x, y, bins=n_bins)
    pxy = hist_xy / hist_xy.sum()
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    mask = pxy > 1e-30
    mi = np.sum(pxy[mask] * np.log2(pxy[mask] / (px[:, None] * py[None, :])[mask] + 1e-30))
    return float(max(0, mi))


# ---------------------------------------------------------------------------
# Main pair metrics
# ---------------------------------------------------------------------------

def compute_pair_metrics(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    signal_a: str = "",
    signal_b: str = "",
) -> Dict[str, float]:
    """
    Compute all pairwise metrics between two feature vectors.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        1D feature vectors (from signal_vector row). Same length.
    signal_a, signal_b : str
        Signal identifiers (passed through to output).

    Returns
    -------
    dict with:
        signal_a, signal_b : str identifiers
        correlation : float — Pearson correlation
        correlation_abs : float — |correlation|
        distance : float — Euclidean distance
        cosine_similarity : float — cosine similarity [-1, 1]
        mutual_info : float — mutual information (bits)
    """
    vec_a = np.asarray(vec_a, dtype=np.float64).flatten()
    vec_b = np.asarray(vec_b, dtype=np.float64).flatten()

    # Align lengths
    n = min(len(vec_a), len(vec_b))
    va, vb = vec_a[:n], vec_b[:n]

    # Remove NaN positions (pairwise)
    valid = np.isfinite(va) & np.isfinite(vb)
    va_clean = va[valid]
    vb_clean = vb[valid]

    result = {
        'signal_a': signal_a,
        'signal_b': signal_b,
    }

    if len(va_clean) < 3:
        result.update({
            'correlation': np.nan,
            'correlation_abs': np.nan,
            'distance': np.nan,
            'cosine_similarity': np.nan,
            'mutual_info': np.nan,
        })
        return result

    corr = _correlation(va_clean, vb_clean)
    result['correlation'] = corr
    result['correlation_abs'] = abs(corr)
    result['distance'] = _euclidean_distance(va_clean, vb_clean)
    result['cosine_similarity'] = _cosine_similarity(va_clean, vb_clean)
    result['mutual_info'] = _mutual_information(va_clean, vb_clean)

    return result


def compute_pair_metrics_with_context(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
    centroid: np.ndarray,
    signal_a: str = "",
    signal_b: str = "",
) -> Dict[str, float]:
    """
    Compute pair metrics plus centroid-relative context.

    Adds: whether both signals are close/far from centroid,
    whether they're on the same side of the centroid.

    Parameters
    ----------
    vec_a, vec_b : np.ndarray
        1D feature vectors.
    centroid : np.ndarray
        Cohort centroid vector.
    """
    result = compute_pair_metrics(vec_a, vec_b, signal_a, signal_b)

    centroid = np.asarray(centroid, dtype=np.float64).flatten()
    va = np.asarray(vec_a, dtype=np.float64).flatten()
    vb = np.asarray(vec_b, dtype=np.float64).flatten()

    n = min(len(va), len(vb), len(centroid))
    va, vb, c = va[:n], vb[:n], centroid[:n]

    valid = np.isfinite(va) & np.isfinite(vb) & np.isfinite(c)
    if valid.sum() < 2:
        result['dist_a_to_centroid'] = np.nan
        result['dist_b_to_centroid'] = np.nan
        result['same_side'] = np.nan
        return result

    va_c, vb_c, cc = va[valid], vb[valid], c[valid]

    dist_a = float(np.sqrt(np.sum((va_c - cc) ** 2)))
    dist_b = float(np.sqrt(np.sum((vb_c - cc) ** 2)))

    # Same side = dot product of (a-centroid) and (b-centroid) > 0
    da = va_c - cc
    db = vb_c - cc
    same_side = float(np.dot(da, db) > 0)

    result['dist_a_to_centroid'] = dist_a
    result['dist_b_to_centroid'] = dist_b
    result['same_side'] = same_side

    return result
