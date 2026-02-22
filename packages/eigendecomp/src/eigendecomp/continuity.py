"""
Eigenvector continuity enforcement across windows.

Eigenvectors have sign ambiguity — eigh/SVD can flip signs arbitrarily.
When tracking eigenvalue trajectories across windows, we need consistent
orientation so that PC1 at window t+1 points "the same way" as at window t.

Method: dot product between consecutive eigenvectors. If negative, flip.
"""

import numpy as np


def enforce_eigenvector_continuity(
    eigenvectors_current: np.ndarray,
    eigenvectors_previous: np.ndarray,
) -> np.ndarray:
    """
    Ensure eigenvectors maintain consistent orientation across windows.

    For each principal component, check if the dot product with the
    previous window's version is negative. If so, flip the sign.

    Parameters
    ----------
    eigenvectors_current : np.ndarray
        Current window's eigenvectors. Shape (n_components, n_features)
        if row vectors (PCs), or (n_features, n_components) if column vectors.
    eigenvectors_previous : np.ndarray
        Previous window's eigenvectors. Same shape.

    Returns
    -------
    np.ndarray
        Eigenvectors with consistent signs.
    """
    if eigenvectors_previous is None:
        return eigenvectors_current

    if eigenvectors_current.shape != eigenvectors_previous.shape:
        return eigenvectors_current

    corrected = eigenvectors_current.copy()

    if corrected.ndim == 1:
        if np.dot(corrected, eigenvectors_previous) < 0:
            corrected *= -1
        return corrected

    # Determine orientation: rows or columns?
    # Convention: PCs as rows (n_components × n_features)
    n_vecs = corrected.shape[0]

    for j in range(n_vecs):
        dot = np.dot(eigenvectors_previous[j], corrected[j])
        if dot < 0:
            corrected[j] *= -1

    return corrected
