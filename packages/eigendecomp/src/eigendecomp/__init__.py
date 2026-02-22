"""
Eigendecomposition package for the Rudder Framework.

Computes eigenvalue structure of the signal ensemble at cohort scale.
Input: signal_vector feature matrix (n_signals × n_features per window).
Output: eigenvalues, effective dimensionality, condition number, loadings.

This is the core "geometry" — eigenvalue trajectories predict failure.
"""

from eigendecomp.decompose import (
    compute_eigendecomp,
    compute_eigendecomp_batch,
)
from eigendecomp.continuity import enforce_eigenvector_continuity
from eigendecomp.bootstrap import jackknife_effective_dim
from eigendecomp.flatten import flatten_result, flatten_batch

__all__ = [
    'compute_eigendecomp',
    'compute_eigendecomp_batch',
    'enforce_eigenvector_continuity',
    'jackknife_effective_dim',
    'flatten_result',
    'flatten_batch',
]
