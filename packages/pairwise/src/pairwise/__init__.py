"""
Pairwise signal analysis for the Rudder Framework.

Computes relationships between all pairs of signals (or cohorts)
at each window index. Metrics include correlation, distance,
cosine similarity, cross-correlation lag, and mutual information.

N signals → C(N,2) pairs per window. For 14 sensors → 91 pairs.

Eigenvector co-loading gates expensive operations (Granger, DTW)
to pairs that actually share principal component variance.
"""

from pairwise.metrics import (
    compute_pair_metrics,
)
from pairwise.signal import (
    compute_signal_pairwise,
    compute_signal_pairwise_batch,
)
from pairwise.cohort import (
    compute_cohort_pairwise,
)
from pairwise.coloading import (
    compute_coloading_flags,
)

__all__ = [
    'compute_pair_metrics',
    'compute_signal_pairwise',
    'compute_signal_pairwise_batch',
    'compute_cohort_pairwise',
    'compute_coloading_flags',
]
