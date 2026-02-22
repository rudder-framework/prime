"""
Divergence package for the Rudder Framework.

Directional information flow between signals:
- Granger causality (does A predict B beyond B's own history?)
- Transfer entropy (information flow A→B)
- KL/JS divergence between signal distributions

Only computed for pairs flagged by pairwise.coloading (needs_granger=True).
"""

from divergence.causality import compute_granger, compute_transfer_entropy
from divergence.divergence import kl_divergence, js_divergence

__all__ = ['compute_granger', 'compute_transfer_entropy', 'kl_divergence', 'js_divergence']
