"""
Thermodynamics package for the Rudder Framework.

Statistical mechanics analogs from eigenvalue spectra:
- Entropy: Shannon entropy of eigenvalue distribution
- Energy: total_variance (sum of eigenvalues)
- Temperature: variance of effective_dim velocity
- Free energy: F = E - T·S
"""

from thermodynamics.thermo import compute_thermodynamics

__all__ = ['compute_thermodynamics']
