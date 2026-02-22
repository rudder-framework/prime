"""
Stability package for the Rudder Framework.

Per-signal rolling stability metrics:
- Hilbert envelope (instantaneous amplitude)
- Instantaneous frequency
- Wavelet energy
- Amplitude variance ratio (stability indicator)

These are signal-level features, not system-level.
"""

from stability.rolling import compute_signal_stability

__all__ = ['compute_signal_stability']
