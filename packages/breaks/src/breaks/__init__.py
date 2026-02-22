"""
Breaks package for the Rudder Framework.

Detects regime changes (structural breaks) in signals:
- CUSUM: cumulative sum test for mean shift
- Pettitt: nonparametric test for change point

Breaks feed into departure scoring — a signal that has undergone
a structural break is behaving differently from its baseline.
"""

from breaks.detection import detect_breaks_cusum, detect_breaks_pettitt

__all__ = ['detect_breaks_cusum', 'detect_breaks_pettitt']
