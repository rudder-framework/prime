"""Engine: wavelet_stability — multi-scale energy distribution."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    keys = ['wavelet_energy_low', 'wavelet_energy_mid', 'wavelet_energy_high',
            'wavelet_energy_ratio', 'wavelet_entropy', 'wavelet_concentration',
            'wavelet_dominant_scale', 'wavelet_energy_drift',
            'wavelet_temporal_std', 'wavelet_intermittency']
    nan = {k: np.nan for k in keys}
    if len(y) < 32:
        return nan

    try:
        from pmtvs import wavelet_stability
        result = wavelet_stability(y)
        if not isinstance(result, dict):
            return nan
        return {
            'wavelet_energy_low': float(result.get('energy_low', np.nan)),
            'wavelet_energy_mid': float(result.get('energy_mid', np.nan)),
            'wavelet_energy_high': float(result.get('energy_high', np.nan)),
            'wavelet_energy_ratio': float(result.get('energy_ratio', np.nan)),
            'wavelet_entropy': float(result.get('entropy', np.nan)),
            'wavelet_concentration': float(result.get('concentration', np.nan)),
            'wavelet_dominant_scale': float(result.get('dominant_scale', np.nan)),
            'wavelet_energy_drift': float(result.get('energy_drift', np.nan)),
            'wavelet_temporal_std': float(result.get('temporal_std', np.nan)),
            'wavelet_intermittency': float(result.get('intermittency', np.nan)),
        }
    except (ImportError, Exception):
        return nan
