"""Engine: hilbert_stability — instantaneous frequency/amplitude analysis."""
import numpy as np
from typing import Dict


def compute(y: np.ndarray) -> Dict[str, float]:
    keys = ['hilbert_freq_mean', 'hilbert_freq_std', 'hilbert_freq_stability',
            'hilbert_freq_kurtosis', 'hilbert_freq_skewness', 'hilbert_freq_range',
            'hilbert_freq_drift', 'hilbert_amp_cv', 'hilbert_amp_trend',
            'hilbert_phase_coherence', 'hilbert_am_fm_ratio']
    nan = {k: np.nan for k in keys}
    if len(y) < 32:
        return nan

    try:
        from pmtvs import hilbert_stability
        result = hilbert_stability(y)
        if not isinstance(result, dict):
            return nan
        return {
            'hilbert_freq_mean': float(result.get('inst_freq_mean', np.nan)),
            'hilbert_freq_std': float(result.get('inst_freq_std', np.nan)),
            'hilbert_freq_stability': float(result.get('inst_freq_stability', np.nan)),
            'hilbert_freq_kurtosis': float(result.get('inst_freq_kurtosis', np.nan)),
            'hilbert_freq_skewness': float(result.get('inst_freq_skewness', np.nan)),
            'hilbert_freq_range': float(result.get('inst_freq_range', np.nan)),
            'hilbert_freq_drift': float(result.get('inst_freq_drift', np.nan)),
            'hilbert_amp_cv': float(result.get('inst_amp_cv', np.nan)),
            'hilbert_amp_trend': float(result.get('inst_amp_trend', np.nan)),
            'hilbert_phase_coherence': float(result.get('phase_coherence', np.nan)),
            'hilbert_am_fm_ratio': float(result.get('am_fm_ratio', np.nan)),
        }
    except (ImportError, Exception):
        return nan
