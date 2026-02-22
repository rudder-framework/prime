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
        from scipy.signal import hilbert
        analytic = hilbert(y)
        inst_amp = np.abs(analytic)
        phase = np.unwrap(np.angle(analytic))
        inst_freq = np.diff(phase) / (2 * np.pi)
        if len(inst_freq) < 4:
            return nan
        fm, fs = np.mean(inst_freq), np.std(inst_freq)
        if fs < 1e-15:
            return nan
        z = (inst_freq - fm) / fs
        am, ast = np.mean(inst_amp), np.std(inst_amp)
        return {
            'hilbert_freq_mean': float(fm),
            'hilbert_freq_std': float(fs),
            'hilbert_freq_stability': float(1.0 / (1.0 + fs)),
            'hilbert_freq_kurtosis': float(np.mean(z ** 4) - 3.0),
            'hilbert_freq_skewness': float(np.mean(z ** 3)),
            'hilbert_freq_range': float(np.max(inst_freq) - np.min(inst_freq)),
            'hilbert_freq_drift': float(np.polyfit(np.arange(len(inst_freq)), inst_freq, 1)[0]),
            'hilbert_amp_cv': float(ast / am) if am > 1e-15 else 0.0,
            'hilbert_amp_trend': float(np.polyfit(np.arange(len(inst_amp)), inst_amp, 1)[0]),
            'hilbert_phase_coherence': float(1.0 / (1.0 + fs)),
            'hilbert_am_fm_ratio': float(ast / fs) if fs > 1e-15 else 0.0,
        }
    except ImportError:
        return nan
